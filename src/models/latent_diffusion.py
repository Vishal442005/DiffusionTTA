"""
Latent Diffusion Model for Personalized EEG Motor Imagery Synthesis.

Architecture:
1. EEG VQ-VAE: Compresses (B, 64, 640) multi-channel EEG into latent space (B, 16, 160).
2. Conditional 1D Latent U-Net Denoiser with AdaGN/FiLM conditioning on:
   - Diffusion timestep t
   - Motor Imagery class label y in {0, 1, 2, 3}
   - Passive resting-state subject style vector s in R^d
3. DDPM noise schedule + DDIM fast sampling (4 to 20 steps).
4. Classifier-Free Guidance (CFG).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. Sinusoidal Positional Embeddings
# ==============================================================================

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# ==============================================================================
# 2. Vector Quantizer & VQ-VAE for EEG
# ==============================================================================

class VectorQuantizer(nn.Module):
    """
    Vector Quantizer with straight-through gradient estimation.
    """

    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 16,
        commitment_cost: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize continuous latent tensor.

        Args:
            z: Latent tensor of shape (B, C_latent, T_latent)

        Returns:
            z_q: Quantized latent tensor (B, C_latent, T_latent)
            loss: VQ + commitment loss
            encoding_indices: Codebook indices (B, T_latent)
        """
        # Permute to (B, T, C) for codebook distance computation
        z_perm = z.permute(0, 2, 1).contiguous()
        flat_z = z_perm.view(-1, self.embedding_dim)

        # Distances: (z - e)^2 = z^2 + e^2 - 2 z e
        d = (
            torch.sum(flat_z ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_z, self.embedding.weight.t())
        )

        encoding_indices = torch.argmin(d, dim=1)
        quantized = self.embedding(encoding_indices).view(z_perm.shape)

        # Commitment loss + embedding loss
        loss = torch.mean((quantized.detach() - z_perm) ** 2) + self.commitment_cost * torch.mean(
            (quantized - z_perm.detach()) ** 2
        )

        # Straight-through gradient estimator
        quantized = z_perm + (quantized - z_perm).detach()

        # Permute back to (B, C, T)
        z_q = quantized.permute(0, 2, 1).contiguous()
        encoding_indices = encoding_indices.view(z.shape[0], z.shape[2])

        return z_q, loss, encoding_indices


class EEG_VQVAE(nn.Module):
    """
    1D Residual Convolutional VQ-VAE for compressing multi-channel EEG.
    Downsamples temporal dimension by factor of 4 (e.g. 640 -> 160).
    """

    def __init__(
        self,
        in_channels: int = 64,
        latent_channels: int = 16,
        num_embeddings: int = 512,
        commitment_cost: float = 0.25,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels

        # Encoder: (B, 64, 640) -> (B, 64, 320) -> (B, 32, 160) -> (B, 16, 160)
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, latent_channels, kernel_size=3, stride=1, padding=1),
        )

        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=latent_channels,
            commitment_cost=commitment_cost,
        )

        # Decoder: (B, 16, 160) -> (B, 32, 320) -> (B, 64, 640)
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.ConvTranspose1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.ConvTranspose1d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, in_channels, kernel_size=5, stride=1, padding=2),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_e = self.encoder(x)
        z_q, vq_loss, indices = self.quantizer(z_e)
        return z_q, vq_loss, indices

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_q)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_q, vq_loss, indices = self.encode(x)
        x_recon = self.decode(z_q)
        recon_loss = F.mse_loss(x_recon, x)
        total_loss = recon_loss + vq_loss
        return x_recon, total_loss, recon_loss


# ==============================================================================
# 3. Conditional 1D U-Net Denoiser (AdaGN / FiLM)
# ==============================================================================

class ResBlock1D(nn.Module):
    """
    1D Residual Block with Adaptive Group Normalization (AdaGN) for condition injection.
    """

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=min(8, in_ch), num_channels=in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

        # AdaGN condition projection produces scale and shift parameters
        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, out_ch * 2),
        )

        self.residual_conv = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = self.norm2(h)

        # Apply FiLM / AdaGN conditioning
        cond_out = self.cond_proj(cond).unsqueeze(-1)  # (B, 2 * out_ch, 1)
        scale, shift = torch.chunk(cond_out, 2, dim=1)
        h = h * (1.0 + scale) + shift

        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.residual_conv(x)


class SelfAttention1D(nn.Module):
    """Multi-Head Self-Attention along 1D temporal dimension."""

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.norm = nn.GroupNorm(min(8, channels), channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        x_norm = self.norm(x)
        # Permute to (B, T, C) for PyTorch MHA
        x_perm = x_norm.permute(0, 2, 1)
        attn_out, _ = self.mha(x_perm, x_perm, x_perm)
        attn_out = attn_out.permute(0, 2, 1)
        return x + attn_out


class ConditionalUNet1D(nn.Module):
    """
    1D U-Net Denoiser operating on latent EEG tensors.
    Conditioned on timestep t, MI class y, and subject resting-state style vector s.
    """

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        base_channels: int = 64,
        channel_mult: Tuple[int, ...] = (1, 2, 4),
        time_emb_dim: int = 128,
        style_emb_dim: int = 64,
        num_classes: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        cond_dim = time_emb_dim

        # Timestep embedding
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Class embedding (num_classes + 1 for null token in CFG)
        self.class_embed = nn.Embedding(num_classes + 1, time_emb_dim)

        # Subject style vector projection
        self.style_proj = nn.Sequential(
            nn.Linear(style_emb_dim, time_emb_dim),
            nn.LayerNorm(time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Condition fusion MLP
        self.cond_fusion = nn.Sequential(
            nn.Linear(time_emb_dim * 3, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # Input projection
        self.init_conv = nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1)

        # Downsampling path
        # Level 0: base_channels * 1
        # Level 1: base_channels * 2
        # Level 2: base_channels * 4
        c1 = base_channels * channel_mult[0]  # 64
        c2 = base_channels * channel_mult[1]  # 128
        c3 = base_channels * channel_mult[2]  # 256

        self.down_block0 = ResBlock1D(base_channels, c1, cond_dim, dropout)
        self.down_conv0 = nn.Conv1d(c1, c1, kernel_size=4, stride=2, padding=1)  # 160 -> 80

        self.down_block1 = ResBlock1D(c1, c2, cond_dim, dropout)
        self.down_conv1 = nn.Conv1d(c2, c2, kernel_size=4, stride=2, padding=1)  # 80 -> 40

        self.down_block2 = ResBlock1D(c2, c3, cond_dim, dropout)
        self.down_attn2 = SelfAttention1D(c3)

        # Mid Block
        self.mid_block1 = ResBlock1D(c3, c3, cond_dim, dropout)
        self.mid_attn = SelfAttention1D(c3)
        self.mid_block2 = ResBlock1D(c3, c3, cond_dim, dropout)

        # Upsampling path
        self.up_conv1 = nn.ConvTranspose1d(c3, c2, kernel_size=4, stride=2, padding=1)  # 40 -> 80
        self.up_block1 = ResBlock1D(c2 + c2, c2, cond_dim, dropout)

        self.up_conv0 = nn.ConvTranspose1d(c2, c1, kernel_size=4, stride=2, padding=1)  # 80 -> 160
        self.up_block0 = ResBlock1D(c1 + c1, c1, cond_dim, dropout)

        # Final projection
        self.final_norm = nn.GroupNorm(min(8, c1), c1)
        self.final_conv = nn.Conv1d(c1, out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        s: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict noise epsilon_theta(z_t, t, y, s).

        Args:
            z_t: Latent tensor (B, C_latent, T_latent)
            t: Timesteps (B,)
            y: Class labels (B,) with null class = self.num_classes
            s: Subject style vectors (B, style_emb_dim)
        """
        # Condition embedding fusion
        t_emb = self.time_embed(t)
        y_emb = self.class_embed(y)
        s_emb = self.style_proj(s)

        cond = self.cond_fusion(torch.cat([t_emb, y_emb, s_emb], dim=-1))

        # Down path
        x0 = self.init_conv(z_t)
        d0 = self.down_block0(x0, cond)
        x1 = self.down_conv0(d0)

        d1 = self.down_block1(x1, cond)
        x2 = self.down_conv1(d1)

        d2 = self.down_block2(x2, cond)
        d2 = self.down_attn2(d2)

        # Mid path
        m = self.mid_block1(d2, cond)
        m = self.mid_attn(m)
        m = self.mid_block2(m, cond)

        # Up path
        u1 = self.up_conv1(m)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.up_block1(u1, cond)

        u0 = self.up_conv0(u1)
        u0 = torch.cat([u0, d0], dim=1)
        u0 = self.up_block0(u0, cond)

        out = self.final_conv(F.gelu(self.final_norm(u0)))
        return out


# ==============================================================================
# 4. Latent Diffusion Engine & DDIM Fast Sampler
# ==============================================================================

def make_beta_schedule(schedule: str = "cosine", timesteps: int = 1000) -> torch.Tensor:
    """Generate linear or cosine beta schedule."""
    if schedule == "linear":
        return torch.linspace(0.0001, 0.02, timesteps)
    elif schedule == "cosine":
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / (1 + 0.008) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)
    else:
        raise ValueError(f"Unknown beta schedule: {schedule}")


class EEGLatentDiffusion(nn.Module):
    """
    Complete Latent Diffusion Pipeline for EEG Motor Imagery personalization.
    Combines VQ-VAE + 1D Conditional Latent U-Net + DDPM / DDIM engine.
    """

    def __init__(
        self,
        in_channels: int = 64,
        latent_channels: int = 16,
        style_dim: int = 64,
        num_classes: int = 4,
        timesteps: int = 1000,
        beta_schedule: str = "cosine",
        ddim_steps: int = 20,
        guidance_scale: float = 2.5,
        uncond_prob: float = 0.15,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.style_dim = style_dim
        self.num_classes = num_classes
        self.timesteps = timesteps
        self.ddim_steps = ddim_steps
        self.guidance_scale = guidance_scale
        self.uncond_prob = uncond_prob

        # Pretrained or jointly trained VQ-VAE
        self.vqvae = EEG_VQVAE(
            in_channels=in_channels,
            latent_channels=latent_channels,
            num_embeddings=512,
            commitment_cost=0.25,
        )

        # 1D Latent U-Net Denoiser
        self.unet = ConditionalUNet1D(
            in_channels=latent_channels,
            out_channels=latent_channels,
            base_channels=64,
            channel_mult=(1, 2, 4),
            time_emb_dim=128,
            style_emb_dim=style_dim,
            num_classes=num_classes,
        )

        # Diffusion parameters
        betas = make_beta_schedule(schedule=beta_schedule, timesteps=timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def q_sample(self, z_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward diffusion process q(z_t | z_0)."""
        if noise is None:
            noise = torch.randn_like(z_0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sqrt_alpha * z_0 + sqrt_one_minus_alpha * noise

    def compute_loss(
        self,
        x_real: torch.Tensor,
        y: torch.Tensor,
        subject_style: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total training loss: VQ-VAE loss + Latent Diffusion denoising loss.

        Args:
            x_real: Real EEG trial (B, 64, 640)
            y: Class labels (B,)
            subject_style: Subject resting style vector (B, style_dim)
        """
        B = x_real.shape[0]
        device = x_real.device

        # 1. Encode real EEG into continuous latent
        with torch.no_grad():
            z_0, _, _ = self.vqvae.encode(x_real)

        # 2. Sample random timesteps
        t = torch.randint(0, self.timesteps, (B,), device=device).long()
        noise = torch.randn_like(z_0)
        z_t = self.q_sample(z_0, t, noise)

        # 3. Classifier-Free Guidance dropout during training
        # Drop conditioning with probability uncond_prob
        y_cond = y.clone()
        s_cond = subject_style.clone()
        drop_mask = torch.rand(B, device=device) < self.uncond_prob
        y_cond[drop_mask] = self.num_classes  # Null class token
        s_cond[drop_mask] = 0.0                # Null style vector

        # 4. Predict noise with U-Net
        noise_pred = self.unet(z_t, t, y_cond, s_cond)
        diff_loss = F.mse_loss(noise_pred, noise)

        # Also compute VQ-VAE reconstruction loss
        x_recon, vq_total_loss, recon_loss = self.vqvae(x_real)

        total_loss = diff_loss + 0.1 * vq_total_loss

        return {
            "loss": total_loss,
            "diffusion_loss": diff_loss,
            "vq_loss": vq_total_loss,
            "recon_loss": recon_loss,
        }

    @torch.no_grad()
    def sample(
        self,
        subject_style: torch.Tensor,
        class_labels: torch.Tensor,
        num_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Generate subject-personalized synthetic EEG motor imagery trials using DDIM fast sampling.

        Args:
            subject_style: Target subject style vector (B, style_dim)
            class_labels: Target MI classes (B,)
            num_steps: Number of DDIM steps (default self.ddim_steps = 20)
            guidance_scale: Classifier-free guidance scale (default self.guidance_scale = 2.5)

        Returns:
            x_synthetic: Synthesized EEG trials of shape (B, 64, 640)
        """
        B = subject_style.shape[0]
        device = subject_style.device
        steps = num_steps or self.ddim_steps
        cfg_scale = guidance_scale if guidance_scale is not None else self.guidance_scale

        # Start from pure Gaussian noise in latent space (B, 16, 160)
        z = torch.randn(B, self.latent_channels, 160, device=device)

        # DDIM timestep sequence
        times = torch.linspace(self.timesteps - 1, 0, steps + 1).long().to(device)

        # Null conditioning for CFG
        null_labels = torch.full((B,), self.num_classes, device=device, dtype=torch.long)
        null_style = torch.zeros_like(subject_style)

        for i in range(len(times) - 1):
            t_curr = times[i].repeat(B)
            t_prev = times[i + 1].repeat(B)

            alpha_curr = self.alphas_cumprod[t_curr].view(-1, 1, 1)
            alpha_prev = self.alphas_cumprod[t_prev].view(-1, 1, 1) if times[i + 1] >= 0 else torch.ones_like(alpha_curr)

            # Conditional and Unconditional noise predictions
            noise_cond = self.unet(z, t_curr, class_labels, subject_style)
            noise_uncond = self.unet(z, t_curr, null_labels, null_style)

            # Classifier-Free Guidance formula
            noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)

            # DDIM deterministic update step (eta = 0)
            pred_z0 = (z - torch.sqrt(1.0 - alpha_curr) * noise_pred) / torch.sqrt(alpha_curr)
            dir_zt = torch.sqrt(1.0 - alpha_prev) * noise_pred
            z = torch.sqrt(alpha_prev) * pred_z0 + dir_zt

        # Decode quantized/continuous latent z to raw EEG trials
        x_synthetic = self.vqvae.decode(z)
        return x_synthetic
