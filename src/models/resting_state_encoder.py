"""
Resting-State Spatial-Spectral Variational Autoencoder (VAE).

Encodes 30s of passive resting-state EEG into a compact subject-style vector s in R^d.
Captures individual skull attenuation, electrode impedance patterns, and individual alpha
peak frequency (IAF) to condition the latent diffusion model.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralFeatureExtractor(nn.Module):
    """
    Extracts multi-band power spectral densities and individual alpha peak features (IAF)
    from resting-state EEG using differentiable FFT operations.
    """

    def __init__(
        self,
        sfreq: float = 160.0,
        n_channels: int = 64,
        bands: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> None:
        super().__init__()
        self.sfreq = sfreq
        self.n_channels = n_channels
        self.bands = bands or {
            "delta": (0.5, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 13.0),
            "beta": (13.0, 30.0),
            "gamma": (30.0, 45.0),
        }
        self.num_bands = len(self.bands)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract band power from resting EEG.

        Args:
            x: Tensor of shape (B, C, T)

        Returns:
            band_powers: Tensor of shape (B, C, num_bands)
        """
        B, C, T = x.shape
        # Real FFT along time dimension
        fft_vals = torch.fft.rfft(x, dim=-1)
        psd = (torch.abs(fft_vals) ** 2) / T  # (B, C, T_freq)
        freqs = torch.fft.rfftfreq(T, d=1.0 / self.sfreq).to(x.device)

        band_powers = []
        for low, high in self.bands.values():
            mask = (freqs >= low) & (freqs < high)
            if mask.sum() > 0:
                # Mean power in band, log-transformed for scale stabilization
                p = torch.log(torch.mean(psd[:, :, mask], dim=-1) + 1e-8)
            else:
                p = torch.zeros(B, C, device=x.device)
            band_powers.append(p)

        # Shape: (B, C, num_bands)
        return torch.stack(band_powers, dim=-1)


class SpatialCovarianceExtractor(nn.Module):
    """
    Extracts normalized spatial covariance matrices capturing inter-electrode volume conduction
    and individual skull attenuation profiles.
    """

    def __init__(self, n_channels: int = 64) -> None:
        super().__init__()
        self.n_channels = n_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial covariance matrix.

        Args:
            x: Tensor of shape (B, C, T)

        Returns:
            cov: Normalized covariance matrix of shape (B, C, C)
        """
        B, C, T = x.shape
        x_centered = x - torch.mean(x, dim=-1, keepdim=True)
        cov = torch.bmm(x_centered, x_centered.transpose(1, 2)) / (T - 1)
        # Trace normalization for scale invariance
        trace = torch.diagonal(cov, dim1=1, dim2=2).sum(-1, keepdim=True).unsqueeze(-1) + 1e-6
        cov_norm = cov / trace
        return cov_norm


class RestingStateEncoder(nn.Module):
    """
    Spatial-Spectral Variational Autoencoder (VAE) for resting-state EEG.

    Compresses 30s passive resting-state EEG (shape: B x 64 x 4800) into a subject-style
    embedding vector s in R^d.
    """

    def __init__(
        self,
        in_channels: int = 64,
        latent_dim: int = 64,
        sfreq: float = 160.0,
        spatial_hidden_dim: int = 128,
        spectral_hidden_dim: int = 128,
        fusion_hidden_dim: int = 256,
        kl_weight: float = 1e-4,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.sfreq = sfreq
        self.kl_weight = float(kl_weight)

        # Extractors
        self.spectral_extractor = SpectralFeatureExtractor(sfreq=sfreq, n_channels=in_channels)
        self.spatial_extractor = SpatialCovarianceExtractor(n_channels=in_channels)
        self.num_bands = self.spectral_extractor.num_bands

        # Spatial Encoder: processes C x C covariance matrix
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, spatial_hidden_dim),
            nn.LayerNorm(spatial_hidden_dim),
            nn.GELU(),
        )

        # Spectral Encoder: processes (C x num_bands) spectral profile
        self.spectral_encoder = nn.Sequential(
            nn.Linear(in_channels * self.num_bands, spectral_hidden_dim),
            nn.LayerNorm(spectral_hidden_dim),
            nn.GELU(),
            nn.Linear(spectral_hidden_dim, spectral_hidden_dim),
            nn.LayerNorm(spectral_hidden_dim),
            nn.GELU(),
        )

        # Fusion & Bottleneck
        self.fusion = nn.Sequential(
            nn.Linear(spatial_hidden_dim + spectral_hidden_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
        )

        self.fc_mu = nn.Linear(fusion_hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(fusion_hidden_dim, latent_dim)

        # Decoders for self-supervised VAE reconstruction
        self.decoder_fusion = nn.Sequential(
            nn.Linear(latent_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
        )

        # Spatial Decoder (reconstructs C x C normalized covariance)
        self.spatial_decoder = nn.Sequential(
            nn.Linear(fusion_hidden_dim, 64 * 16 * 16),
            nn.GELU(),
            nn.Unflatten(1, (64, 16, 16)),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # (32, 32, 32)
            nn.GELU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),   # (1, 64, 64)
        )

        # Spectral Decoder (reconstructs C x num_bands power profile)
        self.spectral_decoder = nn.Sequential(
            nn.Linear(fusion_hidden_dim, spectral_hidden_dim),
            nn.GELU(),
            nn.Linear(spectral_hidden_dim, in_channels * self.num_bands),
            nn.Unflatten(1, (in_channels, self.num_bands)),
        )

    def encode(self, x: torch.Tensor, sample: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode resting EEG to latent subject-style vector.

        Args:
            x: EEG tensor (B, C, T)
            sample: If True, sample from q(s|x); if False, return mean mu.

        Returns:
            s: Latent subject style vector (B, latent_dim)
            mu: Mean vector (B, latent_dim)
            logvar: Log-variance vector (B, latent_dim)
        """
        # 1. Spatial Covariance
        cov = self.spatial_extractor(x).unsqueeze(1)  # (B, 1, C, C)
        h_spatial = self.spatial_encoder(cov)          # (B, spatial_hidden_dim)

        # 2. Spectral Power
        spec = self.spectral_extractor(x)              # (B, C, num_bands)
        spec_flat = spec.flatten(start_dim=1)
        h_spectral = self.spectral_encoder(spec_flat)  # (B, spectral_hidden_dim)

        # 3. Fusion & VAE Bottleneck
        h = self.fusion(torch.cat([h_spatial, h_spectral], dim=-1))
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), min=-10.0, max=10.0)

        if sample or self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            s = mu + eps * std
        else:
            s = mu

        return s, mu, logvar

    def decode(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode subject-style vector into reconstructed spatial covariance and spectral profile.

        Args:
            s: Subject style vector (B, latent_dim)

        Returns:
            recon_cov: Reconstructed covariance (B, C, C)
            recon_spec: Reconstructed spectral profile (B, C, num_bands)
        """
        h_dec = self.decoder_fusion(s)
        recon_cov = self.spatial_decoder(h_dec).squeeze(1)  # (B, C, C)
        # Symmetrize reconstructed covariance
        recon_cov = 0.5 * (recon_cov + recon_cov.transpose(1, 2))
        recon_spec = self.spectral_decoder(h_dec)           # (B, C, num_bands)
        return recon_cov, recon_spec

    def forward(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with autoencoder reconstruction.

        Args:
            x: Resting EEG tensor (B, C, T)

        Returns:
            Dict containing:
                - 's': Subject style vector (B, latent_dim)
                - 'mu': Latent mean
                - 'logvar': Latent log-variance
                - 'recon_cov': Reconstructed spatial covariance
                - 'recon_spec': Reconstructed spectral power profile
                - 'target_cov': Target spatial covariance
                - 'target_spec': Target spectral power profile
        """
        s, mu, logvar = self.encode(x, sample=self.training)
        recon_cov, recon_spec = self.decode(s)

        target_cov = self.spatial_extractor(x)
        target_spec = self.spectral_extractor(x)

        return {
            "s": s,
            "mu": mu,
            "logvar": logvar,
            "recon_cov": recon_cov,
            "recon_spec": recon_spec,
            "target_cov": target_cov,
            "target_spec": target_spec,
        }

    def compute_loss(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute total VAE loss: Spatial MSE + Spectral MSE + KL Divergence.
        """
        recon_cov_loss = F.mse_loss(outputs["recon_cov"], outputs["target_cov"])
        recon_spec_loss = F.mse_loss(outputs["recon_spec"], outputs["target_spec"])
        recon_loss = recon_cov_loss + recon_spec_loss

        # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        mu = outputs["mu"]
        logvar = outputs["logvar"]
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))

        total_loss = recon_loss + self.kl_weight * kl_loss

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "cov_loss": recon_cov_loss,
            "spec_loss": recon_spec_loss,
            "kl_loss": kl_loss,
        }
