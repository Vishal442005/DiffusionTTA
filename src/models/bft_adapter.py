"""
Backpropagation-Free Test-Time Adaptation (MoE-BFT).

Performs real-time test-time adaptation on incoming unlabeled EEG trials with:
1. K=6 Neurophysiologically sound transformations.
2. Parallel forward passes through frozen Channel-Wise MoE backbone.
3. Source-trained Learning-to-Rank (LtR) reliability evaluation.
4. Confidence-weighted ensemble aggregation in <5ms per trial.

Zero gradient updates. Zero catastrophic forgetting.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.moe_backbone import ChannelWiseMoE


class NeuroTransformBank:
    """
    Neurophysiologically sound EEG transformation bank for test-time augmentation.
    
    Transformations:
    1. Spatial jitter (neighboring electrode linear interpolation)
    2. Narrowband phase shifting (FFT phase perturbation)
    3. Spectral masking (non-motor frequency attenuation)
    4. Time warping (slight temporal resampling)
    5. Amplitude scaling (electrode impedance drift simulation)
    6. Channel subset dropout (electrode disconnection simulation)
    """

    def __init__(
        self,
        sfreq: float = 160.0,
        spatial_jitter_factor: float = 0.05,
        phase_shift_max: float = math.pi / 6.0,
        time_warp_factor: float = 0.08,
        amplitude_scale_range: Tuple[float, float] = (0.85, 1.15),
        channel_dropout_ratio: float = 0.10,
    ) -> None:
        self.sfreq = sfreq
        self.spatial_jitter_factor = spatial_jitter_factor
        self.phase_shift_max = phase_shift_max
        self.time_warp_factor = time_warp_factor
        self.amplitude_scale_range = amplitude_scale_range
        self.channel_dropout_ratio = channel_dropout_ratio

    def spatial_jitter(self, x: torch.Tensor) -> torch.Tensor:
        """
        Spatial jitter: slight spatial blend with adjacent channels (simulates electrode shift).
        """
        # x: (B, C, T)
        alpha = self.spatial_jitter_factor
        rolled_fwd = torch.roll(x, shifts=1, dims=1)
        rolled_bwd = torch.roll(x, shifts=-1, dims=1)
        return (1.0 - alpha) * x + 0.5 * alpha * (rolled_fwd + rolled_bwd)

    def narrowband_phase_shift(self, x: torch.Tensor) -> torch.Tensor:
        """
        Narrowband phase shift: adds small random phase shift across frequencies in Fourier domain.
        Preserves power spectral density and ERD/ERS magnitude.
        """
        B, C, T = x.shape
        # Real FFT: (B, C, T_freq)
        x_fft = torch.fft.rfft(x, dim=-1)
        mag = torch.abs(x_fft)
        phase = torch.angle(x_fft)

        # Random phase delta in [-max_shift, max_shift]
        delta_phase = (torch.rand((B, C, 1), device=x.device) * 2.0 - 1.0) * self.phase_shift_max
        new_phase = phase + delta_phase

        # Polar back to complex
        x_fft_shifted = torch.polar(mag, new_phase)
        x_shifted = torch.fft.irfft(x_fft_shifted, n=T, dim=-1)
        return x_shifted

    def spectral_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Spectral masking: attenuates non-motor frequencies (e.g. sub-delta <1Hz, high gamma >35Hz).
        Emphasizes sensorimotor rhythms (mu 8-12Hz, beta 13-30Hz).
        """
        B, C, T = x.shape
        x_fft = torch.fft.rfft(x, dim=-1)
        freqs = torch.fft.rfftfreq(T, d=1.0 / self.sfreq).to(x.device)

        # Attenuation mask: attenuate noise bands (<1Hz and >35Hz)
        mask = torch.ones_like(freqs)
        mask[freqs < 1.0] = 0.2
        mask[freqs > 35.0] = 0.3

        x_fft_masked = x_fft * mask.view(1, 1, -1)
        return torch.fft.irfft(x_fft_masked, n=T, dim=-1)

    def time_warp(self, x: torch.Tensor) -> torch.Tensor:
        """
        Time warping: slight linear temporal stretching/compression (simulates cognitive latency jitter).
        """
        B, C, T = x.shape
        factor = 1.0 + (torch.rand(1).item() * 2.0 - 1.0) * self.time_warp_factor
        new_T = max(16, int(T * factor))

        # Interpolate along time
        x_reshaped = x.view(B * C, 1, T)
        x_warped = F.interpolate(x_reshaped, size=new_T, mode="linear", align_corners=False)
        # Resample back to original length T
        x_restored = F.interpolate(x_warped, size=T, mode="linear", align_corners=False)
        return x_restored.view(B, C, T)

    def amplitude_scale(self, x: torch.Tensor) -> torch.Tensor:
        """
        Amplitude scaling: gain jitter simulating electrode impedance and skull resistance drift.
        """
        B, C, T = x.shape
        low, high = self.amplitude_scale_range
        scales = torch.empty(B, C, 1, device=x.device).uniform_(low, high)
        return x * scales

    def channel_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """
        Channel subset dropout: randomly drops 10% of channels (simulates poor contact or detachment).
        """
        B, C, T = x.shape
        drop_mask = (torch.rand(B, C, 1, device=x.device) > self.channel_dropout_ratio).float()
        return x * drop_mask

    def apply_all(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Generate original trial + 6 transformed versions.
        Returns list of K+1 = 7 tensors, each of shape (B, C, T).
        """
        return [
            x,
            self.spatial_jitter(x),
            self.narrowband_phase_shift(x),
            self.spectral_mask(x),
            self.time_warp(x),
            self.amplitude_scale(x),
            self.channel_dropout(x),
        ]


class ReliabilityRanker(nn.Module):
    """
    Learning-to-Rank (LtR) neural module.
    Evaluates the prediction confidence, gating consistency, and representation quality
    of each candidate transformed trial to assign a non-negative reliability weight r_k.
    """

    def __init__(self, hidden_dim: int = 64, num_classes: int = 4, num_experts: int = 4) -> None:
        super().__init__()
        # Input features per prediction:
        # - Entropy of class probabilities (1)
        # - Maximum predicted probability (1)
        # - Class distribution vector (num_classes)
        # - Expert gating vector (num_experts)
        # - Representation norm (1)
        feature_dim = 1 + 1 + num_classes + num_experts + 1

        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),  # Ensures strictly positive reliability scores
        )

    def extract_features(
        self,
        probs: torch.Tensor,
        gates: torch.Tensor,
        rep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            probs: Class probabilities (N, num_classes)
            gates: Expert gating weights (N, num_experts)
            rep: Fused latent representation (N, hidden_dim)

        Returns:
            features: Tensor of shape (N, feature_dim)
        """
        eps = 1e-8
        entropy = -torch.sum(probs * torch.log(probs + eps), dim=-1, keepdim=True)
        max_prob, _ = torch.max(probs, dim=-1, keepdim=True)
        rep_norm = torch.norm(rep, p=2, dim=-1, keepdim=True)

        return torch.cat([entropy, max_prob, probs, gates, rep_norm], dim=-1)

    def forward(
        self,
        probs: torch.Tensor,
        gates: torch.Tensor,
        rep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            probs: (N, num_classes)
            gates: (N, num_experts)
            rep: (N, hidden_dim)

        Returns:
            scores: Reliability scores of shape (N, 1)
        """
        feats = self.extract_features(probs, gates, rep)
        return self.net(feats)


class BFTAdapter(nn.Module):
    """
    Backpropagation-Free Test-Time Adaptation on Channel-Wise MoE.

    Zero gradient updates at inference. Latency <5ms.
    Aggregates predictions from transformation bank using Learning-to-Rank reliability scores.
    """

    def __init__(
        self,
        moe_backbone: ChannelWiseMoE,
        hidden_dim: int = 64,
        num_classes: int = 4,
        ensemble_method: str = "confidence_weighted",
        transform_bank: Optional[NeuroTransformBank] = None,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.moe_backbone = moe_backbone
        # Optionally freeze backbone parameters (for inference/TTA)
        if freeze_backbone:
            for param in self.moe_backbone.parameters():
                param.requires_grad = False

        self.num_classes = num_classes
        self.ensemble_method = ensemble_method
        self.transforms = transform_bank or NeuroTransformBank()

        # Learning-to-Rank module
        self.ranker = ReliabilityRanker(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_experts=4,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_timing: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, float]]:
        """
        Perform backpropagation-free test-time adaptation for batch x.

        Args:
            x: Input EEG trials of shape (B, 64, T)
            return_timing: If True, returns elapsed latency in milliseconds.

        Returns:
            final_probs: Adapted class probability distribution (B, num_classes)
            (latency_ms if return_timing=True)
        """
        start_time = time.perf_counter()
        B, C, T = x.shape
        device = x.device

        # 1. Apply K=6 transformations -> K+1 = 7 variants
        # List of 7 tensors of shape (B, C, T)
        variants = self.transforms.apply_all(x)
        K = len(variants)

        # 2. Stack into single batched tensor for maximum GPU/CPU parallel throughput
        # Shape: (7 * B, C, T)
        x_stacked = torch.cat(variants, dim=0)

        # 3. Single forward pass through frozen MoE backbone
        with torch.no_grad():
            out = self.moe_backbone(x_stacked, return_details=True)

        all_probs = out["probs"]       # (7*B, num_classes)
        all_gates = out["gates"]       # (7*B, 4)
        all_rep = out["fused_rep"]     # (7*B, hidden_dim)

        # 4. Evaluate reliability of each candidate prediction
        # Shape: (7*B, 1)
        reliability = self.ranker(all_probs, all_gates, all_rep)

        # Reshape to (K, B, ...)
        probs_reshaped = all_probs.view(K, B, self.num_classes)   # (7, B, num_classes)
        rel_reshaped = reliability.view(K, B, 1)                  # (7, B, 1)

        # 5. Ensemble Aggregation
        if self.ensemble_method == "confidence_weighted":
            # Weighted average: sum_k (r_k * p_k) / sum_k (r_k)
            weights = rel_reshaped / (torch.sum(rel_reshaped, dim=0, keepdim=True) + 1e-8)
            final_probs = torch.sum(weights * probs_reshaped, dim=0)  # (B, num_classes)

        elif self.ensemble_method == "top_k":
            # Top-3 predictions by reliability
            _, top_indices = torch.topk(rel_reshaped.squeeze(-1), k=3, dim=0)  # (3, B)
            top_probs = torch.stack([probs_reshaped[top_indices[i, b], b] for i in range(3) for b in range(B)])
            final_probs = top_probs.view(3, B, self.num_classes).mean(dim=0)

        else:
            # Uniform ensemble
            final_probs = probs_reshaped.mean(dim=0)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if return_timing:
            return final_probs, elapsed_ms

        return final_probs

    def train_ranker_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        margin: float = 0.2,
    ) -> float:
        """
        Train the Learning-to-Rank module on source data using pairwise margin ranking loss.
        Pairs with higher ground-truth confidence get higher rank scores.
        """
        self.ranker.train()
        self.moe_backbone.eval()

        B = x.shape[0]
        variants = self.transforms.apply_all(x)
        x_stacked = torch.cat(variants, dim=0)

        with torch.no_grad():
            out = self.moe_backbone(x_stacked, return_details=True)

        all_probs = out["probs"]
        all_gates = out["gates"]
        all_rep = out["fused_rep"]

        scores = self.ranker(all_probs, all_gates, all_rep).squeeze(-1)  # (7*B,)

        # Ground truth target accuracy/confidence of each transformed trial
        y_repeated = y.repeat(len(variants))
        target_conf = all_probs[torch.arange(len(y_repeated)), y_repeated]  # (7*B,)

        # Create pairs across transformations
        scores_matrix = scores.view(len(variants), B)
        conf_matrix = target_conf.view(len(variants), B)

        # Pairwise loss between original (index 0) and transformed variants (indices 1..6)
        loss = torch.tensor(0.0, device=x.device, requires_grad=True)
        pair_count = 0

        for k in range(1, len(variants)):
            target_sign = torch.sign(conf_matrix[0] - conf_matrix[k])
            # If target_sign > 0, score 0 should be > score k
            valid = target_sign != 0
            if valid.sum() > 0:
                pair_loss = F.margin_ranking_loss(
                    scores_matrix[0][valid],
                    scores_matrix[k][valid],
                    target_sign[valid],
                    margin=margin,
                )
                loss = loss + pair_loss
                pair_count += 1

        if pair_count > 0:
            loss = loss / pair_count
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            return loss.item()

        return 0.0
