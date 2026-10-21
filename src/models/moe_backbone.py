"""
Channel-Wise Mixture of Experts (MoE) Backbone for EEG Motor Imagery Classification.

Architecture:
- Expert 1: Left-hemisphere sensorimotor specialist (C3, FC3, CP3, etc.)
- Expert 2: Right-hemisphere sensorimotor specialist (C4, FC4, CP4, etc.)
- Expert 3: Midline sensorimotor specialist (Cz, FCz, CPz, Fz, Pz)
- Expert 4: Global spatio-temporal coordinator (all 64 channels)
- Gating: Spatial-spectral gating with lateralization-aware routing
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.physionet_loader import STANDARD_CHANNELS


class SensorimotorSpecialist(nn.Module):
    """
    Regional sensorimotor specialist network processing a localized subset of channels.
    Combines temporal frequency filtering, spatial depthwise convolution, and temporal pooling.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        kernel_length: int = 32,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        # Temporal convolution (extracts sensorimotor rhythm frequency patterns: mu and beta)
        self.temporal_conv = nn.Conv2d(1, 16, kernel_size=(1, kernel_length), padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # Spatial depthwise convolution across regional channels
        self.spatial_conv = nn.Conv2d(16, 32, kernel_size=(in_channels, 1), groups=16, bias=False)
        self.bn2 = nn.BatchNorm2d(32)

        # Temporal pooling & subsampling
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout1 = nn.Dropout(dropout)

        # Separable convolution for higher-level temporal dynamics
        self.separable_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 16), padding=(0, 8), groups=32, bias=False),
            nn.Conv2d(32, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Regional EEG tensor of shape (B, C_subset, T)

        Returns:
            h: Representation vector of shape (B, hidden_dim)
        """
        # Add channel dimension: (B, 1, C_subset, T)
        h = x.unsqueeze(1)
        h = self.bn1(self.temporal_conv(h))
        h = self.bn2(self.spatial_conv(h))
        h = F.gelu(h)
        h = self.dropout1(self.pool1(h))

        h = self.separable_conv(h)
        h = self.global_pool(h).squeeze(-1).squeeze(-1)  # (B, hidden_dim)
        return h


class GlobalCoordinator(nn.Module):
    """
    Global spatio-temporal coordinator expert operating on all 64 channels.
    Captures inter-hemispheric phase coupling, bilateral synchronization, and whole-head topology.
    """

    def __init__(
        self,
        in_channels: int = 64,
        hidden_dim: int = 64,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.Conv2d(24, 48, kernel_size=(in_channels, 1), bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(48, hidden_dim, kernel_size=(1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Whole-scalp EEG tensor of shape (B, 64, T)

        Returns:
            h: Representation vector of shape (B, hidden_dim)
        """
        h = x.unsqueeze(1)  # (B, 1, 64, T)
        h = self.conv1(h)
        h = self.conv2(h).squeeze(-1).squeeze(-1)
        return h


class SpatialSpectralGating(nn.Module):
    """
    Dynamic gating router based on spatial-spectral features and hemispheric lateralization.
    Computes routing weights g_k(x) = Softmax(W_g * phi(x) / tau).
    """

    def __init__(
        self,
        in_channels: int = 64,
        num_experts: int = 4,
        c3_idx: int = 8,
        c4_idx: int = 12,
        cz_idx: int = 10,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.c3_idx = c3_idx
        self.c4_idx = c4_idx
        self.cz_idx = cz_idx
        self.temperature = temperature

        # Input feature dimension:
        # 64 (channel variances) + 3 (C3, C4, Cz energy) + 1 (lateralization index) = 68
        feature_dim = in_channels + 4

        self.router = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, num_experts),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gating probabilities and routing logits.

        Args:
            x: EEG tensor of shape (B, 64, T)

        Returns:
            gates: Routing probabilities of shape (B, num_experts)
            logits: Unnormalized routing logits (B, num_experts)
        """
        # Channel-wise variances
        var = torch.var(x, dim=-1)  # (B, 64)

        # Hemispheric lateralization: (Var(C3) - Var(C4)) / (Var(C3) + Var(C4) + eps)
        var_c3 = var[:, self.c3_idx : self.c3_idx + 1]
        var_c4 = var[:, self.c4_idx : self.c4_idx + 1]
        var_cz = var[:, self.cz_idx : self.cz_idx + 1]
        lat_index = (var_c3 - var_c4) / (var_c3 + var_c4 + 1e-6)

        phi = torch.cat([var, var_c3, var_c4, var_cz, lat_index], dim=-1)
        logits = self.router(phi)
        gates = F.softmax(logits / self.temperature, dim=-1)

        return gates, logits


class ChannelWiseMoE(nn.Module):
    """
    Channel-Wise Mixture of Experts backbone for EEG Motor Imagery classification.

    Structure:
    - Expert 0: Left-hemisphere specialist (C3, FC3, CP3, etc.)
    - Expert 1: Right-hemisphere specialist (C4, FC4, CP4, etc.)
    - Expert 2: Midline specialist (Cz, FCz, CPz, etc.)
    - Expert 3: Global coordinator (all 64 channels)
    """

    def __init__(
        self,
        in_channels: int = 64,
        num_classes: int = 4,
        hidden_dim: int = 64,
        dropout: float = 0.25,
        temperature: float = 1.0,
        load_balance_weight: float = 0.01,
        channel_names: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.load_balance_weight = load_balance_weight

        channels = channel_names or STANDARD_CHANNELS
        ch_to_idx = {name: i for i, name in enumerate(channels)}

        # Define channel subsets for each specialist
        left_ch_names = ["FC5", "FC3", "FC1", "C5", "C3", "C1", "CP5", "CP3", "CP1"]
        right_ch_names = ["FC6", "FC4", "FC2", "C6", "C4", "C2", "CP6", "CP4", "CP2"]
        midline_ch_names = ["FCZ", "CZ", "CPZ", "FZ", "PZ"]

        self.left_indices = [ch_to_idx[ch] for ch in left_ch_names if ch in ch_to_idx]
        self.right_indices = [ch_to_idx[ch] for ch in right_ch_names if ch in ch_to_idx]
        self.midline_indices = [ch_to_idx[ch] for ch in midline_ch_names if ch in ch_to_idx]

        # Key landmark indices for gating
        c3_idx = ch_to_idx.get("C3", 8)
        c4_idx = ch_to_idx.get("C4", 12)
        cz_idx = ch_to_idx.get("CZ", 10)

        # 4 Specialists
        self.expert_left = SensorimotorSpecialist(
            in_channels=len(self.left_indices), hidden_dim=hidden_dim, dropout=dropout
        )
        self.expert_right = SensorimotorSpecialist(
            in_channels=len(self.right_indices), hidden_dim=hidden_dim, dropout=dropout
        )
        self.expert_midline = SensorimotorSpecialist(
            in_channels=len(self.midline_indices), hidden_dim=hidden_dim, dropout=dropout
        )
        self.expert_global = GlobalCoordinator(
            in_channels=in_channels, hidden_dim=hidden_dim, dropout=dropout
        )

        self.experts = nn.ModuleList([
            self.expert_left,
            self.expert_right,
            self.expert_midline,
            self.expert_global,
        ])

        # Gating network
        self.gating = SpatialSpectralGating(
            in_channels=in_channels,
            num_experts=4,
            c3_idx=c3_idx,
            c4_idx=c4_idx,
            cz_idx=cz_idx,
            temperature=temperature,
        )

        # Combined Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # Individual expert classification heads for auxiliary supervision and BFT analysis
        self.expert_classifiers = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes) for _ in range(4)
        ])

    def forward(
        self,
        x: torch.Tensor,
        return_details: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through Channel-Wise MoE.

        Args:
            x: EEG tensor (B, 64, T)
            return_details: If True, returns dictionary with gating values and individual expert logits.

        Returns:
            logits: Final class logits (B, num_classes)
            (or Dict if return_details=True)
        """
        B = x.shape[0]

        # Route channel subsets to regional specialists
        x_left = x[:, self.left_indices, :]
        x_right = x[:, self.right_indices, :]
        x_midline = x[:, self.midline_indices, :]
        x_global = x

        h_left = self.expert_left(x_left)        # (B, hidden_dim)
        h_right = self.expert_right(x_right)      # (B, hidden_dim)
        h_midline = self.expert_midline(x_midline)# (B, hidden_dim)
        h_global = self.expert_global(x_global)    # (B, hidden_dim)

        h_stack = torch.stack([h_left, h_right, h_midline, h_global], dim=1)  # (B, 4, hidden_dim)

        # Gating weights: (B, 4)
        gates, gate_logits = self.gating(x)

        # Combined representation: sum_k g_k * h_k
        h_fused = torch.sum(gates.unsqueeze(-1) * h_stack, dim=1)  # (B, hidden_dim)

        # Final classification logits
        logits = self.classifier(h_fused)

        if not return_details:
            return logits

        # Individual expert predictions
        expert_logits = [self.expert_classifiers[i](h_stack[:, i, :]) for i in range(4)]
        expert_logits_tensor = torch.stack(expert_logits, dim=1)  # (B, 4, num_classes)

        return {
            "logits": logits,
            "probs": F.softmax(logits, dim=-1),
            "gates": gates,
            "gate_logits": gate_logits,
            "expert_logits": expert_logits_tensor,
            "expert_reps": h_stack,
            "fused_rep": h_fused,
        }

    def compute_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        label_smoothing: float = 0.05,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total MoE loss: Cross-Entropy + Auxiliary Load Balancing Loss.
        """
        out = self.forward(x, return_details=True)
        logits = out["logits"]
        gates = out["gates"]

        # Classification Cross-Entropy
        cls_loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)

        # Load balancing loss: encourages uniform gate utilization across batch
        mean_gates = torch.mean(gates, dim=0)  # (4,)
        load_balance_loss = 4.0 * torch.sum(mean_gates ** 2)

        total_loss = cls_loss + self.load_balance_weight * load_balance_loss

        return {
            "loss": total_loss,
            "cls_loss": cls_loss,
            "balance_loss": load_balance_loss,
            "logits": logits,
            "gates": gates,
        }
