"""
Model architectures, autoencoders, diffusion networks, and adapters for Paper 2.
"""

from src.models.bft_adapter import BFTAdapter, NeuroTransformBank, ReliabilityRanker
from src.models.latent_diffusion import (
    EEG_VQVAE,
    ConditionalUNet1D,
    EEGLatentDiffusion,
    VectorQuantizer,
)
from src.models.lightning_module import (
    DiffusionLightningModule,
    EndToEndZeroCalibrationPipeline,
    MoEClassifierLightningModule,
    RestingStateLightningModule,
)
from src.models.moe_backbone import (
    ChannelWiseMoE,
    GlobalCoordinator,
    SensorimotorSpecialist,
    SpatialSpectralGating,
)
from src.models.resting_state_encoder import (
    RestingStateEncoder,
    SpatialCovarianceExtractor,
    SpectralFeatureExtractor,
)

__all__ = [
    "RestingStateEncoder",
    "SpectralFeatureExtractor",
    "SpatialCovarianceExtractor",
    "VectorQuantizer",
    "EEG_VQVAE",
    "ConditionalUNet1D",
    "EEGLatentDiffusion",
    "SensorimotorSpecialist",
    "GlobalCoordinator",
    "SpatialSpectralGating",
    "ChannelWiseMoE",
    "NeuroTransformBank",
    "ReliabilityRanker",
    "BFTAdapter",
    "RestingStateLightningModule",
    "DiffusionLightningModule",
    "MoEClassifierLightningModule",
    "EndToEndZeroCalibrationPipeline",
]
