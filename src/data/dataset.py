"""
PyTorch Datasets and Lightning DataModule for Paper 2 (Rest-Diff-BCI + MoE-BFT).

Supports:
1. Standard Motor Imagery (MI) trials with subject labels.
2. Resting-state segments paired with subject MI trials.
3. Mixed real and synthetic trial datasets for diffusion-augmented classifier training.
4. PyTorch Lightning DataModule for reproducible experiment workflows.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.physionet_loader import PhysioNetLoader

logger = logging.getLogger(__name__)


def z_score_normalize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply channel-wise Z-score normalization along temporal dimension.
    
    Args:
        x: EEG array of shape (..., C, T)
        eps: Small constant for numerical stability
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True) + eps
    return (x - mean) / std


class EEGDataset(Dataset):
    """
    PyTorch Dataset for EEG Motor Imagery trials paired with resting-state baseline EEG.
    """

    def __init__(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        subject_ids: Optional[Union[np.ndarray, torch.Tensor]] = None,
        resting_dict: Optional[Dict[int, Union[np.ndarray, torch.Tensor]]] = None,
        normalize: bool = True,
    ) -> None:
        """
        Args:
            X: EEG trials of shape (N, C, T)
            y: Class labels of shape (N,)
            subject_ids: Subject indices of shape (N,)
            resting_dict: Dictionary mapping subject_id -> resting state tensor (C, T_rest)
            normalize: Whether to apply temporal z-score normalization
        """
        if isinstance(X, np.ndarray):
            X_arr = z_score_normalize(X) if normalize else X
            self.X = torch.from_numpy(X_arr).float()
        else:
            self.X = X.float()

        if isinstance(y, np.ndarray):
            self.y = torch.from_numpy(y).long()
        else:
            self.y = y.long()

        if subject_ids is not None:
            if isinstance(subject_ids, np.ndarray):
                self.subject_ids = torch.from_numpy(subject_ids).long()
            else:
                self.subject_ids = subject_ids.long()
        else:
            self.subject_ids = torch.zeros(len(self.X), dtype=torch.long)

        # Store resting state EEG per subject
        self.resting_dict: Dict[int, torch.Tensor] = {}
        if resting_dict is not None:
            for sub_id, rest in resting_dict.items():
                if isinstance(rest, np.ndarray):
                    rest_arr = z_score_normalize(rest) if normalize else rest
                    self.resting_dict[int(sub_id)] = torch.from_numpy(rest_arr).float()
                else:
                    self.resting_dict[int(sub_id)] = rest.float()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        trial = self.X[idx]
        label = self.y[idx]
        sub_id = self.subject_ids[idx].item()

        item = {
            "eeg": trial,
            "label": label,
            "subject_id": torch.tensor(sub_id, dtype=torch.long),
        }

        # Include resting state if available
        if sub_id in self.resting_dict:
            item["resting"] = self.resting_dict[sub_id]
        else:
            # Fallback zero tensor of resting length (default 4800 samples)
            item["resting"] = torch.zeros((trial.shape[0], 4800), dtype=torch.float32)

        return item


class RestingStateDataset(Dataset):
    """
    Dataset containing continuous or windowed resting-state EEG segments across subjects.
    Used for training the spatial-spectral VAE subject-style encoder.
    """

    def __init__(
        self,
        resting_dict: Dict[int, Union[np.ndarray, torch.Tensor]],
        window_samples: int = 4800,
        normalize: bool = True,
    ) -> None:
        """
        Args:
            resting_dict: Mapping from subject_id to resting EEG (C, T)
            window_samples: Target sample window (e.g. 30s * 160Hz = 4800)
            normalize: Apply channel-wise z-score normalization
        """
        self.data: List[torch.Tensor] = []
        self.subject_ids: List[int] = []

        for sub_id, rest in resting_dict.items():
            if isinstance(rest, np.ndarray):
                rest_arr = z_score_normalize(rest) if normalize else rest
                rest_tensor = torch.from_numpy(rest_arr).float()
            else:
                rest_tensor = rest.float()

            C, T = rest_tensor.shape
            if T >= window_samples:
                # Can extract non-overlapping windows or single centered window
                start = (T - window_samples) // 2
                window = rest_tensor[:, start : start + window_samples]
                self.data.append(window)
                self.subject_ids.append(int(sub_id))
            else:
                pad = window_samples - T
                window = torch.nn.functional.pad(rest_tensor, (0, pad), mode="replicate")
                self.data.append(window)
                self.subject_ids.append(int(sub_id))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "resting": self.data[idx],
            "subject_id": torch.tensor(self.subject_ids[idx], dtype=torch.long),
        }


class MixedEEGDataset(Dataset):
    """
    Dataset combining real EEG trials and diffusion-generated synthetic EEG trials
    to train downstream classifiers with controlled synthetic augmentation ratios.
    """

    def __init__(
        self,
        real_X: Union[np.ndarray, torch.Tensor],
        real_y: Union[np.ndarray, torch.Tensor],
        synthetic_X: Optional[Union[np.ndarray, torch.Tensor]] = None,
        synthetic_y: Optional[Union[np.ndarray, torch.Tensor]] = None,
        synthetic_ratio: float = 0.5,
        normalize: bool = True,
    ) -> None:
        """
        Args:
            real_X: Real EEG trials (N_real, C, T)
            real_y: Real labels (N_real,)
            synthetic_X: Generated synthetic EEG trials (N_syn, C, T)
            synthetic_y: Generated synthetic labels (N_syn,)
            synthetic_ratio: Desired proportion of synthetic trials in mixed batches (0.0 to 1.0)
            normalize: Apply normalization
        """
        if isinstance(real_X, np.ndarray):
            real_X = z_score_normalize(real_X) if normalize else real_X
            self.real_X = torch.from_numpy(real_X).float()
        else:
            self.real_X = real_X.float()

        if isinstance(real_y, np.ndarray):
            self.real_y = torch.from_numpy(real_y).long()
        else:
            self.real_y = real_y.long()

        if synthetic_X is not None and len(synthetic_X) > 0 and synthetic_ratio > 0.0:
            if isinstance(synthetic_X, np.ndarray):
                synthetic_X = z_score_normalize(synthetic_X) if normalize else synthetic_X
                self.syn_X = torch.from_numpy(synthetic_X).float()
            else:
                self.syn_X = synthetic_X.float()

            if isinstance(synthetic_y, np.ndarray):
                self.syn_y = torch.from_numpy(synthetic_y).long()
            else:
                self.syn_y = synthetic_y.long()

            # Compute combined sizes
            n_real = len(self.real_X)
            # Desired n_syn = n_real * (synthetic_ratio / (1 - synthetic_ratio))
            if synthetic_ratio >= 1.0:
                self.combined_X = self.syn_X
                self.combined_y = self.syn_y
                self.is_synthetic = torch.ones(len(self.syn_X), dtype=torch.bool)
            else:
                n_syn_needed = int(n_real * (synthetic_ratio / (1.0 - synthetic_ratio)))
                # Subsample or repeat synthetic trials to reach n_syn_needed
                syn_indices = torch.randint(0, len(self.syn_X), (n_syn_needed,))
                syn_sampled_X = self.syn_X[syn_indices]
                syn_sampled_y = self.syn_y[syn_indices]

                self.combined_X = torch.cat([self.real_X, syn_sampled_X], dim=0)
                self.combined_y = torch.cat([self.real_y, syn_sampled_y], dim=0)
                self.is_synthetic = torch.cat([
                    torch.zeros(n_real, dtype=torch.bool),
                    torch.ones(n_syn_needed, dtype=torch.bool),
                ], dim=0)
        else:
            self.combined_X = self.real_X
            self.combined_y = self.real_y
            self.is_synthetic = torch.zeros(len(self.real_X), dtype=torch.bool)

    def __len__(self) -> int:
        return len(self.combined_X)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "eeg": self.combined_X[idx],
            "label": self.combined_y[idx],
            "is_synthetic": self.is_synthetic[idx],
        }


class EEGDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule handling PhysioNet EEG data loading,
    subject splitting (train / val / test), and resting-state associations.
    """

    def __init__(
        self,
        data_dir: str = "c:/Users/Visha/Desktop/Cybernetics Motor/Motor",
        train_subjects: Optional[List[int]] = None,
        val_subjects: Optional[List[int]] = None,
        test_subjects: Optional[List[int]] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        sfreq: float = 160.0,
        trial_duration: float = 4.0,
        resting_duration: float = 30.0,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sfreq = sfreq
        self.trial_duration = trial_duration
        self.resting_duration = resting_duration

        self.train_subjects = train_subjects or list(range(11, 31))
        self.val_subjects = val_subjects or list(range(1, 6))
        self.test_subjects = test_subjects or list(range(6, 11))

        self.loader = PhysioNetLoader(
            data_dir=self.data_dir,
            sfreq=self.sfreq,
            trial_duration=self.trial_duration,
            resting_duration=self.resting_duration,
        )

        self.train_dataset: Optional[EEGDataset] = None
        self.val_dataset: Optional[EEGDataset] = None
        self.test_dataset: Optional[EEGDataset] = None
        self.resting_dataset: Optional[RestingStateDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets for training, validation, testing, and resting state."""
        if stage in ("fit", None):
            train_data = self.loader.load_dataset(self.train_subjects, include_resting=True)
            self.train_dataset = EEGDataset(
                X=train_data["X"],
                y=train_data["y"],
                subject_ids=train_data["subject_idx"],
                resting_dict=train_data["resting_dict"],
            )

            val_data = self.loader.load_dataset(self.val_subjects, include_resting=True)
            self.val_dataset = EEGDataset(
                X=val_data["X"],
                y=val_data["y"],
                subject_ids=val_data["subject_idx"],
                resting_dict=val_data["resting_dict"],
            )

            # Combined resting dataset from train subjects
            self.resting_dataset = RestingStateDataset(
                resting_dict=train_data["resting_dict"],
                window_samples=int(self.resting_duration * self.sfreq),
            )

        if stage in ("test", None):
            test_data = self.loader.load_dataset(self.test_subjects, include_resting=True)
            self.test_dataset = EEGDataset(
                X=test_data["X"],
                y=test_data["y"],
                subject_ids=test_data["subject_idx"],
                resting_dict=test_data["resting_dict"],
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def resting_dataloader(self) -> DataLoader:
        """DataLoader specifically for resting-state encoder training."""
        if self.resting_dataset is None:
            self.setup("fit")
        return DataLoader(
            self.resting_dataset,
            batch_size=min(self.batch_size, len(self.resting_dataset)),
            shuffle=True,
            num_workers=self.num_workers,
        )
