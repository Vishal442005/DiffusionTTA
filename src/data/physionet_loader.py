"""
PhysioNet EEG Motor Movement/Imagery Dataset Loader.

Self-contained dataset loader for Paper 2 (Rest-Diff-BCI + MoE-BFT).
Handles motor imagery trials (runs 4, 6, 8, 10, 12, 14) and resting-state
baseline runs (runs 1 and 2: eyes open and eyes closed).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import mne
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Excluded subjects due to missing/corrupted runs in PhysioNet
EXCLUDED_SUBJECTS = {88, 92, 100}

# Standard 64-channel 10-10 montage names in order of PhysioNet EDF files
STANDARD_CHANNELS = [
    "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "CZ", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
    "FP1", "FPZ", "FP2",
    "AF7", "AF3", "AFZ", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FT8",
    "T7", "T8",
    "T9", "T10",
    "TP7", "TP8",
    "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POZ", "PO4", "PO8",
    "O1", "OZ", "O2",
    "IZ"
]


def clean_channel_name(ch_name: str) -> str:
    """
    Standardize EDF channel names by removing trailing dots and whitespace.
    
    Example:
        'Fc5.' -> 'FC5'
        'C3..' -> 'C3'
        'Cz..' -> 'CZ'
    """
    return ch_name.rstrip(".").strip().upper()


def euclidean_alignment(X: np.ndarray) -> np.ndarray:
    """
    Apply Euclidean Alignment (EA) to EEG trials from a single subject.
    Normalizes the subject reference covariance matrix to identity:
        R = (1/N) * sum_i (1/T) X_i X_i^T
        X_tilde = R^(-1/2) X_i
    
    Reference:
        He & Wu, "Transfer Learning for Brain-Computer Interfaces: A Euclidean Space
        Data Alignment Approach", IEEE Transactions on Biomedical Engineering, 2020.
    
    Args:
        X: EEG array of shape (N, C, T) in Volts or microvolts.
    Returns:
        X_aligned: Aligned EEG array of shape (N, C, T) with mean covariance = I.
    """
    if len(X) == 0:
        return X
    X_uV = (X * 1e6) if np.max(np.abs(X)) < 0.1 else X
    N, C, T = X_uV.shape
    # Compute mean trial covariance matrix: (N, C, C) -> (C, C)
    covs = np.matmul(X_uV, X_uV.transpose(0, 2, 1)) / float(T)
    R = np.mean(covs, axis=0)
    eps = np.trace(R) * 1e-4 / C
    R_reg = R + eps * np.eye(C, dtype=R.dtype)
    U, S, Vh = np.linalg.svd(R_reg)
    R_inv_sqrt = U @ np.diag(1.0 / np.sqrt(S)) @ Vh
    X_aligned = np.einsum('ij,njt->nit', R_inv_sqrt, X_uV).astype(np.float32)
    return X_aligned


class PhysioNetLoader:
    """
    Self-contained loader for the PhysioNet EEG Motor Movement/Imagery Dataset.
    
    Data parameters:
        - 109 subjects (S001 - S109)
        - 64 channels, 160 Hz sampling rate
        - Baseline runs: R01 (eyes open), R02 (eyes closed)
        - Task 2 runs (Left / Right fist imagery): R04, R08, R12
        - Task 4 runs (Both fists / Both feet imagery): R06, R10, R14
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = "c:/Users/Visha/Desktop/Cybernetics Motor/Motor",
        sfreq: float = 160.0,
        l_freq: float = 0.5,
        h_freq: float = 45.0,
        trial_duration: float = 4.0,
        resting_duration: float = 30.0,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Initialize PhysioNet loader.

        Args:
            data_dir: Path to directory containing subject subfolders (S001, S002, etc.)
            sfreq: Sampling rate (PhysioNet native is 160.0 Hz)
            l_freq: Low cutoff frequency for bandpass filter in Hz
            h_freq: High cutoff frequency for bandpass filter in Hz
            trial_duration: Length of motor imagery trials in seconds (default 4.0s = 640 samples)
            resting_duration: Duration of resting-state baseline window in seconds (default 30.0s)
            cache_dir: Optional directory to cache preprocessed NumPy arrays
        """
        self.data_dir = Path(data_dir)
        self.sfreq = sfreq
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.trial_duration = trial_duration
        self.resting_duration = resting_duration
        self.trial_samples = int(self.trial_duration * self.sfreq)
        self.resting_samples = int(self.resting_duration * self.sfreq)
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        if not self.data_dir.exists():
            raise FileNotFoundError(f"PhysioNet data directory not found: {self.data_dir}")

    def get_subject_folder(self, subject_id: int) -> Path:
        """Return path to subject directory (e.g. Motor/S001)."""
        return self.data_dir / f"S{subject_id:03d}"

    def get_edf_path(self, subject_id: int, run_id: int) -> Path:
        """Return path to specific subject run EDF file."""
        sub_folder = self.get_subject_folder(subject_id)
        return sub_folder / f"S{subject_id:03d}R{run_id:02d}.edf"

    def _read_and_filter_raw(self, edf_path: Path) -> mne.io.Raw:
        """
        Load EDF file with MNE, clean channel names, and apply bandpass filter.
        """
        if not edf_path.exists():
            raise FileNotFoundError(f"EDF file does not exist: {edf_path}")

        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
        
        # Clean channel names
        rename_mapping = {ch: clean_channel_name(ch) for ch in raw.ch_names}
        raw.rename_channels(rename_mapping)

        # Apply bandpass filter
        raw.filter(
            l_freq=self.l_freq,
            h_freq=self.h_freq,
            fir_design="firwin",
            skip_by_annotation="edge",
            verbose=False,
        )
        return raw

    def load_subject_resting_state(
        self,
        subject_id: int,
        baseline_runs: Tuple[int, ...] = (1, 2),
        duration_sec: Optional[float] = None,
    ) -> np.ndarray:
        """
        Extract resting-state EEG epoch(s) from baseline runs R01 (eyes open) and/or R02 (eyes closed).

        Args:
            subject_id: Subject index (1-109)
            baseline_runs: Runs to load, default (1, 2)
            duration_sec: Duration in seconds to extract (default self.resting_duration = 30.0s)

        Returns:
            np.ndarray of shape (C, T) representing resting state baseline (e.g. 64, 4800)
        """
        if subject_id in EXCLUDED_SUBJECTS:
            raise ValueError(f"Subject {subject_id} is in excluded list.")

        duration = duration_sec or self.resting_duration
        target_samples = int(duration * self.sfreq)

        resting_segments: List[np.ndarray] = []

        for run_id in baseline_runs:
            edf_path = self.get_edf_path(subject_id, run_id)
            if not edf_path.exists():
                logger.warning("Baseline file %s not found, skipping", edf_path)
                continue

            try:
                raw = self._read_and_filter_raw(edf_path)
                data = raw.get_data()  # shape: (n_channels, n_times)

                if data.shape[1] >= target_samples:
                    # Take central window to avoid filter transient artifacts at borders
                    start_idx = (data.shape[1] - target_samples) // 2
                    seg = data[:, start_idx : start_idx + target_samples]
                    resting_segments.append(seg)
                else:
                    # Pad if shorter than target samples
                    pad_width = target_samples - data.shape[1]
                    seg = np.pad(data, ((0, 0), (0, pad_width)), mode="edge")
                    resting_segments.append(seg)
            except Exception as e:
                logger.error("Failed loading resting state for S%03d run %02d: %s", subject_id, run_id, str(e))

        if not resting_segments:
            # Fallback zero tensor if files unavailable
            logger.warning("No resting segments found for S%03d. Returning zero array.", subject_id)
            return np.zeros((64, target_samples), dtype=np.float32)

        # Average available baseline runs (eyes open + eyes closed) to get representative subject resting state
        combined = np.mean(resting_segments, axis=0).astype(np.float32)
        return combined

    def load_subject_mi_trials(
        self,
        subject_id: int,
        runs: Optional[List[int]] = None,
        align_euclidean: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load motor imagery trials for a given subject.

        Classes:
            0: Left fist (Task 2, T1)
            1: Right fist (Task 2, T2)
            2: Both fists (Task 4, T1)
            3: Both feet (Task 4, T2)

        Args:
            subject_id: Subject index (1-109)
            runs: Optional list of MI runs. Default: [4, 6, 8, 10, 12, 14]

        Returns:
            X: np.ndarray of shape (N_trials, 64, trial_samples)
            y: np.ndarray of shape (N_trials,) with integer labels 0..3
        """
        if subject_id in EXCLUDED_SUBJECTS:
            raise ValueError(f"Subject {subject_id} is in excluded list.")

        if runs is None:
            runs = [4, 6, 8, 10, 12, 14]

        all_epochs_data: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []

        for run_id in runs:
            edf_path = self.get_edf_path(subject_id, run_id)
            if not edf_path.exists():
                logger.warning("MI file %s not found, skipping", edf_path)
                continue

            try:
                raw = self._read_and_filter_raw(edf_path)
                events, event_dict = mne.events_from_annotations(raw, verbose=False)

                t1_code = event_dict.get("T1", None)
                t2_code = event_dict.get("T2", None)

                if t1_code is None and t2_code is None:
                    continue

                # Determine class mapping based on run type
                # Runs 4, 8, 12: Task 2 -> T1=Left Fist (0), T2=Right Fist (1)
                # Runs 6, 10, 14: Task 4 -> T1=Both Fists (2), T2=Both Feet (3)
                if run_id in [4, 8, 12]:
                    mapping = {t1_code: 0, t2_code: 1}
                else:
                    mapping = {t1_code: 2, t2_code: 3}

                # Filter valid MI events
                valid_mask = np.isin(events[:, 2], list(mapping.keys()))
                target_events = events[valid_mask]

                if len(target_events) == 0:
                    continue

                # Create epochs of exact duration (0 to trial_duration - 1/sfreq)
                tmax = self.trial_duration - (1.0 / self.sfreq)
                epochs = mne.Epochs(
                    raw,
                    target_events,
                    tmin=0.0,
                    tmax=tmax,
                    baseline=None,
                    preload=True,
                    verbose=False,
                )

                epochs_data = epochs.get_data()  # (N, C, T)

                # Ensure exact sample length
                if epochs_data.shape[2] > self.trial_samples:
                    epochs_data = epochs_data[:, :, : self.trial_samples]
                elif epochs_data.shape[2] < self.trial_samples:
                    pad_len = self.trial_samples - epochs_data.shape[2]
                    epochs_data = np.pad(epochs_data, ((0, 0), (0, 0), (0, pad_len)), mode="edge")

                labels = np.array([mapping[event_code] for event_code in target_events[:, 2]], dtype=np.int64)

                all_epochs_data.append(epochs_data)
                all_labels.append(labels)

            except Exception as e:
                logger.error("Error processing S%03d run %02d: %s", subject_id, run_id, str(e))

        if not all_epochs_data:
            empty_X = np.empty((0, 64, self.trial_samples), dtype=np.float32)
            empty_y = np.empty((0,), dtype=np.int64)
            return empty_X, empty_y

        X = np.concatenate(all_epochs_data, axis=0).astype(np.float32)
        y = np.concatenate(all_labels, axis=0).astype(np.int64)

        if align_euclidean and len(X) > 0:
            X = euclidean_alignment(X)

        return X, y

    def load_dataset(
        self,
        subject_ids: List[int],
        include_resting: bool = True,
        align_euclidean: bool = False,
    ) -> Dict[str, Union[np.ndarray, Dict[int, np.ndarray]]]:
        """
        Load dataset across multiple subjects.

        Args:
            subject_ids: List of subject IDs to load.
            include_resting: Whether to load resting-state vectors for each subject.
            align_euclidean: Whether to apply Euclidean Alignment (EA) per subject.

        Returns:
            Dict containing:
                - 'X': np.ndarray of shape (Total_trials, 64, T)
                - 'y': np.ndarray of shape (Total_trials,)
                - 'subject_idx': np.ndarray of shape (Total_trials,)
                - 'resting_dict': Dict[subject_id -> np.ndarray of shape (64, T_rest)]
        """
        valid_subjects = [s for s in subject_ids if s not in EXCLUDED_SUBJECTS]

        X_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []
        sub_list: List[np.ndarray] = []
        resting_dict: Dict[int, np.ndarray] = {}

        for sub_id in valid_subjects:
            X_sub, y_sub = self.load_subject_mi_trials(sub_id, align_euclidean=align_euclidean)
            if len(X_sub) > 0:
                X_list.append(X_sub)
                y_list.append(y_sub)
                sub_list.append(np.full((len(y_sub),), sub_id, dtype=np.int64))

            if include_resting:
                rest_data = self.load_subject_resting_state(sub_id)
                resting_dict[sub_id] = rest_data

        if not X_list:
            raise RuntimeError("No valid trials loaded for specified subjects.")

        X_all = np.concatenate(X_list, axis=0)
        y_all = np.concatenate(y_list, axis=0)
        sub_all = np.concatenate(sub_list, axis=0)

        return {
            "X": X_all,
            "y": y_all,
            "subject_idx": sub_all,
            "resting_dict": resting_dict,
        }


# Class alias for backward compatibility
PhysioNetMIDataset = PhysioNetLoader
