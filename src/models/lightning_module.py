"""
PyTorch Lightning Modules for Paper 2 (Rest-Diff-BCI + MoE-BFT).

Includes:
1. RestingStateLightningModule: Trains the Spatial-Spectral VAE for subject style vectors.
2. DiffusionLightningModule: Trains the Conditional EEG Latent Diffusion Model.
3. MoEClassifierLightningModule: Trains the Channel-Wise Mixture of Experts classifier.
4. EndToEndZeroCalibrationPipeline: Orchestrates synthetic calibration + MoE-BFT adaptation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy, MulticlassCohenKappa, MulticlassF1Score

from src.models.bft_adapter import BFTAdapter
from src.models.latent_diffusion import EEGLatentDiffusion
from src.models.moe_backbone import ChannelWiseMoE
from src.models.resting_state_encoder import RestingStateEncoder


class RestingStateLightningModule(pl.LightningModule):
    """
    Lightning Module for training the RestingStateEncoder (Spatial-Spectral VAE).
    """

    def __init__(
        self,
        in_channels: int = 64,
        latent_dim: int = 64,
        sfreq: float = 160.0,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        kl_weight: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.encoder = RestingStateEncoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            sfreq=sfreq,
            kl_weight=kl_weight,
        )
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.encoder(x)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x_rest = batch["resting"]
        out = self.encoder(x_rest)
        loss_dict = self.encoder.compute_loss(out)

        for k, v in loss_dict.items():
            self.log(f"train_rest_{k}", v, prog_bar=(k == "loss"), on_step=True, on_epoch=True)

        return loss_dict["loss"]

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x_rest = batch["resting"]
        out = self.encoder(x_rest)
        loss_dict = self.encoder.compute_loss(out)

        for k, v in loss_dict.items():
            self.log(f"val_rest_{k}", v, prog_bar=(k == "loss"), on_epoch=True)

        return loss_dict["loss"]

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(
            self.encoder.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class DiffusionLightningModule(pl.LightningModule):
    """
    Lightning Module for training the EEGLatentDiffusion model conditioned on
    MI classes and subject resting-state embeddings.
    """

    def __init__(
        self,
        resting_encoder: RestingStateEncoder,
        in_channels: int = 64,
        latent_channels: int = 16,
        style_dim: int = 64,
        num_classes: int = 4,
        timesteps: int = 1000,
        beta_schedule: str = "cosine",
        ddim_steps: int = 20,
        guidance_scale: float = 2.5,
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["resting_encoder"])

        self.resting_encoder = resting_encoder
        # Freeze resting encoder during diffusion training
        for param in self.resting_encoder.parameters():
            param.requires_grad = False

        self.diffusion = EEGLatentDiffusion(
            in_channels=in_channels,
            latent_channels=latent_channels,
            style_dim=style_dim,
            num_classes=num_classes,
            timesteps=timesteps,
            beta_schedule=beta_schedule,
            ddim_steps=ddim_steps,
            guidance_scale=guidance_scale,
        )
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(
        self,
        subject_style: torch.Tensor,
        class_labels: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        return self.diffusion.sample(subject_style, class_labels, num_steps=num_steps)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x_real = batch["eeg"]
        y = batch["label"]
        x_rest = batch["resting"]

        # Extract subject style from resting state
        with torch.no_grad():
            subject_style, _, _ = self.resting_encoder.encode(x_rest, sample=False)

        loss_dict = self.diffusion.compute_loss(x_real, y, subject_style)

        for k, v in loss_dict.items():
            self.log(f"train_diff_{k}", v, prog_bar=(k == "loss"), on_step=True, on_epoch=True)

        return loss_dict["loss"]

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x_real = batch["eeg"]
        y = batch["label"]
        x_rest = batch["resting"]

        with torch.no_grad():
            subject_style, _, _ = self.resting_encoder.encode(x_rest, sample=False)

        loss_dict = self.diffusion.compute_loss(x_real, y, subject_style)

        for k, v in loss_dict.items():
            self.log(f"val_diff_{k}", v, prog_bar=(k == "loss"), on_epoch=True)

        return loss_dict["loss"]

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(
            self.diffusion.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class MoEClassifierLightningModule(pl.LightningModule):
    """
    Lightning Module for training the Channel-Wise Mixture of Experts (MoE) classifier.
    """

    def __init__(
        self,
        in_channels: int = 64,
        num_classes: int = 4,
        hidden_dim: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-3,
        label_smoothing: float = 0.05,
        load_balance_weight: float = 0.01,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.moe = ChannelWiseMoE(
            in_channels=in_channels,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            load_balance_weight=load_balance_weight,
        )
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing

        # Metrics
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_kappa = MulticlassCohenKappa(num_classes=num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_kappa = MulticlassCohenKappa(num_classes=num_classes)
        self.test_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")

    def forward(self, x: torch.Tensor, return_details: bool = False) -> Any:
        return self.moe(x, return_details=return_details)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x = batch["eeg"]
        y = batch["label"]

        loss_dict = self.moe.compute_loss(x, y, label_smoothing=self.label_smoothing)
        loss = loss_dict["loss"]
        logits = loss_dict["logits"]

        preds = torch.argmax(logits, dim=-1)
        self.train_acc(preds, y)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x = batch["eeg"]
        y = batch["label"]

        loss_dict = self.moe.compute_loss(x, y, label_smoothing=self.label_smoothing)
        loss = loss_dict["loss"]
        logits = loss_dict["logits"]

        preds = torch.argmax(logits, dim=-1)
        self.val_acc(preds, y)
        self.val_kappa(preds, y)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", self.val_acc, prog_bar=True, on_epoch=True)
        self.log("val_kappa", self.val_kappa, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x = batch["eeg"]
        y = batch["label"]

        logits = self.moe(x)
        preds = torch.argmax(logits, dim=-1)

        self.test_acc(preds, y)
        self.test_kappa(preds, y)
        self.test_f1(preds, y)

        self.log("test_acc", self.test_acc, on_epoch=True)
        self.log("test_kappa", self.test_kappa, on_epoch=True)
        self.log("test_f1", self.test_f1, on_epoch=True)
        return self.test_acc.compute()

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(
            self.moe.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class EndToEndZeroCalibrationPipeline:
    """
    End-to-End Orchestrator for Zero-Calibration BCI:
    1. Extracts subject-style vector from 30s resting state.
    2. Generates personalized synthetic motor imagery trials.
    3. Fine-tunes/calibrates MoE backbone on synthetic data.
    4. Evaluates online using BFT test-time adaptation.
    """

    def __init__(
        self,
        resting_encoder: RestingStateEncoder,
        diffusion_model: EEGLatentDiffusion,
        moe_classifier: ChannelWiseMoE,
        bft_adapter: Optional[BFTAdapter] = None,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.resting_encoder = resting_encoder.to(self.device).eval()
        self.diffusion_model = diffusion_model.to(self.device).eval()
        self.moe_classifier = moe_classifier.to(self.device)
        self.bft_adapter = bft_adapter or BFTAdapter(self.moe_classifier)
        self.bft_adapter.to(self.device)

    def generate_synthetic_calibration_data(
        self,
        resting_eeg: torch.Tensor,
        num_trials_per_class: int = 100,
        ddim_steps: int = 20,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate personalized synthetic calibration trials from 30s passive resting EEG.

        Args:
            resting_eeg: Resting state EEG (1, 64, T_rest) or (64, T_rest)
            num_trials_per_class: Number of synthetic trials per MI class
            ddim_steps: Sampling steps for DDIM

        Returns:
            syn_X: Synthesized trials of shape (4 * num_trials_per_class, 64, 640)
            syn_y: Corresponding class labels
        """
        if resting_eeg.dim() == 2:
            resting_eeg = resting_eeg.unsqueeze(0)
        resting_eeg = resting_eeg.to(self.device)

        with torch.no_grad():
            style_vector, _, _ = self.resting_encoder.encode(resting_eeg, sample=False)

        syn_X_list: List[torch.Tensor] = []
        syn_y_list: List[torch.Tensor] = []

        batch_size = 25
        for cls_id in range(4):
            remaining = num_trials_per_class
            while remaining > 0:
                cur_batch = min(remaining, batch_size)
                style_batch = style_vector.repeat(cur_batch, 1)
                labels = torch.full((cur_batch,), cls_id, dtype=torch.long, device=self.device)

                with torch.no_grad():
                    gen_trials = self.diffusion_model.sample(
                        style_batch, labels, num_steps=ddim_steps
                    )

                syn_X_list.append(gen_trials.cpu())
                syn_y_list.append(labels.cpu())
                remaining -= cur_batch

        syn_X = torch.cat(syn_X_list, dim=0)
        syn_y = torch.cat(syn_y_list, dim=0)

        return syn_X, syn_y

    def calibrate_on_synthetic_data(
        self,
        syn_X: torch.Tensor,
        syn_y: torch.Tensor,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-4,
    ) -> None:
        """
        Quick calibration/fine-tuning of MoE classifier on personalized synthetic data.
        Requires ZERO active motor execution from the user!
        """
        self.moe_classifier.train()
        optimizer = torch.optim.Adam(self.moe_classifier.parameters(), lr=lr)

        dataset = torch.utils.data.TensorDataset(syn_X, syn_y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for _ in range(epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                loss_dict = self.moe_classifier.compute_loss(batch_x, batch_y)
                optimizer.zero_grad()
                loss_dict["loss"].backward()
                optimizer.step()

        self.moe_classifier.eval()

    def evaluate_test_trials(
        self,
        test_X: torch.Tensor,
        test_y: torch.Tensor,
        use_bft: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate real test trials with or without BFT test-time adaptation.
        """
        self.moe_classifier.eval()
        test_X = test_X.to(self.device)
        test_y = test_y.to(self.device)

        start = time.perf_counter()
        with torch.no_grad():
            if use_bft:
                probs = self.bft_adapter(test_X)
            else:
                probs = self.moe_classifier(test_X, return_details=True)["probs"]

        elapsed_ms = (time.perf_counter() - start) * 1000.0 / len(test_X)

        preds = torch.argmax(probs, dim=-1)
        acc = (preds == test_y).float().mean().item()

        # Compute Cohen's Kappa
        kappa_metric = MulticlassCohenKappa(num_classes=4).to(self.device)
        kappa = kappa_metric(preds, test_y).item()

        return {
            "accuracy": acc,
            "kappa": kappa,
            "latency_ms_per_trial": elapsed_ms,
        }
