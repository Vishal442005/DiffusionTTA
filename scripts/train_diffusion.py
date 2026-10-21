"""
Training Script for Rest-Diff-BCI Latent Diffusion Model.

Pipeline:
1. Loads PhysioNet dataset (MI runs + R01/R02 resting state baselines).
2. Stage 1: Trains the RestingStateEncoder (Spatial-Spectral VAE) on 30s resting EEG.
3. Stage 2: Trains EEGLatentDiffusion (VQ-VAE + 1D Conditional Latent U-Net)
   conditioned on subject resting style vector and MI class labels.
4. Generates validation samples and saves checkpoints.

Usage:
    python train_diffusion.py --config configs/config.yaml --epochs 100 --batch_size 32
"""

from __future__ import annotations

import os
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import sys
import argparse
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "scripts" else Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
CONFIGS_DIR = PROJECT_ROOT / "configs"
DEFAULT_DATA_DIR = Path("c:/Users/Visha/Desktop/Cybernetics Motor/Motor") if Path("c:/Users/Visha/Desktop/Cybernetics Motor/Motor").exists() else PROJECT_ROOT.parent / "Motor"

from typing import Any, Dict

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import EEGDataset, RestingStateDataset
from src.data.physionet_loader import PhysioNetLoader
from src.models.latent_diffusion import EEGLatentDiffusion
from src.models.resting_state_encoder import RestingStateEncoder

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("TrainDiffusion")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Rest-Diff-BCI Latent Diffusion Model")
    parser.add_argument("--config", type=str, default=str(CONFIGS_DIR / "config.yaml"), help="Path to config file")
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR), help="Override PhysioNet data path")
    parser.add_argument("--epochs", type=int, default=None, help="Number of diffusion training epochs")
    parser.add_argument("--rest_epochs", type=int, default=20, help="Epochs for resting-state VAE")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda', 'cpu', or 'auto')")
    parser.add_argument("--checkpoint_dir", type=str, default=str(CHECKPOINTS_DIR), help="Directory to save weights")
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    logger.warning("Config file %s not found. Using defaults.", config_path)
    return {}


def train_resting_encoder(
    encoder: RestingStateEncoder,
    train_rest_dataset: RestingStateDataset,
    val_rest_dataset: RestingStateDataset,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    save_path: Path,
) -> RestingStateEncoder:
    """Train the Spatial-Spectral VAE on 30s passive resting state EEG."""
    logger.info("--> Starting Stage 1: Training Resting-State Spatial-Spectral VAE...")
    train_loader = DataLoader(train_rest_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_rest_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(encoder.parameters(), lr=float(lr), weight_decay=1e-4)
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        encoder.train()
        train_loss = 0.0

        for batch in train_loader:
            x_rest = batch["resting"].to(device)
            out = encoder(x_rest)
            loss_dict = encoder.compute_loss(out)

            optimizer.zero_grad()
            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss_dict["loss"].item() * len(x_rest)

        train_loss /= len(train_rest_dataset)

        # Validation
        encoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x_rest = batch["resting"].to(device)
                out = encoder(x_rest)
                loss_dict = encoder.compute_loss(out)
                val_loss += loss_dict["loss"].item() * len(x_rest)

        val_loss /= max(1, len(val_rest_dataset))

        if epoch % 5 == 0 or epoch == epochs:
            logger.info("Resting VAE Epoch [%02d/%02d] - Train Loss: %.4f | Val Loss: %.4f", epoch, epochs, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(encoder.state_dict(), save_path)

    logger.info("Stage 1 complete! Saved best Resting Encoder to %s", save_path)
    encoder.load_state_dict(torch.load(save_path, map_location=device))
    return encoder


def train_latent_diffusion(
    diffusion: EEGLatentDiffusion,
    resting_encoder: RestingStateEncoder,
    train_dataset: EEGDataset,
    val_dataset: EEGDataset,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    save_path: Path,
) -> None:
    """Train the conditional Latent Diffusion Model."""
    logger.info("--> Starting Stage 2: Training Conditional Latent Diffusion Model...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=float(lr), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    resting_encoder.eval()

    for epoch in range(1, epochs + 1):
        diffusion.train()
        train_loss = 0.0

        for batch in train_loader:
            x_real = batch["eeg"].to(device)
            y = batch["label"].to(device)
            x_rest = batch["resting"].to(device)

            with torch.no_grad():
                subject_style, _, _ = resting_encoder.encode(x_rest, sample=False)

            loss_dict = diffusion.compute_loss(x_real, y, subject_style)
            loss = loss_dict["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(x_real)

        scheduler.step()
        train_loss /= len(train_dataset)

        # Validation
        diffusion.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x_real = batch["eeg"].to(device)
                y = batch["label"].to(device)
                x_rest = batch["resting"].to(device)

                subject_style, _, _ = resting_encoder.encode(x_rest, sample=False)
                loss_dict = diffusion.compute_loss(x_real, y, subject_style)
                val_loss += loss_dict["loss"].item() * len(x_real)

        val_loss /= len(val_dataset)

        if epoch % 5 == 0 or epoch == epochs:
            logger.info(
                "Diffusion Epoch [%03d/%03d] - Train Loss: %.4f | Val Loss: %.4f (Diff: %.4f, VQ: %.4f)",
                epoch,
                epochs,
                train_loss,
                val_loss,
                loss_dict["diffusion_loss"].item(),
                loss_dict["vq_loss"].item(),
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "diffusion_state": diffusion.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                save_path,
            )

    logger.info("Stage 2 complete! Saved best Latent Diffusion model to %s", save_path)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # Resolve arguments with configuration
    data_cfg = cfg.get("data", {})
    rest_cfg = cfg.get("resting_encoder", {})
    diff_cfg = cfg.get("latent_diffusion", {})
    train_cfg = cfg.get("training", {}).get("diffusion", {})

    data_dir = args.data_dir or data_cfg.get("data_dir", "c:/Users/Visha/Desktop/Cybernetics Motor/Motor")
    epochs = args.epochs or train_cfg.get("epochs", 100)
    batch_size = args.batch_size or train_cfg.get("batch_size", 32)
    lr = float(args.lr or train_cfg.get("learning_rate", 2e-4))

    # Device selection
    if args.device == "cuda" or (args.device is None and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("Using compute device: %s", device)

    # Checkpoint path
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rest_save_path = checkpoint_dir / "resting_encoder_best.pt"
    diff_save_path = checkpoint_dir / "diffusion_best.pt"

    # Data loading
    train_subjects = data_cfg.get("train_subjects", list(range(11, 21)))
    val_subjects = data_cfg.get("val_subjects", list(range(1, 6)))

    logger.info("Loading PhysioNet data from: %s", data_dir)
    loader = PhysioNetLoader(data_dir=data_dir)

    logger.info("Loading training subjects: %s", train_subjects)
    train_data = loader.load_dataset(train_subjects, include_resting=True)
    train_dataset = EEGDataset(
        X=train_data["X"],
        y=train_data["y"],
        subject_ids=train_data["subject_idx"],
        resting_dict=train_data["resting_dict"],
    )

    logger.info("Loading validation subjects: %s", val_subjects)
    val_data = loader.load_dataset(val_subjects, include_resting=True)
    val_dataset = EEGDataset(
        X=val_data["X"],
        y=val_data["y"],
        subject_ids=val_data["subject_idx"],
        resting_dict=val_data["resting_dict"],
    )

    train_rest_dataset = RestingStateDataset(train_data["resting_dict"])
    val_rest_dataset = RestingStateDataset(val_data["resting_dict"])

    logger.info("Train trials: %d | Val trials: %d", len(train_dataset), len(val_dataset))

    # Initialize models
    style_dim = rest_cfg.get("latent_dim", 64)
    resting_encoder = RestingStateEncoder(
        in_channels=64,
        latent_dim=style_dim,
        sfreq=160.0,
        kl_weight=float(rest_cfg.get("kl_weight", 1e-4)),
    ).to(device)

    diffusion_model = EEGLatentDiffusion(
        in_channels=64,
        latent_channels=diff_cfg.get("vqvae", {}).get("latent_channels", 16),
        style_dim=style_dim,
        num_classes=4,
        timesteps=diff_cfg.get("diffusion", {}).get("timesteps", 1000),
        beta_schedule=diff_cfg.get("diffusion", {}).get("beta_schedule", "cosine"),
        ddim_steps=diff_cfg.get("diffusion", {}).get("ddim_steps", 20),
        guidance_scale=diff_cfg.get("diffusion", {}).get("guidance_scale", 2.5),
    ).to(device)

    # Execute Stage 1
    resting_encoder = train_resting_encoder(
        encoder=resting_encoder,
        train_rest_dataset=train_rest_dataset,
        val_rest_dataset=val_rest_dataset,
        epochs=args.rest_epochs,
        batch_size=batch_size,
        lr=float(rest_cfg.get("learning_rate", 3e-4)),
        device=device,
        save_path=rest_save_path,
    )

    # Execute Stage 2
    train_latent_diffusion(
        diffusion=diffusion_model,
        resting_encoder=resting_encoder,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        save_path=diff_save_path,
    )

    # Quick test sampling
    logger.info("Testing fast 4-step DDIM trial synthesis...")
    sample_rest = val_rest_dataset[0]["resting"].unsqueeze(0).to(device)
    with torch.no_grad():
        style_vec, _, _ = resting_encoder.encode(sample_rest, sample=False)
        test_labels = torch.tensor([0, 1, 2, 3], device=device)
        test_styles = style_vec.repeat(4, 1)
        generated_trials = diffusion_model.sample(test_styles, test_labels, num_steps=4)

    logger.info("Successfully synthesized trials of shape: %s", generated_trials.shape)
    logger.info("Training pipeline complete!")


if __name__ == "__main__":
    main()
