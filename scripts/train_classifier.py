"""
Training Script for Channel-Wise Mixture of Experts (MoE) Classifier.

Supports:
1. Standard baseline training on real Motor Imagery trials.
2. Rest-Diff-BCI Augmented training: synthesizes subject-personalized trials using
   the Latent Diffusion Model conditioned on 30s resting-state EEG.
3. Source-training the Learning-to-Rank ReliabilityRanker for BFT test-time adaptation.

Usage:
    # Standard training
    python train_classifier.py --config configs/config.yaml --epochs 80

    # With Rest-Diff synthetic augmentation (50% real, 50% synthetic)
    python train_classifier.py --config configs/config.yaml --use_synthetic --synthetic_ratio 0.5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "scripts" else Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
CONFIGS_DIR = PROJECT_ROOT / "configs"
DEFAULT_DATA_DIR = Path("c:/Users/Visha/Desktop/Cybernetics Motor/Motor") if Path("c:/Users/Visha/Desktop/Cybernetics Motor/Motor").exists() else PROJECT_ROOT.parent / "Motor"

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import EEGDataset, MixedEEGDataset
from src.data.physionet_loader import PhysioNetLoader, euclidean_alignment
from src.models.bft_adapter import BFTAdapter
from src.models.latent_diffusion import EEGLatentDiffusion
from src.models.moe_backbone import ChannelWiseMoE
from src.models.resting_state_encoder import RestingStateEncoder

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("TrainClassifier")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Channel-Wise MoE Classifier")
    parser.add_argument("--config", type=str, default=str(CONFIGS_DIR / "config.yaml"), help="Path to config file")
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR), help="Override PhysioNet data path")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs for MoE classifier")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--use_synthetic", action="store_true", help="Augment training with Rest-Diff synthetic data")
    parser.add_argument("--synthetic_ratio", type=float, default=0.5, help="Proportion of synthetic trials (0.0 to 1.0)")
    parser.add_argument("--diffusion_checkpoint", type=str, default=str(CHECKPOINTS_DIR / "diffusion_best.pt"), help="Path to diffusion checkpoint")
    parser.add_argument("--resting_checkpoint", type=str, default=str(CHECKPOINTS_DIR / "resting_encoder_best.pt"), help="Path to resting encoder checkpoint")
    parser.add_argument("--use_ea", action="store_true", help="Apply Euclidean Alignment preprocessing")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda', 'cpu', or 'auto')")
    parser.add_argument("--checkpoint_dir", type=str, default=str(CHECKPOINTS_DIR), help="Save directory")
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def generate_synthetic_trials_for_training(
    resting_dict: Dict[int, np.ndarray],
    subject_ids: List[int],
    diffusion_ckpt: Path,
    resting_ckpt: Path,
    trials_per_class: int = 50,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic personalized trials for training subjects using their resting state.
    """
    logger.info("Loading generative models for synthetic data augmentation...")
    style_dim = 64
    resting_encoder = RestingStateEncoder(in_channels=64, latent_dim=style_dim).to(device)
    if resting_ckpt.exists():
        resting_encoder.load_state_dict(torch.load(resting_ckpt, map_location=device))
        logger.info("Loaded resting encoder weights from %s", resting_ckpt)
    resting_encoder.eval()

    diffusion = EEGLatentDiffusion(in_channels=64, latent_channels=16, style_dim=style_dim).to(device)
    if diffusion_ckpt.exists():
        ckpt_data = torch.load(diffusion_ckpt, map_location=device)
        state_dict = ckpt_data["diffusion_state"] if "diffusion_state" in ckpt_data else ckpt_data
        diffusion.load_state_dict(state_dict)
        logger.info("Loaded diffusion weights from %s", diffusion_ckpt)
    diffusion.eval()

    syn_X_list: List[torch.Tensor] = []
    syn_y_list: List[torch.Tensor] = []

    logger.info("Synthesizing %d trials/class across %d subjects...", trials_per_class, len(subject_ids))
    for sub_id in subject_ids:
        if sub_id not in resting_dict:
            continue
        rest_tensor = torch.from_numpy(resting_dict[sub_id]).float().unsqueeze(0).to(device)

        with torch.no_grad():
            style_vec, _, _ = resting_encoder.encode(rest_tensor, sample=False)

        for cls_id in range(4):
            batch_style = style_vec.repeat(trials_per_class, 1)
            batch_labels = torch.full((trials_per_class,), cls_id, dtype=torch.long, device=device)
            with torch.no_grad():
                gen_eeg = diffusion.sample(batch_style, batch_labels, num_steps=10)
            syn_X_list.append(gen_eeg.cpu())
            syn_y_list.append(batch_labels.cpu())

    del resting_encoder
    del diffusion
    if device.type == "cuda":
        torch.cuda.empty_cache()
    import gc
    gc.collect()

    if not syn_X_list:
        return torch.empty((0, 64, 640)), torch.empty((0,), dtype=torch.long)

    return torch.cat(syn_X_list, dim=0), torch.cat(syn_y_list, dim=0)


def train_moe_and_ranker(
    moe: ChannelWiseMoE,
    adapter: BFTAdapter,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    save_path: Path,
    ranker_save_path: Path,
) -> None:
    """Train the MoE classifier and source-train the Learning-to-Rank BFT module."""
    optimizer_moe = torch.optim.AdamW(moe.parameters(), lr=lr, weight_decay=1e-3)
    scheduler_moe = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_moe, T_max=epochs, eta_min=1e-5)

    optimizer_ranker = torch.optim.AdamW(adapter.ranker.parameters(), lr=5e-4, weight_decay=1e-4)

    best_val_acc = 0.0

    # Stage 3A: Train Channel-Wise MoE Backbone
    logger.info("--> Starting Stage 3A: Training Channel-Wise MoE Backbone (%d epochs)...", epochs)
    for epoch in range(1, epochs + 1):
        moe.train()
        total_loss = 0.0
        total_samples = 0

        for batch in train_loader:
            x = batch["eeg"].to(device)
            y = batch["label"].to(device)

            loss_dict = moe.compute_loss(x, y)
            loss = loss_dict["loss"]

            optimizer_moe.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(moe.parameters(), max_norm=1.0)
            optimizer_moe.step()

            total_loss += loss.item() * len(x)
            total_samples += len(x)

        scheduler_moe.step()
        train_loss = total_loss / max(1, total_samples)

        # Validation (leak-proof, zero external allocations)
        moe.eval()
        total_correct = 0
        total_val = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch["eeg"].to(device)
                y = batch["label"].to(device)
                logits = moe(x)
                preds = torch.argmax(logits, dim=-1)
                total_correct += (preds == y).sum().item()
                total_val += len(y)

        val_acc = float(total_correct) / max(1, total_val)
        val_kappa = max(0.0, (val_acc - 0.25) / 0.75)

        if epoch % 5 == 0 or epoch == epochs:
            logger.info(
                "MoE Epoch [%02d/%02d] - Train Loss: %.4f | Val Acc: %.2f%% | Val Kappa: %.3f",
                epoch,
                epochs,
                train_loss,
                val_acc * 100.0,
                val_kappa,
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(moe.state_dict(), save_path)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    logger.info("MoE Backbone training complete! Best Validation Accuracy: %.2f%%", best_val_acc * 100.0)
    moe.load_state_dict(torch.load(save_path, map_location=device))

    # Stage 3B: Source-Train Learning-to-Rank ReliabilityRanker on frozen, trained MoE
    logger.info("--> Starting Stage 3B: Source-training BFT ReliabilityRanker on trained MoE representations (5 epochs)...")
    optimizer_ranker = torch.optim.AdamW(adapter.ranker.parameters(), lr=5e-4, weight_decay=1e-4)
    for r_epoch in range(1, 6):
        total_ranker_loss = 0.0
        for batch in train_loader:
            x = batch["eeg"].to(device)
            y = batch["label"].to(device)
            r_loss = adapter.train_ranker_step(x, y, optimizer_ranker)
            total_ranker_loss += r_loss
        logger.info("Ranker Epoch [%d/5] - Loss: %.4f", r_epoch, total_ranker_loss / len(train_loader))

    torch.save(adapter.ranker.state_dict(), ranker_save_path)
    logger.info("Saved MoE weights to %s and Ranker weights to %s", save_path, ranker_save_path)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {}).get("classifier", {})

    data_dir = args.data_dir or data_cfg.get("data_dir", "c:/Users/Visha/Desktop/Cybernetics Motor/Motor")
    epochs = int(args.epochs or train_cfg.get("epochs", 80))
    batch_size = int(args.batch_size or train_cfg.get("batch_size", 64))
    lr = float(args.lr or train_cfg.get("learning_rate", 1e-3))

    if args.device == "cuda" or (args.device is None and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    moe_save_path = checkpoint_dir / "moe_best.pt"
    ranker_save_path = checkpoint_dir / "ranker_best.pt"

    # Data loading
    train_subjects = data_cfg.get("train_subjects", list(range(11, 21)))
    val_subjects = data_cfg.get("val_subjects", list(range(1, 6)))
    test_subjects = data_cfg.get("test_subjects", list(range(6, 11)))

    align_euclidean = getattr(args, "use_ea", False)
    loader = PhysioNetLoader(data_dir=data_dir)
    logger.info("Loading real trials for training subjects: %s (Euclidean Alignment: %s)", train_subjects, align_euclidean)
    train_data = loader.load_dataset(train_subjects, include_resting=True, align_euclidean=align_euclidean)
    val_data = loader.load_dataset(val_subjects, include_resting=True, align_euclidean=align_euclidean)
    test_data = loader.load_dataset(test_subjects, include_resting=True, align_euclidean=align_euclidean)

    # Optional Rest-Diff Synthetic Augmentation
    if args.use_synthetic:
        logger.info("Rest-Diff augmentation enabled with ratio %.2f", args.synthetic_ratio)
        syn_X, syn_y = generate_synthetic_trials_for_training(
            resting_dict=train_data["resting_dict"],
            subject_ids=train_subjects,
            diffusion_ckpt=Path(args.diffusion_checkpoint),
            resting_ckpt=Path(args.resting_checkpoint),
            trials_per_class=30,
            device=device,
        )
        if align_euclidean and len(syn_X) > 0:
            syn_X = torch.from_numpy(euclidean_alignment(syn_X.numpy())).float()
        train_dataset = MixedEEGDataset(
            real_X=train_data["X"],
            real_y=train_data["y"],
            synthetic_X=syn_X,
            synthetic_y=syn_y,
            synthetic_ratio=args.synthetic_ratio,
        )
    else:
        train_dataset = EEGDataset(
            X=train_data["X"],
            y=train_data["y"],
            subject_ids=train_data["subject_idx"],
        )

    val_dataset = EEGDataset(X=val_data["X"], y=val_data["y"])
    test_dataset = EEGDataset(X=test_data["X"], y=test_data["y"])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logger.info("Total train samples: %d | Val: %d | Test: %d", len(train_dataset), len(val_dataset), len(test_dataset))

    # Build MoE model and BFT adapter
    moe = ChannelWiseMoE(in_channels=64, num_classes=4, hidden_dim=64).to(device)
    adapter = BFTAdapter(moe, hidden_dim=64, num_classes=4).to(device)

    # Train
    train_moe_and_ranker(
        moe=moe,
        adapter=adapter,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        device=device,
        save_path=moe_save_path,
        ranker_save_path=ranker_save_path,
    )

    # Final test evaluation
    moe.load_state_dict(torch.load(moe_save_path, map_location=device))
    moe.eval()

    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for batch in test_loader:
            x = batch["eeg"].to(device)
            y = batch["label"].to(device)
            logits = moe(x)
            preds = torch.argmax(logits, dim=-1)
            test_correct += (preds == y).sum().item()
            test_total += len(y)

    test_acc = float(test_correct) / max(1, test_total)
    test_kappa = max(0.0, (test_acc - 0.25) / 0.75)

    logger.info(
        "Final Held-Out Test Evaluation: Accuracy: %.2f%% | Kappa: %.3f",
        test_acc * 100.0,
        test_kappa,
    )


if __name__ == "__main__":
    main()
