"""
Paper 2: Test-Time Adaptation (BFT) Evaluation
==============================================
Evaluates the Channel-Wise Mixture of Experts (MoE) classifier on unseen
target subjects / sessions with and without Backpropagation-Free Transformation (BFT).

Usage:
    python evaluate_tta.py
    python evaluate_tta.py --checkpoint checkpoints/moe_best.pt --num_transforms 6
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

# Add Paper 2 root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "scripts" else Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
CONFIGS_DIR = PROJECT_ROOT / "configs"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("c:/Users/Visha/Desktop/Cybernetics Motor/Motor") if Path("c:/Users/Visha/Desktop/Cybernetics Motor/Motor").exists() else PROJECT_ROOT.parent / "Motor"

from src.models.moe_backbone import ChannelWiseMoE
from src.models.bft_adapter import BFTAdapter
from src.data.physionet_loader import PhysioNetLoader, PhysioNetMIDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_subject_tta(model, bft_adapter, X_test, y_test, device):
    """
    Evaluate test accuracy with and without BFT.
    
    Args:
        model: Base classifier (ChannelWiseMoE)
        bft_adapter: BFTAdapter module
        X_test: Tensor of shape (N, C, T)
        y_test: Tensor of shape (N,)
        device: torch.device
        
    Returns:
        dict: {'acc_baseline': float, 'acc_bft': float, 'improvement': float}
    """
    model.eval()
    bft_adapter.eval()
    
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    with torch.no_grad():
        # 1. Baseline prediction (Standard single forward pass)
        logits_base = model(X_test)
        preds_base = torch.argmax(logits_base, dim=-1)
        acc_base = (preds_base == y_test).float().mean().item()
        
        # 2. BFT prediction (Ensemble over K neurophysiological transformations)
        logits_bft = bft_adapter(X_test)
        preds_bft = torch.argmax(logits_bft, dim=-1)
        acc_bft = (preds_bft == y_test).float().mean().item()
        
    return {
        "acc_baseline": acc_base * 100.0,
        "acc_bft": acc_bft * 100.0,
        "improvement": (acc_bft - acc_base) * 100.0
    }


def main():
    parser = argparse.ArgumentParser(description="Paper 2: MoE-BFT Test-Time Adaptation Evaluation")
    parser.add_argument("--config", type=str, default=str(CONFIGS_DIR / "config.yaml"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_test_subjects", type=int, default=5, help="Number of subjects to evaluate")
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINTS_DIR / "moe_best.pt"), help="Path to MoE checkpoint")
    parser.add_argument("--use_ea", action="store_true", help="Apply Euclidean Alignment preprocessing (He & Wu 2020)")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device(args.device)
    logger.info(f"Running BFT Evaluation on device: {device} (Euclidean Alignment = {args.use_ea})")

    # Initialize model
    model = ChannelWiseMoE(
        in_channels=64,
        num_classes=4,
    ).to(device)

    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        logger.info(f"Successfully loaded trained MoE weights from {ckpt_path}")

    # Initialize BFT adapter
    bft_adapter = BFTAdapter(
        moe_backbone=model,
        num_classes=4,
    ).to(device)

    ranker_path = CHECKPOINTS_DIR / "ranker_best.pt"
    if ranker_path.exists():
        bft_adapter.ranker.load_state_dict(torch.load(ranker_path, map_location=device))
        logger.info(f"Successfully loaded trained BFT ReliabilityRanker weights from {ranker_path}")

    # Load dataset
    data_path = cfg.data.get("data_dir", cfg.data.get("physionet_dir", str(DATA_DIR)))
    loader = PhysioNetLoader(data_dir=data_path)
    test_subjects = [s for s in range(1, 110) if s not in (88, 92, 100)][:args.num_test_subjects]

    results = []
    logger.info(f"Evaluating {len(test_subjects)} target subjects...")

    for subj in test_subjects:
        try:
            X, y = loader.load_subject_mi_trials(subj, align_euclidean=args.use_ea)
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.long)

            res = evaluate_subject_tta(model, bft_adapter, X_tensor, y_tensor, device)
            results.append(res)
            logger.info(
                f"Subject {subj}: Baseline Acc = {res['acc_baseline']:.2f}%, "
                f"BFT Acc = {res['acc_bft']:.2f}% (Δ = {res['improvement']:+.2f}%)"
            )
        except Exception as e:
            logger.warning(f"Could not evaluate subject {subj}: {e}")

    if results:
        mean_base = np.mean([r["acc_baseline"] for r in results])
        mean_bft = np.mean([r["acc_bft"] for r in results])
        mean_gain = np.mean([r["improvement"] for r in results])

        logger.info("=" * 60)
        logger.info(f"Average Baseline Accuracy : {mean_base:.2f}%")
        logger.info(f"Average BFT Accuracy      : {mean_bft:.2f}%")
        logger.info(f"Average Absolute Gain     : {mean_gain:+.2f}%")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
