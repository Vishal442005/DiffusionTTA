"""
Paper 2: Master End-to-End Pipeline Execution Script
====================================================
Runs the complete sequence:
1. Stage 1 & 2: Trains Resting-State VAE (20 epochs) & Latent Diffusion (100 epochs) on CUDA.
2. Stage 3: Trains Channel-Wise MoE Classifier with 50% Real + 50% Diffusion-Synthesized Trials (80 epochs).
3. Stage 4: Evaluates Backpropagation-Free Test-Time Adaptation (BFT) on held-out test subjects.

Usage:
    python run_paper2_full.py
"""

import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [Paper2_Pipeline]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Paper2_Pipeline")

PROJECT_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "scripts" else Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def run_stage(stage_name: str, cmd_args: list[str]):
    logger.info("=" * 80)
    logger.info("STARTING: %s", stage_name)
    logger.info("Command: %s", " ".join(cmd_args))
    logger.info("=" * 80)
    
    proc = subprocess.Popen(
        cmd_args,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    
    for line in iter(proc.stdout.readline, ""):
        print(line, end="", flush=True)
        
    proc.stdout.close()
    return_code = proc.wait()
    
    if return_code != 0:
        logger.error("Stage failed with exit code %d: %s", return_code, stage_name)
        sys.exit(return_code)
        
    logger.info("FINISHED SUCCESSFULLY: %s", stage_name)


def main():
    logger.info("Starting Paper 2 Full Training and Evaluation on GPU...")
    
    diff_ckpt = PROJECT_ROOT / "checkpoints" / "diffusion_best.pt"
    rest_ckpt = PROJECT_ROOT / "checkpoints" / "resting_encoder_best.pt"

    force_retrain = "--force_retrain" in sys.argv
    if diff_ckpt.exists() and rest_ckpt.exists() and not force_retrain:
        logger.info("Found trained diffusion checkpoints (%s, %s). Proceeding directly to Classifier & TTA training.", rest_ckpt.name, diff_ckpt.name)
    else:
        # 1. Diffusion Training (Resting VAE + Latent Diffusion)
        run_stage(
            "Stage 1 & 2: Resting VAE + Latent Diffusion Training",
            [
                sys.executable,
                str(SCRIPTS_DIR / "train_diffusion.py"),
                "--device", "cuda",
                "--rest_epochs", "20",
                "--epochs", "100",
                "--batch_size", "32",
            ],
        )
    
    # 2. Classifier Training (Channel-Wise MoE with Synthetic Augmentation)
    run_stage(
        "Stage 3: Channel-Wise MoE Classifier Training (Synthetic Augmentation)",
        [
            sys.executable,
            str(SCRIPTS_DIR / "train_classifier.py"),
            "--device", "cuda",
            "--use_synthetic",
            "--synthetic_ratio", "0.5",
            "--epochs", "80",
        ],
    )
    
    # 3. Test-Time Adaptation Evaluation
    run_stage(
        "Stage 4: Test-Time Adaptation (BFT) Zero-Calibration Evaluation",
        [
            sys.executable,
            str(SCRIPTS_DIR / "evaluate_tta.py"),
            "--device", "cuda",
            "--checkpoint", str(PROJECT_ROOT / "checkpoints" / "moe_best.pt"),
            "--num_test_subjects", "5",
        ],
    )
    
    logger.info("=" * 80)
    logger.info("PAPER 2 END-TO-END PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("Checkpoints stored in: %s", PROJECT_ROOT / "checkpoints")
    logger.info("Results stored in: %s", PROJECT_ROOT / "results")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
