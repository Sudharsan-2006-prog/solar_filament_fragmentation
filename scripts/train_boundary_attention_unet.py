"""
scripts/train_boundary_attention_unet.py — Train the Boundary-Aware Attention U-Net.

This is the primary proposed model. It uses the combined
BCE + Dice + Boundary loss configured in config.yaml.

Example:
    python scripts/train_boundary_attention_unet.py --config configs/config.yaml --preprocessed_dir "..."
    python scripts/train_boundary_attention_unet.py --config configs/config.yaml --sanity
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import load_config
from src.train import run_training


def main():
    parser = argparse.ArgumentParser(description="Train Boundary-Aware Attention U-Net")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--preprocessed_dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None, dest="learning_rate")
    parser.add_argument("--bce_weight", type=float, default=None)
    parser.add_argument("--dice_weight", type=float, default=None)
    parser.add_argument("--boundary_weight", type=float, default=None)
    parser.add_argument("--sanity", action="store_true",
                        help="Run a 1-epoch sanity check on a tiny subset")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.preprocessed_dir:
        cfg["preprocessed_dir"] = args.preprocessed_dir
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        cfg["learning_rate"] = args.learning_rate
    if args.bce_weight is not None:
        cfg.setdefault("loss", {})["bce_weight"] = args.bce_weight
    if args.dice_weight is not None:
        cfg.setdefault("loss", {})["dice_weight"] = args.dice_weight
    if args.boundary_weight is not None:
        cfg.setdefault("loss", {})["boundary_weight"] = args.boundary_weight

    run_training(model_name="boundary_attention_unet", cfg=cfg, sanity=args.sanity)


if __name__ == "__main__":
    main()
