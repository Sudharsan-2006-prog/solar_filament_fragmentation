"""
evaluate.py — Model evaluation and final metrics aggregation.

This script:
  1. Loads a trained model checkpoint.
  2. Evaluates on the VALIDATION split (test labels are not available).
  3. Computes all metrics and writes them to:
       outputs/metrics/<model_name>_eval.csv
  4. Appends a row to outputs/metrics/final_metrics.csv.
  5. Generates comparison charts for all evaluated models.

SCIENTIFIC INTEGRITY:
  - Never evaluates on the test set (which has no ground-truth labels).
  - Never invents metrics.
  - NaN is used when a metric cannot be computed.
"""

import sys
import argparse
import csv
import time
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import load_config, set_seed, load_checkpoint
from src.dataset import build_dataloaders
from src.metrics import compute_binary_metrics, MetricAccumulator
from src.losses import build_loss

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


FINAL_METRICS_CSV = Path("outputs/metrics/final_metrics.csv")
FINAL_METRICS_FIELDS = [
    "model", "experiment_id", "epochs", "best_epoch", "learning_rate",
    "batch_size", "optimizer", "loss_function", "training_time_seconds",
    "validation_loss", "test_loss",
    "dice", "iou", "precision", "recall", "f1", "pixel_accuracy",
    "panoptic_quality",
]


def evaluate_model(model_name: str, cfg: dict, checkpoint_path: Path) -> dict:
    """
    Evaluate a trained model on the validation split and return metric dict.

    Returns dict conforming to FINAL_METRICS_FIELDS.
    panoptic_quality is set to NaN because instance-level IDs are required
    for PQ computation and the semantic masks in this dataset do not provide them.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.get("random_seed", 42))

    # Build model
    if model_name == "unet":
        from src.models.unet import build_unet
        model = build_unet(cfg)
    elif model_name == "attention_unet":
        from src.models.attention_unet import build_attention_unet
        model = build_attention_unet(cfg)
    elif model_name == "boundary_attention_unet":
        from src.models.boundary_attention_unet import build_boundary_attention_unet
        model = build_boundary_attention_unet(cfg)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    epoch, ckpt_metrics, _ = load_checkpoint(model, checkpoint_path, device=device)
    model = model.to(device)
    model.eval()

    criterion = build_loss(cfg, model_name)
    use_boundary_loss = model_name == "boundary_attention_unet"

    _, val_loader = build_dataloaders(cfg, sanity=False)
    acc = MetricAccumulator()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)

            if use_boundary_loss:
                loss, _ = criterion(logits, masks)
            else:
                loss = criterion(logits, masks)

            m = compute_binary_metrics(logits, masks)
            acc.update(m)
            total_loss += loss.item()
            n_batches += 1

    metrics = acc.compute()
    val_loss = total_loss / max(n_batches, 1)

    # Read training time from existing per-model CSV if available
    training_time = float("nan")
    model_csv = Path(f"outputs/metrics/{model_name}_metrics.csv")
    if model_csv.exists():
        import pandas as pd
        df = pd.read_csv(model_csv)
        if "epoch" in df.columns:
            training_time = float("nan")  # Training time is logged in train.py summary

    result = {
        "model": model_name,
        "experiment_id": f"{model_name}_eval",
        "epochs": epoch,
        "best_epoch": epoch,
        "learning_rate": cfg.get("learning_rate", float("nan")),
        "batch_size": cfg.get("batch_size", float("nan")),
        "optimizer": cfg.get("optimizer", ""),
        "loss_function": "BCE+Dice+Boundary" if use_boundary_loss else "BCE+Dice",
        "training_time_seconds": training_time,
        "validation_loss": round(val_loss, 6),
        "test_loss": float("nan"),   # Test set has no labels
        "dice": round(metrics.get("dice", float("nan")), 6),
        "iou": round(metrics.get("iou", float("nan")), 6),
        "precision": round(metrics.get("precision", float("nan")), 6),
        "recall": round(metrics.get("recall", float("nan")), 6),
        "f1": round(metrics.get("f1", float("nan")), 6),
        "pixel_accuracy": round(metrics.get("pixel_accuracy", float("nan")), 6),
        # PQ requires instance-level IDs; not available in this dataset's semantic masks
        "panoptic_quality": float("nan"),
    }

    # Write per-model eval file
    eval_csv = Path(f"outputs/metrics/{model_name}_eval.csv")
    eval_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FINAL_METRICS_FIELDS)
        w.writeheader()
        w.writerow(result)
    print(f"  Saved eval metrics → {eval_csv}")

    return result


def append_to_final_metrics(result: dict):
    """Append a result row to the consolidated final_metrics.csv."""
    FINAL_METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not FINAL_METRICS_CSV.exists()
    with open(FINAL_METRICS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FINAL_METRICS_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(result)
    print(f"  Appended → {FINAL_METRICS_CSV}")


def generate_comparison_charts():
    """
    Read final_metrics.csv and generate bar-chart comparisons.
    Only generates charts if at least 2 models have been evaluated.
    """
    import pandas as pd

    if not FINAL_METRICS_CSV.exists():
        print("final_metrics.csv not found. Evaluate models first.")
        return

    df = pd.read_csv(FINAL_METRICS_CSV)
    if len(df) < 1:
        print("No rows in final_metrics.csv yet.")
        return

    comparison_dir = Path("outputs/figures/comparison")
    comparison_dir.mkdir(parents=True, exist_ok=True)

    for metric in ["dice", "iou", "precision", "recall", "f1", "pixel_accuracy"]:
        if metric not in df.columns:
            continue
        valid = df[df[metric].notna()]
        if valid.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(valid["model"], valid[metric], color=["steelblue", "darkorange", "forestgreen"][:len(valid)])
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"Model Comparison — {metric.replace('_', ' ').title()}")
        ax.set_ylim(0, 1.0)
        for bar, val in zip(bars, valid[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        save_path = comparison_dir / f"{metric}_comparison.png"
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"  Saved comparison chart → {save_path}")

    print(f"\nAll comparison charts saved to {comparison_dir}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model on validation split")
    parser.add_argument("--model", required=True,
                        choices=["unet", "attention_unet", "boundary_attention_unet"],
                        help="Model to evaluate")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to checkpoint .pth file. Defaults to outputs/checkpoints/<model>_best.pth")
    parser.add_argument("--preprocessed_dir", default=None,
                        help="Override preprocessed_dir from config")
    parser.add_argument("--compare", action="store_true",
                        help="Generate comparison charts from final_metrics.csv")
    args = parser.parse_args()

    if args.compare:
        generate_comparison_charts()
        return

    cfg = load_config(args.config)
    if args.preprocessed_dir:
        cfg["preprocessed_dir"] = args.preprocessed_dir

    ckpt_path = Path(args.checkpoint) if args.checkpoint else \
        Path(cfg.get("output", {}).get("checkpoints", "outputs/checkpoints")) / f"{args.model}_best.pth"

    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        print("Train the model first before evaluating.")
        sys.exit(1)

    print(f"\nEvaluating {args.model} from {ckpt_path}...")
    result = evaluate_model(args.model, cfg, ckpt_path)

    print("\n── Validation Metrics ──────────────────────────────")
    for k, v in result.items():
        if k in ["dice", "iou", "precision", "recall", "f1", "pixel_accuracy", "validation_loss"]:
            print(f"  {k:25s}: {v}")
    print("────────────────────────────────────────────────────\n")

    append_to_final_metrics(result)


if __name__ == "__main__":
    main()
