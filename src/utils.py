"""
utils.py — Shared utility functions for the training pipeline.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe for headless servers
import matplotlib.pyplot as plt


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int = 42):
    """
    Set all random seeds for reproducibility across CPU, CUDA, NumPy, and Python.

    Note: Full determinism on GPU requires CUDA_LAUNCH_BLOCKING=1 and may reduce
    performance.  We seed everything but do not enforce full CUDA determinism
    to keep training speed acceptable.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Config loading
# =============================================================================

def load_config(config_path: str) -> dict:
    """Load a YAML config file and return as a Python dict."""
    import yaml
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def override_config(cfg: dict, overrides: dict) -> dict:
    """
    Apply command-line overrides to the config dict.

    Supports top-level keys and nested keys with '.' notation.
    Example: --data_dir "/path" overrides cfg['preprocessed_dir'].
    """
    for key, val in overrides.items():
        if val is None:
            continue
        if "." in key:
            parts = key.split(".", 1)
            if parts[0] not in cfg:
                cfg[parts[0]] = {}
            cfg[parts[0]][parts[1]] = val
        else:
            cfg[key] = val
    return cfg


# =============================================================================
# Optimizer factory
# =============================================================================

def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """Build optimizer from config."""
    lr = float(cfg.get("learning_rate", 1e-4))
    wd = float(cfg.get("weight_decay", 1e-4))
    opt_name = cfg.get("optimizer", "adamw").lower()

    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}. Choose from: adam, adamw, sgd")


# =============================================================================
# Scheduler factory
# =============================================================================

def build_scheduler(optimizer, cfg: dict, n_epochs: int):
    """Build learning rate scheduler from config."""
    sched_name = cfg.get("scheduler", "cosine").lower()
    min_lr = float(cfg.get("scheduler_min_lr", 1e-6))

    if sched_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=min_lr
        )
    elif sched_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(cfg.get("scheduler_step_size", 20)),
            gamma=float(cfg.get("scheduler_gamma", 0.5)),
        )
    elif sched_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=5, factor=0.5, min_lr=min_lr
        )
    elif sched_name == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {sched_name}. Choose from: cosine, step, plateau, none")


# =============================================================================
# Checkpointing
# =============================================================================

def save_checkpoint(
    model: nn.Module,
    optimizer,
    epoch: int,
    metrics: dict,
    path: Path,
    scheduler=None,
    scaler=None,
    best_val_dice: float = None,
):
    """Save model checkpoint with optimizer, scheduler, scaler state and metrics."""
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        ckpt["scaler_state_dict"] = scaler.state_dict()
    if best_val_dice is not None:
        ckpt["best_val_dice"] = best_val_dice
    torch.save(ckpt, path)


def load_checkpoint(
    model: nn.Module,
    path: Path,
    optimizer=None,
    device=None,
    scheduler=None,
    scaler=None,
):
    """
    Load model (and optionally optimizer, scheduler, scaler) state from checkpoint.

    Returns:
        epoch (int), metrics (dict), best_val_dice (float)
    """
    if device is None:
        device = torch.device("cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    
    model.load_state_dict(ckpt["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    elif scheduler is not None:
        print("  [Warn] No scheduler_state_dict in checkpoint. Initialized safely.")
        
    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    elif scaler is not None:
        print("  [Warn] No scaler_state_dict in checkpoint. Initialized safely.")
        
    epoch = ckpt.get("epoch", 0)
    metrics = ckpt.get("metrics", {})
    
    best_val_dice = ckpt.get("best_val_dice")
    if best_val_dice is None:
        best_val_dice = metrics.get("dice", -1.0)
        print(f"  [Warn] No best_val_dice in checkpoint. Inferred {best_val_dice:.4f} from metrics.")
        
    return epoch, metrics, best_val_dice


# =============================================================================
# Training curve plotting
# =============================================================================

def save_training_curves(history: dict, model_name: str, output_dir: Path):
    """
    Save training/validation metric curves as PNG figures.

    Saves one figure per metric pair (train + val on the same plot):
      - loss vs epoch
      - dice vs epoch
      - iou vs epoch
      - precision vs epoch
      - recall vs epoch
      - lr vs epoch
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = history.get("epoch", list(range(1, len(history["train_loss"]) + 1)))

    # --- Loss ---
    _plot_pair(
        epochs,
        history["train_loss"], history["val_loss"],
        "Loss", model_name,
        output_dir / "loss_curve.png",
    )
    # --- Dice ---
    _plot_pair(
        epochs,
        history["train_dice"], history["val_dice"],
        "Dice Score", model_name,
        output_dir / "dice_curve.png",
    )
    # --- IoU ---
    _plot_pair(
        epochs,
        history["train_iou"], history["val_iou"],
        "IoU Score", model_name,
        output_dir / "iou_curve.png",
    )
    # --- Precision ---
    _plot_pair(
        epochs,
        history["train_precision"], history["val_precision"],
        "Precision", model_name,
        output_dir / "precision_curve.png",
    )
    # --- Recall ---
    _plot_pair(
        epochs,
        history["train_recall"], history["val_recall"],
        "Recall", model_name,
        output_dir / "recall_curve.png",
    )
    # --- LR ---
    _plot_single(
        epochs, history["lr"],
        "Learning Rate", model_name,
        output_dir / "lr_curve.png",
    )


def _plot_pair(epochs, train_vals, val_vals, metric_name, model_name, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_vals, label=f"Train {metric_name}", marker=".", markersize=3)
    ax.plot(epochs, val_vals, label=f"Val {metric_name}", marker=".", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{model_name} — {metric_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_single(epochs, vals, metric_name, model_name, save_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, vals, label=metric_name, color="green", marker=".", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{model_name} — {metric_name}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
