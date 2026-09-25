"""
train.py — Core training engine for all three models.

This module implements the full training loop with:
  - Automatic CPU/GPU detection
  - Mixed-precision training (torch.cuda.amp) when CUDA is available
  - Reproducible seeding
  - Early stopping on validation Dice
  - Best and latest checkpoint saving
  - Per-epoch metric CSV writing
  - Training curve plot saving
  - Clear progress display via tqdm

Usage: This module is called by the individual scripts in scripts/.
"""

import os
import sys
import time
import random
import csv
import json
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import yaml

# ── Project imports ──────────────────────────────────────────────────────────
# Allow running from project root: python src/train.py ...
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import build_dataloaders
from src.losses import build_loss
from src.metrics import compute_binary_metrics, MetricAccumulator
from src.utils import (
    set_seed, build_optimizer, build_scheduler,
    save_checkpoint, load_checkpoint,
    save_training_curves,
)


# =============================================================================
# Training loop
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    scaler: Optional[GradScaler],
    device: torch.device,
    cfg: dict,
    epoch: int,
    use_boundary_loss: bool = False,
) -> dict:
    """Run one training epoch and return mean metrics."""
    model.train()
    acc = MetricAccumulator()
    total_loss = 0.0
    n_batches = 0
    grad_clip = cfg.get("grad_clip_norm", 1.0)

    bar = tqdm(loader, desc=f"[Epoch {epoch}] Train", leave=False, dynamic_ncols=True)
    for images, masks in bar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            with autocast():
                logits = model(images)
                if use_boundary_loss:
                    loss, _ = criterion(logits, masks)
                else:
                    loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            if use_boundary_loss:
                loss, _ = criterion(logits, masks)
            else:
                loss = criterion(logits, masks)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        with torch.no_grad():
            m = compute_binary_metrics(logits.detach(), masks)
        acc.update(m)
        total_loss += loss.item()
        n_batches += 1
        bar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{m['dice']:.4f}")

    metrics = acc.compute()
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
    use_boundary_loss: bool = False,
) -> dict:
    """Run one validation epoch and return mean metrics."""
    model.eval()
    acc = MetricAccumulator()
    total_loss = 0.0
    n_batches = 0

    bar = tqdm(loader, desc="[Val]    ", leave=False, dynamic_ncols=True)
    for images, masks in bar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

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
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


# =============================================================================
# Main training orchestrator
# =============================================================================

def run_training(
    model_name: str,
    cfg: dict,
    sanity: bool = False,
    resume: Optional[str] = None,
):
    """
    Full training pipeline for a given model.

    Args:
        model_name: 'unet' | 'attention_unet' | 'boundary_attention_unet'
        cfg:        Parsed config.yaml as dict.
        sanity:     If True, run a minimal 1-epoch check.
    """
    # ── Reproducibility ───────────────────────────────────────────────────────
    seed = cfg.get("random_seed", 42)
    set_seed(seed)

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None
    print(f"\n{'='*60}")
    print(f"  Model   : {model_name}")
    print(f"  Device  : {device}  |  AMP: {use_amp}")
    print(f"  Sanity  : {sanity}")
    print(f"{'='*60}\n")

    # ── Build model ───────────────────────────────────────────────────────────
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
        raise ValueError(f"Unknown model_name: {model_name}")

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(cfg, sanity=sanity)
    print(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── Loss, optimizer, scheduler ────────────────────────────────────────────
    criterion = build_loss(cfg, model_name)
    use_boundary_loss = model_name == "boundary_attention_unet"

    optimizer = build_optimizer(model, cfg)
    epochs = cfg["sanity"]["epochs"] if sanity else cfg["epochs"]
    scheduler = build_scheduler(optimizer, cfg, epochs)

    # ── Output paths ──────────────────────────────────────────────────────────
    out_cfg = cfg.get("output", {})
    ckpt_dir = Path(out_cfg.get("checkpoints", "outputs/checkpoints"))
    metrics_dir = Path(out_cfg.get("metrics", "outputs/metrics"))
    figures_dir = Path(out_cfg.get("figures", "outputs/figures")) / model_name
    logs_dir = Path(out_cfg.get("logs", "outputs/logs"))
    for d in [ckpt_dir, metrics_dir, figures_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    ckpt_best = ckpt_dir / f"{model_name}_best.pth"
    ckpt_last = ckpt_dir / f"{model_name}_last.pth"
    metrics_csv = metrics_dir / f"{model_name}_metrics.csv"
    log_file = logs_dir / f"{model_name}_log.txt"

    # ── Resume logic ──────────────────────────────────────────────────────────
    start_epoch = 0
    patience = cfg.get("early_stopping_patience", 15)
    best_val_dice = -1.0
    epochs_no_improve = 0
    best_epoch = 0

    if resume:
        resume_path = Path(resume)
        if resume_path.exists():
            print(f"  Resuming from: {resume_path}")
            epoch_loaded, metrics_loaded, best_val_dice_loaded = load_checkpoint(
                model, resume_path, optimizer, device, scheduler, scaler
            )
            start_epoch = epoch_loaded
            best_val_dice = best_val_dice_loaded
            best_epoch = start_epoch
            print(f"  ✓ Resumed at Epoch {start_epoch} | Best val Dice so far: {best_val_dice:.4f}")
        else:
            print(f"  ✗ Resume path not found: {resume_path}")
            sys.exit(1)

    # ── CSV header ────────────────────────────────────────────────────────────
    csv_fields = [
        "epoch", "train_loss", "val_loss",
        "train_dice", "val_dice",
        "train_iou", "val_iou",
        "train_precision", "val_precision",
        "train_recall", "val_recall",
        "train_pixel_accuracy", "val_pixel_accuracy",
        "lr",
    ]
    
    open_mode = "a" if resume and start_epoch > 0 else "w"
    with open(metrics_csv, open_mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        if open_mode == "w":
            writer.writeheader()

    # ── Training history (for plots) ─────────────────────────────────────────
    history = {k: [] for k in csv_fields}
    
    # If resuming, read existing history so plots are complete
    if resume and metrics_csv.exists():
        import pandas as pd
        try:
            df = pd.read_csv(metrics_csv)
            for col in csv_fields:
                if col in df.columns:
                    history[col] = df[col].tolist()
        except Exception as e:
            print(f"  [Warn] Failed to load history for plots: {e}")

    start_time = time.time()

    for epoch in range(start_epoch + 1, epochs + 1):
        # --- Train ---
        t_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, cfg, epoch, use_boundary_loss=use_boundary_loss,
        )
        # --- Validate ---
        v_metrics = validate_one_epoch(
            model, val_loader, criterion, device,
            use_boundary_loss=use_boundary_loss,
        )

        # --- LR step ---
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            if cfg.get("scheduler") == "plateau":
                scheduler.step(v_metrics["dice"])
            else:
                scheduler.step()

        # --- Print summary ---
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Loss {t_metrics['loss']:.4f}/{v_metrics['loss']:.4f} | "
            f"Dice {t_metrics['dice']:.4f}/{v_metrics['dice']:.4f} | "
            f"IoU {t_metrics['iou']:.4f}/{v_metrics['iou']:.4f} | "
            f"LR {current_lr:.2e}"
        )

        # --- Log to file ---
        with open(log_file, "a") as lf:
            lf.write(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Loss {t_metrics['loss']:.4f}/{v_metrics['loss']:.4f} | "
                f"Dice {t_metrics['dice']:.4f}/{v_metrics['dice']:.4f} | "
                f"IoU {t_metrics['iou']:.4f}/{v_metrics['iou']:.4f} | "
                f"LR {current_lr:.2e}\n"
            )

        # --- Save CSV row ---
        row = {
            "epoch": epoch,
            "train_loss": round(t_metrics["loss"], 6),
            "val_loss": round(v_metrics["loss"], 6),
            "train_dice": round(t_metrics.get("dice", float("nan")), 6),
            "val_dice": round(v_metrics.get("dice", float("nan")), 6),
            "train_iou": round(t_metrics.get("iou", float("nan")), 6),
            "val_iou": round(v_metrics.get("iou", float("nan")), 6),
            "train_precision": round(t_metrics.get("precision", float("nan")), 6),
            "val_precision": round(v_metrics.get("precision", float("nan")), 6),
            "train_recall": round(t_metrics.get("recall", float("nan")), 6),
            "val_recall": round(v_metrics.get("recall", float("nan")), 6),
            "train_pixel_accuracy": round(t_metrics.get("pixel_accuracy", float("nan")), 6),
            "val_pixel_accuracy": round(v_metrics.get("pixel_accuracy", float("nan")), 6),
            "lr": current_lr,
        }
        with open(metrics_csv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writerow(row)
        for k, v in row.items():
            history[k].append(v)

        # --- Checkpoint: latest ---
        save_checkpoint(
            model, optimizer, epoch, v_metrics, ckpt_last,
            scheduler=scheduler, scaler=scaler, best_val_dice=best_val_dice
        )

        # --- Checkpoint: best ---
        val_dice = v_metrics.get("dice", 0.0)
        if not np.isnan(val_dice) and val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch
            epochs_no_improve = 0
            save_checkpoint(
                model, optimizer, epoch, v_metrics, ckpt_best,
                scheduler=scheduler, scaler=scaler, best_val_dice=best_val_dice
            )
            print(f"  ✓ New best val Dice: {best_val_dice:.4f} (epoch {best_epoch}) — checkpoint saved.")
        else:
            epochs_no_improve += 1

        # --- Early stopping ---
        if not sanity and epochs_no_improve >= patience:
            print(f"\nEarly stopping after {patience} epochs without improvement.")
            break

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  Training complete: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Best val Dice: {best_val_dice:.4f} at epoch {best_epoch}")
    print(f"  Best checkpoint: {ckpt_best}")
    print(f"  Metrics CSV: {metrics_csv}")
    print(f"{'='*60}\n")

    # ── Save training curves ──────────────────────────────────────────────────
    if not sanity:
        save_training_curves(history, model_name, figures_dir)

    # Return summary for aggregation
    return {
        "model": model_name,
        "best_epoch": best_epoch,
        "best_val_dice": best_val_dice,
        "training_time_seconds": round(elapsed, 1),
    }
