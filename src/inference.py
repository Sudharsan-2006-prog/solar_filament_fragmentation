"""
inference.py — Generate qualitative predictions and overlay visualisations.

For each model, this script:
  1. Loads the best checkpoint.
  2. Runs inference on the VALIDATION split (ground truth available).
  3. Saves a configurable number of side-by-side figures:
       [Original Image | Ground Truth | Predicted Mask | Overlay]
  4. Optionally annotates each sample with its metrics.

Qualitative cases are selected to cover:
  - Images with large filaments (easy cases)
  - Images with thin/sparse filaments
  - Images with low filament pixel coverage (low contrast / hard cases)
  This is done by sorting the validation manifest by filament area.

Saves to: outputs/predictions/<model_name>/
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import load_config, set_seed, load_checkpoint
from src.dataset import FilamentDataset
from src.metrics import compute_binary_metrics


def get_val_samples_sorted(cfg: dict, n: int):
    """
    Return a list of (image_tensor, mask_tensor, metadata_dict) tuples.

    Samples are selected across the range of filament area percentages to
    include both easy (large filament) and hard (small/thin filament) cases.
    """
    preprocessed_dir = cfg["preprocessed_dir"]
    use_clahe = cfg.get("use_clahe", True)
    mask_type = cfg.get("mask_type", "binary")
    image_size = cfg.get("image_size", 512)

    import pandas as pd
    manifest_path = Path(preprocessed_dir) / "metadata" / "val_manifest.csv"
    df = pd.read_csv(manifest_path)

    # Sort by filament area (ascending) to get a range from hard to easy
    if "filament_area_pct" in df.columns:
        df = df.sort_values("filament_area_pct", ascending=True)

    # Pick n evenly-spaced samples across the sorted list to represent diversity
    total = len(df)
    indices = np.linspace(0, total - 1, min(n, total), dtype=int)

    dataset = FilamentDataset(
        preprocessed_dir=preprocessed_dir,
        split="val",
        use_clahe=use_clahe,
        mask_type=mask_type,
        transform=None,  # No augmentation for qualitative display
        image_size=image_size,
    )

    samples = []
    for idx in indices:
        img, mask = dataset[int(idx)]
        info = dataset.get_sample_info(int(idx))
        samples.append((img, mask, info))
    return samples


def save_prediction_figure(
    img: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    metrics: dict,
    save_path: Path,
    filename: str,
    filament_area_pct: float,
):
    """
    Save a 4-panel figure: Original | Ground Truth | Prediction | Overlay.

    Args:
        img: (H, W) float32 in [0, 1]
        gt_mask: (H, W) uint8 binary ground truth
        pred_mask: (H, W) uint8 binary prediction
        metrics: dict with dice, iou etc.
        save_path: Where to save the PNG.
        filename: Image filename for the figure title.
        filament_area_pct: Percentage of image that is filament (for difficulty label).
    """
    fig = plt.figure(figsize=(16, 4.5))
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.04)

    difficulty = (
        "easy" if filament_area_pct > 0.5 else
        "medium" if filament_area_pct > 0.15 else
        "hard"
    )
    title = (
        f"{filename}  [{difficulty}, area={filament_area_pct:.3f}%]  "
        f"Dice={metrics.get('dice', float('nan')):.3f}  "
        f"IoU={metrics.get('iou', float('nan')):.3f}"
    )
    fig.suptitle(title, fontsize=9, y=1.01)

    # Panel 1: Original image
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax0.set_title("Input H-α", fontsize=8)
    ax0.axis("off")

    # Panel 2: Ground truth mask
    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(gt_mask, cmap="hot", vmin=0, vmax=1)
    ax1.set_title("Ground Truth", fontsize=8)
    ax1.axis("off")

    # Panel 3: Predicted mask
    ax2 = fig.add_subplot(gs[2])
    ax2.imshow(pred_mask, cmap="hot", vmin=0, vmax=1)
    ax2.set_title("Prediction", fontsize=8)
    ax2.axis("off")

    # Panel 4: Overlay (img in gray, GT in green, prediction in red)
    ax3 = fig.add_subplot(gs[3])
    overlay = np.stack([img, img, img], axis=-1)
    overlay[..., 1] = np.clip(overlay[..., 1] + gt_mask.astype(float) * 0.5, 0, 1)  # GT in green
    overlay[..., 0] = np.clip(overlay[..., 0] + pred_mask.astype(float) * 0.4, 0, 1)  # Pred in red
    ax3.imshow(overlay)
    ax3.set_title("Overlay (G=GT, R=Pred)", fontsize=8)
    ax3.axis("off")

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_inference(model_name: str, cfg: dict, checkpoint_path: Path, n_samples: int):
    """Load checkpoint and save qualitative prediction images."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.get("random_seed", 42))

    # Build and load model
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

    load_checkpoint(model, checkpoint_path, device=device)
    model = model.to(device)
    model.eval()

    threshold = float(cfg.get("inference_threshold", 0.5))
    out_dir = Path(cfg.get("output", {}).get("predictions", "outputs/predictions")) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {n_samples} qualitative predictions for {model_name}...")
    samples = get_val_samples_sorted(cfg, n_samples)

    with torch.no_grad():
        for i, (img_tensor, mask_tensor, info) in enumerate(samples):
            img_batch = img_tensor.unsqueeze(0).to(device)
            logits = model(img_batch)
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred = (prob > threshold).astype(np.uint8)

            gt = mask_tensor.numpy().astype(np.uint8)
            img_np = img_tensor.squeeze().numpy()

            # Compute per-sample metrics
            pred_t = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).float()
            gt_t = torch.from_numpy(gt).unsqueeze(0).long()
            m = compute_binary_metrics(pred_t, gt_t, threshold=threshold)

            filename = info.get("filename", f"sample_{i:03d}")
            area_pct = float(info.get("filament_area_pct", 0.0))
            save_path = out_dir / f"{Path(filename).stem}_pred.png"
            save_prediction_figure(img_np, gt, pred, m, save_path, filename, area_pct)
            print(f"  [{i+1:3d}/{len(samples)}] {filename}  Dice={m.get('dice', float('nan')):.4f}")

    print(f"\nQualitative predictions saved to: {out_dir}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate qualitative segmentation predictions")
    parser.add_argument("--model", required=True,
                        choices=["unet", "attention_unet", "boundary_attention_unet"])
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to .pth checkpoint. Defaults to outputs/checkpoints/<model>_best.pth")
    parser.add_argument("--preprocessed_dir", default=None)
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Number of qualitative images to save (overrides config)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.preprocessed_dir:
        cfg["preprocessed_dir"] = args.preprocessed_dir

    ckpt_path = Path(args.checkpoint) if args.checkpoint else \
        Path(cfg.get("output", {}).get("checkpoints", "outputs/checkpoints")) / f"{args.model}_best.pth"

    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    n_samples = args.n_samples if args.n_samples is not None else \
        int(cfg.get("num_prediction_samples", 20))

    run_inference(args.model, cfg, ckpt_path, n_samples)


if __name__ == "__main__":
    main()
