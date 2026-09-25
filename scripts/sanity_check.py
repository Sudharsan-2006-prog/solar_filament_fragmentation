"""
scripts/sanity_check.py — Minimal end-to-end pipeline verification.

Verifies:
  1. Dataset loads correctly (images and masks)
  2. Each model's forward pass produces correct output shape
  3. Each loss function computes a scalar without errors
  4. Backward pass runs without errors
  5. GPU is available and used if present
  6. Output predictions have the correct spatial dimensions

This script runs 1 epoch over a VERY small subset of the data.
It does NOT write checkpoints or metrics files.

Usage:
    python scripts/sanity_check.py --preprocessed_dir "..."
    python scripts/sanity_check.py --config configs/config.yaml
"""

import sys
import argparse
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import set_seed, load_config
from src.dataset import FilamentDataset
from src.losses import build_loss


MODELS = ["unet", "attention_unet", "boundary_attention_unet"]


def build_model(model_name: str, cfg: dict) -> torch.nn.Module:
    if model_name == "unet":
        from src.models.unet import build_unet
        return build_unet(cfg)
    elif model_name == "attention_unet":
        from src.models.attention_unet import build_attention_unet
        return build_attention_unet(cfg)
    elif model_name == "boundary_attention_unet":
        from src.models.boundary_attention_unet import build_boundary_attention_unet
        return build_boundary_attention_unet(cfg)


def run_sanity_check(cfg: dict, model_names: list):
    set_seed(cfg.get("random_seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    max_s = int(cfg["sanity"]["max_samples"])
    preprocessed_dir = cfg["preprocessed_dir"]
    if not preprocessed_dir:
        print("\nERROR: preprocessed_dir is empty in config.yaml")
        print("Please set it to the root of the preprocessed dataset first.")
        sys.exit(1)

    print(f"\nLoading {max_s} samples from: {preprocessed_dir}")

    # Load a tiny dataset
    try:
        dataset = FilamentDataset(
            preprocessed_dir=preprocessed_dir,
            split="train",
            use_clahe=cfg.get("use_clahe", True),
            mask_type=cfg.get("mask_type", "binary"),
            max_samples=max_s,
        )
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Run preprocessing first, then re-run this sanity check.")
        sys.exit(1)

    print(f"  Dataset size: {len(dataset)}")
    img0, mask0 = dataset[0]
    print(f"  Image shape : {img0.shape}, dtype: {img0.dtype}")
    print(f"  Mask shape  : {mask0.shape}, dtype: {mask0.dtype}")
    print(f"  Mask unique : {torch.unique(mask0).tolist()}")

    loader = torch.utils.data.DataLoader(dataset, batch_size=min(2, max_s), shuffle=False)
    images, masks = next(iter(loader))
    images = images.to(device)
    masks = masks.to(device)
    print(f"\n  Batch images: {images.shape}")
    print(f"  Batch masks : {masks.shape}")

    all_passed = True
    for model_name in model_names:
        print(f"\n{'─'*50}")
        print(f"  Checking model: {model_name}")
        try:
            model = build_model(model_name, cfg).to(device)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"    Parameters  : {n_params:,}")

            criterion = build_loss(cfg, model_name)
            use_boundary = model_name == "boundary_attention_unet"

            # Forward pass
            model.train()
            logits = model(images)
            print(f"    Logits shape: {logits.shape}")
            assert logits.shape[2:] == images.shape[2:], \
                f"Output spatial size mismatch: {logits.shape} vs input {images.shape}"

            # Loss
            if use_boundary:
                loss, comps = criterion(logits, masks)
                print(f"    Loss components: {comps}")
            else:
                loss = criterion(logits, masks)
            print(f"    Loss value  : {loss.item():.6f}")
            assert torch.isfinite(loss), "Loss is NaN or Inf!"

            # Backward pass
            loss.backward()
            print(f"    Backward    : OK")
            model.zero_grad()

            # Prediction shape
            model.eval()
            with torch.no_grad():
                pred = torch.sigmoid(model(images))
            print(f"    Pred shape  : {pred.shape}  range=[{pred.min():.3f}, {pred.max():.3f}]")
            print(f"  ✓ {model_name}: ALL CHECKS PASSED")

        except Exception as e:
            print(f"  ✗ {model_name}: FAILED — {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    print(f"\n{'='*50}")
    if all_passed:
        print("  ✓ SANITY CHECK PASSED — All models verified.")
    else:
        print("  ✗ SANITY CHECK FAILED — Fix errors before full training.")
    print("="*50)
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Sanity check: verify dataset + all models")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--preprocessed_dir", default=None,
                        help="Override preprocessed_dir from config")
    parser.add_argument("--model", default=None,
                        choices=MODELS,
                        help="Test only one model (default: test all 3)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.preprocessed_dir:
        cfg["preprocessed_dir"] = args.preprocessed_dir

    models_to_check = [args.model] if args.model else MODELS
    run_sanity_check(cfg, models_to_check)


if __name__ == "__main__":
    main()
