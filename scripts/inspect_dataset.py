"""
scripts/inspect_dataset.py — Inspect raw and/or preprocessed dataset structure.

Runs WITHOUT the dataset present and reports clearly what is missing.
When the dataset IS present, prints statistics from the manifest CSVs.

Usage:
    # Inspect raw dataset (before preprocessing)
    python scripts/inspect_dataset.py --raw_dir "..." 

    # Inspect preprocessed dataset (after preprocessing)
    python scripts/inspect_dataset.py --preprocessed_dir "..."

    # Inspect both
    python scripts/inspect_dataset.py --raw_dir "..." --preprocessed_dir "..."
"""

import sys
import os
import json
import argparse
from pathlib import Path


def check_raw_dataset(raw_dir: str):
    """Inspect the raw MAGFiLO dataset directory."""
    raw = Path(raw_dir)
    print("\n" + "="*60)
    print(f"RAW DATASET INSPECTION: {raw}")
    print("="*60)

    if not raw.exists():
        print("  ✗ Directory does NOT exist.")
        print(f"\n  Expected location:\n    {raw}")
        print("\n  ACTION REQUIRED:")
        print("    Download the MAGFiLO_1.0_Kaggle_2026 dataset from Kaggle")
        print("    and place it at the path above.")
        return False

    # Check required subdirectories
    train_img_dir = raw / "train" / "train_images"
    train_json = raw / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    test_img_dir = raw / "test" / "test_images"

    checks = [
        (train_img_dir, "train/train_images/"),
        (train_json, "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"),
        (test_img_dir, "test/test_images/"),
    ]

    all_ok = True
    for path, name in checks:
        if path.exists():
            if path.is_dir():
                n = len(list(path.glob("*.jpeg")) + list(path.glob("*.jpg")) + list(path.glob("*.png")))
                print(f"  ✓ {name}  ({n} image files)")
            else:
                size_mb = path.stat().st_size / 1e6
                print(f"  ✓ {name}  ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {name}  MISSING")
            all_ok = False

    if all_ok and train_json.exists():
        print("\n  Parsing annotation JSON (this may take a moment)...")
        with open(train_json, "r") as f:
            coco = json.load(f)
        print(f"    Image records in JSON : {len(coco['images'])}")
        print(f"    Annotation records    : {len(coco['annotations'])}")
        # Count unique filenames
        unique_fns = set(img["file_name"] for img in coco["images"])
        print(f"    Unique physical images: {len(unique_fns)}")
        cats = {c["id"]: c["name"] for c in coco.get("categories", [])}
        print(f"    Categories            : {cats}")

    return all_ok


def check_preprocessed_dataset(preprocessed_dir: str):
    """Inspect the preprocessed dataset directory."""
    prep = Path(preprocessed_dir)
    print("\n" + "="*60)
    print(f"PREPROCESSED DATASET INSPECTION: {prep}")
    print("="*60)

    if not prep.exists():
        print("  ✗ Directory does NOT exist.")
        print("\n  ACTION REQUIRED:")
        print("    Run preprocessing first:")
        print(f"    python solar_filament_segmentation_repo/preprocess_filament_dataset.py \\")
        print(f"        --raw_dir <RAW_DIR> \\")
        print(f"        --output_dir \"{preprocessed_dir}\"")
        return False

    # Check key subdirectories
    expected = [
        "images_512/train", "images_512/val", "images_512/test",
        "images_clahe_512/train", "images_clahe_512/val",
        "masks_512/binary/train", "masks_512/binary/val",
        "masks_512/multiclass/train", "masks_512/multiclass/val",
        "masks_512/spine/train", "masks_512/spine/val",
        "masks_512/agreement/train", "masks_512/agreement/val",
        "metadata",
    ]
    for rel in expected:
        full = prep / rel
        if full.exists():
            if full.is_dir():
                n = len(list(full.glob("*.png")))
                print(f"  ✓ {rel}/  ({n} PNG files)")
            else:
                print(f"  ✓ {rel}")
        else:
            print(f"  ✗ {rel}  MISSING")

    # Load manifest CSVs
    for split in ["train", "val", "test"]:
        csv_path = prep / "metadata" / f"{split}_manifest.csv"
        if csv_path.exists():
            import pandas as pd
            df = pd.read_csv(csv_path)
            print(f"\n  {split.capitalize()} manifest: {len(df)} samples")
            if "filament_area_pct" in df.columns and split != "test":
                print(f"    Filament area: min={df['filament_area_pct'].min():.3f}%  "
                      f"mean={df['filament_area_pct'].mean():.3f}%  "
                      f"max={df['filament_area_pct'].max():.3f}%")
            if "station" in df.columns:
                print(f"    Stations     : {dict(df['station'].value_counts())}")

    # Load summary JSON
    summary_json = prep / "metadata" / "dataset_summary.json"
    if summary_json.exists():
        with open(summary_json, "r") as f:
            summary = json.load(f)
        print(f"\n  Dataset Summary:")
        print(f"    Resolution  : {summary.get('resolution_scaled')} (scaled), {summary.get('resolution_raw')} (raw)")
        print(f"    Splits      : {summary.get('split_counts')}")
        print(f"    Filament coverage: {summary.get('filament_pixel_coverage_pct')}%")

    return True


def main():
    parser = argparse.ArgumentParser(description="Inspect MAGFiLO dataset structure")
    parser.add_argument("--raw_dir", default=None,
                        help="Path to raw MAGFiLO_1.0_Kaggle_2026 directory")
    parser.add_argument("--preprocessed_dir", default=None,
                        help="Path to preprocessed dataset directory")
    parser.add_argument("--config", default="configs/config.yaml",
                        help="Config file to read paths from (if not provided via args)")
    args = parser.parse_args()

    raw_dir = args.raw_dir
    prep_dir = args.preprocessed_dir

    # Fall back to config if paths not given
    if not raw_dir and not prep_dir:
        try:
            import yaml
            with open(args.config) as f:
                cfg = yaml.safe_load(f)
            raw_dir = cfg.get("raw_data_dir", "")
            prep_dir = cfg.get("preprocessed_dir", "")
            print(f"Reading paths from {args.config}")
        except Exception:
            pass

    if not raw_dir and not prep_dir:
        print("Usage: python scripts/inspect_dataset.py --raw_dir <PATH> [--preprocessed_dir <PATH>]")
        sys.exit(1)

    if raw_dir:
        check_raw_dataset(raw_dir)

    if prep_dir:
        check_preprocessed_dataset(prep_dir)

    print("\n" + "="*60)
    print("Inspection complete.")
    print("="*60)


if __name__ == "__main__":
    main()
