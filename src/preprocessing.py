"""
src/preprocessing.py — Thin bridge to the repository's preprocessing script.

The actual preprocessing is implemented in:
    solar_filament_segmentation_repo/preprocess_filament_dataset.py

This module documents:
  1. What the preprocessing pipeline does
  2. How to invoke it
  3. What output it produces

Do NOT re-implement the preprocessing here.  The repository's implementation
is complete and well-documented.  Re-implementing it would risk introducing
inconsistencies.
"""

# =============================================================================
# Preprocessing documentation (informational — see repository script)
# =============================================================================

PREPROCESSING_SUMMARY = """
PREPROCESSING PIPELINE (implemented in solar_filament_segmentation_repo/preprocess_filament_dataset.py)
==========================================

INPUT:
  - Raw MAGFiLO_1.0_Kaggle_2026 dataset:
    <raw_dir>/train/train_images/         # 707 JPEG files (2048×2048, 8-bit grayscale)
    <raw_dir>/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json  # COCO-format
    <raw_dir>/test/test_images/           # 180 JPEG files (unannotated)

STAGES:
  Stage 1: Parse COCO JSON, group multi-annotator records by unique image filename
  Stage 2: Fuse multi-annotator masks via majority/union voting
           - K=1: single ground truth used directly
           - K=2: union (pixel filament if ≥1 annotator marked it)
           - K=3: majority vote (pixel filament if ≥2 annotators marked it)
  Stage 3: Generate full-resolution 2048×2048 masks (multiclass, binary, spine)
  Stage 4: Resize to 512×512 (INTER_AREA for images, INTER_NEAREST for masks)
           Apply CLAHE (clipLimit=2.0, tileGrid=8×8)
  Stage 5: Export YOLOv8/v11 polygon labels
  Stage 6: Observatory-stratified 80/20 train/val split (seed=42)
  Stage 7: Process test images (resize + CLAHE, no masks)

OUTPUT STRUCTURE:
  <preprocessed_dir>/
    images_512/{train,val,test}/        # 512×512 grayscale PNGs
    images_clahe_512/{train,val,test}/  # 512×512 CLAHE-enhanced PNGs  ← used for training
    masks_512/
      binary/{train,val}/               # 512×512 PNGs, values {0,255}  ← 0=bg, 255=filament
      multiclass/{train,val}/           # 512×512 PNGs, values {0,1,2,3,4}
      spine/{train,val}/                # 512×512 skeleton PNGs
      agreement/{train,val}/            # 512×512 annotator agreement count PNGs
    masks_full_2048/{multiclass,binary,spine}/
    labels_yolo/{train,val}/            # YOLO-format .txt files
    metadata/
      train_manifest.csv
      val_manifest.csv
      test_manifest.csv
      full_manifest.csv
      dataset_summary.json

MASK SEMANTICS (verified):
  Binary masks: stored as {0, 255}; dataset.py converts to {0, 1}
  Multiclass:   pixel values {0=Background, 1=Left, 2=Right, 3=Unidentifiable, 4=Ambiguous}
  Spine masks:  {0, 255}; centerline skeleton of filament
  Agreement:    {0, 1, 2, 3}; count of annotators agreeing at each pixel

CLASS IMBALANCE:
  Filaments occupy ~0.4–1.5% of pixels.
  Background: >98.5% of pixels.
  This is why BCE alone fails — Dice loss and Boundary loss are critical.

HOW TO RUN:
  python solar_filament_segmentation_repo/preprocess_filament_dataset.py \\
      --raw_dir "<RAW_DIR>" \\
      --output_dir "<PREPROCESSED_DIR>"

COMMAND TO REPRODUCE WITH CUSTOM SIZE:
  python solar_filament_segmentation_repo/preprocess_filament_dataset.py \\
      --raw_dir "<RAW_DIR>" \\
      --output_dir "<PREPROCESSED_DIR>" \\
      --target_size 512 \\
      --val_ratio 0.20 \\
      --seed 42
"""


def print_preprocessing_guide():
    """Print the preprocessing guide to stdout."""
    print(PREPROCESSING_SUMMARY)


if __name__ == "__main__":
    print_preprocessing_guide()
