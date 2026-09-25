# Solar Filament Segmentation — IEEE Big Data Cup 2026, Challenge 02

## Problem Statement

This project addresses automated segmentation of solar filaments in full-disk H-α (656.3 nm) solar imagery from the GONG telescope network, submitted for the **IEEE Big Data Cup 2026 – Challenge 02: Solar Filament Segmentation** as a Principles of Machine Learning assignment.

Solar filaments are dense plasma clouds suspended in the solar corona by magnetic fields. Their eruptions trigger Coronal Mass Ejections (CMEs) that can disrupt satellite operations and power grids on Earth. Accurate automated segmentation is critical for space weather forecasting.

### Challenges
- Filaments occupy only **0.4–1.5%** of image pixels (extreme class imbalance)
- Thin, elongated structures (often 1–3 pixels wide at 512×512 resolution)
- Low contrast against the solar chromospheric background
- Multi-annotator disagreement in ground truth

---

## Dataset

**MAGFiLO 1.0 (MLEcoFi Kaggle 2026)** — available on Kaggle.

| Split | Images | Source |
|-------|--------|--------|
| Train | 565 | 80% of 707 GONG images |
| Val | 142 | 20% of 707 GONG images |
| Test | 180 | Competition evaluation set (no labels) |

**Raw structure:**
```
MAGFiLO_1.0_Kaggle_2026/
├── train/
│   ├── train_images/              # 707 JPEG, 2048×2048, 8-bit grayscale
│   └── MAGFiLO_1.0_Annotations_kaggle2026_train.json
└── test/
    └── test_images/               # 180 JPEG, unannotated
```

**Mask classes:**
| Value | Class | Meaning |
|-------|-------|---------|
| 0 | Background | Solar disk / off-limb |
| 1 | Left | Sinistral chirality filament |
| 2 | Right | Dextral chirality filament |
| 3 | Unidentifiable | Chirality unclear |
| 4 | Ambiguous | Conflicting chirality |

> **Note:** Binary training (classes 0/1) is the default. Multiclass can be enabled by changing `mask_type: "multiclass"` in `configs/config.yaml`.

---

## Repository Setup

```bash
# 1. Clone the preprocessing repository
git clone https://github.com/Sudharsan-2006-prog/solar_filament_fragmentation solar_filament_segmentation_repo

# 2. Navigate to the project directory
cd solar_filament_segmentation

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Preprocessing

The preprocessing is **already implemented** in `solar_filament_segmentation_repo/preprocess_filament_dataset.py`.

**After downloading the dataset:**

```bash
python ../solar_filament_segmentation_repo/preprocess_filament_dataset.py ^
    --raw_dir "C:\Users\sivas\OneDrive\Desktop\vs code files\ML\MAGFiLO_1.0_Kaggle_2026" ^
    --output_dir "C:\Users\sivas\OneDrive\Desktop\vs code files\ML\MAGFiLO_1.0_Kaggle_2026\preprocessed"
```

This generates:
- 512×512 PNG images (raw + CLAHE-enhanced)
- Binary masks `{0, 255}` → converted to `{0, 1}` at load time
- Multiclass masks `{0, 1, 2, 3, 4}`
- Spine skeleton masks
- Annotator agreement heatmaps
- Manifest CSVs for reproducible train/val splits

**Edit `configs/config.yaml` and set:**
```yaml
preprocessed_dir: "C:/Users/sivas/OneDrive/Desktop/vs code files/ML/MAGFiLO_1.0_Kaggle_2026/preprocessed"
```

---

## Models

### Model 1: U-Net (Baseline)
Standard encoder–decoder with skip connections.  
**Loss:** BCE + Dice  
**File:** `src/models/unet.py`

### Model 2: Attention U-Net
Adds soft attention gates to all skip connections, allowing the network to suppress irrelevant background (chromospheric network, sunspots) and focus on filament-bearing regions.  
**Loss:** BCE + Dice  
**File:** `src/models/attention_unet.py`

### Model 3: Boundary-Aware Attention U-Net (Proposed)
Extends Attention U-Net with a **Boundary-Aware Refinement Module (BARM)** — a parallel set of dilated convolutions (dilation 1, 2, 4) that explicitly learns multi-scale edge context. This is combined with a boundary supervision loss term.  
**Loss:** α·BCE + β·Dice + γ·BoundaryLoss (configurable)  
**File:** `src/models/boundary_attention_unet.py`

---

## Loss Functions

| Loss | Formula | Purpose |
|------|---------|---------|
| BCE | Standard binary cross-entropy | Per-pixel classification signal |
| Dice | 1 − (2·\|P∩G\| / (\|P\|+\|G\|)) | Handles class imbalance |
| Boundary | BCE on morphological boundary pixels | Thin structure edge supervision |

**Boundary extraction:** `Boundary = Mask − Erode(Mask)` via max-pool on inverted mask.  
This is valid for the MAGFiLO binary masks (verified: values are {0,255} stored, {0,1} after loading).

---

## Metrics

All metrics computed on the **validation set** (never the test set):

| Metric | Description |
|--------|-------------|
| Dice / F1 | 2·TP / (2·TP+FP+FN) |
| IoU | TP / (TP+FP+FN) |
| Precision | TP / (TP+FP) |
| Recall | TP / (TP+FN) |
| Pixel Accuracy | (TP+TN) / Total |
| Panoptic Quality | **NaN** — requires instance-level IDs not present in semantic masks |

> NaN is used honestly for any metric that cannot be computed.

---

## Training Commands

**See Section A–D in the terminal commands below.**

Quick reference:
```bash
python scripts/train_unet.py --config configs/config.yaml
python scripts/train_attention_unet.py --config configs/config.yaml
python scripts/train_boundary_attention_unet.py --config configs/config.yaml
```

---

## Evaluation Commands

```bash
# Evaluate all three models
python src/evaluate.py --model unet --config configs/config.yaml
python src/evaluate.py --model attention_unet --config configs/config.yaml
python src/evaluate.py --model boundary_attention_unet --config configs/config.yaml

# Generate comparison charts
python src/evaluate.py --compare
```

---

## Output Directory

```
outputs/
├── checkpoints/
│   ├── unet_best.pth
│   ├── attention_unet_best.pth
│   └── boundary_attention_unet_best.pth
├── metrics/
│   ├── unet_metrics.csv              (per-epoch training metrics)
│   ├── attention_unet_metrics.csv
│   ├── boundary_attention_unet_metrics.csv
│   └── final_metrics.csv             (consolidated evaluation results)
├── figures/
│   ├── unet/                         (loss, dice, iou, lr curves)
│   ├── attention_unet/
│   ├── boundary_attention_unet/
│   └── comparison/                   (bar charts across all models)
├── predictions/
│   ├── unet/
│   ├── attention_unet/
│   └── boundary_attention_unet/
└── logs/
    ├── unet_log.txt
    ├── attention_unet_log.txt
    └── boundary_attention_unet_log.txt
```

---

## Reproducibility

- Random seed: `42` (set in `config.yaml`, applied to Python, NumPy, PyTorch, CUDA)
- Train/val split: fixed, observatory-stratified, generated by preprocessing script with `--seed 42`
- All configurations in `configs/config.yaml`
- Do not tune hyperparameters on the test set

---

## Hardware Requirements

- **Minimum:** 8 GB RAM, any CPU (training will be slow)
- **Recommended:** NVIDIA GPU with ≥ 6 GB VRAM (e.g., RTX 3060)
- **AMP:** Automatically enabled when CUDA is available
- **Image size:** 512×512 with batch_size=8 requires ~3 GB VRAM

## Streamlit Live Demo

A Streamlit dashboard is included for live interactive segmentation of solar filaments. It allows uploading custom test images or selecting existing validation images, automatically applies the necessary CLAHE preprocessing, and runs inference using the trained models.

`ash
# Run the UI
streamlit run app.py
`

## U-Net Baseline Validation Results

The current U-Net baseline achieved the following metrics on the held-out validation set (Epoch 19):

- **Dice / F1:** 0.6132
- **IoU:** 0.4452
- **Precision:** 0.6488
- **Recall:** 0.5868
- **Pixel Accuracy:** 99.71%
