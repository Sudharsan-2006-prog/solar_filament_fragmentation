# MAGFiLO Solar Filament Segmentation Dataset: Preprocessing & Architecture Guide

## 1. Executive Summary

This document provides a comprehensive specification and technical guide for the preprocessing of the **MAGFiLO 1.0 (MLEcoFi Kaggle 2026)** dataset located at `D:\Downloads\filament-segmentation-2026`. 

The dataset is dedicated to the automated detection, semantic segmentation, and magnetic chirality classification of **solar filaments** observed in full-disk solar H-$\alpha$ ($656.3\text{ nm}$) imagery acquired by the **Global Oscillation Network Group (GONG)** telescope network.

The preprocessing pipeline resolves key challenges inherent to the raw dataset:
1. **Multi-Annotator Ambiguity**: Reconciles subjective annotations from multiple expert solar physicists via an automated consensus / majority-voting fusion algorithm.
2. **Resolution & Memory Optimization**: Generates both full-scale ($2048 \times 2048$) masks and ML-optimized ($512 \times 512$) images and masks with nearest-neighbor interpolation.
3. **Contrast & Chromospheric Feature Enhancement**: Applies Contrast-Limited Adaptive Histogram Equalization (CLAHE) to reveal subtle absorption fibrils.
4. **Leakage-Free Observatory-Stratified Splitting**: Ensures zero data leakage across multi-annotator entries while stratifying identically across all 6 planetary solar observatories.
5. **Multi-Framework Format Export**: Produces standard semantic segmentation masks ($0\dots 4$), binary masks, filament spine (skeleton) centerlines, inter-annotator agreement heatmaps, YOLOv8/v11 instance segmentation polygon labels, and metadata manifests.

---

## 2. Astrophysical Background & Domain Context

### 2.1 Solar Filaments & Chirality
Solar filaments (termed *prominences* when seen projected beyond the solar limb) are dense, cool clouds of plasma suspended in the hot solar corona by complex magnetic fields. When their magnetic structures destabilize, filaments erupt, triggering **Coronal Mass Ejections (CMEs)** and geomagnetic storms that impact satellites, power grids, and high-frequency communication on Earth.

A critical physical property of a filament is its **magnetic chirality** (handedness of the magnetic field):
* **Left-bearing (Sinistral)**: Barbs extend to the left of the main spine when viewed from positive to negative polarity.
* **Right-bearing (Dextral)**: Barbs extend to the right of the main spine.
* **Unidentifiable**: Barb structures are obscured, unresolved, or diffuse.
* **Ambiguous**: Conflicting chirality indicators along different segments of the filament.

Accurately segmenting the filament body and determining its chirality is essential for predicting the direction of CME magnetic fields (flux ropes) approaching Earth.

### 2.2 The GONG Observatory Network
The images originate from the **National Solar Observatory (NSO) / GONG** network, which operates six identical solar telescopes around the globe to ensure uninterrupted 24/7 monitoring of the Sun:

| Station Code | Observatory Name | Location | Train+Val Count | Test Count |
| :---: | :--- | :--- | :---: | :---: |
| **Bh** | Big Bear Solar Observatory | California, USA | 118 | 33 |
| **Ch** | Cerro Tololo Inter-American Observatory | La Serena, Chile | 125 | 33 |
| **Lh** | Learmonth Solar Observatory | Western Australia | 127 | 34 |
| **Mh** | Mauna Loa Solar Observatory | Hawaii, USA | 125 | 26 |
| **Th** | Teide Observatory | Canary Islands, Spain | 110 | 27 |
| **Uh** | Udaipur Solar Observatory | Rajasthan, India | 102 | 27 |
| **Total** | | | **707** | **180** |

---

## 3. Raw Dataset Analysis & Structural Properties

### 3.1 Raw Directory Structure
```
D:\Downloads\filament-segmentation-2026\
└── MAGFiLO_1.0_Kaggle_2026/
    ├── train/
    │   ├── train_images/                                 # 707 JPEG files (2048 x 2048, 8-bit grayscale)
    │   └── MAGFiLO_1.0_Annotations_kaggle2026_train.json # 48.6 MB COCO-format JSON
    └── test/
        └── test_images/                                  # 180 JPEG files (2048 x 2048, unannotated)
```

### 3.2 Multi-Annotator Multiplicity
The raw JSON contains **1,154 image records** and **8,199 filament annotations**, yet only **707 unique physical images** exist on disk. This reflects a multi-rater design:
* **Single Annotator**: 411 images ($58.1\%$)
* **Two Independent Annotators**: 145 images ($20.5\%$)
* **Three Independent Annotators**: 151 images ($21.4\%$)

Each annotator ID contains an identifier prefix (e.g., `050101-20111116063134Lh`). Pairwise inter-annotator Intersection-over-Union (IoU) on sample images averages **$\sim 51.9\%$**, demonstrating that solar physicists often disagree on fuzzy filament boundaries and faint chromospheric tails.

### 3.3 Target Categories

| Category ID | Category Name | Scientific Meaning | Mask Pixel Value | YOLO Class ID |
| :---: | :--- | :--- | :---: | :---: |
| `0` | **Background** | Solar disk photosphere / off-limb space | `0` | *N/A* |
| `1` | **Left** | Sinistral chirality filament | `1` | `0` |
| `2` | **Right** | Dextral chirality filament | `2` | `1` |
| `3` | **Unidentifiable** | Unidentifiable chirality | `3` | `2` |
| `4` | **Ambiguous** | Ambiguous chirality indicators | `4` | `3` |

### 3.4 Extreme Class Imbalance
Filaments occupy only **$0.40\% - 1.50\%$** of total image pixels. Over $98.5\%$ of every solar disk image is background. Models trained without class-balanced loss functions (e.g., Focal Tversky, Generalized Dice) will collapse toward predicting background.

---

## 4. Preprocessing Methodology

The end-to-end preprocessing pipeline is implemented in [preprocess_filament_dataset.py](file:///c:/Users/sudha/OneDrive%20-%20SSN-Institute/Documents/ML%20THEORY%20ASS/preprocess_filament_dataset.py). The pipeline consists of 7 modular stages:

```
Raw COCO JSON & 2048x2048 JPEGs
               │
               ▼
[Stage 1] Multi-Annotator Grouping & Polygon Parsing
               │
               ▼
[Stage 2] Consensus Fusion (Majority / Plurality Voting)
               │
         ┌─────┴─────────────────────────┐
         ▼                               ▼
[Stage 3] Full 2048x2048 Masks    [Stage 4] 512x512 Rescaling & CLAHE
  - Multiclass                     - Grayscale & CLAHE Images
  - Binary                         - Nearest-Neighbor Discrete Masks
  - Spine                          - Agreement Confidence Heatmaps
         │                               │
         └───────────────┬───────────────┘
                         ▼
[Stage 5] YOLOv8/v11-seg Polygon Contour Extraction
                         │
                         ▼
[Stage 6] Observatory-Stratified 80/20 Train/Val Split
                         │
                         ▼
[Stage 7] Test Image Preprocessing & Manifest Export
```

### Stage 1: Annotation Parsing & Multi-Rater Grouping
For each unique physical image $I$, all associated annotator IDs $\{id_1, \dots, id_K\}$ ($K \in \{1, 2, 3\}$) are extracted from the COCO metadata. The polygon coordinate lists $\mathbf{P}_k = \{P_{k,1}, \dots, P_{k,m}\}$ and spine coordinates $\mathbf{S}_k = \{S_{k,1}, \dots, S_{k,m}\}$ are parsed for each annotator $k$.

### Stage 2: Consensus Fusion Algorithm
Let $C_k(x, y) \in \{0, 1, 2, 3, 4\}$ denote the raster class assigned by annotator $k$ at pixel $(x, y)$, and $B_k(x, y) = \mathbb{I}(C_k(x, y) > 0)$ denote binary presence.

1. **Agreement Heatmap**:
   $$A(x, y) = \sum_{k=1}^K B_k(x, y) \in \{0, 1, \dots, K\}$$
2. **Consensus Binary Mask**:
   $$M_{\text{binary}}(x, y) = \begin{cases} 
   \mathbb{I}(A(x, y) \ge 1) & \text{if } K \le 2 \text{ (Union / Sensitive)} \\
   \mathbb{I}(A(x, y) \ge 2) & \text{if } K = 3 \text{ (Majority Vote)}
   \end{cases}$$
3. **Multiclass Consensus (Chirality Plurality)**:
   For every pixel where $M_{\text{binary}}(x, y) = 1$, the consensus class is computed as:
   $$M_{\text{multi}}(x, y) = \operatorname{mode}\left(\{ C_k(x, y) \mid C_k(x, y) > 0 \}\right)$$
   *Tie-breaker*: If two annotators vote for distinct classes (e.g. one votes Left ($1$) and one votes Right ($2$)), the pixel is classified as **Ambiguous ($4$)** if present, or assigned to the primary candidate.
4. **Filament Spine Union**:
   The spine polylines trace the magnetic neutral line of the filament. The consensus spine mask is the union of all annotator spines drawn with a line width of 2 pixels.

### Stage 3 & 4: Dual-Scale Image & Mask Rescaling
* **Full Resolution ($2048 \times 2048$)**: Exported as lossless 8-bit single-channel PNGs into `masks_full_2048/`. These preserve the raw instrument resolution for tiling, sliding-window inference, or high-capacity architectures.
* **Target Scale ($512 \times 512$)**:
  * **Images**: Downsampled using area interpolation (`cv2.INTER_AREA`), optimal for decimation without aliasing.
  * **Masks**: Downsampled strictly using **nearest-neighbor interpolation** (`cv2.INTER_NEAREST`). Bilinear or bicubic interpolation would generate non-existent intermediate values (e.g., fractional classes like $1.4$ or $2.7$), corrupting the ground truth.
  * **CLAHE Enhancement**: Standard full-disk solar images suffer from center-to-limb darkening and low contrast in faint fibrils. We apply CLAHE with clip limit $2.0$ and tile grid $8 \times 8$ (`cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))`), producing significantly crisper absorption boundaries.

### Stage 5: YOLOv8 / YOLOv11 Instance Segmentation Polygon Export
To support instance segmentation architectures (YOLOv8x-seg, YOLOv11x-seg), exterior contours are extracted from each isolated filament component using the Teh-Chin chain approximation algorithm (`cv2.CHAIN_APPROX_TC89_KCOS`). Coordinates are normalized to $[0.0, 1.0]$ and formatted as:
```
<class_id_0_indexed> <x_1> <y_1> <x_2> <y_2> ... <x_n> <y_n>
```

### Stage 6: Observatory-Stratified 80/20 Train/Validation Split
Splitting randomly on raw JSON entries would cause severe data leakage because the same physical image annotated by multiple raters would appear in both train and validation sets.
Instead:
1. Split is performed at the **unique physical image** level.
2. Split is **stratified across all 6 GONG observatories** (Bh, Ch, Lh, Mh, Th, Uh), guaranteeing that atmospheric conditions, telescope optics, and instrument transfer functions are equally represented in both splits.
3. Seed is fixed to `42` for exact reproducibility.

Split partition:
* **Train Set**: 565 images ($80.0\%$)
* **Validation Set**: 142 images ($20.0\%$)
* **Test Set**: 180 images (Evaluation/Competition set)

### Stage 7: Manifest & Metadata Compilation
The pipeline exports complete metadata catalogs:
* `metadata/train_manifest.csv`
* `metadata/val_manifest.csv`
* `metadata/test_manifest.csv`
* `metadata/full_manifest.csv`
* `metadata/dataset_summary.json`

Every manifest record includes: `filename`, `split`, `station`, `station_name`, `date_captured`, `annotator_count`, `filament_pixel_count`, `filament_area_pct`, `classes_present`, class pixel breakdowns, and relative paths to images, CLAHE images, multiclass masks, binary masks, spine masks, and YOLO label files.

---

## 5. Preprocessed Directory Layout

The preprocessed dataset is structured under `D:\Downloads\filament-segmentation-2026\preprocessed\`:

```
D:\Downloads\filament-segmentation-2026\preprocessed/
├── images_512/
│   ├── train/                  # 565 normalized 512x512 grayscale PNG images
│   ├── val/                    # 142 normalized 512x512 grayscale PNG images
│   └── test/                   # 180 normalized 512x512 test PNG images
├── images_clahe_512/
│   ├── train/                  # 565 CLAHE-enhanced 512x512 PNG images
│   ├── val/                    # 142 CLAHE-enhanced 512x512 PNG images
│   └── test/                   # 180 CLAHE-enhanced 512x512 test PNG images
├── masks_512/
│   ├── multiclass/
│   │   ├── train/              # 512x512 PNG masks (values: 0=Bg, 1=Left, 2=Right, 3=Unid, 4=Ambig)
│   │   └── val/
│   ├── binary/
│   │   ├── train/              # 512x512 PNG masks (values: 0=Background, 255=Filament)
│   │   └── val/
│   ├── spine/
│   │   ├── train/              # 512x512 PNG centerline skeletons (values: 0 or 255)
│   │   └── val/
│   └── agreement/
│       ├── train/              # 512x512 PNG rater agreement counts (values: 0, 1, 2, 3)
│       └── val/
├── masks_full_2048/
│   ├── multiclass/             # Full resolution 2048x2048 multiclass PNG masks
│   ├── binary/                 # Full resolution 2048x2048 binary PNG masks
│   └── spine/                  # Full resolution 2048x2048 spine skeleton PNG masks
├── labels_yolo/
│   ├── train/                  # 565 TXT files with normalized polygon contours
│   └── val/                    # 142 TXT files with normalized polygon contours
└── metadata/
    ├── dataset_summary.json    # Complete global statistics and parameters
    ├── train_manifest.csv      # Train split sample-level metadata
    ├── val_manifest.csv        # Validation split sample-level metadata
    ├── test_manifest.csv       # Test set sample-level metadata
    └── full_manifest.csv       # Unified dataset catalog
```

---

## 6. How to Use the Preprocessed Data in ML Workflows

### 6.1 Loading with PyTorch `Dataset` & `DataLoader`
A complete PyTorch data loader interface is provided in [dataset_loader.py](file:///c:/Users/sudha/OneDrive%20-%20SSN-Institute/Documents/ML%20THEORY%20ASS/dataset_loader.py):

```python
from dataset_loader import FilamentSegmentationDataset, get_data_loaders

# 1. Direct Dataset Access
train_dataset = FilamentSegmentationDataset(
    preprocessed_dir=r"D:\Downloads\filament-segmentation-2026\preprocessed",
    split="train",
    use_clahe=True,          # Uses CLAHE-enhanced contrast images
    mask_type="multiclass"   # 'multiclass', 'binary', or 'spine'
)

image, mask = train_dataset[0]
print(f"Image tensor shape : {image.shape}")   # torch.Size([1, 512, 512])
print(f"Mask tensor shape  : {mask.shape}")    # torch.Size([512, 512])
print(f"Unique class labels: {mask.unique()}") # e.g. tensor([0, 1, 2])

# 2. Batched DataLoaders
train_loader, val_loader = get_data_loaders(
    preprocessed_dir=r"D:\Downloads\filament-segmentation-2026\preprocessed",
    batch_size=16,
    use_clahe=True,
    mask_type="multiclass"
)

for images, masks in train_loader:
    # Forward pass in U-Net / SegFormer / DeepLabV3+
    # outputs = model(images)
    break
```

### 6.2 Training YOLOv8 / YOLOv11 Segmentation
To train a YOLO segmentation model using the generated `labels_yolo/`:
Create a `dataset.yaml`:
```yaml
path: D:/Downloads/filament-segmentation-2026/preprocessed
train: images_512/train
val: images_512/val
test: images_512/test

names:
  0: Left
  1: Right
  2: Unidentifiable
  3: Ambiguous
```
Then train via CLI or Python:
```bash
yolo segment train data=dataset.yaml model=yolov8m-seg.pt epochs=100 imgsz=512 batch=16
```

---

## 7. Recommended Training Strategies & Loss Functions

1. **Handling Extreme Class Imbalance**:
   Because background accounts for $>98.5\%$ of all pixels, standard Cross-Entropy loss will converge to trivial background-only predictions. 
   * **Recommended Loss**: Combination of Focal Loss and Dice Loss:
     $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Focal}}(\gamma=2.0) + \mathcal{L}_{\text{Dice}}$$
   * Alternatively, use **Focal Tversky Loss** with $\alpha=0.3, \beta=0.7$ to heavily penalize false negatives (missed filaments).
2. **Curriculum Strategy (Two-Stage Modeling)**:
   * **Stage 1 (Binary Detection)**: Train a binary segmentation network on `masks_512/binary/` to isolate filament presence from solar disk background.
   * **Stage 2 (Chirality Classification)**: Classify the extracted filament regions into Left, Right, Unidentifiable, or Ambiguous.
3. **Data Augmentation**:
   * Solar images have natural rotational invariance. Use random rotations ($0^\circ - 360^\circ$), horizontal and vertical flips, and random affine scaling.
   * *Caution on Horizontal Flips*: Flipping an image horizontally **inverts the chirality** (a Left-bearing filament becomes Right-bearing!). When applying horizontal flips (`hflip`), you must swap class labels $1 \leftrightarrow 2$!
4. **Leveraging Agreement Maps for Soft Labels**:
   Use the `masks_512/agreement/` heatmaps as confidence weights during loss calculation. Pixels agreed upon by all 3 annotators can be weighted $3\times$ higher than pixels marked by only 1 annotator.

---

## 8. Command-Line Reproduction

To re-run or customize the preprocessing pipeline with different image dimensions or validation ratios:

```bash
# Re-run at 512x512 (default)
python preprocess_filament_dataset.py

# Re-run at 1024x1024 resolution
python preprocess_filament_dataset.py --target_size 1024

# Re-run with custom split ratio and seed
python preprocess_filament_dataset.py --val_ratio 0.15 --seed 123
```
