"""
augmentations.py — Albumentations transforms for solar filament segmentation.

KEY DOMAIN CONSTRAINT:
  Horizontal flipping INVERTS the chirality label of a filament.
  A sinistral (Left, class 1) filament becomes dextral (Right, class 2) and
  vice versa.  The custom ChiralityFlipTransform below swaps labels 1 ↔ 2
  immediately after any horizontal flip, so that multiclass masks remain
  physically correct.

  This concern only applies to "multiclass" mask_type.  For binary masks
  the flip is a standard augmentation with no special handling needed.
"""

import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import DualTransform


# =============================================================================
# Custom transform: swap chirality labels after horizontal flip
# =============================================================================

class ChiralityAwareHorizontalFlip(DualTransform):
    """
    Horizontal flip that also swaps multiclass chirality labels 1 ↔ 2.

    In the MAGFiLO dataset:
        class 1 = Left (sinistral)
        class 2 = Right (dextral)
    Mirroring the image reverses the apparent handedness, so the label must
    be swapped to remain physically correct.

    If you are training with binary masks (mask_type="binary"), this transform
    behaves identically to a standard HorizontalFlip.
    """

    def __init__(self, p: float = 0.5):
        super().__init__(p=p)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return np.fliplr(img)

    def apply_to_mask(self, mask: np.ndarray, **params) -> np.ndarray:
        flipped = np.fliplr(mask)
        # Swap chirality labels 1 ↔ 2 (only meaningful for multiclass masks)
        swapped = flipped.copy()
        swapped[flipped == 1] = 2
        swapped[flipped == 2] = 1
        return swapped

    def get_transform_init_args_names(self):
        return ()


# =============================================================================
# Public transform factories
# =============================================================================

def get_train_transform(image_size: int = 512) -> A.Compose:
    """
    Training augmentation pipeline for solar H-alpha imagery.

    Rationale for each augmentation:
    - Random rotations (0–360°): The Sun has no fixed orientation in GONG
      telescope images, making full rotational augmentation physically valid.
    - Vertical flip: Equivalent to viewing from the southern hemisphere;
      physically valid and doubles effective dataset size.
    - ChiralityAwareHorizontalFlip: Flips the image while correcting the
      multiclass label to preserve chirality semantics.
    - Elastic distortion: Simulates seeing-induced wavefront aberrations
      that cause minor structural deformations in H-alpha observations.
    - GridDistortion: Approximates atmospheric dispersion effects.
    - RandomBrightnessContrast + CLAHE: Mimics exposure variations across
      GONG stations with different sky conditions.
    - GaussianBlur: Models point-spread-function (PSF) variation between
      telescopes, which affects filament boundary sharpness.
    - GaussNoise: Models photon noise and CCD readout noise.
    """
    return A.Compose([
        # --- Geometric transforms ---
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=180, border_mode=0, p=0.7),  # Full 360° coverage via two 180° ranges
        A.VerticalFlip(p=0.5),
        ChiralityAwareHorizontalFlip(p=0.5),
        A.ElasticTransform(alpha=1.0, sigma=50, p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.05, p=0.2),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=0, p=0.3),

        # --- Photometric transforms (image only; masks unaffected) ---
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(var_limit=(5.0, 25.0), p=0.2),

        # --- Ensure output is exactly image_size × image_size ---
        A.Resize(image_size, image_size, always_apply=True),
    ])


def get_val_transform(image_size: int = 512) -> A.Compose:
    """
    Validation/test transform: only resize (no stochastic augmentations).
    Using augmentations at validation time would make metrics noisy and
    non-reproducible across runs.
    """
    return A.Compose([
        A.Resize(image_size, image_size, always_apply=True),
    ])
