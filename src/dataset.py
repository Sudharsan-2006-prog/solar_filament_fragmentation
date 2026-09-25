"""
dataset.py — PyTorch Dataset for the preprocessed MAGFiLO solar filament dataset.

This module wraps the preprocessed output of preprocess_filament_dataset.py
(from the cloned repository) and exposes a clean Dataset/DataLoader interface
that reads the manifest CSVs rather than re-discovering files by glob.
Using manifests guarantees that the exact same observatory-stratified split
produced by the preprocessing script is used during training and evaluation.
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class FilamentDataset(Dataset):
    """
    PyTorch Dataset for solar filament segmentation (MAGFiLO / MLEcoFi 2026).

    Reads image and mask paths from the manifest CSV files that were generated
    by ``preprocess_filament_dataset.py``.  This is intentional: it guarantees
    that we use the exact observatory-stratified 80/20 split from preprocessing
    and avoids accidental re-splitting that would cause data leakage.

    Args:
        preprocessed_dir (str): Root of the preprocessed dataset, e.g.
            ``…/preprocessed``.  Must contain ``metadata/``, ``images_*``, ``masks_*``.
        split (str): One of ``'train'``, ``'val'``, ``'test'``.
        use_clahe (bool): Use CLAHE contrast-enhanced images (recommended for solar imagery).
        mask_type (str): ``'binary'`` (0/1) or ``'multiclass'`` (0–4).
            Must match ``model.output_channels`` in config.
        transform (callable, optional): Albumentations-style transform that
            accepts ``image=`` and ``mask=`` keyword arguments and returns a dict.
        image_size (int): Target resize dimension (default 512). Preprocessing
            already outputs this size; this is a safety guard.
        max_samples (int or None): If set, truncate the dataset to the first N
            samples. Used for sanity checks only.
    """

    def __init__(
        self,
        preprocessed_dir: str,
        split: str = "train",
        use_clahe: bool = True,
        mask_type: str = "binary",
        transform=None,
        image_size: int = 512,
        max_samples: int = None,
    ):
        super().__init__()
        self.preprocessed_dir = preprocessed_dir
        self.split = split
        self.use_clahe = use_clahe
        self.mask_type = mask_type
        self.transform = transform
        self.image_size = image_size

        manifest_path = os.path.join(preprocessed_dir, "metadata", f"{split}_manifest.csv")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"Manifest CSV not found: {manifest_path}\n"
                "Did you run the preprocessing script first?\n"
                "  python solar_filament_segmentation_repo/preprocess_filament_dataset.py "
                f"--raw_dir <RAW_DIR> --output_dir <PREPROCESSED_DIR>"
            )

        self.df = pd.read_csv(manifest_path)

        # Sanity-check: drop rows with missing image paths
        self.df = self.df.dropna(subset=["rel_path_clahe", "rel_path_img"])

        if max_samples is not None:
            # Truncate for sanity checks; reproducible because manifest is sorted
            self.df = self.df.iloc[:max_samples].reset_index(drop=True)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.df)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # ── Load image ────────────────────────────────────────────────
        img_rel = row["rel_path_clahe"] if self.use_clahe else row["rel_path_img"]
        img_path = os.path.join(self.preprocessed_dir, img_rel)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Could not read image: {img_path}")

        # ── Test split: no ground-truth mask ──────────────────────────
        if self.split == "test":
            img = self._to_tensor(img)
            return img, row["filename"]

        # ── Load mask ─────────────────────────────────────────────────
        if self.mask_type == "binary":
            mask_rel = row["rel_path_mask_binary"]
            mask_path = os.path.join(self.preprocessed_dir, mask_rel)
            mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_raw is None:
                raise RuntimeError(f"Could not read mask: {mask_path}")
            # Binary masks are stored as {0, 255}; convert to {0, 1}
            mask = (mask_raw > 127).astype(np.uint8)

        elif self.mask_type == "multiclass":
            mask_rel = row["rel_path_mask_multiclass"]
            mask_path = os.path.join(self.preprocessed_dir, mask_rel)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"Could not read mask: {mask_path}")
            # Multiclass masks store pixel values in {0, 1, 2, 3, 4} directly
            mask = mask.astype(np.uint8)

        else:
            raise ValueError(
                f"Unknown mask_type='{self.mask_type}'. Choose 'binary' or 'multiclass'."
            )

        # ── Augmentations ─────────────────────────────────────────────
        # NOTE: Horizontal flips INVERT chirality (Left↔Right).
        # The augmentation pipeline in augmentations.py swaps class labels 1↔2
        # automatically when a horizontal flip is applied.  Do NOT apply a
        # raw hflip here without that label-swap logic.
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        # ── To tensors ────────────────────────────────────────────────
        img_tensor = self._to_tensor(img)                           # (1, H, W) float32 in [0, 1]
        mask_tensor = torch.from_numpy(mask.copy()).long()          # (H, W) int64

        return img_tensor, mask_tensor

    # ------------------------------------------------------------------
    @staticmethod
    def _to_tensor(img: np.ndarray) -> torch.Tensor:
        """Normalize grayscale uint8 → float32 [0, 1] and add channel dim."""
        img_f = img.astype(np.float32) / 255.0
        return torch.from_numpy(img_f).unsqueeze(0)  # (1, H, W)

    # ------------------------------------------------------------------
    def get_sample_info(self, idx: int) -> dict:
        """Return manifest metadata for sample at index (useful for qualitative display)."""
        return self.df.iloc[idx].to_dict()


# =============================================================================
# DataLoader factory
# =============================================================================

def build_dataloaders(cfg: dict, sanity: bool = False):
    """
    Build train and validation DataLoaders from the project config dict.

    Args:
        cfg: Parsed config.yaml as a Python dict.
        sanity: If True, restricts each split to ``cfg['sanity']['max_samples']``
                samples so that a full forward/backward pass can be quickly verified.

    Returns:
        (train_loader, val_loader)
    """
    from src.augmentations import get_train_transform, get_val_transform

    preprocessed_dir = cfg["preprocessed_dir"]
    if not preprocessed_dir:
        raise ValueError(
            "config.yaml: 'preprocessed_dir' is empty.\n"
            "Set it to the root of the preprocessed dataset before training."
        )

    mask_type = cfg.get("mask_type", "binary")
    use_clahe = cfg.get("use_clahe", True)
    image_size = cfg.get("image_size", 512)
    max_samples = cfg["sanity"]["max_samples"] if sanity else None

    train_ds = FilamentDataset(
        preprocessed_dir=preprocessed_dir,
        split="train",
        use_clahe=use_clahe,
        mask_type=mask_type,
        transform=get_train_transform(image_size),
        image_size=image_size,
        max_samples=max_samples,
    )
    val_ds = FilamentDataset(
        preprocessed_dir=preprocessed_dir,
        split="val",
        use_clahe=use_clahe,
        mask_type=mask_type,
        transform=get_val_transform(image_size),
        image_size=image_size,
        max_samples=max_samples,
    )

    # num_workers=0 is safe on Windows when workers cause spawn errors.
    # Increase to cfg['num_workers'] if your OS supports it.
    nw = cfg.get("num_workers", 0)
    batch_size = cfg["sanity"]["max_samples"] if sanity else cfg["batch_size"]

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return train_loader, val_loader
