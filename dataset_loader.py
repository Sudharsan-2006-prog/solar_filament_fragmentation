"""
Solar Filament Segmentation Dataset Loader (PyTorch & NumPy).

Provides convenient Dataset and DataLoader interfaces for loading the preprocessed
MAGFiLO filament segmentation images, consensus masks, spine masks, and CLAHE enhancements.
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class Dataset: pass
    class DataLoader: pass


class FilamentSegmentationDataset(Dataset):
    """
    Dataset class for MAGFiLO Solar Filament Segmentation.
    
    Args:
        preprocessed_dir (str): Root directory of preprocessed data.
        split (str): 'train', 'val', or 'test'.
        use_clahe (bool): Whether to use CLAHE contrast-enhanced images (True) or raw grayscale (False).
        mask_type (str): 'multiclass' (0..4), 'binary' (0 or 1), or 'spine' (0 or 1).
        transform (callable, optional): Optional transform / augmentation to apply.
    """
    def __init__(
        self,
        preprocessed_dir=r"D:\Downloads\filament-segmentation-2026\preprocessed",
        split="train",
        use_clahe=True,
        mask_type="multiclass",
        transform=None
    ):
        self.preprocessed_dir = preprocessed_dir
        self.split = split
        self.use_clahe = use_clahe
        self.mask_type = mask_type
        self.transform = transform
        
        manifest_path = os.path.join(preprocessed_dir, "metadata", f"{split}_manifest.csv")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found at {manifest_path}")
            
        self.df = pd.read_csv(manifest_path)
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_rel = row["rel_path_clahe"] if self.use_clahe else row["rel_path_img"]
        img_full_path = os.path.join(self.preprocessed_dir, img_rel)
        img = cv2.imread(img_full_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to load image: {img_full_path}")
            
        # Test images don't have ground truth masks
        if self.split == "test":
            img_norm = img.astype(np.float32) / 255.0
            if HAS_TORCH:
                img_tensor = torch.from_numpy(img_norm).unsqueeze(0)
                return img_tensor, row["filename"]
            return img_norm, row["filename"]
            
        # Load mask
        if self.mask_type == "multiclass":
            mask_rel = row["rel_path_mask_multiclass"]
            mask_full_path = os.path.join(self.preprocessed_dir, mask_rel)
            mask = cv2.imread(mask_full_path, cv2.IMREAD_GRAYSCALE)
        elif self.mask_type == "binary":
            mask_rel = row["rel_path_mask_binary"]
            mask_full_path = os.path.join(self.preprocessed_dir, mask_rel)
            mask = (cv2.imread(mask_full_path, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        elif self.mask_type == "spine":
            mask_rel = row["rel_path_mask_spine"]
            mask_full_path = os.path.join(self.preprocessed_dir, mask_rel)
            mask = (cv2.imread(mask_full_path, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        else:
            raise ValueError(f"Unknown mask_type: {self.mask_type}")
            
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]
            
        img_norm = img.astype(np.float32) / 255.0
        
        if HAS_TORCH:
            img_tensor = torch.from_numpy(img_norm).unsqueeze(0)  # Shape: (1, H, W)
            mask_tensor = torch.from_numpy(mask).long()          # Shape: (H, W)
            return img_tensor, mask_tensor
            
        return img_norm, mask


def get_data_loaders(preprocessed_dir, batch_size=16, use_clahe=True, mask_type="multiclass"):
    """
    Helper function to instantiate train and val PyTorch DataLoaders.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required to build DataLoaders.")
        
    train_dataset = FilamentSegmentationDataset(
        preprocessed_dir=preprocessed_dir,
        split="train",
        use_clahe=use_clahe,
        mask_type=mask_type
    )
    val_dataset = FilamentSegmentationDataset(
        preprocessed_dir=preprocessed_dir,
        split="val",
        use_clahe=use_clahe,
        mask_type=mask_type
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader


if __name__ == "__main__":
    print("Testing FilamentSegmentationDataset...")
    ds = FilamentSegmentationDataset(
        preprocessed_dir=r"D:\Downloads\filament-segmentation-2026\preprocessed",
        split="train",
        mask_type="multiclass"
    )
    print(f"Dataset length: {len(ds)}")
    img, mask = ds[0]
    print(f"Sample 0 -> Image shape: {img.shape}, dtype: {img.dtype}")
    print(f"Sample 0 -> Mask shape: {mask.shape}, unique values: {np.unique(mask) if not HAS_TORCH else torch.unique(mask)}")
