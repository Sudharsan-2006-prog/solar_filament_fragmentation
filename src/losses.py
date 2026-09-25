"""
losses.py — Loss functions for solar filament segmentation.

DESIGN NOTES
------------
Binary Cross-Entropy alone is inadequate for this dataset because solar
filaments occupy only ~0.4–1.5% of image pixels.  A model that predicts
all-background scores ~98.5% pixel accuracy while detecting zero filaments.

We therefore combine:
  1. BCE Loss     – pixel-wise binary classification signal
  2. Dice Loss    – region-overlap loss; robust to foreground–background imbalance
  3. Boundary Loss – explicit supervision on filament edges, extracted from
                     ground-truth masks via morphological erosion.

The boundary term is critical because filaments are thin, elongated structures
and models without boundary supervision tend to produce blurry, over-dilated
predictions with poor delineation of the filament spine.

Mask format assumptions (VERIFIED from repository):
  - Binary masks: uint8/int64, values in {0, 1} after conversion by dataset.py
  - Binary masks are stored as {0, 255} on disk; dataset.py converts to {0, 1}
  - Morphological boundary extraction (Mask − Erode(Mask)) is therefore VALID
    and appropriate for this dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# =============================================================================
# Dice Loss
# =============================================================================

class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.

    Dice = 2 * |P ∩ G| / (|P| + |G|)

    The smooth term prevents division-by-zero when a batch contains images
    with no filament pixels (which is common given <1.5% foreground coverage).
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model output (B, 1, H, W) — NOT sigmoid-activated.
            targets: Ground truth (B, H, W) int64 with values {0, 1}.
        Returns:
            Scalar Dice loss in [0, 1].
        """
        probs = torch.sigmoid(logits).squeeze(1)      # (B, H, W)
        targets_f = targets.float()

        intersection = (probs * targets_f).sum(dim=(1, 2))
        union = probs.sum(dim=(1, 2)) + targets_f.sum(dim=(1, 2))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


# =============================================================================
# BCE + Dice combined (for U-Net and Attention U-Net)
# =============================================================================

class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross-Entropy + Dice Loss.

    BCE provides per-pixel gradient signal (necessary for convergence from
    random init), while Dice corrects the class imbalance by focusing on
    the foreground-overlap ratio rather than pixel-level accuracy.

    Args:
        bce_weight:  Coefficient for BCE term.
        dice_weight: Coefficient for Dice term.
        dice_smooth: Smoothing factor in Dice numerator/denominator.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, dice_smooth: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=dice_smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_f = targets.float()
        bce_loss = self.bce(logits.squeeze(1), targets_f)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# =============================================================================
# Boundary Loss
# =============================================================================

class BoundaryLoss(nn.Module):
    """
    Boundary-Aware Loss for thin, elongated structure segmentation.

    The boundary ground truth is extracted from the binary mask by:
        Boundary = Mask − Erode(Mask, kernel_size)

    This gives a thin ring around each filament region.  The model is then
    penalised for missing or incorrectly predicting these edge pixels.

    Implementation note:
    We perform morphological erosion with a max-pool trick on the inverted mask,
    which is fully differentiable-friendly (the erosion itself operates on the
    numpy ground truth, not the predictions — the loss is standard BCE on
    boundary pixels).

    Validity for this dataset:
    The binary masks in this dataset are stored as {0, 255} → {0, 1} (see
    DATASET_PREPROCESSING_GUIDE.md section 5 and masks_512/binary/).
    Morphological erosion of {0, 1} arrays is physically meaningful because
    the mask represents a crisp consensus binary filament region.
    """

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.bce = nn.BCEWithLogitsLoss()

    def _extract_boundary(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Extract boundary pixels from a binary mask using morphological erosion.

        Erosion via max-pooling on the inverted mask:
            eroded = 1 − MaxPool(1 − mask)
            boundary = mask − eroded
        """
        mask_f = mask.float().unsqueeze(1)          # (B, 1, H, W)
        pad = self.kernel_size // 2

        # Erode by min-pooling (equivalent to max-pool on the complement)
        inv = 1.0 - mask_f
        inv_dilated = F.max_pool2d(inv, kernel_size=self.kernel_size, stride=1, padding=pad)
        eroded = 1.0 - inv_dilated                  # (B, 1, H, W)

        boundary = (mask_f - eroded).clamp(0.0, 1.0)   # thin ring
        return boundary.squeeze(1)                   # (B, H, W)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, 1, H, W) raw model outputs.
            targets: (B, H, W) int64 binary ground truth {0, 1}.
        Returns:
            Boundary BCE loss focused on edge pixels.
        """
        boundary_gt = self._extract_boundary(targets)   # (B, H, W) float
        boundary_pred = logits.squeeze(1)               # (B, H, W)

        # Only supervise where there is a boundary in the ground truth
        # (background boundary pixels carry no information)
        if boundary_gt.sum() < 1.0:
            # No filament pixels in this batch — return zero boundary loss
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        return self.bce(boundary_pred, boundary_gt)


# =============================================================================
# Boundary-Aware combined loss (for Boundary-Aware Attention U-Net)
# =============================================================================

class BoundaryAwareLoss(nn.Module):
    """
    Combined BCE + Dice + Boundary Loss for the proposed model.

    L_total = α * BCE + β * Dice + γ * Boundary

    All three coefficients (α, β, γ) are loaded from config.yaml so that
    they can be tuned without code changes.

    The boundary term provides explicit edge supervision which is especially
    important for:
    - Thin filaments (1–3 pixel wide) where boundaries carry most diagnostic info
    - Low-contrast filaments where the network tends to predict blurred regions
    - Fragmented predictions where the topology is incorrect
    """

    def __init__(
        self,
        bce_weight: float = 0.4,
        dice_weight: float = 0.4,
        boundary_weight: float = 0.2,
        dice_smooth: float = 1.0,
        boundary_kernel: int = 3,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=dice_smooth)
        self.boundary = BoundaryLoss(kernel_size=boundary_kernel)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Returns total loss and component dict for logging.
        """
        targets_f = targets.float()
        bce_loss = self.bce(logits.squeeze(1), targets_f)
        dice_loss = self.dice(logits, targets)
        boundary_loss = self.boundary(logits, targets)

        total = (
            self.bce_weight * bce_loss
            + self.dice_weight * dice_loss
            + self.boundary_weight * boundary_loss
        )

        components = {
            "loss_bce": bce_loss.item(),
            "loss_dice": dice_loss.item(),
            "loss_boundary": boundary_loss.item(),
            "loss_total": total.item(),
        }
        return total, components


# =============================================================================
# Loss factory
# =============================================================================

def build_loss(cfg: dict, model_name: str):
    """
    Instantiate the appropriate loss function from config.

    Args:
        cfg: Parsed config.yaml dict.
        model_name: One of 'unet', 'attention_unet', 'boundary_attention_unet'.
    Returns:
        Loss module.
    """
    loss_cfg = cfg.get("loss", {})
    bce_w = float(loss_cfg.get("bce_weight", 0.4))
    dice_w = float(loss_cfg.get("dice_weight", 0.4))
    bnd_w = float(loss_cfg.get("boundary_weight", 0.2))
    smooth = float(loss_cfg.get("dice_smooth", 1.0))

    if model_name == "boundary_attention_unet":
        return BoundaryAwareLoss(
            bce_weight=bce_w,
            dice_weight=dice_w,
            boundary_weight=bnd_w,
            dice_smooth=smooth,
        )
    else:
        # Standard models use BCE + Dice
        return BCEDiceLoss(bce_weight=bce_w, dice_weight=dice_w, dice_smooth=smooth)
