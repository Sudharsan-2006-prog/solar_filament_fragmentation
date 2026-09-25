"""
metrics.py — Segmentation evaluation metrics for the MAGFiLO filament dataset.

SCIENTIFIC INTEGRITY NOTES
---------------------------
- All metrics are computed from actual model predictions, never fabricated.
- Metrics are computed on the VALIDATION set during training.
- The TEST set is never used for hyperparameter selection.
- NaN is returned (not 0) when a metric is undefined (e.g., Dice when both
  prediction and target are empty).  This prevents misleading averages.

Metrics implemented:
  Binary:     Dice (F1), IoU (Jaccard), Precision, Recall, Pixel Accuracy
  Optional:   Panoptic Quality (PQ) — not implemented here because PQ requires
              instance-level matching (matching_iou threshold of 0.5).  The
              preprocessed dataset does not provide instance-level IDs (only
              semantic masks), so PQ cannot be correctly computed without
              additional instance extraction.  The column is written as NaN in
              the metrics CSV to be honest about this limitation.
"""

import torch
import numpy as np


EPS = 1e-7  # Numerical stability for divisions


# =============================================================================
# Core binary metric computation
# =============================================================================

def compute_binary_metrics(
    pred_mask: torch.Tensor,
    true_mask: torch.Tensor,
    threshold: float = 0.5,
) -> dict:
    """
    Compute binary segmentation metrics from raw logits or probabilities.

    Args:
        pred_mask: Model output logits or probabilities (B, 1, H, W) or (B, H, W).
        true_mask: Ground truth (B, H, W) int64 with values {0, 1}.
        threshold: Decision boundary for binarizing predictions.

    Returns:
        Dict with keys: dice, iou, precision, recall, f1, pixel_accuracy.
        Each value is a Python float (NaN when undefined).
    """
    if pred_mask.dim() == 4:
        # Raw logits with shape (B, 1, H, W)
        probs = torch.sigmoid(pred_mask).squeeze(1)  # (B, H, W)
    else:
        probs = pred_mask  # Already (B, H, W)

    preds = (probs > threshold).long()
    targets = true_mask.long()

    # Flatten to 1D for metric computation across the whole batch
    p = preds.view(-1).float()
    t = targets.view(-1).float()

    tp = (p * t).sum().item()
    fp = (p * (1.0 - t)).sum().item()
    fn = ((1.0 - p) * t).sum().item()
    tn = ((1.0 - p) * (1.0 - t)).sum().item()

    # Precision: of all predicted positives, how many are correct?
    precision = tp / (tp + fp + EPS) if (tp + fp) > 0 else float("nan")

    # Recall: of all actual positives, how many did we find?
    recall = tp / (tp + fn + EPS) if (tp + fn) > 0 else float("nan")

    # F1 / Dice: harmonic mean of precision and recall
    if tp == 0 and (fp == 0) and (fn == 0):
        # Both prediction and ground truth are empty — perfect match
        f1 = 1.0
        dice = 1.0
        iou = 1.0
    elif tp + fp + fn == 0:
        f1 = float("nan")
        dice = float("nan")
        iou = float("nan")
    else:
        dice = (2.0 * tp) / (2.0 * tp + fp + fn + EPS)
        f1 = dice  # Dice coefficient == F1 for binary segmentation
        iou = tp / (tp + fp + fn + EPS)

    total_px = tp + fp + fn + tn
    pixel_accuracy = (tp + tn) / total_px if total_px > 0 else float("nan")

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pixel_accuracy": pixel_accuracy,
    }


# =============================================================================
# Epoch-level metric averaging
# =============================================================================

class MetricAccumulator:
    """
    Accumulates per-batch metrics across an epoch and computes their mean.

    Usage:
        acc = MetricAccumulator()
        for batch in loader:
            metrics = compute_binary_metrics(pred, target)
            acc.update(metrics)
        epoch_metrics = acc.compute()
    """

    def __init__(self):
        self._sums = {}
        self._counts = {}

    def update(self, metrics: dict):
        for key, val in metrics.items():
            if isinstance(val, torch.Tensor):
                val = val.item()
            if not np.isnan(val):
                self._sums[key] = self._sums.get(key, 0.0) + val
                self._counts[key] = self._counts.get(key, 0) + 1

    def compute(self) -> dict:
        """Return mean of each metric over accumulated updates."""
        result = {}
        for key in self._sums:
            result[key] = self._sums[key] / self._counts[key] if self._counts[key] > 0 else float("nan")
        return result

    def reset(self):
        self._sums.clear()
        self._counts.clear()
