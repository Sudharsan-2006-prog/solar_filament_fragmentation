"""
boundary_attention_unet.py — Boundary-Aware Attention U-Net (proposed model).

This is our primary proposed architecture for the IEEE Big Data Cup 2026 –
Challenge 02: Solar Filament Segmentation.

MOTIVATION
----------
Solar filaments are challenging to segment because they are:
  1. Thin and elongated  → boundaries carry most of the structural information
  2. Low contrast        → edges blur into chromospheric background features
  3. Fragmented predictions common → network tends to break thin structures
  4. Confused with fibril network → attention needed to suppress background

ARCHITECTURE OVERVIEW
---------------------

  H-alpha image (1, 512, 512)
        │
  ┌─────▼─────────────────────────────────────────────────────────────────┐
  │                        ENCODER                                         │
  │   ConvBlock(1→64) → Down(64→128) → Down(128→256) → Down(256→512)     │
  └─────┬─────────────────────────────────────────────────────────────────┘
        │ skip connections s1, s2, s3, s4
  ┌─────▼─────┐
  │ Bottleneck│ Down(512→1024) with dropout
  └─────┬─────┘
        │
  ┌─────▼──────────────────────────────────────────────────────────────────┐
  │                DECODER WITH ATTENTION GATES                             │
  │  AttentionUpBlock × 4 (same as Attention U-Net)                        │
  └─────┬──────────────────────────────────────────────────────────────────┘
        │ decoder features d
        │
  ┌─────▼──────────────────────────────────────────────────────────────────┐
  │            BOUNDARY-AWARE REFINEMENT MODULE (BARM)                      │
  │                                                                          │
  │  Motivation: After the standard decoder, the feature map still has       │
  │  coarse representation of filament boundaries.  BARM applies a dedicated │
  │  branch of convolutions with dilated receptive fields that captures       │
  │  multi-scale edge context — crucial for thin (1–3px) filament spines.    │
  │                                                                          │
  │  Architecture:                                                            │
  │    d → Conv3×3(dilation=1) → BN → ReLU    (local edges)                 │
  │      → Conv3×3(dilation=2) → BN → ReLU    (thin structure context)       │
  │      → Conv3×3(dilation=4) → BN → ReLU    (filament body context)        │
  │    Concatenate 3 scales → 1×1 conv → boundary-refined features           │
  │    Add residual from d → out_conv(1×1) → logits                          │
  └─────┬──────────────────────────────────────────────────────────────────┘
        │
  Segmentation mask logits (B, out_ch, H, W)

LOSS FUNCTION
-------------
  L_total = α * BCE + β * Dice + γ * BoundaryLoss
  All weights are configurable in config.yaml under loss.{bce,dice,boundary}_weight.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.unet import ConvBlock, DownBlock
from src.models.attention_unet import AttentionGate


# =============================================================================
# Attention Up Block (shared with attention_unet)
# =============================================================================

class AttentionUpBlock(nn.Module):
    """Decoder block with attention gate — see attention_unet.py for full docs."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.attn = AttentionGate(f_g=in_ch, f_l=skip_ch, f_int=max(skip_ch // 2, 1))
        self.conv = ConvBlock(in_ch + skip_ch, out_ch, dropout=dropout)

    def forward(self, g: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        g_up = self.up(g)
        if g_up.shape[2:] != skip.shape[2:]:
            g_up = F.interpolate(g_up, size=skip.shape[2:], mode="bilinear", align_corners=True)
        skip_gated = self.attn(g_up, skip)
        x = torch.cat([g_up, skip_gated], dim=1)
        return self.conv(x)


# =============================================================================
# Boundary-Aware Refinement Module (BARM)
# =============================================================================

class BoundaryAwareRefinementModule(nn.Module):
    """
    Multi-scale dilated convolution branch for boundary-aware feature refinement.

    Uses dilated convolutions at three scales to capture:
      - dilation=1: Local pixel-level edges (1-2px filament boundaries)
      - dilation=2: Short-range context (3-4px thin filament spines)
      - dilation=4: Medium-range context (filament body width)

    The three scales are concatenated and compressed with a 1×1 conv, then
    added to the original decoder features as a residual.  This means the
    BARM can only ADD boundary refinement and cannot destroy existing features,
    making training stable.

    Args:
        in_ch: Number of input channels (= base_features from final decoder stage).
    """

    def __init__(self, in_ch: int):
        super().__init__()
        mid_ch = in_ch // 4  # Compressed channel count per dilation branch

        # Three parallel dilated conv branches
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )

        # Fuse the 3 branches back to in_ch channels
        self.fuse = nn.Sequential(
            nn.Conv2d(mid_ch * 3, in_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b4 = self.branch4(x)
        fused = self.fuse(torch.cat([b1, b2, b4], dim=1))
        # Residual: add back the original features so training is stable
        return x + fused


# =============================================================================
# Boundary-Aware Attention U-Net
# =============================================================================

class BoundaryAttentionUNet(nn.Module):
    """
    Boundary-Aware Attention U-Net: our proposed model for solar filament segmentation.

    Combines:
      1. U-Net encoder–decoder with skip connections
      2. Soft attention gates on all skip connections (suppresses background noise)
      3. Boundary-Aware Refinement Module (BARM) after the final decoder stage
         (enhances thin filament edges using multi-scale dilated convolutions)

    Trained with a combined loss:
      L = α·BCE + β·Dice + γ·BoundaryLoss
    (weights configurable in config.yaml)

    Args:
        in_channels:   Input channels (1 for grayscale H-alpha).
        out_channels:  Output channels (1 for binary segmentation).
        base_features: Feature width at encoder stage 1.
        dropout:       Bottleneck dropout rate.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_features: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        f = base_features

        # --- Encoder (identical to U-Net) ---
        self.enc1 = ConvBlock(in_channels, f)
        self.enc2 = DownBlock(f, f * 2)
        self.enc3 = DownBlock(f * 2, f * 4)
        self.enc4 = DownBlock(f * 4, f * 8)

        # --- Bottleneck ---
        self.bottleneck = DownBlock(f * 8, f * 16, dropout=dropout)

        # --- Decoder with Attention Gates (identical to Attention U-Net) ---
        self.dec4 = AttentionUpBlock(f * 16, f * 8, f * 8)
        self.dec3 = AttentionUpBlock(f * 8, f * 4, f * 4)
        self.dec2 = AttentionUpBlock(f * 4, f * 2, f * 2)
        self.dec1 = AttentionUpBlock(f * 2, f, f)

        # --- Boundary-Aware Refinement Module ---
        # Applied to the final decoder output before the output projection.
        # This is the architectural novelty beyond Attention U-Net.
        self.barm = BoundaryAwareRefinementModule(in_ch=f)

        # --- Output projection ---
        self.out_conv = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W)
        Returns:
            logits: (B, out_channels, H, W)
        """
        # Encoder
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        # Bottleneck
        b = self.bottleneck(s4)

        # Attention-gated decoder
        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        # Boundary-Aware Refinement
        d_refined = self.barm(d1)

        return self.out_conv(d_refined)


# =============================================================================
# Factory
# =============================================================================

def build_boundary_attention_unet(cfg: dict) -> BoundaryAttentionUNet:
    """Build BoundaryAttentionUNet from project config dict."""
    m = cfg.get("model", {})
    return BoundaryAttentionUNet(
        in_channels=int(m.get("input_channels", 1)),
        out_channels=int(m.get("output_channels", 1)),
        base_features=int(m.get("base_features", 64)),
    )
