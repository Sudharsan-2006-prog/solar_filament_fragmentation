"""
attention_unet.py — Attention U-Net for solar filament segmentation.

Reference: Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas",
MIDL 2018.

Motivation for this task:
  Solar filaments are small, irregularly shaped structures scattered across
  a large, mostly featureless solar disk.  Attention gates learn to suppress
  the response of irrelevant background regions (chromospheric network,
  sunspot penumbrae, etc.) and amplify the feature activations in regions
  that are likely to contain filament material.

  This is the key difference from standard U-Net: the skip connection is
  gated by an attention coefficient α ∈ (0, 1) computed from both the
  decoder feature map (which carries coarse spatial context) and the encoder
  skip connection (which carries fine local detail).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.unet import ConvBlock, DownBlock


# =============================================================================
# Attention Gate
# =============================================================================

class AttentionGate(nn.Module):
    """
    Soft attention gate from Oktay et al. 2018.

    Computes a spatial attention map α from:
      - g: gating signal (coarse, from decoder)
      - x: skip connection (fine, from encoder)

    The gate suppresses irrelevant background features so that the decoder
    only receives filament-relevant information through the skip path.

    Formula:
        q = W_x(x) + W_g(g)        [additive attention]
        α = σ(W_ψ(ReLU(q)))        [soft attention coefficient]
        output = α ⊙ x              [element-wise gating]
    """

    def __init__(self, f_g: int, f_l: int, f_int: int):
        """
        Args:
            f_g:   Number of channels in the gating signal (from decoder).
            f_l:   Number of channels in the skip feature (from encoder).
            f_int: Intermediate channel width for the attention computation.
        """
        super().__init__()
        # 1×1 conv to project gating signal
        self.W_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(f_int),
        )
        # 1×1 conv to project skip features
        self.W_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(f_int),
        )
        # 1×1 conv to compute scalar attention map
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g: Gating signal (B, f_g, H', W') — typically from the up-sampled decoder.
            x: Skip connection (B, f_l, H, W) — from the encoder.
        Returns:
            Attention-gated skip connection (B, f_l, H, W).
        """
        # Upsample g to match x if sizes differ (due to MaxPool truncation on odd dims)
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=True)

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)          # α ∈ (0, 1)
        return x * psi               # Element-wise gating


# =============================================================================
# Attention U-Net Up Block
# =============================================================================

class AttentionUpBlock(nn.Module):
    """
    Decoder block with attention gate on the skip connection.

    Pipeline:
        1. Upsample decoder feature map g.
        2. Apply attention gate to skip connection x using g as gating signal.
        3. Concatenate gated skip + upsampled g.
        4. Apply double convolution.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.attn = AttentionGate(f_g=in_ch, f_l=skip_ch, f_int=skip_ch // 2)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch, dropout=dropout)

    def forward(self, g: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        g_up = self.up(g)
        if g_up.shape[2:] != skip.shape[2:]:
            g_up = F.interpolate(g_up, size=skip.shape[2:], mode="bilinear", align_corners=True)
        skip_gated = self.attn(g_up, skip)
        x = torch.cat([g_up, skip_gated], dim=1)
        return self.conv(x)


# =============================================================================
# Attention U-Net
# =============================================================================

class AttentionUNet(nn.Module):
    """
    Attention U-Net: U-Net augmented with soft attention gates on all skip connections.

    The encoder and bottleneck are identical to standard U-Net.
    The decoder replaces each plain UpBlock with an AttentionUpBlock.

    Args:
        in_channels:   Input channel count (1 for grayscale).
        out_channels:  Output class count (1 for binary segmentation).
        base_features: Feature width at the first encoder stage.
        dropout:       Dropout in the bottleneck.
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

        # --- Encoder (same as standard U-Net) ---
        self.enc1 = ConvBlock(in_channels, f)
        self.enc2 = DownBlock(f, f * 2)
        self.enc3 = DownBlock(f * 2, f * 4)
        self.enc4 = DownBlock(f * 4, f * 8)

        # --- Bottleneck ---
        self.bottleneck = DownBlock(f * 8, f * 16, dropout=dropout)

        # --- Decoder with Attention Gates ---
        self.dec4 = AttentionUpBlock(f * 16, f * 8, f * 8)
        self.dec3 = AttentionUpBlock(f * 8, f * 4, f * 4)
        self.dec2 = AttentionUpBlock(f * 4, f * 2, f * 2)
        self.dec1 = AttentionUpBlock(f * 2, f, f)

        # --- Output projection ---
        self.out_conv = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W)
        Returns:
            logits: (B, out_channels, H, W)
        """
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        b = self.bottleneck(s4)

        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        return self.out_conv(d1)


# =============================================================================
# Factory
# =============================================================================

def build_attention_unet(cfg: dict) -> AttentionUNet:
    """Build AttentionUNet from project config dict."""
    m = cfg.get("model", {})
    return AttentionUNet(
        in_channels=int(m.get("input_channels", 1)),
        out_channels=int(m.get("output_channels", 1)),
        base_features=int(m.get("base_features", 64)),
    )
