"""
unet.py — Standard U-Net baseline for solar filament segmentation.

Reference: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical
Image Segmentation", MICCAI 2015.

Architecture:
  Encoder (4 downsampling stages) → Bottleneck → Decoder (4 upsampling stages)
  with skip connections from encoder to decoder at each scale.

Design choices for this task:
  - BatchNorm after every Conv: helps with the extreme class imbalance by
    keeping activations stable even when most of the feature maps are
    dominated by background features.
  - ReLU activation: standard; can be swapped to LeakyReLU if desired.
  - Configurable input/output channels and base feature width.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Building blocks
# =============================================================================

class ConvBlock(nn.Module):
    """
    Double convolution block: Conv → BN → ReLU → Conv → BN → ReLU.

    This is the standard U-Net building block.  Two convolutions increase
    the receptive field at each scale, which is important for capturing the
    extended structure of solar filaments.
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """MaxPool2d downsampling followed by a ConvBlock."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """
    Bilinear upsampling + skip connection concatenation + ConvBlock.

    We use bilinear upsampling rather than transposed convolutions because:
    - Transposed convolutions can produce checkerboard artifacts on thin structures
    - Bilinear + 1×1 conv produces cleaner outputs for filament edge detail
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle potential size mismatch from odd input dimensions
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# =============================================================================
# U-Net
# =============================================================================

class UNet(nn.Module):
    """
    Standard U-Net for binary (or multiclass) solar filament segmentation.

    Args:
        in_channels:    Number of input channels (1 for grayscale H-alpha).
        out_channels:   Number of output channels (1 for binary, N for multiclass).
        base_features:  Width of the first encoder stage.  Each subsequent stage
                        doubles the channels up to base_features * 8.
        dropout:        Dropout rate applied in the bottleneck (0 = no dropout).
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

        # --- Encoder ---
        self.enc1 = ConvBlock(in_channels, f)
        self.enc2 = DownBlock(f, f * 2)
        self.enc3 = DownBlock(f * 2, f * 4)
        self.enc4 = DownBlock(f * 4, f * 8)

        # --- Bottleneck ---
        self.bottleneck = DownBlock(f * 8, f * 16, dropout=dropout)

        # --- Decoder ---
        self.dec4 = UpBlock(f * 16, f * 8, f * 8)
        self.dec3 = UpBlock(f * 8, f * 4, f * 4)
        self.dec2 = UpBlock(f * 4, f * 2, f * 2)
        self.dec1 = UpBlock(f * 2, f, f)

        # --- Output projection ---
        # 1×1 conv maps from feature space to class logits
        self.out_conv = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W)
        Returns:
            logits: (B, out_channels, H, W)
                    Apply sigmoid (binary) or softmax (multiclass) externally.
        """
        # Encoder path — save skip connections
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        # Bottleneck
        b = self.bottleneck(s4)

        # Decoder path — fuse skip connections
        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        return self.out_conv(d1)


# =============================================================================
# Factory
# =============================================================================

def build_unet(cfg: dict) -> UNet:
    """Build UNet from project config dict."""
    m = cfg.get("model", {})
    return UNet(
        in_channels=int(m.get("input_channels", 1)),
        out_channels=int(m.get("output_channels", 1)),
        base_features=int(m.get("base_features", 64)),
    )
