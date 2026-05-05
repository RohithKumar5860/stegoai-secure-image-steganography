"""
Texture-Aware Dual-Path CNN (TA-DPCNN)
Generates a Dynamic Embedding Strength Map (DESM) for adaptive steganography.

Architecture:
  - Texture Path:  Edge/Entropy/Texture detection CNN
  - Distortion Path: Pixel sensitivity prediction CNN
  - Fusion:         Concatenate -> 1x1 conv -> Sigmoid -> DESM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ──────────────────────────────────────────────
#  Sub-Blocks
# ──────────────────────────────────────────────

class ConvBNReLU(nn.Module):
    """Conv2d → BatchNorm2d → ReLU"""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """Residual block with two ConvBNReLU layers"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBNReLU(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.relu(out + residual)
        return out


# ──────────────────────────────────────────────
#  Texture Path
# ──────────────────────────────────────────────

class TexturePath(nn.Module):
    """
    Detects edges, local entropy, and texture complexity.
    Uses Sobel-like initial filters followed by residual blocks.
    Input: [B, 3, H, W]
    Output: [B, 64, H, W]
    """
    def __init__(self):
        super().__init__()
        # Initial edge/texture detection
        self.entry = nn.Sequential(
            ConvBNReLU(3, 32, kernel=3),
            ConvBNReLU(32, 32, kernel=3),
        )
        # High-frequency feature extraction
        self.high_freq = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False, groups=32),  # depthwise
            nn.Conv2d(32, 32, kernel_size=1, bias=False),                         # pointwise
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # Residual refinement
        self.res1 = ResidualBlock(32)
        self.res2 = ResidualBlock(32)
        # Upscale to 64 channels
        self.expand = ConvBNReLU(32, 64, kernel=1, padding=0)

    def forward(self, x):
        e = self.entry(x)
        hf = self.high_freq(e)
        r = self.res1(hf)
        r = self.res2(r)
        return self.expand(r)


# ──────────────────────────────────────────────
#  Distortion Path
# ──────────────────────────────────────────────

class DistortionPath(nn.Module):
    """
    Predicts pixel-level sensitivity to distortion.
    Uses multi-scale features to capture smooth areas (high sensitivity) vs
    textured areas (low sensitivity).
    Input: [B, 3, H, W]
    Output: [B, 64, H, W]
    """
    def __init__(self):
        super().__init__()
        # Multi-scale branches
        self.branch1 = ConvBNReLU(3, 16, kernel=1, padding=0)   # 1×1 context
        self.branch3 = ConvBNReLU(3, 24, kernel=3)               # 3×3 context
        self.branch5 = ConvBNReLU(3, 24, kernel=5, padding=2)    # 5×5 context

        # Merge branches
        self.merge = ConvBNReLU(64, 64, kernel=3)
        self.res1  = ResidualBlock(64)
        self.res2  = ResidualBlock(64)

        # Attention gate (channel-wise)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 64),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        cat = torch.cat([b1, b3, b5], dim=1)  # [B, 64, H, W]
        m = self.merge(cat)
        m = self.res1(m)
        m = self.res2(m)

        # Channel attention
        attn_w = self.attn(m).view(m.size(0), 64, 1, 1)
        return m * attn_w


# ──────────────────────────────────────────────
#  Fusion Head → DESM
# ──────────────────────────────────────────────

class FusionHead(nn.Module):
    """
    Concatenates Texture + Distortion features and produces DESM.
    Input: two [B, 64, H, W] tensors
    Output: [B, 1, H, W]  values in [0,1]
    """
    def __init__(self):
        super().__init__()
        self.fuse = nn.Sequential(
            ConvBNReLU(128, 64, kernel=1, padding=0),
            ResidualBlock(64),
            ConvBNReLU(64, 32, kernel=3),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, tex, dist):
        x = torch.cat([tex, dist], dim=1)  # [B, 128, H, W]
        return self.fuse(x)


# ──────────────────────────────────────────────
#  Full TA-DPCNN
# ──────────────────────────────────────────────

class TADPCNN(nn.Module):
    """
    Texture-Aware Dual-Path CNN

    Forward pass:
      image [B, 3, H, W] → DESM [B, 1, H, W]

    DESM values closer to 1 → safe to embed more bits (complex texture)
    DESM values closer to 0 → avoid embedding (smooth / sensitive areas)
    """
    def __init__(self):
        super().__init__()
        self.texture_path    = TexturePath()
        self.distortion_path = DistortionPath()
        self.fusion          = FusionHead()

    def forward(self, x):
        tex  = self.texture_path(x)
        dist = self.distortion_path(x)
        desm = self.fusion(tex, dist)
        return desm

    @torch.no_grad()
    def predict_desm(self, image_np: np.ndarray) -> np.ndarray:
        """
        Convenience wrapper: numpy RGB image [H,W,3] uint8 → numpy DESM [H,W].
        """
        self.eval()
        img = image_np.astype(np.float32) / 255.0
        t   = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
        desm = self(t)                                               # [1,1,H,W]
        return desm.squeeze().cpu().numpy()                          # [H,W]


# ──────────────────────────────────────────────
#  DESM-based embedding strength
# ──────────────────────────────────────────────

def compute_embedding_capacity(desm: np.ndarray, threshold: float = 0.3) -> int:
    """
    Count how many pixels are 'safe' for embedding.
    threshold: minimum DESM value to consider a pixel embeddable.
    """
    return int(np.sum(desm >= threshold))


def get_embedding_order(desm: np.ndarray) -> np.ndarray:
    """
    Return pixel indices sorted by DESM value (descending).
    Used to prioritise which pixels carry bits.
    """
    flat = desm.flatten()
    return np.argsort(flat)[::-1]   # highest DESM first
