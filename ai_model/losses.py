"""
Loss Functions for TA-DPCNN Training
L = α*MSE + β*(1-SSIM) + γ*EntropyPenalty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


# ──────────────────────────────────────────────
#  SSIM Loss
# ──────────────────────────────────────────────

def _gaussian_kernel(window_size: int, sigma: float, channels: int) -> torch.Tensor:
    """Create a Gaussian kernel for SSIM computation."""
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    gauss  = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    gauss  = gauss / gauss.sum()
    kernel = gauss.unsqueeze(0) * gauss.unsqueeze(1)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    return kernel.expand(channels, 1, window_size, window_size).contiguous()


def ssim_loss(pred: torch.Tensor, target: torch.Tensor,
              window_size: int = 11, sigma: float = 1.5,
              C1: float = 0.01 ** 2, C2: float = 0.03 ** 2) -> torch.Tensor:
    """
    Returns 1 - SSIM (lower is better when used as a loss).
    pred, target: [B, C, H, W] in [0, 1]
    """
    B, C, H, W = pred.shape
    kernel = _gaussian_kernel(window_size, sigma, C).to(pred.device)

    mu1 = F.conv2d(pred,   kernel, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(target, kernel, padding=window_size // 2, groups=C)

    mu1_sq  = mu1 * mu1
    mu2_sq  = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq  = F.conv2d(pred   * pred,   kernel, padding=window_size // 2, groups=C) - mu1_sq
    sigma2_sq  = F.conv2d(target * target, kernel, padding=window_size // 2, groups=C) - mu2_sq
    sigma12    = F.conv2d(pred   * target, kernel, padding=window_size // 2, groups=C) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return 1.0 - ssim_map.mean()


# ──────────────────────────────────────────────
#  Entropy Penalty
# ──────────────────────────────────────────────

def entropy_penalty(desm: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Penalises DESM maps that are either all-zero (embed nowhere)
    or all-one (embed everywhere without discrimination).
    Encourages a diverse, information-rich DESM.
    penalty = -H(desm) where H is information entropy.
    """
    flat = desm.view(desm.size(0), -1)           # [B, H*W]
    flat = flat.clamp(eps, 1 - eps)
    # Treat each pixel as a Bernoulli r.v.
    h    = -(flat * flat.log() + (1 - flat) * (1 - flat).log())
    # We MAXIMISE entropy → MINIMISE negative entropy
    return -h.mean()


# ──────────────────────────────────────────────
#  Composite Loss
# ──────────────────────────────────────────────

class DESMLoss(nn.Module):
    """
    Composite loss for DESM supervision:

    L = α * MSE(stego, cover)
      + β * (1 - SSIM(stego, cover))
      + γ * EntropyPenalty(DESM)

    Inputs:
      stego:  stego image tensor [B, 3, H, W]
      cover:  cover image tensor [B, 3, H, W]
      desm:   DESM output        [B, 1, H, W]
    """
    def __init__(self, alpha: float = 1.0, beta: float = 0.5, gamma: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.mse   = nn.MSELoss()

    def forward(self, stego: torch.Tensor, cover: torch.Tensor,
                desm: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        mse_val     = self.mse(stego, cover)
        ssim_val    = ssim_loss(stego, cover)
        entropy_val = entropy_penalty(desm)

        total = (self.alpha * mse_val +
                 self.beta  * ssim_val +
                 self.gamma * entropy_val)

        return total, {
            "mse":     mse_val.item(),
            "ssim":    1.0 - ssim_val.item(),
            "entropy": -entropy_val.item(),
            "total":   total.item(),
        }
