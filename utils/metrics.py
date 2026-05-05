"""
Image quality metrics: PSNR, SSIM, MSE
Also includes comparison against naive LSB (baseline).
"""

import math
import numpy as np
from typing import Optional

try:
    from skimage.metrics import structural_similarity as _ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


# ──────────────────────────────────────────────
#  MSE
# ──────────────────────────────────────────────

def compute_mse(original: np.ndarray, stego: np.ndarray) -> float:
    """Mean Squared Error between two images."""
    diff = original.astype(np.float64) - stego.astype(np.float64)
    return float(np.mean(diff ** 2))


# ──────────────────────────────────────────────
#  PSNR
# ──────────────────────────────────────────────

def compute_psnr(original: np.ndarray, stego: np.ndarray,
                 max_val: float = 255.0) -> float:
    """
    Peak Signal-to-Noise Ratio (dB).
    Higher → more imperceptible embedding (>40 dB is excellent).
    """
    mse = compute_mse(original, stego)
    if mse == 0.0:
        return float("inf")
    return 10 * math.log10(max_val ** 2 / mse)


# ──────────────────────────────────────────────
#  SSIM
# ──────────────────────────────────────────────

def compute_ssim(original: np.ndarray, stego: np.ndarray) -> float:
    """
    Structural Similarity Index (0–1).
    Closer to 1 → visually indistinguishable.
    """
    if SKIMAGE_AVAILABLE:
        # channel_axis for colour images
        return float(_ssim(original, stego, channel_axis=2, data_range=255))

    # Fallback: simplified single-channel SSIM on luminance
    def _to_grey(img):
        return (0.2989 * img[:, :, 0] +
                0.5870 * img[:, :, 1] +
                0.1140 * img[:, :, 2]).astype(np.float64)

    orig_g = _to_grey(original)
    steg_g = _to_grey(stego)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    mu1, mu2   = orig_g.mean(), steg_g.mean()
    sigma1_sq  = orig_g.var()
    sigma2_sq  = steg_g.var()
    sigma12    = float(np.mean((orig_g - mu1) * (steg_g - mu2)))

    num  = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    den  = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    return num / den


# ──────────────────────────────────────────────
#  Baseline naive LSB
# ──────────────────────────────────────────────

def naive_lsb_embed(cover: np.ndarray, n_bytes: int) -> np.ndarray:
    """
    Sequentially embed n_bytes of zeros into the LSB of every pixel channel
    (no DESM guidance) for baseline comparison.
    Note: uses .copy() then .ravel() to avoid mutating the input.
    """
    stego = cover.copy()                  # never mutate the original
    flat  = stego.ravel()                 # flat VIEW into the copy (uint8)
    n_bits = n_bytes * 8
    limit  = min(n_bits, len(flat))
    # Embed 0-bit into LSB: clear bit-0 using bitwise AND
    flat[:limit] = (flat[:limit].astype(np.uint16) & np.uint16(0xFE)).astype(np.uint8)
    return stego


def compute_all_metrics(original: np.ndarray, stego: np.ndarray,
                        payload_bytes: int = 0) -> dict:
    """
    Compute PSNR, SSIM, MSE and compare against naive LSB baseline.
    PSNR is capped at 100.0 when images are identical (avoids JSON inf).
    """
    psnr_val = compute_psnr(original, stego)
    ssim_val = compute_ssim(original, stego)
    mse_val  = compute_mse(original, stego)

    # JSON cannot serialise float('inf'); cap it
    psnr_display = min(psnr_val, 100.0) if psnr_val != float("inf") else 100.0

    result = {
        "psnr": round(psnr_display, 4),
        "ssim": round(ssim_val, 6),
        "mse":  round(mse_val,  6),
    }

    if payload_bytes > 0:
        baseline = naive_lsb_embed(original, payload_bytes)
        b_psnr   = compute_psnr(original, baseline)
        result["baseline_psnr"] = round(min(b_psnr, 100.0), 4)
        result["baseline_ssim"] = round(compute_ssim(original, baseline), 6)
        result["baseline_mse"]  = round(compute_mse(original,  baseline), 6)

    return result
