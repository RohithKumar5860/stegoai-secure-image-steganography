"""
Model Loader - Load TA-DPCNN from checkpoint or create a fresh instance.

If PyTorch is NOT installed, a lightweight numpy-based fallback DESM generator
is used. It estimates texture complexity via local gradient magnitude, giving a
reasonable (though not optimal) embedding strength map without any ML dependency.

If PyTorch IS installed but no checkpoint exists, the TA-DPCNN runs with random
weights in eval mode (still functional; train via ai_model/train.py).
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# ── Optional PyTorch import ───────────────────────────────────────────────────
try:
    import torch
    from .model import TADPCNN
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None          # type: ignore[assignment]
    TADPCNN = None        # type: ignore[assignment,misc]
    _TORCH_AVAILABLE = False
    log.warning(
        "PyTorch not installed. Using numpy-based fallback DESM generator. "
        "Install with: pip install torch torchvision"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Resolve checkpoint paths relative to this file so they work regardless of CWD
_MODULE_DIR = Path(__file__).resolve().parent
_CHECKPOINT_PATHS = [
    str(_MODULE_DIR / "checkpoints" / "best_tadpcnn.pth"),
    str(_MODULE_DIR.parent / "checkpoints" / "best_tadpcnn.pth"),
]

_MODEL_INSTANCE: Optional[object] = None   # TADPCNN or None


# ─────────────────────────────────────────────────────────────────────────────
#  Numpy fallback DESM generator
# ─────────────────────────────────────────────────────────────────────────────

def _numpy_desm(image_np: np.ndarray) -> np.ndarray:
    """
    Lightweight fallback DESM without PyTorch.

    Estimates per-pixel embedding safety using:
      1. Sobel gradient magnitude (edges = safe to embed)
      2. Local variance via box-filter approximation (texture = safe)
    Both signals are normalised to [0,1] and averaged.

    Args:
        image_np: RGB uint8 [H, W, 3]

    Returns:
        DESM [H, W] float32 in [0, 1]
    """
    # Convert to greyscale float
    grey = (0.2989 * image_np[:, :, 0] +
            0.5870 * image_np[:, :, 1] +
            0.1140 * image_np[:, :, 2]).astype(np.float32) / 255.0

    # Sobel-like gradient magnitude (finite differences)
    gx = np.zeros_like(grey)
    gy = np.zeros_like(grey)
    gx[:, 1:-1] = grey[:, 2:] - grey[:, :-2]          # horizontal
    gy[1:-1, :] = grey[2:, :] - grey[:-2, :]          # vertical
    grad = np.sqrt(gx ** 2 + gy ** 2)

    # Local variance via sliding window approximation (5×5 box)
    def _box_mean(x: np.ndarray, k: int = 5) -> np.ndarray:
        from numpy.lib.stride_tricks import sliding_window_view
        try:
            win = sliding_window_view(x, (k, k))           # [H-k+1, W-k+1, k, k]
            out = win.mean(axis=(-1, -2))
            # Pad back to original size
            pad_h = (x.shape[0] - out.shape[0])
            pad_w = (x.shape[1] - out.shape[1])
            return np.pad(out, ((pad_h // 2, pad_h - pad_h // 2),
                                (pad_w // 2, pad_w - pad_w // 2)), mode='edge')
        except Exception:
            # fallback: return global mean if stride_tricks fails
            return np.full_like(x, x.mean())

    mean  = _box_mean(grey, k=5)
    sq_mean = _box_mean(grey ** 2, k=5)
    variance = np.clip(sq_mean - mean ** 2, 0, None)

    # Normalise both signals independently
    def _norm(x: np.ndarray) -> np.ndarray:
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-8)

    desm = 0.5 * _norm(grad) + 0.5 * _norm(variance)
    return desm.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  PyTorch-based loader (only used when torch is available)
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: Optional[str] = None, device: str = "cpu"):
    """
    Load the TA-DPCNN model (requires PyTorch).

    Args:
        checkpoint_path: explicit path; if None, tries default locations.
        device: 'cpu' or 'cuda'

    Returns:
        TADPCNN in eval mode, or None if torch is unavailable.
    """
    global _MODEL_INSTANCE

    if not _TORCH_AVAILABLE:
        log.warning("load_model() called but PyTorch is not installed. Returning None.")
        return None

    model = TADPCNN().to(device)
    model.eval()

    paths_to_try = ([checkpoint_path] if checkpoint_path else []) + _CHECKPOINT_PATHS

    for path in paths_to_try:
        if path and Path(path).is_file():
            try:
                ckpt  = torch.load(path, map_location=device, weights_only=True)
                state = ckpt.get("model_state", ckpt)
                model.load_state_dict(state)
                log.info("TA-DPCNN loaded from checkpoint: %s", path)
                _MODEL_INSTANCE = model
                return model
            except Exception as exc:
                log.warning("Failed to load checkpoint %s: %s", path, exc)

    log.warning(
        "No checkpoint found. Using untrained TA-DPCNN. "
        "DESM will approximate texture with random weights. "
        "Run 'python -m ai_model.train' to train the model."
    )
    _MODEL_INSTANCE = model
    return model


def get_model(device: str = "cpu"):
    """Singleton accessor — loads once per process. Returns None without torch."""
    global _MODEL_INSTANCE
    if not _TORCH_AVAILABLE:
        return None
    if _MODEL_INSTANCE is None:
        _MODEL_INSTANCE = load_model(device=device)
    return _MODEL_INSTANCE


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_desm(image_np: np.ndarray, device: str = "cpu") -> np.ndarray:
    """
    Generate a Dynamic Embedding Strength Map for a cover image.

    Uses the TA-DPCNN when PyTorch is available; otherwise falls back to
    a fast numpy-based texture estimator.

    Args:
        image_np: RGB image [H, W, 3] uint8
        device:   torch device string ('cpu' or 'cuda')

    Returns:
        DESM [H, W] float32 in [0, 1]
        (higher → more suitable for embedding)
    """
    if _TORCH_AVAILABLE:
        model = get_model(device=device)
        if model is not None:
            return model.predict_desm(image_np).astype(np.float32)

    # Numpy fallback
    log.debug("Using numpy fallback DESM generator.")
    return _numpy_desm(image_np)
