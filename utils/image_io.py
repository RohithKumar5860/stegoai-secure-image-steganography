"""
Image I/O utilities – load, save, convert between numpy/base64/PIL.
"""

import io
import base64
import logging
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from PIL import Image

log = logging.getLogger(__name__)


def load_image(path: str) -> np.ndarray:
    """Load image as RGB uint8 numpy array [H, W, 3]."""
    if CV2_AVAILABLE:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        return np.array(Image.open(path).convert("RGB"))


def save_image(img: np.ndarray, path: str) -> None:
    """Save RGB uint8 numpy array to file."""
    if CV2_AVAILABLE:
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        Image.fromarray(img).save(path)
    log.debug("Image saved to %s", path)


def numpy_to_base64(img: np.ndarray, fmt: str = "PNG") -> str:
    """Convert numpy array [H,W,3] to base64-encoded PNG/JPEG string."""
    pil = Image.fromarray(img.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def base64_to_numpy(b64: str) -> np.ndarray:
    """Convert base64 image string to numpy array [H,W,3]."""
    data = base64.b64decode(b64)
    pil  = Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(pil)


def bytes_to_numpy(raw_bytes: bytes) -> np.ndarray:
    """Convert raw image file bytes to numpy array."""
    pil = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    return np.array(pil)
