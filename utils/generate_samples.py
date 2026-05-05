"""
Generate synthetic sample images for testing.
Run: python utils/generate_samples.py
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'samples')


def make_gradient(h=512, w=512):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for x in range(w):
        img[:, x, 0] = int(x / w * 255)   # R gradient
    for y in range(h):
        img[y, :, 1] = int(y / h * 255)   # G gradient
    img[:, :, 2] = 120
    return img


def make_textured(h=512, w=512):
    np.random.seed(7)
    base  = np.random.randint(40, 200, (h, w, 3), dtype=np.uint8)
    # Add grid lines
    base[::32, :, :] = 0
    base[:, ::32, :] = 0
    return base


def make_portrait(h=512, w=512):
    """Simple geometric 'portrait' (circles, rectangles)."""
    img = Image.new("RGB", (w, h), color=(30, 30, 50))
    draw = ImageDraw.Draw(img)
    # Background gradient-like rectangles
    for i in range(0, h, 4):
        c = int(30 + i * 0.1)
        draw.rectangle([0, i, w, i+4], fill=(c, c, c+30))
    # Head
    draw.ellipse([156, 80, 356, 280], fill=(220, 180, 140))
    # Body
    draw.rectangle([186, 280, 326, 460], fill=(60, 80, 160))
    # Eyes
    draw.ellipse([210, 145, 240, 175], fill=(60, 40, 20))
    draw.ellipse([272, 145, 302, 175], fill=(60, 40, 20))
    return np.array(img)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    samples = [
        ("gradient.png",  make_gradient()),
        ("textured.png",  make_textured()),
        ("portrait.png",  make_portrait()),
    ]
    for name, arr in samples:
        path = os.path.join(OUTPUT_DIR, name)
        Image.fromarray(arr).save(path)
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
