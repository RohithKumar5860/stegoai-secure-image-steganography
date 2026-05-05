"""
Training Script for TA-DPCNN

Usage:
  python -m ai_model.train --data_dir ./data/train --epochs 50 --batch_size 8

The training pairs cover images with synthetically embedded stego images
so that the model learns where embedding is perceptually safe.
"""

import argparse
import os
import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

try:
    import cv2
except ImportError:
    cv2 = None

from .model import TADPCNN
from .losses import DESMLoss

# ──────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────────

class SteganographyDataset(Dataset):
    """
    Loads cover images and creates synthetic stego images by randomly
    embedding bits, so the model learns perceptual embedding quality.
    """
    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    def __init__(self, data_dir: str, size: int = 256):
        self.paths = [
            p for p in Path(data_dir).rglob("*")
            if p.suffix.lower() in self.EXTENSIONS
        ]
        if not self.paths:
            raise ValueError(f"No images found in {data_dir}")
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])
        log.info("Dataset: %d images in %s", len(self.paths), data_dir)

    def _load_image(self, path: Path) -> np.ndarray:
        if cv2:
            img = cv2.imread(str(path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            from PIL import Image
            img = np.array(Image.open(path).convert("RGB"))
        return img

    def _embed_synthetic(self, cover: torch.Tensor) -> torch.Tensor:
        """Random LSB embedding to simulate a stego image."""
        noise_mask = torch.rand_like(cover)                # uniform noise
        bits       = torch.randint(0, 2, cover.shape).float() / 255.0
        stego = cover.clone()
        stego[noise_mask > 0.7] += bits[noise_mask > 0.7]
        return stego.clamp(0, 1)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_np = self._load_image(self.paths[idx])
        cover  = self.transform(img_np)                    # [3, H, W]
        stego  = self._embed_synthetic(cover)              # [3, H, W]
        return cover, stego


# ──────────────────────────────────────────────
#  Training Loop
# ──────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Training device: %s", device)

    # Data
    dataset = SteganographyDataset(args.data_dir, size=args.image_size)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=True, num_workers=args.workers, pin_memory=True)

    # Model
    model = TADPCNN().to(device)
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        log.info("Resumed from %s", args.resume)

    # Loss + Optimiser
    criterion = DESMLoss(alpha=args.alpha, beta=args.beta, gamma=args.gamma)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss = float("inf")
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for cover, stego in loader:
            cover = cover.to(device)
            stego = stego.to(device)

            optimizer.zero_grad()
            desm = model(cover)                        # [B, 1, H, W]
            loss, metrics = criterion(stego, cover, desm)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg = epoch_loss / len(loader)
        elapsed = time.time() - t0

        log.info(
            "Epoch %3d/%d | loss=%.4f | MSE=%.4f | SSIM=%.4f | Entropy=%.4f | %.1fs",
            epoch, args.epochs, avg,
            metrics["mse"], metrics["ssim"], metrics["entropy"], elapsed,
        )

        # Checkpoint
        if avg < best_loss:
            best_loss = avg
            ckpt_path = os.path.join(args.save_dir, "best_tadpcnn.pth")
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "loss":        best_loss,
                "args":        vars(args),
            }, ckpt_path)
            log.info("  → Saved best model to %s", ckpt_path)

        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.save_dir, f"tadpcnn_ep{epoch:03d}.pth")
            torch.save({"epoch": epoch, "model_state": model.state_dict()}, ckpt_path)

    log.info("Training complete. Best loss: %.4f", best_loss)


# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("TA-DPCNN Training")
    p.add_argument("--data_dir",   default="./data/train")
    p.add_argument("--save_dir",   default="./ai_model/checkpoints")
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch_size", type=int,   default=8)
    p.add_argument("--image_size", type=int,   default=256)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--alpha",      type=float, default=1.0,  help="MSE weight")
    p.add_argument("--beta",       type=float, default=0.5,  help="SSIM weight")
    p.add_argument("--gamma",      type=float, default=0.1,  help="Entropy weight")
    p.add_argument("--workers",    type=int,   default=2)
    p.add_argument("--save_every", type=int,   default=10)
    p.add_argument("--resume",     default="", help="Path to checkpoint to resume from")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
