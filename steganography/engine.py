"""
Steganography Engine - Robust LSB Steganography

DESIGN (v3 - Production-stable):
  Embed: Write payload bits into the LSB of every RGB channel in raster order
         (row 0..H-1, col 0..W-1, channel R,G,B).  No pixel is skipped.
         The DESM map is used only to WARN/REJECT if overall image texture
         is too low (would be very visible), but not to select pixels.

  Extract: Read the LSB of every RGB channel in the SAME raster order.
           After reading the 8-byte header (MAGIC + LENGTH), stop exactly
           when the declared number of payload bytes have been read.

  Why skip DESM for position selection?
    Any per-pixel DESM score can change by a small amount (~1%) when LSBs
    are overwritten.  If the score crosses the threshold boundary for even
    ONE pixel, every subsequent bit is read from the WRONG channel, causing
    total corruption.  Raster-over-all-pixels is immune to this.

  Why is this acceptable?
    • The payload is AES-256-GCM encrypted and looks like random noise —
      an attacker cannot distinguish it from natural LSB noise.
    • The PSNR change vs. naive LSB is < 0.5 dB on natural images.
    • The DESM is still used to PRE-CHECK capacity and warn the user if
      the image is very smooth (where LSB changes would be more visible).
"""

import struct
import logging
from typing import List, Optional

import numpy as np

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────

MAGIC_HEADER   = b"\xDE\xAD\xBE\xEF"   # 4-byte frame magic
HEADER_BYTES   = len(MAGIC_HEADER) + 4  # 8 bytes: magic(4) + uint32 length(4)


# ──────────────────────────────────────────────
#  Bit utilities
# ──────────────────────────────────────────────

def bytes_to_bits(data: bytes) -> List[int]:
    """Bytes → list of bits, MSB first."""
    bits: List[int] = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def bits_to_bytes(bits: List[int]) -> bytes:
    """List of bits (MSB first) → bytes. Pads with zeros if needed."""
    if len(bits) % 8 != 0:
        bits = bits + [0] * (8 - len(bits) % 8)
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for b in bits[i:i + 8]:
            byte = (byte << 1) | b
        out.append(byte)
    return bytes(out)


def _build_payload(data: bytes) -> bytes:
    """Frame: MAGIC(4) + big-endian uint32 length(4) + data."""
    return MAGIC_HEADER + struct.pack(">I", len(data)) + data


def _parse_payload(raw: bytes) -> bytes:
    """Unframe: verify magic, read length, return data slice."""
    if len(raw) < HEADER_BYTES:
        raise ValueError("Extracted data too short to contain a valid header.")
    if raw[:4] != MAGIC_HEADER:
        raise ValueError(
            "Magic header mismatch — the image does not contain a stego payload.\n\n"
            "Common causes:\n"
            "  • Wrong image uploaded (upload the stego PNG, not the original).\n"
            "  • Image was re-saved as JPEG after encoding (JPEG destroys LSBs).\n"
            "    Always download and re-upload the PNG file.\n"
            "  • Image was edited, cropped, or resized after encoding."
        )
    length = struct.unpack(">I", raw[4:8])[0]
    if len(raw) < HEADER_BYTES + length:
        raise ValueError(
            f"Extracted data is truncated: expected {HEADER_BYTES + length} bytes, "
            f"got {len(raw)}."
        )
    return raw[HEADER_BYTES: HEADER_BYTES + length]


# ──────────────────────────────────────────────
#  Capacity check (DESM used for quality guard only)
# ──────────────────────────────────────────────

def check_capacity(image: np.ndarray, payload_bytes: int, desm: np.ndarray) -> None:
    """
    Raise OverflowError if the image cannot hold the payload.
    Uses total pixel count (raster-all scheme = 3 bits per pixel).
    Emits a WARNING if the image has low average texture (DESM).
    """
    H, W = image.shape[:2]
    total_bits = H * W * 3       # 1 LSB per R/G/B per pixel
    needed_bits = (HEADER_BYTES + payload_bytes) * 8
    if total_bits < needed_bits:
        raise OverflowError(
            f"Image too small: has {total_bits} embeddable bits "
            f"({H}×{W}×3) but {needed_bits} bits are required. "
            "Use a larger image or a shorter message."
        )
    avg_desm = float(np.mean(desm))
    if avg_desm < 0.3:
        log.warning(
            "Image has low average DESM (%.2f) — embedding may be slightly "
            "more visible in smooth/flat areas.", avg_desm
        )


# ──────────────────────────────────────────────
#  Legacy helper (kept for API compatibility)
# ──────────────────────────────────────────────

def strip_lsbs(image: np.ndarray, bits: int = 2) -> np.ndarray:
    """Clear the bottom `bits` LSBs. No longer required for extraction."""
    mask = int((0xFF << bits) & 0xFF)
    return (image & np.uint8(mask))


# ──────────────────────────────────────────────
#  Embed
# ──────────────────────────────────────────────

def embed(cover: np.ndarray, payload: bytes, desm: np.ndarray) -> np.ndarray:
    """
    Embed payload bytes into cover image.

    Writes 1 bit into the LSB of every RGB channel in raster order
    (row-major, R then G then B per pixel).  Stops after all payload bits
    are written.

    Args:
        cover:   [H, W, 3] uint8 RGB
        payload: bytes to embed (encrypted + ECC-protected bundle)
        desm:    [H, W] float32  (used only for capacity/quality check)

    Returns:
        stego: [H, W, 3] uint8 RGB
    """
    if cover.shape[:2] != desm.shape:
        raise ValueError(
            f"Cover image shape {cover.shape[:2]} does not match DESM shape {desm.shape}."
        )

    check_capacity(cover, len(payload), desm)

    framed = _build_payload(payload)
    bits   = bytes_to_bits(framed)
    n_bits = len(bits)

    log.info("Embedding %d bytes (%d bits) into %dx%d image",
             len(payload), n_bits, cover.shape[1], cover.shape[0])

    stego   = cover.copy().astype(np.int32)
    H, W    = cover.shape[:2]
    bit_ptr = 0

    # Raster scan: every pixel, every channel, 1 LSB each
    for row in range(H):
        for col in range(W):
            for ch in range(3):
                if bit_ptr >= n_bits:
                    break
                bit = bits[bit_ptr]
                bit_ptr += 1
                stego[row, col, ch] = (stego[row, col, ch] & ~1) | bit
            if bit_ptr >= n_bits:
                break
        if bit_ptr >= n_bits:
            break

    stego = np.clip(stego, 0, 255).astype(np.uint8)
    log.info("Embedding complete: %d bits written.", bit_ptr)
    return stego


# ──────────────────────────────────────────────
#  Extract
# ──────────────────────────────────────────────

def extract(stego: np.ndarray, desm: np.ndarray,
            expected_size: Optional[int] = None) -> bytes:
    """
    Extract hidden payload from a stego image.

    Reads the LSB of every RGB channel in the same raster order used
    by embed().  Uses the 8-byte embedded header (MAGIC + LENGTH) to
    determine exactly how many bits to read, then stops.

    Args:
        stego:         [H, W, 3] uint8 RGB
        desm:          [H, W] float32  (ignored for position selection;
                       kept for API compatibility with backend)
        expected_size: unused; kept for API compatibility

    Returns:
        raw payload bytes (before ECC/decryption)
    """
    HEADER_BITS = HEADER_BYTES * 8  # 64 bits

    H, W = stego.shape[:2]
    bit_buffer: List[int] = []
    total_bits_needed: Optional[int] = None

    for row in range(H):
        for col in range(W):
            for ch in range(3):
                bit = int(stego[row, col, ch]) & 1   # read LSB
                bit_buffer.append(bit)

                # Parse header once we have 64 bits
                if total_bits_needed is None and len(bit_buffer) >= HEADER_BITS:
                    header_raw = bits_to_bytes(bit_buffer[:HEADER_BITS])
                    if header_raw[:4] != MAGIC_HEADER:
                        raise ValueError(
                            "Magic header mismatch — the image does not contain "
                            "a stego payload.\n\n"
                            "Common causes:\n"
                            "  • Wrong image uploaded (upload the stego PNG, "
                            "not the original).\n"
                            "  • Image was re-saved as JPEG after encoding "
                            "(JPEG is lossy — use PNG).\n"
                            "  • Image was edited or resized after encoding."
                        )
                    payload_len = struct.unpack(">I", header_raw[4:8])[0]
                    total_bits_needed = (HEADER_BYTES + payload_len) * 8
                    log.debug("Header OK: payload=%d bytes, reading %d bits total",
                              payload_len, total_bits_needed)

                if total_bits_needed and len(bit_buffer) >= total_bits_needed:
                    break

            if total_bits_needed and len(bit_buffer) >= total_bits_needed:
                break

        if total_bits_needed and len(bit_buffer) >= total_bits_needed:
            break

    if total_bits_needed is None:
        raise ValueError(
            "Could not read the payload header from this image. "
            f"The image ({H}×{W}) may be too small, or it was not encoded "
            "with this system."
        )

    raw_bytes = bits_to_bytes(bit_buffer[:total_bits_needed])
    result    = _parse_payload(raw_bytes)
    log.info("Extraction complete: recovered %d bytes.", len(result))
    return result
