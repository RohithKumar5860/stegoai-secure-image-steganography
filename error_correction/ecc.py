"""
Error Correction Module
  - Hamming(7,4): fast single-bit correction
  - Reed-Solomon (via reedsolo): burst-error correction

Pipeline:
  encode(data) → protected bytes (RS outer, Hamming inner)
  decode(data) → original bytes
"""

import logging
from typing import Dict, Tuple

try:
    import reedsolo
    RS_AVAILABLE = True
except ImportError:
    RS_AVAILABLE = False
    logging.warning("reedsolo not installed; RS correction disabled. pip install reedsolo")

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
#  Hamming(7,4)
# ──────────────────────────────────────────────

# Generator matrix G (4×7)  and parity check matrix H (3×7)
_G = [
    [1, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1],
]
_H = [
    [1, 1, 0, 1, 1, 0, 0],
    [0, 1, 1, 1, 0, 1, 0],
    [1, 0, 1, 1, 0, 0, 1],
]

# Pre-compute syndrome → error position lookup
_SYNDROME_TABLE: Dict[Tuple, int] = {}
for _pos in range(7):
    _syn = tuple(sum(_H[r][_pos] for r in range(3)) % 2 for r in range(3))
    if any(_syn):
        _SYNDROME_TABLE[_syn] = _pos


def _hamming_encode_nibble(nibble: int) -> int:
    """Encode 4-bit nibble → 7-bit codeword (int)."""
    bits = [(nibble >> (3 - i)) & 1 for i in range(4)]
    codeword = [0] * 7
    for i in range(4):
        for j in range(7):
            codeword[j] ^= bits[i] * _G[i][j]
    result = 0
    for b in codeword:
        result = (result << 1) | b
    return result


def _hamming_decode_nibble(codeword7: int) -> int:
    """Decode 7-bit codeword → 4-bit nibble with single-bit correction."""
    received = [(codeword7 >> (6 - i)) & 1 for i in range(7)]
    syndrome  = tuple(sum(_H[r][j] * received[j] for j in range(7)) % 2 for r in range(3))

    if any(syndrome) and syndrome in _SYNDROME_TABLE:
        err_pos = _SYNDROME_TABLE[syndrome]
        received[err_pos] ^= 1   # correct

    # Extract data bits (positions 0,1,2,3 in G)
    nibble = (received[0] << 3) | (received[1] << 2) | (received[2] << 1) | received[3]
    return nibble


def hamming_encode(data: bytes) -> bytes:
    """
    Encode every byte as two 7-bit codewords (high nibble + low nibble).
    Output: 2× the input length (roughly; stored as full bytes).
    """
    out = bytearray()
    for byte in data:
        hi = _hamming_encode_nibble(byte >> 4)
        lo = _hamming_encode_nibble(byte & 0x0F)
        out.append(hi)
        out.append(lo)
    return bytes(out)


def hamming_decode(data: bytes) -> bytes:
    """
    Decode hamming-encoded bytes back to original.
    Expects pairs of bytes (hi nibble codeword, lo nibble codeword).
    """
    if len(data) % 2 != 0:
        raise ValueError("Hamming-encoded data must have even length.")
    out = bytearray()
    for i in range(0, len(data), 2):
        hi = _hamming_decode_nibble(data[i])
        lo = _hamming_decode_nibble(data[i + 1])
        out.append((hi << 4) | lo)
    return bytes(out)


# ──────────────────────────────────────────────
#  Reed-Solomon
# ──────────────────────────────────────────────

RS_NSYM = 10    # number of error-correction symbols per block


def rs_encode(data: bytes) -> bytes:
    """Encode bytes with Reed-Solomon (10 ECC symbols per block)."""
    if not RS_AVAILABLE:
        log.warning("RS encode skipped (reedsolo unavailable).")
        return data
    rsc    = reedsolo.RSCodec(RS_NSYM)
    return bytes(rsc.encode(data))


def rs_decode(data: bytes) -> bytes:
    """Decode Reed-Solomon encoded bytes, correcting burst errors."""
    if not RS_AVAILABLE:
        log.warning("RS decode skipped (reedsolo unavailable).")
        return data
    rsc = reedsolo.RSCodec(RS_NSYM)
    decoded, _, _ = rsc.decode(data)
    return bytes(decoded)


# ──────────────────────────────────────────────
#  Combined Pipeline
# ──────────────────────────────────────────────

def encode_with_ecc(data: bytes) -> bytes:
    """
    Apply both error-correction layers:
      inner: Hamming(7,4)
      outer: Reed-Solomon

    Returns protected bytes ready to embed.
    """
    hamming_encoded = hamming_encode(data)
    rs_encoded      = rs_encode(hamming_encoded)
    return rs_encoded


def decode_with_ecc(data: bytes) -> bytes:
    """
    Reverse both layers (RS outer first, then Hamming inner).
    """
    rs_decoded      = rs_decode(data)
    hamming_decoded = hamming_decode(rs_decoded)
    return hamming_decoded
