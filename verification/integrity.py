"""
Verification Module - SHA-256 + HMAC
Generates and verifies integrity metadata for embedded payloads.
"""

import hmac as _hmac
import hashlib
import json
import time
import logging
from dataclasses import dataclass, asdict

log = logging.getLogger(__name__)

HMAC_KEY_ENV = "STEGO_HMAC_KEY"
DEFAULT_HMAC_KEY = b"stego-default-hmac-key-2025"


@dataclass
class IntegrityMetadata:
    sha256:    str    # hex digest of the plaintext payload bytes
    hmac:      str    # hex HMAC-SHA256 of sha256||timestamp
    timestamp: float  # Unix epoch float

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    def to_bytes(self) -> bytes:
        return self.to_json().encode()

    @classmethod
    def from_json(cls, s: str) -> "IntegrityMetadata":
        return cls(**json.loads(s))

    @classmethod
    def from_bytes(cls, b: bytes) -> "IntegrityMetadata":
        return cls.from_json(b.decode())


# ──────────────────────────────────────────────
#  Core helpers
# ──────────────────────────────────────────────

def _get_hmac_key() -> bytes:
    import os
    key = os.environ.get(HMAC_KEY_ENV, "").encode()
    return key if key else DEFAULT_HMAC_KEY


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compute_hmac(sha256_hex: str, timestamp: float, key: bytes = None) -> str:
    if key is None:
        key = _get_hmac_key()
    msg = f"{sha256_hex}{timestamp}".encode()
    return _hmac.new(key, msg, hashlib.sha256).hexdigest()


# ──────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────

def create_metadata(payload: bytes) -> IntegrityMetadata:
    """
    Create integrity metadata for a payload.

    Args:
        payload: the raw bytes being embedded (after encryption)

    Returns:
        IntegrityMetadata
    """
    sha = compute_sha256(payload)
    ts  = time.time()
    hmac_val = compute_hmac(sha, ts)
    meta = IntegrityMetadata(sha256=sha, hmac=hmac_val, timestamp=ts)
    log.debug("Metadata created | SHA256=%s...  ts=%.2f", sha[:16], ts)
    return meta


def verify_metadata(payload: bytes, meta: IntegrityMetadata) -> dict:
    """
    Verify integrity of a payload against stored metadata.

    Returns:
        dict with keys:
          authentic (bool)
          hash_match (bool)
          hmac_match (bool)
          reason (str)
    """
    sha_now  = compute_sha256(payload)
    hmac_now = compute_hmac(meta.sha256, meta.timestamp)

    hash_match = _hmac.compare_digest(sha_now, meta.sha256)
    hmac_match = _hmac.compare_digest(hmac_now, meta.hmac)
    authentic  = hash_match and hmac_match

    reason = "Authentic" if authentic else (
        "Hash mismatch — payload may be corrupted or tampered"
        if not hash_match else
        "HMAC mismatch — metadata authentication failed"
    )

    return {
        "authentic":  authentic,
        "hash_match": hash_match,
        "hmac_match": hmac_match,
        "reason":     reason,
    }
