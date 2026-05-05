"""
Encryption Module: AES-256-GCM + RSA-2048
Pipeline: compress → AES-256-GCM encrypt → RSA-2048 encrypt AES key

Key functions:
  generate_rsa_keypair()
  encrypt_message(message, public_key)  → EncryptedPayload
  decrypt_message(payload, private_key) → plaintext
"""

import os
import zlib
import json
import base64
import logging
from dataclasses import dataclass, asdict

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.backends import default_backend

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
#  Data Container
# ──────────────────────────────────────────────

@dataclass
class EncryptedPayload:
    """
    All data needed to decrypt the message.
    Fields are base64-encoded bytes.
    """
    encrypted_aes_key: str   # RSA-encrypted AES key
    nonce:             str   # AES-GCM nonce (12 bytes)
    ciphertext:        str   # AES-GCM ciphertext
    tag:               str   # AES-GCM auth tag (included in ciphertext by AESGCM)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    def to_bytes(self) -> bytes:
        return self.to_json().encode("utf-8")

    @classmethod
    def from_json(cls, data: str) -> "EncryptedPayload":
        d = json.loads(data)
        return cls(**d)

    @classmethod
    def from_bytes(cls, data: bytes) -> "EncryptedPayload":
        return cls.from_json(data.decode("utf-8"))


# ──────────────────────────────────────────────
#  RSA Key Pair Generation
# ──────────────────────────────────────────────

def generate_rsa_keypair() -> tuple[bytes, bytes]:
    """
    Generate an RSA-2048 key pair.

    Returns:
        (private_key_pem, public_key_pem) as bytes
    """
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend(),
    )
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_pem, public_pem


def load_private_key(pem: bytes):
    return serialization.load_pem_private_key(pem, password=None, backend=default_backend())


def load_public_key(pem: bytes):
    return serialization.load_pem_public_key(pem, backend=default_backend())


# ──────────────────────────────────────────────
#  Encrypt
# ──────────────────────────────────────────────

def encrypt_message(message: str, public_key_pem: bytes) -> EncryptedPayload:
    """
    Encrypt message:
      1. zlib compress
      2. AES-256-GCM encrypt (random key)
      3. RSA-2048 encrypt the AES key

    Args:
        message:        Plaintext UTF-8 string
        public_key_pem: RSA public key PEM bytes

    Returns:
        EncryptedPayload
    """
    # Step 1: Compress
    compressed = zlib.compress(message.encode("utf-8"), level=9)
    log.debug("Compressed: %d → %d bytes", len(message), len(compressed))

    # Step 2: AES-256-GCM
    aes_key = os.urandom(32)   # 256-bit
    nonce   = os.urandom(12)   # 96-bit GCM nonce
    aesgcm  = AESGCM(aes_key)
    ciphertext = aesgcm.encrypt(nonce, compressed, None)   # includes auth tag

    # Step 3: RSA-OAEP encrypt AES key
    pub_key = load_public_key(public_key_pem)
    encrypted_aes_key = pub_key.encrypt(
        aes_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )

    return EncryptedPayload(
        encrypted_aes_key=base64.b64encode(encrypted_aes_key).decode(),
        nonce=base64.b64encode(nonce).decode(),
        ciphertext=base64.b64encode(ciphertext).decode(),
        tag="",  # included in ciphertext by AESGCM
    )


# ──────────────────────────────────────────────
#  Decrypt
# ──────────────────────────────────────────────

def decrypt_message(payload: EncryptedPayload, private_key_pem: bytes) -> str:
    """
    Decrypt an EncryptedPayload:
      1. RSA-OAEP decrypt AES key
      2. AES-256-GCM decrypt
      3. zlib decompress

    Args:
        payload:         EncryptedPayload instance
        private_key_pem: RSA private key PEM bytes

    Returns:
        Plaintext UTF-8 string
    """
    # Step 1: Recover AES key
    priv_key = load_private_key(private_key_pem)
    encrypted_aes_key = base64.b64decode(payload.encrypted_aes_key)
    aes_key = priv_key.decrypt(
        encrypted_aes_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )

    # Step 2: AES-256-GCM decrypt
    nonce      = base64.b64decode(payload.nonce)
    ciphertext = base64.b64decode(payload.ciphertext)
    aesgcm     = AESGCM(aes_key)
    compressed = aesgcm.decrypt(nonce, ciphertext, None)

    # Step 3: Decompress
    plaintext = zlib.decompress(compressed).decode("utf-8")
    return plaintext


# ──────────────────────────────────────────────
#  Key Storage Helpers
# ──────────────────────────────────────────────

def save_keypair(private_pem: bytes, public_pem: bytes,
                 private_path: str = "private_key.pem",
                 public_path:  str = "public_key.pem") -> None:
    with open(private_path, "wb") as f:
        f.write(private_pem)
    with open(public_path, "wb") as f:
        f.write(public_pem)
    log.info("Keys saved to %s / %s", private_path, public_path)


def load_keypair(private_path: str = "private_key.pem",
                 public_path:  str = "public_key.pem") -> tuple[bytes, bytes]:
    with open(private_path, "rb") as f:
        private_pem = f.read()
    with open(public_path, "rb") as f:
        public_pem = f.read()
    return private_pem, public_pem
