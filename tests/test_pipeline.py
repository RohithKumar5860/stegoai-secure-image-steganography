"""
Unit Tests - Steganography Pipeline
Tests each module independently and as an end-to-end pipeline.
"""

import os
import sys
import struct
import unittest
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ──────────────────────────────────────────────
#  Encryption Tests
# ──────────────────────────────────────────────

class TestEncryption(unittest.TestCase):
    def setUp(self):
        from encryption.crypto import generate_rsa_keypair
        self.private_pem, self.public_pem = generate_rsa_keypair()

    def test_roundtrip_short(self):
        from encryption.crypto import encrypt_message, decrypt_message
        msg = "Hello, secret world!"
        payload = encrypt_message(msg, self.public_pem)
        result  = decrypt_message(payload, self.private_pem)
        self.assertEqual(result, msg)

    def test_roundtrip_long(self):
        from encryption.crypto import encrypt_message, decrypt_message
        msg = "A" * 5000
        payload = encrypt_message(msg, self.public_pem)
        result  = decrypt_message(payload, self.private_pem)
        self.assertEqual(result, msg)

    def test_wrong_key_fails(self):
        from encryption.crypto import encrypt_message, decrypt_message, generate_rsa_keypair
        msg = "test"
        payload = encrypt_message(msg, self.public_pem)
        other_priv, _ = generate_rsa_keypair()
        with self.assertRaises(Exception):
            decrypt_message(payload, other_priv)

    def test_payload_to_from_bytes(self):
        from encryption.crypto import encrypt_message, EncryptedPayload
        payload = encrypt_message("test", self.public_pem)
        raw     = payload.to_bytes()
        restored = EncryptedPayload.from_bytes(raw)
        self.assertEqual(payload.ciphertext, restored.ciphertext)


# ──────────────────────────────────────────────
#  Error Correction Tests
# ──────────────────────────────────────────────

class TestErrorCorrection(unittest.TestCase):
    def test_hamming_roundtrip(self):
        from error_correction.ecc import hamming_encode, hamming_decode
        data    = b"Test data 12345!@#$%"
        encoded = hamming_encode(data)
        decoded = hamming_decode(encoded)
        self.assertEqual(decoded, data)

    def test_hamming_single_bit_correction(self):
        from error_correction.ecc import hamming_encode, hamming_decode
        data    = b"\xAB"
        encoded = bytearray(hamming_encode(data))
        encoded[0] ^= 0x01   # flip LSB of first codeword
        decoded = hamming_decode(bytes(encoded))
        self.assertEqual(decoded, data)

    def test_combined_ecc_roundtrip(self):
        from error_correction.ecc import encode_with_ecc, decode_with_ecc
        data    = b"Error correction test payload!"
        encoded = encode_with_ecc(data)
        decoded = decode_with_ecc(encoded)
        self.assertEqual(decoded, data)


# ──────────────────────────────────────────────
#  Verification Tests
# ──────────────────────────────────────────────

class TestVerification(unittest.TestCase):
    def test_authentic(self):
        from verification.integrity import create_metadata, verify_metadata
        data = b"payload bytes"
        meta = create_metadata(data)
        res  = verify_metadata(data, meta)
        self.assertTrue(res["authentic"])

    def test_tampered(self):
        from verification.integrity import create_metadata, verify_metadata
        data  = b"original payload"
        meta  = create_metadata(data)
        tampered = b"modified payload!"
        res   = verify_metadata(tampered, meta)
        self.assertFalse(res["authentic"])
        self.assertFalse(res["hash_match"])

    def test_metadata_serialisation(self):
        from verification.integrity import create_metadata, IntegrityMetadata
        data = b"test"
        meta = create_metadata(data)
        raw  = meta.to_bytes()
        restored = IntegrityMetadata.from_bytes(raw)
        self.assertEqual(meta.sha256, restored.sha256)
        self.assertEqual(meta.hmac, restored.hmac)


# ──────────────────────────────────────────────
#  Steganography Engine Tests
# ──────────────────────────────────────────────

class TestSteganographyEngine(unittest.TestCase):
    def _make_cover(self, h=128, w=128):
        np.random.seed(42)
        return (np.random.randint(0, 256, (h, w, 3), dtype=np.uint8))

    def _make_desm(self, h=128, w=128, val=0.8):
        return np.full((h, w), val, dtype=np.float32)

    def test_embed_extract_roundtrip(self):
        from steganography.engine import embed, extract
        cover   = self._make_cover()
        desm    = self._make_desm()
        payload = b"Hello steganography roundtrip test!"
        stego   = embed(cover, payload, desm)
        result  = extract(stego, desm)
        self.assertEqual(result, payload)

    def test_stego_shape_unchanged(self):
        from steganography.engine import embed
        cover  = self._make_cover()
        desm   = self._make_desm()
        stego  = embed(cover, b"test", desm)
        self.assertEqual(cover.shape, stego.shape)
        self.assertEqual(stego.dtype, np.uint8)

    def test_overflow_raises(self):
        from steganography.engine import embed
        cover   = self._make_cover(32, 32)   # tiny image
        desm    = self._make_desm(32, 32, val=0.3)
        huge    = b"X" * 10000               # too large
        with self.assertRaises(OverflowError):
            embed(cover, huge, desm)


# ──────────────────────────────────────────────
#  Metrics Tests
# ──────────────────────────────────────────────

class TestMetrics(unittest.TestCase):
    def test_identical_psnr_inf(self):
        from utils.metrics import compute_psnr
        img  = np.zeros((64, 64, 3), dtype=np.uint8)
        psnr = compute_psnr(img, img)
        self.assertEqual(psnr, float('inf'))

    def test_psnr_positive(self):
        from utils.metrics import compute_psnr
        orig  = np.zeros((64, 64, 3), dtype=np.uint8)
        noisy = np.ones((64, 64, 3), dtype=np.uint8)
        psnr  = compute_psnr(orig, noisy)
        self.assertGreater(psnr, 0)

    def test_ssim_one_for_identical(self):
        from utils.metrics import compute_ssim
        img  = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        ssim = compute_ssim(img, img)
        self.assertAlmostEqual(ssim, 1.0, places=4)


# ──────────────────────────────────────────────
#  End-to-End Pipeline Test
# ──────────────────────────────────────────────

class TestEndToEnd(unittest.TestCase):
    def test_full_pipeline(self):
        """Complete encode → decode pipeline with all modules."""
        import struct
        from encryption.crypto import generate_rsa_keypair, encrypt_message, decrypt_message, EncryptedPayload
        from error_correction.ecc import encode_with_ecc, decode_with_ecc
        from verification.integrity import create_metadata, verify_metadata, IntegrityMetadata
        from steganography.engine import embed, extract

        # Setup
        priv_pem, pub_pem = generate_rsa_keypair()
        message = "End-to-end pipeline test message — 2025!"
        cover   = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
        desm    = np.random.uniform(0.3, 1.0, (256, 256)).astype(np.float32)

        # Encode path
        enc_payload  = encrypt_message(message, pub_pem)
        payload_bytes = enc_payload.to_bytes()
        ecc_payload  = encode_with_ecc(payload_bytes)
        meta         = create_metadata(ecc_payload)
        meta_bytes   = meta.to_bytes()
        bundle       = struct.pack(">I", len(meta_bytes)) + meta_bytes + ecc_payload
        stego        = embed(cover, bundle, desm)

        # Decode path
        bundle_out   = extract(stego, desm)
        meta_len     = struct.unpack(">I", bundle_out[:4])[0]
        meta_bytes_o = bundle_out[4:4 + meta_len]
        ecc_out      = bundle_out[4 + meta_len:]
        meta_out     = IntegrityMetadata.from_bytes(meta_bytes_o)
        v            = verify_metadata(ecc_out, meta_out)
        payload_out  = decode_with_ecc(ecc_out)
        enc_obj      = EncryptedPayload.from_bytes(payload_out)
        recovered    = decrypt_message(enc_obj, priv_pem)

        self.assertTrue(v["authentic"], f"Integrity check failed: {v['reason']}")
        self.assertEqual(recovered, message)


# ──────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
