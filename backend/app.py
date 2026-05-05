"""
Flask Backend - Secure Steganography API
Routes: /encode, /decode, /verify, /generate-keys, /logs, /health
"""

import os
import sys
import struct
import time
import json
import logging
import traceback
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ── Path setup ────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_model.loader import generate_desm
from encryption.crypto import (
    generate_rsa_keypair, encrypt_message, decrypt_message, EncryptedPayload
)
from steganography.engine import embed, extract, strip_lsbs
from error_correction.ecc import encode_with_ecc, decode_with_ecc
from verification.integrity import create_metadata, verify_metadata, IntegrityMetadata
from utils.image_io import bytes_to_numpy, numpy_to_base64, load_image, save_image
from utils.metrics import compute_all_metrics
from database.db import init_db, log_encode, log_decode, save_keypair_db, get_recent_logs

# ──────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
app = Flask(
    __name__,
    static_folder=str(ROOT / "frontend"),
    static_url_path="",
)
CORS(app)

# Init DB using an absolute path anchored to the project root
_DB_PATH = str(ROOT / "stego.db")
init_db(db_path=_DB_PATH)

UPLOAD_DIR = ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

STEGO_DIR = ROOT / "stego_output"
STEGO_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────
#  Serve frontend
# ──────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(str(ROOT / "frontend"), "index.html")


# ──────────────────────────────────────────────
#  Health Check
# ──────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({"status": "ok", "timestamp": time.time()})


# ──────────────────────────────────────────────
#  Generate RSA Key Pair
# ──────────────────────────────────────────────

@app.route("/generate-keys", methods=["POST"])
def generate_keys():
    """Generate a new RSA-2048 key pair, store in DB."""
    try:
        data  = request.get_json(silent=True) or {}
        label = data.get("label", f"key_{int(time.time())}")

        private_pem, public_pem = generate_rsa_keypair()
        save_keypair_db(label, public_pem, private_pem)

        return jsonify({
            "success":    True,
            "label":      label,
            "public_key": public_pem.decode(),
            "private_key": private_pem.decode(),
            "message":    "Key pair generated. Keep the private key safe!",
        })
    except Exception as e:
        log.error("generate-keys error: %s", traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


# ──────────────────────────────────────────────
#  /encode
# ──────────────────────────────────────────────

@app.route("/encode", methods=["POST"])
def encode_route():
    """
    Form-data:
      image:      file (required)
      message:    str  (required)
      public_key: str  (PEM, optional – generates fresh pair if omitted)
    """
    t_start = time.perf_counter()
    try:
        # ── Input validation
        if "image" not in request.files:
            return jsonify({"success": False, "error": "No image file provided"}), 400
        message = request.form.get("message", "").strip()
        if not message:
            return jsonify({"success": False, "error": "Message is empty"}), 400

        img_file = request.files["image"]
        img_bytes = img_file.read()
        image_np  = bytes_to_numpy(img_bytes)
        img_name  = img_file.filename or "unknown.png"

        # ── Key handling
        pub_pem_str  = request.form.get("public_key", "")
        priv_pem_out = None

        if pub_pem_str:
            pub_pem = pub_pem_str.encode()
        else:
            priv_pem_out, pub_pem = generate_rsa_keypair()

        # ── Encryption
        payload_obj  = encrypt_message(message, pub_pem)
        payload_bytes = payload_obj.to_bytes()

        # ── Error correction
        ecc_payload = encode_with_ecc(payload_bytes)

        # ── Integrity metadata
        meta = create_metadata(ecc_payload)

        # Bundle: [4-byte meta_len][meta_json][ecc_payload]
        meta_bytes = meta.to_bytes()
        full_bundle = struct.pack(">I", len(meta_bytes)) + meta_bytes + ecc_payload

        # ── DESM generation
        desm = generate_desm(image_np)

        # ── Embed
        stego_np = embed(image_np, full_bundle, desm)

        # ── Metrics
        metrics = compute_all_metrics(image_np, stego_np, len(full_bundle))

        # ── Save stego image
        stego_path = str(STEGO_DIR / f"stego_{int(time.time()*1000)}.png")
        save_image(stego_np, stego_path)

        # ── Encode images to base64 for frontend
        orig_b64  = numpy_to_base64(image_np)
        stego_b64 = numpy_to_base64(stego_np)

        t_elapsed = round(time.perf_counter() - t_start, 3)

        # ── DB log
        log_encode(
            image_name=img_name,
            message_len=len(message),
            psnr=metrics["psnr"],
            ssim=metrics["ssim"],
            mse=metrics["mse"],
            encode_time=t_elapsed,
            sha256=meta.sha256,
            metadata=metrics,
            db_path=_DB_PATH,
        )

        response = {
            "success":         True,
            "stego_image_b64": stego_b64,
            "original_b64":    orig_b64,
            "metrics":         metrics,
            "sha256":          meta.sha256,
            "timestamp":       meta.timestamp,
            "encode_time":     t_elapsed,
            "message_length":  len(message),
        }
        if priv_pem_out:
            response["private_key"] = priv_pem_out.decode()
            response["public_key"]  = pub_pem.decode()

        return jsonify(response)

    except OverflowError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        log.error("encode error: %s", traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


# ──────────────────────────────────────────────
#  /decode
# ──────────────────────────────────────────────

@app.route("/decode", methods=["POST"])
def decode_route():
    """
    Form-data:
      image:       file (stego image, required)
      private_key: str  (PEM, required)
    """
    t_start = time.perf_counter()
    try:
        if "image" not in request.files:
            return jsonify({"success": False, "error": "No stego image provided"}), 400

        priv_pem_str = request.form.get("private_key", "").strip()
        if not priv_pem_str:
            return jsonify({"success": False, "error": "Private key required for decoding"}), 400

        img_file  = request.files["image"]
        img_bytes = img_file.read()
        img_name  = img_file.filename or "unknown.png"
        stego_np  = bytes_to_numpy(img_bytes)

        # ── DESM — MUST be computed from LSB-stripped stego so the pixel sort
        # order matches what was used during embedding on the original cover.
        desm = generate_desm(strip_lsbs(stego_np))

        # ── Extract raw bundle
        full_bundle = extract(stego_np, desm)

        # ── Parse bundle: meta + ecc_payload
        meta_len    = struct.unpack(">I", full_bundle[:4])[0]
        meta_bytes  = full_bundle[4:4 + meta_len]
        ecc_payload = full_bundle[4 + meta_len:]

        # ── Verify integrity
        meta       = IntegrityMetadata.from_bytes(meta_bytes)
        verify_res = verify_metadata(ecc_payload, meta)

        # ── ECC decode
        payload_bytes = decode_with_ecc(ecc_payload)

        # ── Decrypt
        payload_obj = EncryptedPayload.from_bytes(payload_bytes)
        priv_pem    = priv_pem_str.encode()
        message     = decrypt_message(payload_obj, priv_pem)

        t_elapsed = round(time.perf_counter() - t_start, 3)

        log_decode(
            image_name=img_name,
            decode_time=t_elapsed,
            authentic=verify_res["authentic"],
            reason=verify_res["reason"],
            metadata=verify_res,
            db_path=_DB_PATH,
        )

        return jsonify({
            "success":     True,
            "message":     message,
            "authentic":   verify_res["authentic"],
            "reason":      verify_res["reason"],
            "hash_match":  verify_res["hash_match"],
            "hmac_match":  verify_res["hmac_match"],
            "decode_time": t_elapsed,
            "timestamp":   meta.timestamp,
        })

    except Exception as e:
        log.error("decode error: %s", traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


# ──────────────────────────────────────────────
#  /verify
# ──────────────────────────────────────────────

@app.route("/verify", methods=["POST"])
def verify_route():
    """
    Verify stego image integrity without decrypting.
    Only requires the stego image — no private key needed.
    Checks magic header, SHA-256 hash, and HMAC.
    Form-data:
      image: stego image file (required)
    """
    t_start = time.perf_counter()
    try:
        if "image" not in request.files:
            return jsonify({"success": False, "error": "No stego image provided"}), 400

        stego_np    = bytes_to_numpy(request.files["image"].read())
        # Strip LSBs before DESM so pixel order matches the original embed order
        desm        = generate_desm(strip_lsbs(stego_np))
        full_bundle = extract(stego_np, desm)

        meta_len    = struct.unpack(">I", full_bundle[:4])[0]
        meta_bytes  = full_bundle[4:4 + meta_len]
        ecc_payload = full_bundle[4 + meta_len:]

        meta       = IntegrityMetadata.from_bytes(meta_bytes)
        verify_res = verify_metadata(ecc_payload, meta)

        return jsonify({
            "success":     True,
            "authentic":   verify_res["authentic"],
            "hash_match":  verify_res["hash_match"],
            "hmac_match":  verify_res["hmac_match"],
            "reason":      verify_res["reason"],
            "sha256":      meta.sha256,
            "timestamp":   meta.timestamp,
            "verify_time": round(time.perf_counter() - t_start, 3),
        })

    except Exception as e:
        log.error("verify error: %s", traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


# ──────────────────────────────────────────────
#  /logs
# ──────────────────────────────────────────────

@app.route("/logs")
def logs_route():
    n = int(request.args.get("n", 20))
    return jsonify(get_recent_logs(n, db_path=_DB_PATH))


# ──────────────────────────────────────────────
#  Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    log.info("Starting Steganography API on http://localhost:%d", port)
    app.run(host="0.0.0.0", port=port, debug=False)
