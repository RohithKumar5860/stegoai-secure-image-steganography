# 🛡️ StegoAI — AI-Powered Secure Image Steganography System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=for-the-badge&logo=flask&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Tests](https://img.shields.io/badge/Tests-17%2F17%20Passing-brightgreen?style=for-the-badge)

**Hide encrypted messages inside images using AI-driven adaptive embedding.**  
AES-256 · RSA-2048 · TA-DPCNN · Reed-Solomon ECC · SHA-256 / HMAC

</div>

---

## 📖 Table of Contents

1. [Project Description](#-project-description)
2. [Features](#-features)
3. [System Architecture](#-system-architecture)
4. [Tech Stack](#-tech-stack)
5. [Project Structure](#-project-structure)
6. [Installation](#-installation)
7. [How to Run](#-how-to-run)
8. [Usage Instructions](#-usage-instructions)
9. [API Endpoints](#-api-endpoints)
10. [Running Tests](#-running-tests)
11. [AI Model Training](#-ai-model-training)
12. [Future Improvements](#-future-improvements)
13. [Issues Fixed (Changelog)](#-issues-fixed-changelog)
14. [Contributing](#-contributing)
15. [License](#-license)

---

## 📝 Project Description

**StegoAI** is a production-grade steganography system that conceals encrypted text messages inside cover images using an AI model to determine the safest embedding locations. The result is a visually indistinguishable stego image with:

- **Military-grade encryption** (AES-256-GCM + RSA-2048-OAEP)
- **AI-guided adaptive embedding** (Texture-Aware Dual-Path CNN)
- **Dual error correction** (Hamming(7,4) + Reed-Solomon)
- **Tamper detection** (SHA-256 hash + HMAC-SHA256)
- **A modern web UI** (pure HTML/CSS/JS, no frameworks)

---

## ✨ Features

| Category | Feature |
|---|---|
| 🧠 **AI Model** | TA-DPCNN generates a Dynamic Embedding Strength Map (DESM) |
| 🔐 **Encryption** | AES-256-GCM + RSA-2048-OAEP + zlib compression |
| 🧬 **Steganography** | Adaptive LSB with 1–2 bits per pixel based on DESM score |
| 🛡 **Error Correction** | Hamming(7,4) inner + Reed-Solomon(10) outer |
| ✅ **Verification** | SHA-256 + HMAC-SHA256 tamper detection (no key required) |
| 📊 **Metrics** | PSNR, SSIM, MSE — compared against naive LSB baseline |
| 🗄 **Database** | SQLite logging of all encode/decode operations |
| 🌐 **Frontend** | Dark/light mode, drag-drop upload, toast notifications |
| 📱 **Responsive** | Works on desktop and mobile |

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Browser (Frontend)                       │
│   index.html · style.css · script.js  (Pure Vanilla JS)        │
└────────────────────────┬────────────────────────────────────────┘
                         │  HTTP (multipart/form-data)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Flask Backend  (:5000)                     │
│   /encode  /decode  /verify  /generate-keys  /logs  /health    │
└───┬─────────────┬──────────────┬──────────────┬────────────────┘
    │             │              │              │
    ▼             ▼              ▼              ▼
┌────────┐  ┌─────────┐  ┌──────────┐  ┌──────────────┐
│TA-DPCNN│  │AES+RSA  │  │Hamming + │  │SHA-256+HMAC  │
│  DESM  │  │Encrypt  │  │Reed-Sol. │  │Verification  │
│  Map   │  │Decrypt  │  │  ECC     │  │              │
└────┬───┘  └────┬────┘  └────┬─────┘  └──────┬───────┘
     │           │            │               │
     └───────────┴────────────┴───────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  Adaptive LSB Engine │
              │  embed() / extract() │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │    SQLite Database   │
              │   (stego.db)         │
              └──────────────────────┘
```

### Encode Pipeline
```
Message → zlib compress → AES-256-GCM encrypt → RSA encrypt key
       → Hamming(7,4) → Reed-Solomon(10)
       → create_metadata (SHA-256 + HMAC)
       → Bundle: [meta_len|meta_json|ecc_payload]
       → TA-DPCNN generates DESM
       → Adaptive LSB embed using DESM
       → Stego Image
```

### Decode Pipeline
```
Stego Image → TA-DPCNN DESM → Adaptive LSB extract
           → Parse bundle → verify SHA-256 + HMAC
           → Reed-Solomon decode → Hamming decode
           → RSA decrypt AES key → AES-256-GCM decrypt
           → zlib decompress → Plaintext Message
```

---

## 🔧 Tech Stack

| Layer | Technology |
|---|---|
| **AI Model** | PyTorch, Custom TA-DPCNN |
| **Backend** | Python 3.10+, Flask 3.x, Flask-CORS |
| **Encryption** | `cryptography` library (AES-256-GCM, RSA-2048-OAEP) |
| **Error Correction** | `reedsolo` (Reed-Solomon), Custom Hamming(7,4) |
| **Image Processing** | NumPy, Pillow, OpenCV (optional), scikit-image |
| **Database** | SQLite (built-in) |
| **Frontend** | Vanilla HTML5, CSS3 (glassmorphism), JavaScript (ES6+) |
| **Testing** | pytest |

---

## 📁 Project Structure

```
Steganography_proj/
│
├── ai_model/                   # TA-DPCNN neural network
│   ├── __init__.py
│   ├── model.py                # Full architecture (TexturePath, DistortionPath, FusionHead)
│   ├── losses.py               # Composite loss: MSE + SSIM + EntropyPenalty
│   ├── loader.py               # Singleton model loader with checkpoint discovery
│   └── train.py                # Training script with dataset, optimizer, scheduler
│
├── encryption/                 # AES-256-GCM + RSA-2048-OAEP
│   ├── __init__.py
│   └── crypto.py               # encrypt_message(), decrypt_message(), generate_rsa_keypair()
│
├── steganography/              # Adaptive LSB engine
│   ├── __init__.py
│   └── engine.py               # embed(), extract(), _get_pixel_plan()
│
├── error_correction/           # Dual-layer ECC
│   ├── __init__.py
│   └── ecc.py                  # Hamming(7,4) + Reed-Solomon encode/decode
│
├── verification/               # Integrity checking
│   ├── __init__.py
│   └── integrity.py            # SHA-256 + HMAC-SHA256, IntegrityMetadata
│
├── utils/                      # Shared utilities
│   ├── __init__.py
│   ├── image_io.py             # load/save/base64 image helpers
│   ├── metrics.py              # PSNR, SSIM, MSE + baseline comparison
│   └── generate_samples.py     # Generate test images
│
├── database/                   # SQLite persistence
│   ├── __init__.py
│   └── db.py                   # init_db(), log_encode(), log_decode(), keypair storage
│
├── backend/                    # Flask REST API
│   ├── __init__.py
│   └── app.py                  # Routes: /encode /decode /verify /generate-keys /logs /health
│
├── frontend/                   # Pure HTML/CSS/JS UI
│   ├── index.html              # 5-tab UI: Encode, Decode, Verify, Keys, Logs
│   ├── style.css               # Dark/light theme, glassmorphism, animations
│   └── script.js               # Fetch API calls, drag-drop, toast notifications
│
├── tests/
│   └── test_pipeline.py        # 17 unit + integration tests (all passing)
│
├── requirements.txt            # Python dependencies
├── run.py                      # Entry point: python run.py
└── stego.db                    # Auto-created SQLite database
```

---

## ⚙️ Installation

### Prerequisites

- **Python 3.10 or higher** ([download](https://www.python.org/downloads/))
- **pip** (comes with Python)
- **Git** (optional)

### Step 1 — Clone or Download the Project

```bash
git clone https://github.com/yourname/stegoai.git
cd stegoai
```

Or simply extract the project ZIP into a folder called `Steganography_proj`.

### Step 2 — Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note on PyTorch:** `torch` and `torchvision` are optional. The system works fully without them — the TA-DPCNN runs with random (untrained) weights, which still produces a valid DESM. To install PyTorch:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> ```

### Step 4 — Verify Installation

```bash
python -m pytest tests/test_pipeline.py -v
```

Expected output:
```
17 passed in ~4s
```

---

## 🚀 How to Run

### Start the Server

```bash
python run.py
```

You will see:
```
🚀 StegoAI Server running → http://localhost:5000
```

### Open the Web UI

Open your browser and go to:
```
http://localhost:5000
```

The Flask server serves the frontend automatically from the `/frontend` folder.

### Running on a Different Port

```bash
PORT=8080 python run.py          # Linux/macOS
$env:PORT=8080; python run.py    # Windows PowerShell
```

---

## 📘 Usage Instructions

### 🔒 Encoding a Message

1. Open **http://localhost:5000** in your browser
2. Click the **Encode** tab (active by default)
3. **Upload a cover image** — drag & drop or click to browse (PNG/JPG/BMP, max 20 MB)
4. **Type your secret message** in the text area (up to 10,000 characters)
5. **RSA Public Key** (optional):
   - Leave blank → a fresh RSA-2048 key pair is auto-generated
   - Or paste an existing public key PEM
6. Click **Encode & Embed**
7. The system will:
   - Generate a DESM map using TA-DPCNN
   - Encrypt your message (AES-256-GCM + RSA-2048)
   - Apply error correction (Hamming + Reed-Solomon)
   - Embed adaptively based on DESM scores
8. **Results shown:**
   - Side-by-side original vs. stego image
   - PSNR, SSIM, MSE metrics (vs. naive LSB baseline)
   - SHA-256 hash + timestamp
   - ⚠️ **If no public key was provided:** your auto-generated **Private Key** is shown — **save it immediately**, it is required to decode

### 🔓 Decoding a Message

1. Click the **Decode** tab
2. Upload the **stego image** (the one produced by encoding)
3. Paste your **RSA Private Key** (PEM format)
4. Click **Decode Message**
5. The system will:
   - Extract bits using the same DESM order
   - Verify SHA-256 hash + HMAC (tamper detection)
   - Apply error correction
   - Decrypt the message
6. **Results shown:**
   - Recovered plaintext message
   - ✅ Authentic / ⚠️ Tampered status
   - Hash match, HMAC match details
   - Decode time

### ✅ Verifying Image Integrity (No Decryption)

1. Click the **Verify** tab
2. Upload the stego image
3. Click **Verify Integrity** — **no private key required**
4. Results: authentic/tampered, SHA-256 hash, timestamp

### 🔑 Generating Key Pairs

1. Click the **Keys** tab
2. Optionally enter a label for the key
3. Click **Generate Key Pair**
4. Copy both the public and private keys and store them securely

### 📋 Viewing Logs

1. Click the **Logs** tab
2. Click **↺ Refresh** to load the last 30 encode/decode operations

---

## 🌐 API Endpoints

All endpoints accept and return JSON (except file uploads which use `multipart/form-data`).

### `GET /health`
Check if the server is running.

**Response:**
```json
{ "status": "ok", "timestamp": 1746000000.0 }
```

---

### `POST /generate-keys`
Generate an RSA-2048 key pair.

**Request body (JSON):**
```json
{ "label": "my-key-2025" }
```

**Response:**
```json
{
  "success": true,
  "label": "my-key-2025",
  "public_key": "-----BEGIN PUBLIC KEY-----\n...",
  "private_key": "-----BEGIN RSA PRIVATE KEY-----\n...",
  "message": "Key pair generated. Keep the private key safe!"
}
```

---

### `POST /encode`
Embed an encrypted message into an image.

**Request** (`multipart/form-data`):

| Field | Type | Required | Description |
|---|---|---|---|
| `image` | file | ✅ | Cover image (PNG/JPG/BMP) |
| `message` | string | ✅ | Secret message text |
| `public_key` | string | ❌ | RSA public key PEM (auto-generated if omitted) |

**Response:**
```json
{
  "success": true,
  "stego_image_b64": "<base64 PNG>",
  "original_b64": "<base64 PNG>",
  "metrics": {
    "psnr": 51.23,
    "ssim": 0.9997,
    "mse": 0.048,
    "baseline_psnr": 48.13,
    "baseline_ssim": 0.9991,
    "baseline_mse": 0.100
  },
  "sha256": "a3f2...",
  "timestamp": 1746000000.0,
  "encode_time": 1.234,
  "message_length": 42,
  "private_key": "-----BEGIN RSA PRIVATE KEY-----\n...",
  "public_key": "-----BEGIN PUBLIC KEY-----\n..."
}
```

---

### `POST /decode`
Extract and decrypt a message from a stego image.

**Request** (`multipart/form-data`):

| Field | Type | Required | Description |
|---|---|---|---|
| `image` | file | ✅ | Stego image |
| `private_key` | string | ✅ | RSA private key PEM |

**Response:**
```json
{
  "success": true,
  "message": "Your secret message here",
  "authentic": true,
  "reason": "Authentic",
  "hash_match": true,
  "hmac_match": true,
  "decode_time": 0.987,
  "timestamp": 1746000000.0
}
```

---

### `POST /verify`
Verify image integrity without decryption. **No private key required.**

**Request** (`multipart/form-data`):

| Field | Type | Required | Description |
|---|---|---|---|
| `image` | file | ✅ | Stego image to verify |

**Response:**
```json
{
  "success": true,
  "authentic": true,
  "hash_match": true,
  "hmac_match": true,
  "reason": "Authentic",
  "sha256": "a3f2...",
  "timestamp": 1746000000.0,
  "verify_time": 0.456
}
```

---

### `GET /logs?n=30`
Retrieve the most recent operation logs.

**Response:**
```json
{
  "encode_logs": [ { "id": 1, "image_name": "photo.png", "psnr": 51.23, ... } ],
  "decode_logs": [ { "id": 1, "authentic": 1, "reason": "Authentic", ... } ]
}
```

---

## 🧪 Running Tests

```bash
# Run all 17 tests
python -m pytest tests/test_pipeline.py -v

# Run a specific test class
python -m pytest tests/test_pipeline.py::TestEndToEnd -v

# Run with coverage (requires pytest-cov)
pip install pytest-cov
python -m pytest tests/test_pipeline.py --cov=. --cov-report=term-missing
```

### Test Coverage

| Test Class | Tests | Description |
|---|---|---|
| `TestEncryption` | 4 | AES+RSA encrypt/decrypt roundtrip, serialisation |
| `TestErrorCorrection` | 3 | Hamming roundtrip, single-bit correction, RS roundtrip |
| `TestVerification` | 3 | SHA-256, HMAC, tamper detection |
| `TestSteganographyEngine` | 3 | Embed/extract roundtrip, shape check, overflow |
| `TestMetrics` | 3 | PSNR, SSIM correctness |
| `TestEndToEnd` | 1 | Full pipeline: encrypt→ECC→embed→extract→verify→decrypt |

---

## 🧠 AI Model Training

The TA-DPCNN works without training (random weights still produce a valid DESM). To train for optimal embedding quality:

### Step 1 — Prepare Dataset

Download any image dataset (e.g., [DIV2K](https://data-fiftyone.net/datasets/div2k), [COCO](https://cocodataset.org), or your own collection) into:
```
data/train/
```

### Step 2 — Generate Sample Images (optional)

```bash
python utils/generate_samples.py
```

### Step 3 — Train

```bash
python -m ai_model.train \
  --data_dir ./data/train \
  --save_dir ./ai_model/checkpoints \
  --epochs 50 \
  --batch_size 8 \
  --lr 1e-4
```

Checkpoints are saved to `ai_model/checkpoints/best_tadpcnn.pth`.

### Training Arguments

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | `./data/train` | Path to training images |
| `--save_dir` | `./ai_model/checkpoints` | Where to save checkpoints |
| `--epochs` | `50` | Number of training epochs |
| `--batch_size` | `8` | Images per batch |
| `--image_size` | `256` | Resize all images to this size |
| `--lr` | `1e-4` | Learning rate |
| `--alpha` | `1.0` | MSE loss weight |
| `--beta` | `0.5` | SSIM loss weight |
| `--gamma` | `0.1` | Entropy penalty weight |
| `--resume` | `` | Path to checkpoint to resume from |

---

## 🐛 Issues Fixed (Changelog)

### v1.1 — Bug Fixes & Improvements

| # | File | Issue | Fix |
|---|---|---|---|
| 1 | `ai_model/loader.py` | `TADPCNN \| None` union syntax (Python 3.10+ only) | Changed to `Optional[TADPCNN]` from `typing` |
| 2 | `ai_model/loader.py` | Checkpoint paths relative to CWD (broke when run from different dir) | Paths now resolved relative to `__file__` using `pathlib.Path` |
| 3 | `ai_model/losses.py` | `tuple[Tensor, dict]` return type (Python 3.10+ only) | Changed to `Tuple[Tensor, dict]` from `typing` |
| 4 | `ai_model/model.py` | Unused `h, w = image_np.shape[:2]` variables | Removed |
| 5 | `error_correction/ecc.py` | Unused `math`, `Tuple` imports; `dict[tuple, int]` (Python 3.9+ only) | Replaced with `Dict`, `Tuple` from `typing`; removed `math` |
| 6 | `steganography/engine.py` | `list[int]`, `list[tuple]` type hints (Python 3.9+ only) | Changed to `List`, `Tuple` from `typing` |
| 7 | `steganography/engine.py` | **Critical:** `extract()` called `_parse_payload()` speculatively on a growing buffer, causing silent `ValueError` suppression and potential data loss | Rewrote to a clean two-pass: collect header (64 bits), determine exact length, collect remaining bits, parse once |
| 8 | `steganography/engine.py` | `desm.flatten()[fi]` called inside loop (redundant re-allocation) | Pre-computed `flat_desm = desm.flatten()` once outside loop |
| 9 | `utils/metrics.py` | `naive_lsb_embed` called `stego.reshape(-1)` returning a **view**, then mutated `flat[i]` in a loop — corrupted the original array | Used `stego.ravel()` on a `.copy()` + vectorised `&= 0xFE` |
| 10 | `utils/metrics.py` | `PSNR = float('inf')` when images are identical → JSON serialisation error | Capped at `100.0` before returning |
| 11 | `backend/app.py` | `import struct` inside route handlers (repeated import overhead) | Moved to module top-level |
| 12 | `backend/app.py` | `import tempfile` — unused import | Removed |
| 13 | `backend/app.py` | `init_db()` and `get_recent_logs()` called without `db_path` → DB created relative to CWD | All DB calls now use `_DB_PATH = str(ROOT / "stego.db")` |
| 14 | `backend/app.py` | `/verify` route required `private_key` but verification is hash/HMAC-only (no decryption) | Removed private key requirement from `/verify` |
| 15 | `frontend/index.html` | Verify tab had private key textarea (inconsistent with backend) | Removed private key field; added explanatory hint text |
| 16 | `frontend/script.js` | Verify handler sent `private_key` in FormData | Removed; handler now sends only the image |
| 17 | `frontend/script.js` | Verify result panel did not show `hash_match` / `hmac_match` fields | Added both fields to the result display |
| 18 | `requirements.txt` | Missing packages: `cryptography`, `reedsolo`, `scikit-image`, `flask-cors` | Added all; `torch`/`torchvision` marked optional |

---

## 🔮 Future Improvements

- [ ] **Trained model weights** — Ship a pretrained `best_tadpcnn.pth` for optimal DESM quality
- [ ] **DCT-domain embedding** — Add DCT steganography mode alongside LSB for JPEG images
- [ ] **Multi-image batch processing** — Encode/decode multiple images in one API call
- [ ] **Key management UI** — Full keypair manager with import/export and named profiles
- [ ] **Capacity estimator** — Show maximum message size for a given image before encoding
- [ ] **JPEG support** — Handle lossy JPEG re-compression that currently destroys LSB data
- [ ] **Docker container** — `docker-compose up` one-liner deployment
- [ ] **REST API authentication** — API key middleware for production deployments
- [ ] **MongoDB option** — Pluggable database backend alongside SQLite

---

## 📸 Screenshots

| Encode Tab | Decode Tab |
|---|---|
| *(Upload image, type message, embed)* | *(Upload stego, paste private key, decode)* |

| Verify Tab | Logs Tab |
|---|---|
| *(Upload stego, verify integrity instantly)* | *(View encode/decode operation history)* |

> Run the app and navigate to `http://localhost:5000` to see the live UI.

---

## 🤝 Contributing

Open source contributors are welcome to contribute to this project! We appreciate your help in making this tool better. Please feel free to fork the repository, make your changes, and submit a pull request.

---

## 📄 License

```
MIT License

Copyright (c) 2025 StegoAI Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

<div align="center">

Built with ❤️ using Python · Flask · PyTorch · Pure Vanilla JS

</div>
