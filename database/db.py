"""
SQLite Database Layer
Stores: encode/decode logs, metadata, verification results
"""

import sqlite3
import json
import time
import logging
import os
from contextlib import contextmanager
from typing import Optional

log = logging.getLogger(__name__)

DB_PATH = os.environ.get("STEGO_DB_PATH", "./stego.db")


# ──────────────────────────────────────────────
#  Context Manager
# ──────────────────────────────────────────────

@contextmanager
def get_connection(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ──────────────────────────────────────────────
#  Schema
# ──────────────────────────────────────────────

def init_db(db_path: str = DB_PATH) -> None:
    """Create tables if they don't exist."""
    with get_connection(db_path) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS encode_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   REAL    NOT NULL,
            image_name  TEXT,
            message_len INTEGER,
            psnr        REAL,
            ssim        REAL,
            mse         REAL,
            encode_time REAL,
            sha256      TEXT,
            metadata_json TEXT
        );

        CREATE TABLE IF NOT EXISTS decode_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   REAL    NOT NULL,
            image_name  TEXT,
            decode_time REAL,
            authentic   INTEGER,   -- 0 or 1
            reason      TEXT,
            metadata_json TEXT
        );

        CREATE TABLE IF NOT EXISTS keypairs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at  REAL    NOT NULL,
            label       TEXT UNIQUE,
            public_key  TEXT,       -- PEM
            private_key TEXT        -- PEM (store securely in production!)
        );
        """)
    log.info("Database initialised at %s", db_path)


# ──────────────────────────────────────────────
#  Encode Log
# ──────────────────────────────────────────────

def log_encode(image_name: str, message_len: int, psnr: float,
               ssim: float, mse: float, encode_time: float,
               sha256: str, metadata: dict, db_path: str = DB_PATH) -> int:
    with get_connection(db_path) as conn:
        cur = conn.execute("""
            INSERT INTO encode_log
              (timestamp, image_name, message_len, psnr, ssim, mse,
               encode_time, sha256, metadata_json)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (time.time(), image_name, message_len, psnr, ssim, mse,
              encode_time, sha256, json.dumps(metadata)))
        return cur.lastrowid


# ──────────────────────────────────────────────
#  Decode Log
# ──────────────────────────────────────────────

def log_decode(image_name: str, decode_time: float, authentic: bool,
               reason: str, metadata: dict, db_path: str = DB_PATH) -> int:
    with get_connection(db_path) as conn:
        cur = conn.execute("""
            INSERT INTO decode_log
              (timestamp, image_name, decode_time, authentic, reason, metadata_json)
            VALUES (?,?,?,?,?,?)
        """, (time.time(), image_name, decode_time, int(authentic), reason, json.dumps(metadata)))
        return cur.lastrowid


# ──────────────────────────────────────────────
#  Key Pair Storage
# ──────────────────────────────────────────────

def save_keypair_db(label: str, public_pem: bytes, private_pem: bytes,
                    db_path: str = DB_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO keypairs
              (created_at, label, public_key, private_key)
            VALUES (?,?,?,?)
        """, (time.time(), label, public_pem.decode(), private_pem.decode()))


def load_keypair_db(label: str, db_path: str = DB_PATH) -> Optional[tuple[bytes, bytes]]:
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT public_key, private_key FROM keypairs WHERE label=?", (label,)
        ).fetchone()
    if row:
        return row["private_key"].encode(), row["public_key"].encode()
    return None


# ──────────────────────────────────────────────
#  Recent logs
# ──────────────────────────────────────────────

def get_recent_logs(n: int = 20, db_path: str = DB_PATH) -> dict:
    with get_connection(db_path) as conn:
        encodes = [dict(r) for r in conn.execute(
            "SELECT * FROM encode_log ORDER BY timestamp DESC LIMIT ?", (n,)
        ).fetchall()]
        decodes = [dict(r) for r in conn.execute(
            "SELECT * FROM decode_log ORDER BY timestamp DESC LIMIT ?", (n,)
        ).fetchall()]
    return {"encode_logs": encodes, "decode_logs": decodes}
