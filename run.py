"""
Entry point - run the Flask server.
Usage: python run.py
"""
# -*- coding: utf-8 -*-
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backend.app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Avoid emoji in print to prevent cp1252 encoding errors on Windows
    print(f"\nStegoAI Server running --> http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
