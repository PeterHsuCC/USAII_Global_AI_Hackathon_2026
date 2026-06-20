"""Bootstraps `sys.path` so `risk_detection` is importable.

Mirrors the manual sys.path insertion already used by frontend/app.py and
scripts/*.py in this repo, rather than relying on an editable install.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))