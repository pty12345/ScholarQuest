"""Prepend repo src/ to sys.path so `python scripts/*.py` works without `pip install -e .`."""
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
_src = _root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
