#!/usr/bin/env python3
"""Train RGB heatmap CNN (calls train_cnn.py --modality rgb)."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent / "train_cnn.py"
sys.argv = [str(_SCRIPT), "--modality", "rgb", *sys.argv[1:]]
runpy.run_path(str(_SCRIPT), run_name="__main__")
