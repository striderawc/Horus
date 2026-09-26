#!/usr/bin/env python3
"""Train event-frame heatmap CNN (calls train_cnn.py --modality event_frame)."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent / "train_cnn.py"
sys.argv = [str(_SCRIPT), "--modality", "event_frame", *sys.argv[1:]]
runpy.run_path(str(_SCRIPT), run_name="__main__")
