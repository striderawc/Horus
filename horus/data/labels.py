"""Parse TrackNet Label.csv files."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FrameLabel:
    frame: str
    visibility: int
    x: float
    y: float
    status: int


def _parse_int(value: str, default: int = 0) -> int:
    value = (value or "").strip()
    if value == "":
        return default
    return int(float(value))


def _parse_float(value: str, default: float = 0.0) -> float:
    value = (value or "").strip()
    if value == "":
        return default
    return float(value)


def load_labels(path: str | Path) -> list[FrameLabel]:
    """Load and normalize a clip ``Label.csv``."""
    path = Path(path)
    labels: list[FrameLabel] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return labels
        # Normalize header keys
        field_map = {name.strip().lower(): name for name in reader.fieldnames}

        def col(*candidates: str) -> str | None:
            for c in candidates:
                if c in field_map:
                    return field_map[c]
            return None

        frame_col = col("file name", "filename", "file")
        vis_col = col("visibility", "visibility class", "vc")
        x_col = col("x-coordinate", "x", "x_coordinate")
        y_col = col("y-coordinate", "y", "y_coordinate")
        status_col = col("status", "trajectory pattern", "trajectory")

        if not frame_col or not vis_col or not x_col or not y_col:
            raise ValueError(f"Unrecognized Label.csv columns in {path}: {reader.fieldnames}")

        for row in reader:
            frame = row[frame_col].strip()
            vis = _parse_int(row[vis_col], default=0)
            x = _parse_float(row[x_col], default=0.0)
            y = _parse_float(row[y_col], default=0.0)
            status = _parse_int(row[status_col], default=-1) if status_col else -1
            labels.append(
                FrameLabel(frame=frame, visibility=vis, x=x, y=y, status=status)
            )
    return labels
