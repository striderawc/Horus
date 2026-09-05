"""On-the-fly frame differencing helpers."""

from __future__ import annotations

import cv2
import numpy as np


def load_gray(path: str, width: int, height: int) -> np.ndarray:
    """Load image as float32 grayscale in ``[0, 1]``, resized to ``(H, W)``."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    if bgr.shape[1] != width or bgr.shape[0] != height:
        bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return gray


def load_rgb(path: str, width: int, height: int) -> np.ndarray:
    """Load image as float32 RGB in ``[0, 1]``, CHW layout after caller stacks."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    if bgr.shape[1] != width or bgr.shape[0] != height:
        bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb


def frame_differences(
    frames_gray: list[np.ndarray],
    mode: str = "absolute",
) -> np.ndarray:
    """Compute consecutive diffs for a list of grayscale frames.

    Returns ``(T-1, H, W)`` float32 array.
    """
    if len(frames_gray) < 2:
        raise ValueError("Need at least 2 frames for differencing")
    diffs = []
    for i in range(1, len(frames_gray)):
        d = frames_gray[i] - frames_gray[i - 1]
        if mode == "absolute":
            d = np.abs(d)
        elif mode == "signed":
            pass
        else:
            raise ValueError(f"Unknown frame_diff mode: {mode}")
        diffs.append(d.astype(np.float32))
    return np.stack(diffs, axis=0)
