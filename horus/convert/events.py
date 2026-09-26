"""Bin raw DVS events into event frames and voxel grids."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _as_events_arrays(h5: h5py.File) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract t, x, y, p from common v2e / AEDAT HDF5 layouts.

    SensorsINI/v2e writes dataset ``events`` as ``(N, 4)`` uint32 with columns
    ``[t_us, x, y, p]`` where polarity is ``0/1`` (OFF/ON) and ``t`` is microseconds.
    """
    if "events" in h5:
        ev = h5["events"]
        if isinstance(ev, h5py.Dataset):
            # structured or Nx4
            data = ev[()]
            if data.dtype.names:
                t = np.asarray(data["t"] if "t" in data.dtype.names else data["timestamp"], dtype=np.float64)
                x = np.asarray(data["x"], dtype=np.int32)
                y = np.asarray(data["y"], dtype=np.int32)
                p = np.asarray(data["p"] if "p" in data.dtype.names else data["polarity"], dtype=np.int8)
                return t, x, y, p
            arr = np.asarray(data)
            if arr.ndim == 2 and arr.shape[1] >= 4:
                return (
                    arr[:, 0].astype(np.float64),
                    arr[:, 1].astype(np.int32),
                    arr[:, 2].astype(np.int32),
                    arr[:, 3].astype(np.int8),
                )
        # group with datasets
        if isinstance(ev, h5py.Group):
            t = np.asarray(ev["t"] if "t" in ev else ev["timestamp"], dtype=np.float64)
            x = np.asarray(ev["x"], dtype=np.int32)
            y = np.asarray(ev["y"], dtype=np.int32)
            p = np.asarray(ev["p"] if "p" in ev else ev["polarity"], dtype=np.int8)
            return t, x, y, p

    # flat datasets at root (synthetic / alternate writers)
    for t_key in ("t", "timestamp", "ts"):
        if t_key in h5:
            t = np.asarray(h5[t_key], dtype=np.float64)
            x = np.asarray(h5["x"], dtype=np.int32)
            y = np.asarray(h5["y"], dtype=np.int32)
            p_key = "p" if "p" in h5 else ("polarity" if "polarity" in h5 else None)
            if p_key is None:
                raise KeyError(f"No polarity dataset in {list(h5.keys())}")
            p = np.asarray(h5[p_key], dtype=np.int8)
            return t, x, y, p

    raise KeyError(f"Unrecognized event HDF5 layout; keys={list(h5.keys())}")


def load_raw_events(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load raw events as ``t, x, y, p`` with ``t`` in seconds and ``p`` in ``{0,1}``."""
    path = Path(path)
    with h5py.File(path, "r") as h5:
        t, x, y, p = _as_events_arrays(h5)
    # v2e stores microseconds as uint32; also accept seconds floats
    if t.size and np.nanmax(t) > 1e3:
        t = t * 1e-6
    # Normalize polarity to {0, 1}
    if p.size and p.min() < 0:
        p = (p > 0).astype(np.int8)
    return t, x, y, p


def events_to_frame(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    t0: float,
    t1: float,
    height: int,
    width: int,
) -> np.ndarray:
    """Accumulate events in ``[t0, t1)`` into a ``(2, H, W)`` polarity frame."""
    frame = np.zeros((2, height, width), dtype=np.float32)
    if t.size == 0:
        return frame
    mask = (t >= t0) & (t < t1)
    if not np.any(mask):
        return frame
    xs = x[mask]
    ys = y[mask]
    ps = p[mask]
    in_bounds = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    xs, ys, ps = xs[in_bounds], ys[in_bounds], ps[in_bounds]
    pos = ps > 0
    neg = ~pos
    if np.any(pos):
        np.add.at(frame[0], (ys[pos], xs[pos]), 1.0)
    if np.any(neg):
        np.add.at(frame[1], (ys[neg], xs[neg]), 1.0)
    return frame


def events_to_voxel(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    t0: float,
    t1: float,
    height: int,
    width: int,
    num_bins: int,
) -> np.ndarray:
    """Bilinear-in-time voxel grid ``(num_bins, H, W)`` with polarity as sign."""
    voxel = np.zeros((num_bins, height, width), dtype=np.float32)
    if t.size == 0 or t1 <= t0:
        return voxel
    mask = (t >= t0) & (t < t1)
    if not np.any(mask):
        return voxel
    ts = t[mask]
    xs = x[mask]
    ys = y[mask]
    ps = p[mask]
    in_bounds = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    ts, xs, ys, ps = ts[in_bounds], xs[in_bounds], ys[in_bounds], ps[in_bounds]
    if ts.size == 0:
        return voxel

    duration = t1 - t0
    # Normalize time to [0, num_bins)
    tn = (ts - t0) / duration * num_bins
    pol = np.where(ps > 0, 1.0, -1.0).astype(np.float32)
    t_floor = np.floor(tn).astype(np.int32)
    t_ceil = t_floor + 1
    w_ceil = (tn - t_floor).astype(np.float32)
    w_floor = 1.0 - w_ceil

    valid_f = (t_floor >= 0) & (t_floor < num_bins)
    if np.any(valid_f):
        np.add.at(
            voxel,
            (t_floor[valid_f], ys[valid_f], xs[valid_f]),
            pol[valid_f] * w_floor[valid_f],
        )
    valid_c = (t_ceil >= 0) & (t_ceil < num_bins)
    if np.any(valid_c):
        np.add.at(
            voxel,
            (t_ceil[valid_c], ys[valid_c], xs[valid_c]),
            pol[valid_c] * w_ceil[valid_c],
        )
    return voxel


def precompute_clip_events(
    events_h5: str | Path,
    out_frames_h5: str | Path,
    out_voxels_h5: str | Path,
    num_frames: int,
    height: int,
    width: int,
    frame_dt: float,
    num_bins: int,
    t_start: float = 0.0,
) -> dict[str, Any]:
    """Write per-frame event frames and voxels for one clip."""
    t, x, y, p = load_raw_events(events_h5)
    # If event timestamps don't start near 0, align first event to t_start
    if t.size:
        t0_data = float(t.min())
        # Prefer aligning so frame 0 starts at t_start relative to stream
        offset = t0_data - t_start
        t = t - offset

    frames = np.zeros((num_frames, 2, height, width), dtype=np.float32)
    voxels = np.zeros((num_frames, num_bins, height, width), dtype=np.float32)

    for i in range(num_frames):
        a = t_start + i * frame_dt
        b = a + frame_dt
        frames[i] = events_to_frame(t, x, y, p, a, b, height, width)
        voxels[i] = events_to_voxel(t, x, y, p, a, b, height, width, num_bins)

    out_frames_h5 = Path(out_frames_h5)
    out_voxels_h5 = Path(out_voxels_h5)
    out_frames_h5.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_frames_h5, "w") as h5:
        h5.create_dataset("event_frames", data=frames, compression="gzip", compression_opts=4)
        h5.attrs["num_frames"] = num_frames
        h5.attrs["height"] = height
        h5.attrs["width"] = width
        h5.attrs["channels"] = 2

    with h5py.File(out_voxels_h5, "w") as h5:
        h5.create_dataset("voxels", data=voxels, compression="gzip", compression_opts=4)
        h5.attrs["num_frames"] = num_frames
        h5.attrs["height"] = height
        h5.attrs["width"] = width
        h5.attrs["num_bins"] = num_bins

    return {
        "num_events": int(t.size),
        "num_frames": num_frames,
        "frames_path": str(out_frames_h5),
        "voxels_path": str(out_voxels_h5),
    }


def load_event_frame_window(
    path: str | Path,
    start_idx: int,
    length: int,
) -> np.ndarray:
    """Load ``length`` event frames starting at ``start_idx`` → ``(2*L, H, W)``."""
    with h5py.File(path, "r") as h5:
        data = h5["event_frames"]
        if start_idx + length > data.shape[0]:
            raise IndexError(
                f"{path}: need frames [{start_idx}, {start_idx + length}) "
                f"but only {data.shape[0]} available (truncated conversion?)"
            )
        window = np.asarray(data[start_idx : start_idx + length], dtype=np.float32)
    # (L, 2, H, W) → (2L, H, W); copy so tensors are writable under DataLoader workers
    return window.reshape(-1, window.shape[-2], window.shape[-1]).copy()


def load_voxel_window(
    path: str | Path,
    start_idx: int,
    length: int,
) -> np.ndarray:
    """Load ``length`` voxels starting at ``start_idx`` → ``(bins*L, H, W)``."""
    with h5py.File(path, "r") as h5:
        data = h5["voxels"]
        if start_idx + length > data.shape[0]:
            raise IndexError(
                f"{path}: need voxels [{start_idx}, {start_idx + length}) "
                f"but only {data.shape[0]} available (truncated conversion?)"
            )
        window = np.asarray(data[start_idx : start_idx + length], dtype=np.float32)
    return window.reshape(-1, window.shape[-2], window.shape[-1]).copy()
