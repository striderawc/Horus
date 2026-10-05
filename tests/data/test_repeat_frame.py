"""Tests for event-frame repeat / tile sequence ablation."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

from horus.data.dataset import BallTrackDataset
from horus.data.index import ClipInfo
from horus.data.labels import FrameLabel


def _fake_clip(tmp: Path, n: int = 5) -> ClipInfo:
    frames = tuple(tmp / f"{i:04d}.jpg" for i in range(n))
    for fp in frames:
        fp.write_bytes(b"")
    labels = tuple(
        FrameLabel(frame=fp.name, visibility=1, x=100.0 + i, y=50.0, status=0)
        for i, fp in enumerate(frames)
    )
    return ClipInfo(game="game1", clip="Clip1", path=tmp, frames=frames, labels=labels)


def _write_event_frames(path: Path, n: int, h: int = 8, w: int = 8) -> np.ndarray:
    """Write distinct polarity frames so tiling vs window is detectable."""
    data = np.zeros((n, 2, h, w), dtype=np.float32)
    for i in range(n):
        data[i, 0].fill(float(i + 1))
        data[i, 1].fill(float(10 * (i + 1)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("event_frames", data=data)
    return data


def _cfg(tmp: Path, *, seq_len: int = 3) -> dict:
    return {
        "sequence": {"length": seq_len},
        "image": {"width": 8, "height": 8},
        "source": {"width": 8, "height": 8},
        "paths": {
            "raw_dataset": str(tmp),
            "processed_root": str(tmp / "processed"),
            "events_subdir": "events",
        },
        "v2e": {"output_width": 8, "output_height": 8},
        "heatmap": {"sigma": 1.0, "amplitude": 1.0},
    }


def test_repeat_frame_tiles_same_event(tmp_path: Path):
    clip_dir = tmp_path / "game1" / "Clip1"
    clip_dir.mkdir(parents=True)
    clip = _fake_clip(clip_dir, n=5)
    events = tmp_path / "processed" / "events" / "game1" / "Clip1" / "event_frames.h5"
    raw = _write_event_frames(events, n=5)

    ds = BallTrackDataset(
        [clip],
        _cfg(tmp_path),
        modality="event_frame",
        repeat_frame=True,
        input_steps=4,
    )
    assert ds.seq_len == 1
    assert ds.input_steps == 4
    assert len(ds) == 5  # one sample per frame

    sample = ds[2]  # end_idx == 2 → frame with value 3
    inp = sample["input"]
    assert inp.shape == (8, 8, 8) or inp.shape == (2 * 4, 8, 8)
    assert inp.shape[0] == 8  # 2 * 4
    # All four tiles match frame index 2
    expected = torch.from_numpy(raw[2].reshape(2, 8, 8))
    for t in range(4):
        chunk = inp[t * 2 : (t + 1) * 2]
        assert torch.allclose(chunk, expected)
    assert float(sample["coords"][0]) == 102.0  # label for frame 2


def test_temporal_window_uses_distinct_frames(tmp_path: Path):
    clip_dir = tmp_path / "game1" / "Clip1"
    clip_dir.mkdir(parents=True)
    clip = _fake_clip(clip_dir, n=5)
    events = tmp_path / "processed" / "events" / "game1" / "Clip1" / "event_frames.h5"
    raw = _write_event_frames(events, n=5)

    ds = BallTrackDataset(
        [clip],
        _cfg(tmp_path, seq_len=3),
        modality="event_frame",
        repeat_frame=False,
        input_steps=3,
    )
    # end_idx=2 → frames 0,1,2
    sample = ds[0]
    inp = sample["input"]
    assert inp.shape[0] == 6
    assert not torch.allclose(inp[0:2], inp[2:4])
    assert torch.allclose(inp[0:2], torch.from_numpy(raw[0].reshape(2, 8, 8)))
    assert torch.allclose(inp[4:6], torch.from_numpy(raw[2].reshape(2, 8, 8)))


def test_custom_steps_without_repeat(tmp_path: Path):
    clip_dir = tmp_path / "game1" / "Clip1"
    clip_dir.mkdir(parents=True)
    clip = _fake_clip(clip_dir, n=6)
    events = tmp_path / "processed" / "events" / "game1" / "Clip1" / "event_frames.h5"
    _write_event_frames(events, n=6)

    ds = BallTrackDataset(
        [clip],
        _cfg(tmp_path, seq_len=3),
        modality="event_frame",
        input_steps=5,
    )
    assert ds.seq_len == 5
    assert len(ds) == 2  # end_idx in {4, 5}
    assert ds[0]["input"].shape[0] == 10
