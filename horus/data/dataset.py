"""PyTorch Dataset for RGB / frame-diff / event modalities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset

from horus.config import events_clip_dir
from horus.convert.events import load_event_frame_window, load_voxel_window
from horus.convert.frame_diff import frame_differences, load_gray, load_rgb
from horus.data.index import (
    ClipInfo,
    SampleRef,
    build_sample_refs,
    clips_by_key,
    index_clips,
    split_clips,
)
from horus.data.targets import build_targets

Modality = Literal["rgb", "diff", "event_frame", "voxel"]


class BallTrackDataset(Dataset):
    """Sequence dataset with heatmap + coordinate targets.

    The label is taken from the **last** frame of each window of length ``N``.
    """

    def __init__(
        self,
        clips: list[ClipInfo],
        cfg: dict[str, Any],
        modality: Modality = "rgb",
    ) -> None:
        self.cfg = cfg
        self.modality: Modality = modality
        self.clips = clips
        self.clip_map = clips_by_key(clips)
        self.seq_len = int(cfg["sequence"]["length"])
        self.refs = build_sample_refs(clips, self.seq_len)

        img = cfg["image"]
        self.out_w = int(img["width"])
        self.out_h = int(img["height"])
        src = cfg["source"]
        self.src_w = int(src["width"])
        self.src_h = int(src["height"])

        # Event tensors are stored at v2e output resolution (often == training res)
        v2e = cfg.get("v2e", {})
        self.event_w = int(v2e.get("output_width") or self.out_w)
        self.event_h = int(v2e.get("output_height") or self.out_h)

        if modality in ("event_frame", "voxel"):
            ready: list[ClipInfo] = []
            missing: list[str] = []
            short: list[str] = []
            for clip in clips:
                d = events_clip_dir(cfg, clip.game, clip.clip)
                needed = (
                    d / "event_frames.h5"
                    if modality == "event_frame"
                    else d / "voxels.h5"
                )
                if not needed.is_file():
                    missing.append(str(needed))
                    continue
                # Reject truncated conversions (e.g. early --stop_time smoke runs)
                with h5py.File(needed, "r") as h5:
                    key = "event_frames" if modality == "event_frame" else "voxels"
                    n_h5 = int(h5[key].shape[0])
                if n_h5 < clip.num_frames:
                    short.append(f"{clip.key} (h5={n_h5}, frames={clip.num_frames})")
                    continue
                ready.append(clip)
            if not ready:
                preview = "\n  ".join((missing + short)[:5])
                more = (
                    f"\n  ... and {len(missing) + len(short) - 5} more"
                    if len(missing) + len(short) > 5
                    else ""
                )
                raise FileNotFoundError(
                    f"No usable precomputed event files for modality={modality}. "
                    f"Run scripts/convert_v2e.py then scripts/precompute_events.py.\n"
                    f"  {preview}{more}"
                )
            if missing:
                print(
                    f"[BallTrackDataset] skipping {len(missing)} clip(s) "
                    f"without {modality} files; using {len(ready)}"
                )
            if short:
                print(
                    f"[BallTrackDataset] skipping {len(short)} truncated clip(s): "
                    + ", ".join(short[:3])
                    + ("…" if len(short) > 3 else "")
                )
            clips = ready
            self.clips = clips
            self.clip_map = clips_by_key(clips)
            self.refs = build_sample_refs(clips, self.seq_len)

    def __len__(self) -> int:
        return len(self.refs)

    def _window_indices(self, ref: SampleRef) -> range:
        start = ref.end_idx - self.seq_len + 1
        return range(start, ref.end_idx + 1)

    def _load_rgb(self, clip: ClipInfo, indices: range) -> torch.Tensor:
        frames = [
            load_rgb(str(clip.frames[i]), self.out_w, self.out_h) for i in indices
        ]
        # list of (H, W, 3) → (3N, H, W)
        stacked = np.concatenate([f.transpose(2, 0, 1) for f in frames], axis=0)
        return torch.from_numpy(stacked)

    def _load_diff(self, clip: ClipInfo, indices: range) -> torch.Tensor:
        # Need one extra prior frame when possible for N diffs; plan: stack N-1
        # diffs inside the window (between the N frames).
        grays = [
            load_gray(str(clip.frames[i]), self.out_w, self.out_h) for i in indices
        ]
        mode = self.cfg.get("frame_diff", {}).get("mode", "absolute")
        diffs = frame_differences(grays, mode=mode)  # (N-1, H, W)
        return torch.from_numpy(diffs)

    def _load_event_frames(self, clip: ClipInfo, indices: range) -> torch.Tensor:
        path = events_clip_dir(self.cfg, clip.game, clip.clip) / "event_frames.h5"
        start = indices.start
        arr = load_event_frame_window(path, start, self.seq_len)
        return torch.from_numpy(arr)

    def _load_voxels(self, clip: ClipInfo, indices: range) -> torch.Tensor:
        path = events_clip_dir(self.cfg, clip.game, clip.clip) / "voxels.h5"
        start = indices.start
        arr = load_voxel_window(path, start, self.seq_len)
        return torch.from_numpy(arr)

    def __getitem__(self, index: int) -> dict[str, Any]:
        ref = self.refs[index]
        clip = self.clip_map[ref.clip_key]
        indices = self._window_indices(ref)

        if self.modality == "rgb":
            inp = self._load_rgb(clip, indices)
        elif self.modality == "diff":
            inp = self._load_diff(clip, indices)
        elif self.modality == "event_frame":
            inp = self._load_event_frames(clip, indices)
        elif self.modality == "voxel":
            inp = self._load_voxels(clip, indices)
        else:
            raise ValueError(f"Unknown modality: {self.modality}")

        label = clip.labels[ref.end_idx]
        heat_cfg = self.cfg.get("heatmap", {})
        # For event modalities, heatmaps match event/training resolution
        dst_w, dst_h = self.out_w, self.out_h
        targets = build_targets(
            x=label.x,
            y=label.y,
            visibility=label.visibility,
            status=label.status,
            src_w=self.src_w,
            src_h=self.src_h,
            dst_w=dst_w,
            dst_h=dst_h,
            sigma=float(heat_cfg.get("sigma", 2.5)),
            amplitude=float(heat_cfg.get("amplitude", 1.0)),
        )

        return {
            "input": inp,
            "heatmap": targets["heatmap"],
            "coords": targets["coords"],
            "visibility": targets["visibility"],
            "status": targets["status"],
            "visible": targets["visible"],
            "game": clip.game,
            "clip": clip.clip,
            "end_idx": ref.end_idx,
            "frame": label.frame,
            "modality": self.modality,
        }


def build_dataloader(
    cfg: dict[str, Any],
    split: str = "train",
    modality: Modality = "rgb",
    *,
    batch_size: int | None = None,
    num_workers: int | None = None,
    shuffle: bool | None = None,
) -> torch.utils.data.DataLoader:
    """Build a DataLoader for a named split and modality."""
    from torch.utils.data import DataLoader

    raw_root = Path(cfg["paths"]["raw_dataset"])
    all_clips = index_clips(raw_root)
    split_map = split_clips(all_clips, cfg["splits"])
    if split not in split_map:
        raise KeyError(f"Unknown split '{split}'. Options: {list(split_map)}")
    clips = split_map[split]
    if not clips:
        raise RuntimeError(f"No clips found for split={split}")

    ds = BallTrackDataset(clips, cfg, modality=modality)
    dl_cfg = cfg.get("dataloader", {})
    bs = batch_size if batch_size is not None else int(dl_cfg.get("batch_size", 4))
    nw = num_workers if num_workers is not None else int(dl_cfg.get("num_workers", 0))
    if shuffle is None:
        shuffle = bool(dl_cfg.get("shuffle_train", True)) if split == "train" else False

    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=nw,
        pin_memory=False,
    )
