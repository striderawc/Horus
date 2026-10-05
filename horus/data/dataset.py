"""PyTorch Dataset for RGB / frame-diff / event modalities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Literal

import numpy as np
import h5py
import torch
from torch.utils.data import BatchSampler, Dataset

from horus.config import events_clip_dir
from horus.convert.events import load_event_frame_window, load_voxel_window
from horus.convert.frame_diff import frame_differences, load_gray, load_rgb
from horus.data.crop import (
    crop_chw_stack,
    fullframe_to_crop,
    pick_bucket,
    sample_negative_crop,
    sample_positive_crop,
)
from horus.data.index import (
    ClipInfo,
    SampleRef,
    build_sample_refs,
    clips_by_key,
    index_clips,
    split_clips,
)
from horus.data.targets import build_targets, make_gaussian_heatmap, scale_coords

Modality = Literal["rgb", "diff", "event_frame", "voxel"]


class BallTrackDataset(Dataset):
    """Sequence dataset with heatmap + coordinate targets.

    The label is taken from the **last** frame of each window of length ``N``.

    With ``repeat_frame=True`` (event_frame only), each sample loads a single
    event frame and tiles it ``input_steps`` times → ``(e, e, …, e)`` instead of
    a temporal window. The label is that same frame.

    When ``cfg['crop']['enabled']`` is true, each sample is a jittered square
    crop (bucket sizes) around the ball, or a negative crop without the ball.
    """

    def __init__(
        self,
        clips: list[ClipInfo],
        cfg: dict[str, Any],
        modality: Modality = "rgb",
        *,
        repeat_frame: bool = False,
        input_steps: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.modality: Modality = modality
        self.clips = clips
        self.clip_map = clips_by_key(clips)
        cfg_len = int(cfg["sequence"]["length"])
        self.input_steps = int(input_steps) if input_steps is not None else cfg_len
        if self.input_steps < 1:
            raise ValueError("input_steps must be >= 1")
        self.repeat_frame = bool(repeat_frame)
        if self.repeat_frame and modality != "event_frame":
            raise ValueError("repeat_frame is only supported for modality=event_frame")
        # Window used for indexing / labels. Repeat mode: one frame, then tile.
        self.seq_len = 1 if self.repeat_frame else self.input_steps
        self.refs = build_sample_refs(clips, self.seq_len)

        img = cfg["image"]
        self.out_w = int(img["width"])
        self.out_h = int(img["height"])
        src = cfg["source"]
        self.src_w = int(src["width"])
        self.src_h = int(src["height"])

        crop_cfg = cfg.get("crop") or {}
        self.crop_enabled = bool(crop_cfg.get("enabled", False))
        self.crop_buckets = [int(b) for b in crop_cfg.get("buckets", [64, 128, 256])]
        self.crop_jitter_std_frac = float(crop_cfg.get("jitter_std_frac", 0.15))
        self.crop_negative_prob = float(crop_cfg.get("negative_prob", 0.2))
        self.crop_margin = float(crop_cfg.get("min_ball_margin", 4))
        self._rng = np.random.default_rng(seed)

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
        if self.repeat_frame:
            # Single polarity frame (2, H, W), tiled to (2 * input_steps, H, W).
            frame = load_event_frame_window(path, start, 1)
            arr = np.concatenate([frame] * self.input_steps, axis=0)
            return torch.from_numpy(arr)
        arr = load_event_frame_window(path, start, self.seq_len)
        return torch.from_numpy(arr)

    def _load_voxels(self, clip: ClipInfo, indices: range) -> torch.Tensor:
        path = events_clip_dir(self.cfg, clip.game, clip.clip) / "voxels.h5"
        start = indices.start
        arr = load_voxel_window(path, start, self.seq_len)
        return torch.from_numpy(arr)

    def _apply_crop(
        self,
        inp: torch.Tensor,
        *,
        sx: float,
        sy: float,
        visible: bool,
        heat_cfg: dict[str, Any],
        visibility: int,
        status: int,
    ) -> dict[str, Any]:
        """Jittered / negative crop → crop-space targets."""
        forced = getattr(self, "_force_crop_size", None)
        if forced is not None:
            size = int(forced)
        elif getattr(self, "_getitem_index", None) is not None:
            size = self.crop_buckets[
                int(self._getitem_index) % len(self.crop_buckets)
            ]
        else:
            size = pick_bucket(self.crop_buckets, self._rng)
        want_negative = (not visible) or (
            float(self._rng.random()) < self.crop_negative_prob
        )

        if want_negative:
            x0, y0, sz = sample_negative_crop(
                sx if visible else None,
                sy if visible else None,
                size,
                self.out_w,
                self.out_h,
                self._rng,
                margin=self.crop_margin,
            )
            in_crop = False
        else:
            x0, y0, sz, in_crop = sample_positive_crop(
                sx,
                sy,
                size,
                self.out_w,
                self.out_h,
                self._rng,
                jitter_std_frac=self.crop_jitter_std_frac,
                margin=self.crop_margin,
            )

        cropped = crop_chw_stack(inp, x0, y0, sz)
        sigma_full = float(heat_cfg.get("sigma", 2.5))
        # Scale Gaussian width with crop size relative to full training width.
        sigma_crop = max(sigma_full * (sz / float(self.out_w)), 0.5)
        amplitude = float(heat_cfg.get("amplitude", 1.0))

        if in_crop:
            cx, cy = fullframe_to_crop(
                np.array([sx, sy], dtype=np.float32), x0, y0
            )
            heat = make_gaussian_heatmap(
                float(cx),
                float(cy),
                height=sz,
                width=sz,
                sigma=sigma_crop,
                amplitude=amplitude,
                visible=True,
            )
            coords = torch.tensor(
                [float(cx), float(cy)], dtype=torch.float32
            )
            vis_flag = 1.0
        else:
            heat = np.zeros((sz, sz), dtype=np.float32)
            coords = torch.tensor([-1.0, -1.0], dtype=torch.float32)
            vis_flag = 0.0

        return {
            "input": cropped,
            "heatmap": torch.from_numpy(heat).unsqueeze(0),
            "coords": coords,
            "visibility": torch.tensor(visibility, dtype=torch.int64),
            "status": torch.tensor(status, dtype=torch.int64),
            "visible": torch.tensor(vis_flag, dtype=torch.float32),
            "in_crop": torch.tensor(1.0 if in_crop else 0.0, dtype=torch.float32),
            "crop_origin": torch.tensor([x0, y0], dtype=torch.int64),
            "crop_size": torch.tensor(sz, dtype=torch.int64),
        }

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
        dst_w, dst_h = self.out_w, self.out_h
        sx, sy = scale_coords(
            label.x, label.y, self.src_w, self.src_h, dst_w, dst_h
        )
        visible = label.visibility != 0

        pending = getattr(self, "_pending_crop_size", None)
        if isinstance(pending, dict) and index in pending:
            self._force_crop_size = int(pending.pop(index))
        self._getitem_index = index
        try:
            if self.crop_enabled:
                out = self._apply_crop(
                    inp,
                    sx=sx,
                    sy=sy,
                    visible=visible,
                    heat_cfg=heat_cfg,
                    visibility=label.visibility,
                    status=label.status,
                )
            else:
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
                out = {
                    "input": inp,
                    "heatmap": targets["heatmap"],
                    "coords": targets["coords"],
                    "visibility": targets["visibility"],
                    "status": targets["status"],
                    "visible": targets["visible"],
                    "in_crop": targets["visible"].clone(),
                    "crop_origin": torch.tensor([0, 0], dtype=torch.int64),
                    "crop_size": torch.tensor(dst_w, dtype=torch.int64),
                }
        finally:
            if hasattr(self, "_force_crop_size"):
                delattr(self, "_force_crop_size")
            if hasattr(self, "_getitem_index"):
                delattr(self, "_getitem_index")

        out.update(
            {
                "game": clip.game,
                "clip": clip.clip,
                "end_idx": ref.end_idx,
                "frame": label.frame,
                "modality": self.modality,
            }
        )
        return out


class BucketBatchSampler(BatchSampler):
    """Yield batches where every sample shares the same crop bucket size.

    Bucket size is ``crop_buckets[index % len(buckets)]`` so assignment is
    deterministic and safe under DataLoader workers.
    """

    def __init__(
        self,
        dataset: BallTrackDataset,
        batch_size: int,
        *,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self._rng = np.random.default_rng(seed)

    def _size_for(self, index: int) -> int:
        buckets = self.dataset.crop_buckets
        return buckets[int(index) % len(buckets)]

    def __iter__(self) -> Iterator[list[int]]:
        n = len(self.dataset)
        indices = list(range(n))
        if self.shuffle:
            self._rng.shuffle(indices)

        assigned: dict[int, list[int]] = {b: [] for b in self.dataset.crop_buckets}
        for idx in indices:
            assigned.setdefault(self._size_for(idx), []).append(idx)

        batches: list[list[int]] = []
        for idxs in assigned.values():
            for start in range(0, len(idxs), self.batch_size):
                batch = idxs[start : start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                if batch:
                    batches.append(batch)
        if self.shuffle:
            self._rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size


def build_dataloader(
    cfg: dict[str, Any],
    split: str = "train",
    modality: Modality = "rgb",
    *,
    batch_size: int | None = None,
    num_workers: int | None = None,
    shuffle: bool | None = None,
    repeat_frame: bool = False,
    input_steps: int | None = None,
    seed: int | None = None,
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

    ds = BallTrackDataset(
        clips,
        cfg,
        modality=modality,
        repeat_frame=repeat_frame,
        input_steps=input_steps,
        seed=seed,
    )
    dl_cfg = cfg.get("dataloader", {})
    bs = batch_size if batch_size is not None else int(dl_cfg.get("batch_size", 4))
    nw = num_workers if num_workers is not None else int(dl_cfg.get("num_workers", 0))
    if shuffle is None:
        shuffle = bool(dl_cfg.get("shuffle_train", True)) if split == "train" else False

    crop_enabled = bool((cfg.get("crop") or {}).get("enabled", False))
    if crop_enabled:
        batch_sampler = BucketBatchSampler(
            ds, bs, shuffle=shuffle, drop_last=False, seed=seed
        )
        return DataLoader(
            ds,
            batch_sampler=batch_sampler,
            num_workers=nw,
            pin_memory=False,
        )

    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=nw,
        pin_memory=False,
    )
