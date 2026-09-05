#!/usr/bin/env python3
"""Write a synthetic events.h5 for pipeline smoke tests (not a v2e substitute).

Generates DVS-like events from grayscale frame differences so
``precompute_events.py`` and event dataloaders can be exercised without
installing SensorsINI/v2e. Prefer ``convert_v2e.py`` for real experiments.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from horus.config import events_clip_dir, load_config
from horus.data.index import index_clips


def parse_clip_arg(value: str) -> tuple[str, str]:
    parts = value.strip().replace("\\", "/").split("/")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected form gameN/ClipM")
    return parts[0], parts[1]


def _resize_gray(path: Path, width: int, height: int) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    if bgr.shape[1] != width or bgr.shape[0] != height:
        bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    # log-ish intensity for thresholding
    return np.log1p(gray * 10.0)


def synthesize_clip(
    frame_paths: list[Path],
    width: int,
    height: int,
    frame_dt: float,
    threshold: float = 0.08,
    max_events_per_frame: int = 5000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ts: list[float] = []
    xs: list[int] = []
    ys: list[int] = []
    ps: list[int] = []

    prev = _resize_gray(frame_paths[0], width, height)
    rng = np.random.default_rng(0)
    for i in range(1, len(frame_paths)):
        cur = _resize_gray(frame_paths[i], width, height)
        delta = cur - prev
        pos_yx = np.argwhere(delta > threshold)
        neg_yx = np.argwhere(delta < -threshold)
        t0 = i * frame_dt
        for yx, pol in ((pos_yx, 1), (neg_yx, 0)):
            if yx.shape[0] == 0:
                continue
            if yx.shape[0] > max_events_per_frame:
                choice = rng.choice(yx.shape[0], size=max_events_per_frame, replace=False)
                yx = yx[choice]
            # jitter timestamps within the frame interval
            jit = rng.uniform(0.0, frame_dt, size=yx.shape[0])
            for (y, x), jt in zip(yx, jit, strict=False):
                ts.append(t0 + float(jt))
                xs.append(int(x))
                ys.append(int(y))
                ps.append(pol)
        prev = cur

    return (
        np.asarray(ts, dtype=np.float64),
        np.asarray(xs, dtype=np.int16),
        np.asarray(ys, dtype=np.int16),
        np.asarray(ps, dtype=np.int8),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--clip", type=parse_clip_arg, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.08)
    args = parser.parse_args()

    cfg = load_config(args.config)
    game, clip_name = args.clip
    clips = [
        c
        for c in index_clips(cfg["paths"]["raw_dataset"])
        if c.game == game and c.clip == clip_name
    ]
    if not clips:
        raise SystemExit(f"Clip not found: {game}/{clip_name}")
    clip = clips[0]

    out_dir = events_clip_dir(cfg, clip.game, clip.clip)
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "events.h5"
    if events_path.is_file() and not args.force:
        print(f"exists: {events_path} (use --force to overwrite)")
        return

    v2e = cfg.get("v2e", {})
    width = int(v2e.get("output_width") or cfg["image"]["width"])
    height = int(v2e.get("output_height") or cfg["image"]["height"])
    frame_dt = float(cfg.get("events", {}).get("frame_dt", 1.0 / cfg["source"]["fps"]))

    print(f"Synthesizing events for {clip.key} ({clip.num_frames} frames) @ {width}x{height}…")
    t, x, y, p = synthesize_clip(
        list(clip.frames),
        width=width,
        height=height,
        frame_dt=frame_dt,
        threshold=args.threshold,
    )
    with h5py.File(events_path, "w") as h5:
        h5.create_dataset("t", data=t, compression="gzip")
        h5.create_dataset("x", data=x, compression="gzip")
        h5.create_dataset("y", data=y, compression="gzip")
        h5.create_dataset("p", data=p, compression="gzip")
        h5.attrs["source"] = "synthesize_events"
        h5.attrs["width"] = width
        h5.attrs["height"] = height

    sidecar = out_dir / "v2e_run.yaml"
    sidecar.write_text(
        yaml.safe_dump(
            {
                "note": "Synthetic events from frame differencing (smoke test only)",
                "script": "scripts/synthesize_events.py",
                "threshold": args.threshold,
                "width": width,
                "height": height,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    print(f"wrote {events_path} ({len(t)} events)")


if __name__ == "__main__":
    main()
