#!/usr/bin/env python3
"""Precompute event frames and voxel grids from raw v2e HDF5 files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqdm import tqdm

from horus.config import events_clip_dir, load_config
from horus.convert.events import precompute_clip_events
from horus.data.index import index_clips


def parse_clip_arg(value: str) -> tuple[str, str]:
    parts = value.strip().replace("\\", "/").split("/")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected form gameN/ClipM, e.g. game1/Clip1")
    return parts[0], parts[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--clip", type=parse_clip_arg, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing event_frames.h5 / voxels.h5",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    clips = index_clips(cfg["paths"]["raw_dataset"])
    if args.clip:
        game, clip_name = args.clip
        clips = [c for c in clips if c.game == game and c.clip == clip_name]
        if not clips:
            raise SystemExit(f"Clip not found: {game}/{clip_name}")
    if args.limit is not None:
        clips = clips[: args.limit]

    v2e = cfg.get("v2e", {})
    height = int(v2e.get("output_height") or cfg["image"]["height"])
    width = int(v2e.get("output_width") or cfg["image"]["width"])
    frame_dt = float(cfg.get("events", {}).get("frame_dt", 1.0 / cfg["source"]["fps"]))
    num_bins = int(cfg.get("events", {}).get("voxel", {}).get("num_bins", 5))

    ok, skipped, failed = 0, 0, 0
    for clip in tqdm(clips, desc="precompute"):
        out_dir = events_clip_dir(cfg, clip.game, clip.clip)
        events_h5 = out_dir / "events.h5"
        frames_h5 = out_dir / "event_frames.h5"
        voxels_h5 = out_dir / "voxels.h5"

        if not events_h5.is_file():
            failed += 1
            tqdm.write(f"MISSING raw events: {events_h5} (run convert_v2e.py first)")
            continue

        if frames_h5.is_file() and voxels_h5.is_file() and not args.force:
            skipped += 1
            continue

        try:
            info = precompute_clip_events(
                events_h5=events_h5,
                out_frames_h5=frames_h5,
                out_voxels_h5=voxels_h5,
                num_frames=clip.num_frames,
                height=height,
                width=width,
                frame_dt=frame_dt,
                num_bins=num_bins,
            )
            ok += 1
            tqdm.write(
                f"{clip.key}: events={info['num_events']} frames={info['num_frames']}"
            )
        except Exception as exc:  # noqa: BLE001
            failed += 1
            tqdm.write(f"FAIL {clip.key}: {exc}")

    print(f"done: precomputed={ok} skipped_existing={skipped} failed={failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
