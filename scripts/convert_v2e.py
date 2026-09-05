#!/usr/bin/env python3
"""Convert TrackNet clips to raw DVS events via SensorsINI/v2e."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqdm import tqdm

from horus.config import events_clip_dir, load_config
from horus.convert.v2e_wrapper import run_v2e_on_clip
from horus.data.index import index_clips


def parse_clip_arg(value: str) -> tuple[str, str]:
    parts = value.strip().replace("\\", "/").split("/")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected form gameN/ClipM, e.g. game1/Clip1")
    return parts[0], parts[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--clip",
        type=parse_clip_arg,
        default=None,
        help="Convert a single clip: game1/Clip1",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run v2e even if events.h5 already exists",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N clips (after filtering)",
    )
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

    print(f"Converting {len(clips)} clip(s) with v2e…")
    ok, skipped, failed = 0, 0, 0
    for clip in tqdm(clips, desc="v2e"):
        out_dir = events_clip_dir(cfg, clip.game, clip.clip)
        events_path = out_dir / "events.h5"
        if events_path.is_file() and not args.force:
            skipped += 1
            continue
        try:
            run_v2e_on_clip(clip.path, out_dir, cfg, force=args.force)
            ok += 1
        except Exception as exc:  # noqa: BLE001 — surface per-clip failures in batch
            failed += 1
            tqdm.write(f"FAIL {clip.key}: {exc}")

    print(f"done: converted={ok} skipped_existing={skipped} failed={failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
