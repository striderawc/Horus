#!/usr/bin/env python3
"""Smoke-test BallTrackDataset / DataLoader for a given modality."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without install when developing
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from horus.config import load_config
from horus.data.dataset import build_dataloader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to data.yaml (default: configs/data.yaml)",
    )
    parser.add_argument(
        "--modality",
        choices=["rgb", "diff", "event_frame", "voxel"],
        default="rgb",
    )
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    loader = build_dataloader(
        cfg,
        split=args.split,
        modality=args.modality,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    print(
        f"split={args.split} modality={args.modality} "
        f"batches≈{len(loader)} dataset_len={len(loader.dataset)}"
    )
    batch = next(iter(loader))
    for key in ("input", "heatmap", "coords", "visibility", "status", "visible"):
        val = batch[key]
        print(f"  {key}: shape={tuple(val.shape)} dtype={val.dtype}")
    print(f"  games: {batch['game']}")
    print(f"  frames: {batch['frame']}")
    print("OK")


if __name__ == "__main__":
    main()
