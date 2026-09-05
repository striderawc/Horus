#!/usr/bin/env python3
"""Print dataset index summary (clips, frames, split sizes)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from horus.config import load_config
from horus.data.index import build_sample_refs, index_clips, split_clips


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    clips = index_clips(cfg["paths"]["raw_dataset"])
    seq = int(cfg["sequence"]["length"])
    print(f"clips={len(clips)} total_frames={sum(c.num_frames for c in clips)}")
    split_map = split_clips(clips, cfg["splits"])
    for name, subset in split_map.items():
        n_frames = sum(c.num_frames for c in subset)
        n_samples = len(build_sample_refs(subset, seq))
        print(
            f"  {name}: clips={len(subset)} frames={n_frames} "
            f"seq_samples(len={seq})={n_samples}"
        )
        for c in subset[:3]:
            print(f"    - {c.key} frames={c.num_frames}")
        if len(subset) > 3:
            print(f"    ... ({len(subset) - 3} more)")


if __name__ == "__main__":
    main()
