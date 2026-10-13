#!/usr/bin/env python3
"""Smoke the closed-loop SSM ↔ detector crop pipeline (stub detector)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from horus.config import load_config
from horus.track import (
    build_akf,
    build_ca_akf,
    build_ca_ekf,
    build_ekf,
    build_kf,
    build_pf,
    run_track_detect_loop,
    synthetic_cv_trajectory,
)
from horus.track.loop import stub_detect_fn


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument(
        "--tracker",
        choices=["cv_kf", "cv_akf", "ca_akf", "cv_ekf", "ca_ekf", "cv_pf"],
        default="cv_kf",
    )
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = load_config(args.config)
    img_w = int(cfg["image"]["width"])
    img_h = int(cfg["image"]["height"])
    buckets = list((cfg.get("crop") or {}).get("buckets", [64, 128, 256]))
    dt = 1.0 / float(cfg["source"]["fps"])

    builders = {
        "cv_kf": build_kf,
        "cv_akf": build_akf,
        "ca_akf": build_ca_akf,
        "cv_ekf": build_ekf,
        "ca_ekf": build_ca_ekf,
        "cv_pf": build_pf,
    }
    tracker = builders[args.tracker](cfg)

    true_xy = synthetic_cv_trajectory(
        args.steps, x0=(200.0, 150.0), v0=(30.0, -10.0), dt=dt
    )
    # Dummy modality stacks (C=2) — stub detector ignores pixels.
    stacks = [torch.zeros(2, img_h, img_w) for _ in range(args.steps)]
    detect = stub_detect_fn(
        list(true_xy), noise_sigma=3.0, miss_prob=0.05, rng=np.random.default_rng(args.seed)
    )

    results = run_track_detect_loop(
        tracker,
        z0=true_xy[0],
        frame_stacks=stacks,
        detect_fn=detect,
        img_w=img_w,
        img_h=img_h,
        buckets=buckets,
        presence_thresh=float((cfg.get("detector") or {}).get("presence_thresh", 0.5)),
    )

    n_det = sum(1 for r in results if r.detected)
    errs = []
    for r, xy in zip(results, true_xy):
        errs.append(float(np.linalg.norm(r.state.position - xy)))
    mean_err = float(np.mean(errs[5:])) if len(errs) > 5 else float(np.mean(errs))
    last = results[-1]
    print(
        f"tracker={args.tracker} steps={args.steps} detections={n_det}/{len(results)} "
        f"mean_err[5:]={mean_err:.2f}px last_crop={last.crop_size} "
        f"last_r={last.r_meas}"
    )


if __name__ == "__main__":
    main()
