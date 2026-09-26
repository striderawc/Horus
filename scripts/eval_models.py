#!/usr/bin/env python3
"""Evaluate heatmap models: accuracy, FLOPs, latency."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from horus.config import load_config, repo_root
from horus.eval.bench import ARCH_REGISTRY, bench_models, format_table, save_results


def _parse_model(s: str) -> dict:
    """Parse ``name=...,ckpt=...,arch=...,modality=...`` (ckpt/arch required)."""
    parts: dict[str, str] = {}
    for token in s.split(","):
        if "=" not in token:
            raise argparse.ArgumentTypeError(
                f"Invalid --model fragment '{token}' (expected key=value)"
            )
        k, v = token.split("=", 1)
        parts[k.strip()] = v.strip()
    if "ckpt" not in parts and "checkpoint" not in parts:
        raise argparse.ArgumentTypeError("--model requires ckpt=...")
    if "arch" not in parts:
        raise argparse.ArgumentTypeError("--model requires arch=...")
    ckpt = parts.get("ckpt") or parts["checkpoint"]
    return {
        "name": parts.get("name"),
        "checkpoint": ckpt,
        "arch": parts["arch"],
        "modality": parts.get("modality"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--latency-runs", type=int, default=50)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--no-save", action="store_true", help="Skip writing JSON")

    # Single-model convenience flags
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument(
        "--arch",
        default=None,
        help=f"Architecture name ({', '.join(sorted(ARCH_REGISTRY))})",
    )
    p.add_argument("--modality", default=None)
    p.add_argument("--name", default=None)

    # Multi-model
    p.add_argument(
        "--model",
        action="append",
        type=_parse_model,
        default=[],
        metavar="name=...,ckpt=...,arch=...",
        help="Repeatable model spec (overrides --checkpoint if used)",
    )
    args = p.parse_args()

    specs: list[dict] = list(args.model)
    if not specs:
        if args.checkpoint is None or args.arch is None:
            p.error("Provide --checkpoint and --arch, or one or more --model specs")
        specs = [
            {
                "name": args.name,
                "checkpoint": str(args.checkpoint),
                "arch": args.arch,
                "modality": args.modality,
            }
        ]

    for spec in specs:
        ckpt = Path(spec["checkpoint"])
        if not ckpt.is_absolute():
            ckpt = repo_root() / ckpt
        if not ckpt.is_file():
            p.error(f"Checkpoint not found: {ckpt}")
        spec["checkpoint"] = ckpt
        if not spec.get("name"):
            spec["name"] = ckpt.stem

    cfg = load_config(args.config)
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = bench_models(
        specs,
        cfg,
        split=args.split,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        warmup=args.warmup,
        latency_runs=args.latency_runs,
    )

    print(format_table(results))
    if any(r.gflops is None for r in results):
        print(
            "\nNote: FLOPs skipped (fvcore not installed). "
            "pip install 'horus[eval]' or pip install fvcore"
        )

    if not args.no_save:
        path = save_results(results, args.out_dir)
        print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
