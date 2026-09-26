"""Bench one or more heatmap models: accuracy, FLOPs, latency."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

from horus.config import repo_root
from horus.data.dataset import Modality, build_dataloader
from horus.eval.metrics import (
    AccuracyResult,
    input_channels,
    measure_accuracy,
    measure_flops,
    measure_latency,
)
from horus.models import BallHeatmapCNN, BallHeatmapSNN

ArchFactory = Callable[[], nn.Module]

# name -> (factory, default modality)
ARCH_REGISTRY: dict[str, tuple[ArchFactory, Modality]] = {
    "rgb_cnn": (lambda: BallHeatmapCNN(in_ch=9), "rgb"),
    "diff_cnn": (lambda: BallHeatmapCNN(in_ch=2), "diff"),
    "event_frame_cnn": (lambda: BallHeatmapCNN(in_ch=6), "event_frame"),
    "event_frame_snn": (lambda: BallHeatmapSNN(in_ch=2, steps=3), "event_frame"),
}


@dataclass
class ModelResult:
    name: str
    arch: str
    modality: str
    mean_px: float
    acc_at_5: float
    acc_at_10: float
    n_visible: int
    gflops: float | None
    latency_ms: float
    device: str
    acc_per_gflop: float | None
    acc_per_ms: float | None


def load_model(
    arch: str,
    checkpoint: Path | str,
    device: torch.device,
) -> nn.Module:
    """Build ``arch`` and load ``checkpoint['model']`` weights."""
    if arch not in ARCH_REGISTRY:
        raise KeyError(f"Unknown arch '{arch}'. Known: {sorted(ARCH_REGISTRY)}")
    factory, _ = ARCH_REGISTRY[arch]
    model = factory().to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def _eff_ratios(acc_at_5: float, gflops: float | None, latency_ms: float) -> tuple[float | None, float | None]:
    per_gflop = (acc_at_5 / gflops) if gflops and gflops > 0 else None
    per_ms = (acc_at_5 / latency_ms) if latency_ms > 0 else None
    return per_gflop, per_ms


def bench_one(
    *,
    name: str,
    arch: str,
    checkpoint: Path | str,
    cfg: dict[str, Any],
    split: str = "test",
    modality: Modality | None = None,
    device: torch.device | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    warmup: int = 10,
    latency_runs: int = 50,
    loader: Any | None = None,
) -> ModelResult:
    """Run accuracy + FLOPs + latency for a single checkpoint."""
    if arch not in ARCH_REGISTRY:
        raise KeyError(f"Unknown arch '{arch}'. Known: {sorted(ARCH_REGISTRY)}")
    _, default_mod = ARCH_REGISTRY[arch]
    mod: Modality = modality if modality is not None else default_mod

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(arch, checkpoint, device)
    dl = loader or build_dataloader(
        cfg, split, mod, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    acc: AccuracyResult = measure_accuracy(model, dl, device)

    img = cfg["image"]
    h, w = int(img["height"]), int(img["width"])
    in_ch = input_channels(mod, cfg)
    flops = measure_flops(model, in_ch=in_ch, height=h, width=w, device=device)
    gflops = (flops / 1e9) if flops is not None else None
    latency_ms = measure_latency(
        model,
        in_ch=in_ch,
        height=h,
        width=w,
        device=device,
        warmup=warmup,
        runs=latency_runs,
    )
    per_gflop, per_ms = _eff_ratios(acc.acc_at_5, gflops, latency_ms)

    return ModelResult(
        name=name,
        arch=arch,
        modality=mod,
        mean_px=acc.mean_px,
        acc_at_5=acc.acc_at_5,
        acc_at_10=acc.acc_at_10,
        n_visible=acc.n_visible,
        gflops=gflops,
        latency_ms=latency_ms,
        device=str(device),
        acc_per_gflop=per_gflop,
        acc_per_ms=per_ms,
    )


def bench_models(
    specs: list[dict[str, Any]],
    cfg: dict[str, Any],
    *,
    split: str = "test",
    device: torch.device | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    warmup: int = 10,
    latency_runs: int = 50,
) -> list[ModelResult]:
    """Bench multiple models; reuse one dataloader per modality."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders: dict[Modality, Any] = {}
    results: list[ModelResult] = []
    for spec in specs:
        arch = str(spec["arch"])
        if arch not in ARCH_REGISTRY:
            raise KeyError(f"Unknown arch '{arch}'. Known: {sorted(ARCH_REGISTRY)}")
        _, default_mod = ARCH_REGISTRY[arch]
        mod: Modality = spec.get("modality") or default_mod
        if mod not in loaders:
            loaders[mod] = build_dataloader(
                cfg,
                split,
                mod,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,
            )
        name = str(spec.get("name") or Path(spec["checkpoint"]).stem)
        results.append(
            bench_one(
                name=name,
                arch=arch,
                checkpoint=spec["checkpoint"],
                cfg=cfg,
                split=split,
                modality=mod,
                device=device,
                batch_size=batch_size,
                num_workers=num_workers,
                warmup=warmup,
                latency_runs=latency_runs,
                loader=loaders[mod],
            )
        )
    return results


def _fmt(v: float | None, digits: int = 3) -> str:
    if v is None:
        return "n/a"
    if v != v:  # NaN
        return "nan"
    return f"{v:.{digits}f}"


def format_table(results: list[ModelResult]) -> str:
    """Human-readable comparison table with pure / efficiency winners."""
    headers = [
        "name",
        "modality",
        "mean_px",
        "acc@5",
        "GFLOPs",
        "latency_ms",
        "acc/GFLOP",
        "acc/ms",
    ]
    rows: list[list[str]] = []
    for r in results:
        rows.append(
            [
                r.name,
                r.modality,
                _fmt(r.mean_px, 2),
                _fmt(r.acc_at_5, 3),
                _fmt(r.gflops, 3),
                _fmt(r.latency_ms, 2),
                _fmt(r.acc_per_gflop, 3),
                _fmt(r.acc_per_ms, 4),
            ]
        )

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def line(cells: list[str]) -> str:
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    parts = [line(headers), line(["-" * w for w in widths])]
    parts.extend(line(row) for row in rows)

    if results:
        best_acc = min(results, key=lambda r: r.mean_px if r.mean_px == r.mean_px else float("inf"))
        parts.append("")
        parts.append(f"Best accuracy (lowest mean_px): {best_acc.name} ({_fmt(best_acc.mean_px, 2)} px)")

        with_gflop = [r for r in results if r.acc_per_gflop is not None]
        if with_gflop:
            best_g = max(with_gflop, key=lambda r: r.acc_per_gflop or 0.0)
            parts.append(
                f"Best acc/GFLOP: {best_g.name} ({_fmt(best_g.acc_per_gflop, 3)})"
            )
        else:
            parts.append("Best acc/GFLOP: n/a (install eval extra: pip install 'horus[eval]')")

        with_ms = [r for r in results if r.acc_per_ms is not None]
        if with_ms:
            best_l = max(with_ms, key=lambda r: r.acc_per_ms or 0.0)
            parts.append(f"Best acc/ms: {best_l.name} ({_fmt(best_l.acc_per_ms, 4)})")

    return "\n".join(parts)


def save_results(
    results: list[ModelResult],
    out_dir: Path | str | None = None,
) -> Path:
    """Write JSON under ``results/eval/``; return the file path."""
    root = Path(out_dir) if out_dir else repo_root() / "results" / "eval"
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = root / f"{stamp}.json"
    payload = {
        "created_utc": stamp,
        "models": [asdict(r) for r in results],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
