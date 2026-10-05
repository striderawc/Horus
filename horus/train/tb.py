"""TensorBoard helpers for heatmap training runs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.tensorboard import SummaryWriter


def resolve_log_dir(
    base: str | Path,
    run_name: str,
    *,
    repo: Path,
) -> Path:
    """``base/run_name/<utc-timestamp>`` under the repo unless ``base`` is absolute."""
    root = Path(base)
    if not root.is_absolute():
        root = repo / root
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = root / run_name / stamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_writer(log_dir: Path) -> SummaryWriter:
    """Create a ``SummaryWriter`` writing under ``log_dir``."""
    return SummaryWriter(log_dir=str(log_dir))


def open_run_writer(
    *,
    run_name: str,
    repo: Path,
    log_dir: str | Path | None,
    disabled: bool = False,
    default_root: str | Path = "runs",
) -> SummaryWriter | None:
    """Open a timestamped run writer, or ``None`` if TensorBoard is disabled."""
    if disabled:
        return None
    root = Path(log_dir) if log_dir is not None else Path(default_root)
    run_dir = resolve_log_dir(root, run_name, repo=repo)
    print(f"tensorboard log_dir={run_dir}")
    return make_writer(run_dir)


def log_scalars(
    writer: SummaryWriter,
    tag_to_value: Mapping[str, float],
    step: int,
) -> None:
    """Write a batch of scalar metrics at ``step``."""
    for tag, value in tag_to_value.items():
        writer.add_scalar(tag, float(value), step)


def log_hparams(
    writer: SummaryWriter,
    hparams: Mapping[str, Any],
    metrics: Mapping[str, float],
) -> None:
    """Record run hyperparams with a small metric snapshot (best-effort)."""
    clean = {}
    for key, value in hparams.items():
        if isinstance(value, (bool, int, float, str)):
            clean[key] = value
        elif value is None:
            clean[key] = "None"
        else:
            clean[key] = str(value)
    writer.add_hparams(dict(clean), {k: float(v) for k, v in metrics.items()})


def _to_hw01(heat: torch.Tensor) -> torch.Tensor:
    """Squeeze to ``(H, W)`` float in ``[0, 1]`` for display."""
    x = heat.detach().float().cpu()
    while x.ndim > 2:
        x = x[0]
    return x.clamp(0.0, 1.0)


def heatmap_compare_image(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Build a ``(3, H, 3W)`` RGB strip: GT | pred | |pred−GT|."""
    g = _to_hw01(target)
    p = _to_hw01(pred)
    if g.shape != p.shape:
        raise ValueError(f"heatmap shape mismatch: gt={tuple(g.shape)} pred={tuple(p.shape)}")
    err = (p - g).abs()
    # Fake RGB so TensorBoard shows greyscale panels side-by-side.
    panels = [g, p, err]
    row = torch.cat(panels, dim=1)  # (H, 3W)
    return row.unsqueeze(0).expand(3, -1, -1).contiguous()


def log_heatmap_images(
    writer: SummaryWriter,
    pred: torch.Tensor,
    target: torch.Tensor,
    step: int,
    *,
    tag: str = "debug/heatmap_gt_pred_abserr",
    max_samples: int = 1,
) -> None:
    """Log GT / pred / abs-error strips for the first ``max_samples`` items."""
    if pred.ndim != 4 or target.ndim != 4:
        raise ValueError(
            f"expected pred/target (B,1,H,W), got {tuple(pred.shape)} / {tuple(target.shape)}"
        )
    n = min(int(max_samples), pred.shape[0], target.shape[0])
    for i in range(n):
        image = heatmap_compare_image(pred[i], target[i])
        sample_tag = tag if n == 1 else f"{tag}/sample{i}"
        writer.add_image(sample_tag, image, step, dataformats="CHW")
