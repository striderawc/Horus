"""TensorBoard helpers for heatmap training runs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

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
