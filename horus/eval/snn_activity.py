"""Per-layer spike / membrane activity metrics for BallHeatmapSNN."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from horus.models.snn import LAYER_NAMES, BallHeatmapSNN, LayerActivity, SNNActivityTrace

# Spike rate (spikes / neuron / timestep) at or above this counts as "nearly every
# timestep". With T=3, 0.9 means the unit must fire on all three steps.
DEFAULT_HYPERACTIVE_THRESHOLD = 0.9

METRIC_KEYS = (
    "mean_membrane",
    "spike_frac",
    "spikes_per_neuron_per_timestep",
    "silent_frac",
    "hyperactive_frac",
)


def summarize_layer(
    activity: LayerActivity,
    hyperactive_threshold: float = DEFAULT_HYPERACTIVE_THRESHOLD,
) -> dict[str, float]:
    """Scalar activity stats for one LIF layer.

    Neuron unit = every spatial site ``(c, h, w)``. Fractions are averaged
    over the batch.
    """
    spikes = activity.spikes  # (B, T, C, H, W)
    membrane = activity.membrane
    if spikes.ndim != 5:
        raise ValueError(f"expected spikes (B,T,C,H,W), got {tuple(spikes.shape)}")

    # Per-neuron spike count over time → (B, C, H, W)
    count = spikes.sum(dim=1)
    t = spikes.shape[1]
    rate = count / max(t, 1)

    # Flatten spatial+channel dims for per-sample fractions
    flat_count = count.flatten(1)
    flat_rate = rate.flatten(1)
    n_units = flat_count.shape[1]
    if n_units == 0:
        zeros = {k: 0.0 for k in METRIC_KEYS}
        return zeros

    spiked = (flat_count > 0).float().mean(dim=1)  # (B,)
    silent = (flat_count == 0).float().mean(dim=1)
    hyper = (flat_rate >= hyperactive_threshold).float().mean(dim=1)

    return {
        "mean_membrane": float(membrane.mean().item()),
        "spike_frac": float(spiked.mean().item()),
        "spikes_per_neuron_per_timestep": float(spikes.mean().item()),
        "silent_frac": float(silent.mean().item()),
        "hyperactive_frac": float(hyper.mean().item()),
    }


def summarize_activity(
    trace: SNNActivityTrace,
    hyperactive_threshold: float = DEFAULT_HYPERACTIVE_THRESHOLD,
) -> dict[str, dict[str, float]]:
    """Per-layer activity dict keyed by ``LAYER_NAMES`` (e.g. ``lif1``)."""
    return {
        name: summarize_layer(layer, hyperactive_threshold=hyperactive_threshold)
        for name, layer in trace.items()
    }


def _accumulate(totals: dict[str, dict[str, float]], stats: dict[str, dict[str, float]]) -> None:
    for layer, metrics in stats.items():
        bucket = totals.setdefault(layer, {k: 0.0 for k in METRIC_KEYS})
        for key, value in metrics.items():
            bucket[key] += value


def _average(totals: dict[str, dict[str, float]], n: int) -> dict[str, dict[str, float]]:
    n = max(n, 1)
    return {
        layer: {k: v / n for k, v in metrics.items()}
        for layer, metrics in totals.items()
    }


@torch.inference_mode()
def measure_activity(
    model: BallHeatmapSNN,
    loader: DataLoader,
    device: torch.device,
    *,
    max_batches: int | None = 20,
    hyperactive_threshold: float = DEFAULT_HYPERACTIVE_THRESHOLD,
    samples_per_batch: int = 1,
) -> dict[str, dict[str, float]]:
    """Mean per-layer activity over up to ``max_batches`` of ``loader``.

    Uses at most ``samples_per_batch`` items per loader batch (default 1) so
    activity traces stay small — full-resolution ``(B,T,C,H,W)`` stacks for all
    LIFs OOMs quickly when ``steps`` grows.

    ``max_batches=None`` uses the full loader.
    """
    if not isinstance(model, BallHeatmapSNN):
        raise TypeError(f"expected BallHeatmapSNN, got {type(model).__name__}")

    was_training = model.training
    model.eval()
    totals: dict[str, dict[str, float]] = {}
    n = 0
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        for batch in loader:
            # Activity logging does not need the full training batch.
            x = batch["input"][: max(int(samples_per_batch), 1)].to(device)
            out = model.forward_with_activity(x)
            # With predict_presence: (heat, presence, trace); else (heat, trace).
            trace = out[-1]
            _accumulate(totals, summarize_activity(trace, hyperactive_threshold))
            del trace
            if len(out) == 3:
                del out
            n += 1
            if max_batches is not None and n >= max_batches:
                break
    finally:
        model.train(was_training)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Ensure all active LIF layers present even on empty loader
    for name in model.lif_layer_names:
        totals.setdefault(name, {k: 0.0 for k in METRIC_KEYS})
    return _average(totals, n)


def format_activity_line(layer: str, metrics: dict[str, float]) -> str:
    """One-line stdout summary for training logs."""
    return (
        f"activity {layer}  mean_v={metrics['mean_membrane']:.4f}  "
        f"spike_frac={metrics['spike_frac']:.4f}  "
        f"rate={metrics['spikes_per_neuron_per_timestep']:.4f}  "
        f"silent={metrics['silent_frac']:.4f}  "
        f"hyper={metrics['hyperactive_frac']:.4f}"
    )


def format_activity_report(stats: dict[str, dict[str, float]]) -> str:
    """Multi-line activity report for layers present in ``stats``."""
    order = [name for name in LAYER_NAMES if name in stats]
    lines = [format_activity_line(name, stats[name]) for name in order]
    return "\n".join(lines)
