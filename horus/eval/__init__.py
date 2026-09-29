"""Model evaluation: accuracy, FLOPs, latency, SNN activity."""

from horus.eval.bench import ARCH_REGISTRY, ModelResult, bench_models, load_model
from horus.eval.metrics import AccuracyResult, measure_accuracy, measure_flops, measure_latency
from horus.eval.snn_activity import (
    format_activity_report,
    measure_activity,
    summarize_activity,
)

__all__ = [
    "ARCH_REGISTRY",
    "AccuracyResult",
    "ModelResult",
    "bench_models",
    "format_activity_report",
    "load_model",
    "measure_accuracy",
    "measure_activity",
    "measure_flops",
    "measure_latency",
    "summarize_activity",
]
