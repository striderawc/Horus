"""Model architectures (CNN / SNN)."""

from horus.models.cnn import BallHeatmapCNN, heatmap_to_coords, weighted_heatmap_mse
from horus.models.snn import (
    LAYER_NAMES,
    STAGE_NAMES,
    BallHeatmapSNN,
    LayerActivity,
    SNNActivityTrace,
)

__all__ = [
    "BallHeatmapCNN",
    "BallHeatmapSNN",
    "LAYER_NAMES",
    "LayerActivity",
    "STAGE_NAMES",
    "SNNActivityTrace",
    "heatmap_to_coords",
    "weighted_heatmap_mse",
]
