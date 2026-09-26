"""Model architectures (CNN / SNN)."""

from horus.models.cnn import BallHeatmapCNN, heatmap_to_coords, weighted_heatmap_mse
from horus.models.snn import BallHeatmapSNN

__all__ = [
    "BallHeatmapCNN",
    "BallHeatmapSNN",
    "heatmap_to_coords",
    "weighted_heatmap_mse",
]
