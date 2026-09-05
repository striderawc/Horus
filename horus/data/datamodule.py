"""Convenience factory mirroring a light Lightning-style datamodule."""

from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader

from horus.data.dataset import Modality, build_dataloader


class BallTrackDataModule:
    """Hold config and build train/val/test loaders for a modality."""

    def __init__(self, cfg: dict[str, Any], modality: Modality = "rgb") -> None:
        self.cfg = cfg
        self.modality = modality

    def train_dataloader(self, **kwargs: Any) -> DataLoader:
        return build_dataloader(self.cfg, "train", self.modality, **kwargs)

    def val_dataloader(self, **kwargs: Any) -> DataLoader:
        kwargs.setdefault("shuffle", False)
        return build_dataloader(self.cfg, "val", self.modality, **kwargs)

    def test_dataloader(self, **kwargs: Any) -> DataLoader:
        kwargs.setdefault("shuffle", False)
        return build_dataloader(self.cfg, "test", self.modality, **kwargs)
