"""Sequence RGB dataloader for TrackNet-style ball heatmap training."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

SRC_W, SRC_H = 1280, 720


class BallDataset(Dataset):
    """Sliding windows of N RGB frames → stacked input + Gaussian heatmap."""

    def __init__(self, cfg: dict, split: str = "train") -> None:
        self.num_frames = int(cfg["model"]["num_frames"])
        self.out_w = int(cfg["image"]["width"])
        self.out_h = int(cfg["image"]["height"])
        self.sigma = float(cfg["heatmap"]["sigma"])
        self.amplitude = float(cfg["heatmap"]["amplitude"])

        root = Path(cfg["paths"]["dataset"])
        games = set(cfg["splits"][split])
        self.samples: list[tuple[list[Path], dict]] = []

        for game_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name in games):
            for clip_dir in sorted(p for p in game_dir.iterdir() if p.is_dir()):
                label_path = clip_dir / "Label.csv"
                if not label_path.is_file():
                    continue
                frames = sorted(clip_dir.glob("*.jpg"), key=lambda p: int(p.stem) if p.stem.isdigit() else 10**9)
                if len(frames) < self.num_frames:
                    continue

                labels: dict[str, dict] = {}
                with label_path.open(newline="") as f:
                    for row in csv.DictReader(f):
                        vis = int(row["visibility"] or 0)
                        labels[row["file name"]] = {
                            "visibility": vis,
                            "x": float(row["x-coordinate"] or 0),
                            "y": float(row["y-coordinate"] or 0),
                        }

                for end in range(self.num_frames - 1, len(frames)):
                    window = frames[end - self.num_frames + 1 : end + 1]
                    lb = labels.get(window[-1].name, {"visibility": 0, "x": 0.0, "y": 0.0})
                    self.samples.append((window, lb))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        paths, lb = self.samples[idx]
        stacked = []
        for p in paths:
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(p)
            if bgr.shape[1] != self.out_w or bgr.shape[0] != self.out_h:
                bgr = cv2.resize(bgr, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            stacked.append(rgb.transpose(2, 0, 1))
        inp = torch.from_numpy(np.concatenate(stacked, axis=0))

        heat = np.zeros((self.out_h, self.out_w), dtype=np.float32)
        visible = lb["visibility"] > 0
        if visible:
            cx = lb["x"] * self.out_w / SRC_W
            cy = lb["y"] * self.out_h / SRC_H
            radius = int(max(1, round(3 * self.sigma)))
            x0 = max(0, int(np.floor(cx - radius)))
            x1 = min(self.out_w, int(np.ceil(cx + radius)) + 1)
            y0 = max(0, int(np.floor(cy - radius)))
            y1 = min(self.out_h, int(np.ceil(cy + radius)) + 1)
            if x0 < x1 and y0 < y1:
                xs = np.arange(x0, x1, dtype=np.float32)
                ys = np.arange(y0, y1, dtype=np.float32)
                xx, yy = np.meshgrid(xs, ys)
                heat[y0:y1, x0:x1] = self.amplitude * np.exp(
                    -((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * self.sigma * self.sigma)
                )

        return {"input": inp, "heatmap": torch.from_numpy(heat).unsqueeze(0)}


def get_loader(cfg: dict, split: str = "train") -> DataLoader:
    tcfg = cfg["train"]
    return DataLoader(
        BallDataset(cfg, split),
        batch_size=int(tcfg["batch_size"]),
        shuffle=(split == "train"),
        num_workers=int(tcfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )
