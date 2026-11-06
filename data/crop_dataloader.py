"""Variable-size signed frame-difference crops for the SSM-gate CNN."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

SRC_W, SRC_H = 1280, 720


def side_choices(min_side: int, max_side: int, multiple: int) -> list[int]:
    start = ((min_side + multiple - 1) // multiple) * multiple
    stop = (max_side // multiple) * multiple
    sides = list(range(start, stop + 1, multiple))
    if not sides:
        raise ValueError(f"no crop sides in [{min_side}, {max_side}] on a multiple of {multiple}")
    return sides


def _inside(x0: int, y0: int, h: int, w: int, bx: float, by: float) -> bool:
    return x0 <= bx < x0 + w and y0 <= by < y0 + h


def _crop(img: np.ndarray, x0: int, y0: int, h: int, w: int) -> np.ndarray:
    """Crop ``img`` at ``(x0, y0)``. Pixels outside the frame are 0."""
    out = np.zeros((h, w), dtype=np.float32)
    src_h, src_w = img.shape[:2]
    x_from = max(0, x0)
    y_from = max(0, y0)
    x_to = min(src_w, x0 + w)
    y_to = min(src_h, y0 + h)
    if x_from >= x_to or y_from >= y_to:
        return out
    out[y_from - y0 : y_to - y0, x_from - x0 : x_to - x0] = img[y_from:y_to, x_from:x_to]
    return out


def _gaussian(h: int, w: int, cx: float, cy: float, sigma: float, amplitude: float) -> np.ndarray:
    heat = np.zeros((h, w), dtype=np.float32)
    radius = int(max(1, round(3 * sigma)))
    x0 = max(0, int(np.floor(cx - radius)))
    x1 = min(w, int(np.ceil(cx + radius)) + 1)
    y0 = max(0, int(np.floor(cy - radius)))
    y1 = min(h, int(np.ceil(cy + radius)) + 1)
    if x0 < x1 and y0 < y1:
        xs = np.arange(x0, x1, dtype=np.float32)
        ys = np.arange(y0, y1, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        heat[y0:y1, x0:x1] = amplitude * np.exp(
            -((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma)
        )
    return heat


def _origin(
    rng: np.random.Generator,
    mode: str,
    ball_x: float,
    ball_y: float,
    h: int,
    w: int,
    *,
    center_jitter: float,
    src_w: int = SRC_W,
    src_h: int = SRC_H,
) -> tuple[int, int]:
    """Top-left of a crop that places the ball according to ``mode``."""
    if mode == "random":
        x_hi = max(1, src_w - w + 1)
        y_hi = max(1, src_h - h + 1)
        return int(rng.integers(0, x_hi)), int(rng.integers(0, y_hi))

    if mode == "centered":
        jx = float(rng.uniform(-center_jitter, center_jitter)) * w
        jy = float(rng.uniform(-center_jitter, center_jitter)) * h
        return int(round(ball_x + jx - w / 2)), int(round(ball_y + jy - h / 2))

    if mode == "offset":
        rel_x = float(rng.uniform(0, w))
        rel_y = float(rng.uniform(0, h))
        x0 = int(round(ball_x - rel_x))
        y0 = int(round(ball_y - rel_y))
        if ball_x < x0:
            x0 -= 1
        elif ball_x >= x0 + w:
            x0 += 1
        if ball_y < y0:
            y0 -= 1
        elif ball_y >= y0 + h:
            y0 += 1
        return x0, y0

    # absent: shift the crop so the ball center falls just outside one edge
    side = int(rng.integers(0, 4))
    extra = int(rng.integers(0, max(w, h) // 2 + 1))
    if side == 0:
        x0 = int(np.floor(ball_x)) + 1 + extra
        y0 = int(round(ball_y - float(rng.uniform(0, h))))
    elif side == 1:
        x0 = int(np.floor(ball_x)) - w - extra
        y0 = int(round(ball_y - float(rng.uniform(0, h))))
    elif side == 2:
        y0 = int(np.floor(ball_y)) + 1 + extra
        x0 = int(round(ball_x - float(rng.uniform(0, w))))
    else:
        y0 = int(np.floor(ball_y)) - h - extra
        x0 = int(round(ball_x - float(rng.uniform(0, w))))
    return x0, y0


class CropDataset(Dataset):
    """Stacked grayscale differences on a random crop, plus a Gaussian heatmap.

    Any labeled ball (visibility > 0: visible, hard to see, or occluded) is
    usually near the crop center, sometimes elsewhere in the crop, and
    sometimes outside it. A Gaussian is painted when that point lies inside
    the crop. Visibility 0, not in frame, gets an empty heatmap.
    """

    def __init__(self, cfg: dict, split: str = "train") -> None:
        crop = cfg["crop"]
        self.num_diffs = int(cfg["model"]["num_diffs"])
        self.num_frames = self.num_diffs + 1
        self.sigma = float(cfg["heatmap"]["sigma"])
        self.amplitude = float(cfg["heatmap"]["amplitude"])
        p_centered = float(crop["p_centered"])
        p_offset = float(crop["p_offset"])
        p_absent = float(crop["p_absent"])
        total = p_centered + p_offset + p_absent
        if total <= 0:
            raise ValueError("crop placement probabilities must sum to a positive value")
        self.p_centered = p_centered / total
        self.p_offset = p_offset / total
        self.center_jitter = float(crop["center_jitter"])
        self.sides = side_choices(int(crop["min_side"]), int(crop["max_side"]), int(crop["multiple"]))

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

    def _mode(self, rng: np.random.Generator, visibility: int) -> str:
        # 1 = visible, 2 = hard to see, 3 = occluded. Same placement as the baseline positives.
        if visibility > 0:
            r = float(rng.random())
            if r < self.p_centered:
                return "centered"
            if r < self.p_centered + self.p_offset:
                return "offset"
            return "absent"
        return "random"

    def __getitem__(self, idx: int | tuple[int, int, int]) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng()
        if isinstance(idx, tuple):
            sample_idx, h, w = idx
            h, w = int(h), int(w)
        else:
            sample_idx = int(idx)
            h = int(rng.choice(self.sides))
            w = int(rng.choice(self.sides))

        paths, lb = self.samples[sample_idx]
        grays = []
        for p in paths:
            gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise FileNotFoundError(p)
            if gray.shape[1] != SRC_W or gray.shape[0] != SRC_H:
                gray = cv2.resize(gray, (SRC_W, SRC_H), interpolation=cv2.INTER_AREA)
            grays.append(gray)

        diffs = [
            (grays[i].astype(np.float32) - grays[i - 1].astype(np.float32)) / 255.0
            for i in range(1, len(grays))
        ]

        vis = int(lb["visibility"])
        mode = self._mode(rng, vis)
        x0, y0 = _origin(
            rng,
            mode,
            lb["x"],
            lb["y"],
            h,
            w,
            center_jitter=self.center_jitter,
        )
        cropped = np.stack([_crop(d, x0, y0, h, w) for d in diffs], axis=0)

        # Same rule as the baseline: a peak for every labeled ball whose center is inside the crop.
        paint = vis > 0 and mode != "absent" and _inside(x0, y0, h, w, lb["x"], lb["y"])
        if paint:
            heat = _gaussian(h, w, lb["x"] - x0, lb["y"] - y0, self.sigma, self.amplitude)
        else:
            heat = np.zeros((h, w), dtype=np.float32)

        return {
            "input": torch.from_numpy(cropped),
            "heatmap": torch.from_numpy(heat).unsqueeze(0),
        }


class CropBatchSampler(Sampler[list[tuple[int, int, int]]]):
    """Each batch shares one crop height and width. The size changes between batches."""

    def __init__(self, n: int, batch_size: int, shuffle: bool, sides: list[int]) -> None:
        self.n = n
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sides = sides

    def __len__(self) -> int:
        if self.n == 0:
            return 0
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        rng = np.random.default_rng()
        indices = np.arange(self.n)
        if self.shuffle:
            rng.shuffle(indices)
        for start in range(0, self.n, self.batch_size):
            h = int(rng.choice(self.sides))
            w = int(rng.choice(self.sides))
            yield [(int(i), h, w) for i in indices[start : start + self.batch_size]]


def get_loader(cfg: dict, split: str = "train") -> DataLoader:
    dataset = CropDataset(cfg, split)
    tcfg = cfg["train"]
    sampler = CropBatchSampler(
        len(dataset),
        batch_size=int(tcfg["batch_size"]),
        shuffle=(split == "train"),
        sides=dataset.sides,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=int(tcfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )
