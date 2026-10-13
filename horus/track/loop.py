"""Closed-loop SSM ↔ detector crop pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import numpy as np
import torch

from horus.data.crop import crop_chw_stack
from horus.models.cnn import soft_heatmap_spatial_std
from horus.track.gate import gate_to_square_crop
from horus.track.state import TrackState


class _Tracker(Protocol):
    def initiate(self, z: np.ndarray) -> TrackState: ...

    def predict(self, state: TrackState) -> TrackState: ...

    def update(
        self,
        state: TrackState,
        z: np.ndarray | None,
        *,
        r_meas: float | None = None,
    ) -> TrackState: ...


@dataclass
class TrackDetectResult:
    """One frame of the closed loop."""

    t: int
    state: TrackState
    crop_origin: tuple[int, int]
    crop_size: int
    z: np.ndarray | None
    r_meas: float | None
    detected: bool


def detection_to_measurement(
    heat: torch.Tensor,
    presence: torch.Tensor | float | None,
    *,
    origin: tuple[float, float],
    presence_thresh: float = 0.5,
    sigma_min: float = 0.5,
    sigma_max: float = 64.0,
    temperature: float = 0.1,
) -> tuple[np.ndarray | None, float | None]:
    """Map crop heatmap (+ presence) → full-frame ``(z, r_meas)``.

    ``heat`` is logits or probs ``(1, H, W)`` / ``(H, W)``. Misses when
    ``sigmoid(presence) < presence_thresh``. ``r_meas`` is the soft spatial
    std of the heatmap (peak sharpness → measurement noise).
    """
    if heat.ndim == 3:
        heat2d = heat[0]
    elif heat.ndim == 2:
        heat2d = heat
    else:
        raise ValueError(f"expected (1,H,W) or (H,W), got {tuple(heat.shape)}")

    if presence is not None:
        if isinstance(presence, torch.Tensor):
            p = float(torch.sigmoid(presence.reshape(-1)[0]).item())
        else:
            p = float(1.0 / (1.0 + np.exp(-float(presence))))
        if p < presence_thresh:
            return None, None

    flat = heat2d.reshape(-1)
    idx = int(flat.argmax().item())
    h, w = int(heat2d.shape[0]), int(heat2d.shape[1])
    y_c = float(idx // w)
    x_c = float(idx % w)
    x0, y0 = float(origin[0]), float(origin[1])
    z = np.array([x_c + x0, y_c + y0], dtype=np.float64)

    heat_b = heat2d.detach().float().unsqueeze(0).unsqueeze(0)
    sigma = float(soft_heatmap_spatial_std(heat_b, temperature=temperature)[0].item())
    sigma = max(sigma_min, min(sigma, sigma_max))
    return z, sigma


def run_track_detect_loop(
    tracker: _Tracker,
    *,
    z0: np.ndarray,
    frame_stacks: Sequence[torch.Tensor | np.ndarray],
    detect_fn: Callable[
        [torch.Tensor, int, int, int],
        tuple[torch.Tensor, torch.Tensor | float | None],
    ],
    img_w: int,
    img_h: int,
    buckets: Sequence[int] = (64, 128, 256),
    n_sigma: float = 3.0,
    presence_thresh: float = 0.5,
) -> list[TrackDetectResult]:
    """Predict → gate crop → detect → update for each frame after ``z0``.

    ``detect_fn(crop, x0, y0, size)`` returns ``(heatmap_logits, presence_logit)``.
    """
    if not frame_stacks:
        return []

    state = tracker.initiate(np.asarray(z0, dtype=np.float64).reshape(2))
    results: list[TrackDetectResult] = [
        TrackDetectResult(
            t=0,
            state=state,
            crop_origin=(0, 0),
            crop_size=0,
            z=np.asarray(z0, dtype=np.float64).reshape(2),
            r_meas=None,
            detected=True,
        )
    ]

    for t in range(1, len(frame_stacks)):
        state = tracker.predict(state)
        x0, y0, size = gate_to_square_crop(
            state,
            buckets=buckets,
            img_w=img_w,
            img_h=img_h,
            n_sigma=n_sigma,
        )
        stack = frame_stacks[t]
        if isinstance(stack, np.ndarray):
            stack_t = torch.from_numpy(np.asarray(stack))
        else:
            stack_t = stack
        crop = crop_chw_stack(stack_t, x0, y0, size)
        heat, presence = detect_fn(crop, x0, y0, size)
        z, r = detection_to_measurement(
            heat.detach().cpu() if isinstance(heat, torch.Tensor) else heat,
            presence.detach().cpu()
            if isinstance(presence, torch.Tensor)
            else presence,
            origin=(x0, y0),
            presence_thresh=presence_thresh,
        )
        state = tracker.update(state, z, r_meas=r)
        results.append(
            TrackDetectResult(
                t=t,
                state=state,
                crop_origin=(x0, y0),
                crop_size=size,
                z=z,
                r_meas=r,
                detected=z is not None,
            )
        )
    return results


def stub_detect_fn(
    true_xy: Sequence[np.ndarray | None],
    *,
    noise_sigma: float = 2.0,
    miss_prob: float = 0.0,
    rng: np.random.Generator | None = None,
) -> Callable[
    [torch.Tensor, int, int, int],
    tuple[torch.Tensor, torch.Tensor | float | None],
]:
    """Oracle stub detector for smoke tests (delta peak + presence logit)."""
    rng = rng or np.random.default_rng(0)
    call_t = {"i": 0}

    def _detect(
        crop: torch.Tensor, x0: int, y0: int, size: int
    ) -> tuple[torch.Tensor, float]:
        call_t["i"] += 1
        t = call_t["i"]
        # Logits: large positive at peak, zeros elsewhere.
        heat = torch.full((1, size, size), -4.0, dtype=torch.float32)
        if t >= len(true_xy) or true_xy[t] is None:
            return heat, -4.0  # low presence
        if miss_prob > 0 and float(rng.random()) < miss_prob:
            return heat, -4.0
        xy = np.asarray(true_xy[t], dtype=np.float64).reshape(2)
        if noise_sigma > 0:
            xy = xy + rng.normal(0.0, noise_sigma, size=2)
        xc = float(xy[0] - x0)
        yc = float(xy[1] - y0)
        ix = int(round(xc))
        iy = int(round(yc))
        if 0 <= ix < size and 0 <= iy < size:
            heat[0, iy, ix] = 8.0
            return heat, 4.0  # high presence
        return heat, -4.0

    return _detect
