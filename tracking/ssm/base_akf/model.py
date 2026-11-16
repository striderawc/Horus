"""Constant-velocity SSM with innovation-scaled process noise."""

from __future__ import annotations

import numpy as np

# 95% threshold for a 2-D Gaussian innovation.
_CHI2 = 5.991
# One frame of constant velocity. State is [x, y, vx, vy].
F = np.array(
    [
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


class SSM:
    """State ``[x, y, vx, vy]`` in pixels and pixels per frame.

    AKF with the usual SSM loop: ``initiate`` → ``predict`` → ``update``.
    ``update`` adapts process-noise scale from the innovation; ``update(None)``
    keeps velocity and grows uncertainty so a later gate can reopen.
    ``r_std`` overrides ``r_meas`` for that frame (pixels); omit it to keep the config value.
    """

    def __init__(self, cfg: dict) -> None:
        self.q_pos = float(cfg["q_pos"])
        self.q_vel = float(cfg["q_vel"])
        self.r = float(cfg["r_meas"]) ** 2
        self.p0_pos = float(cfg["init_pos_std"]) ** 2
        self.p0_vel = float(cfg["init_vel_std"]) ** 2
        self.forget = float(cfg["forgetting"])
        self.q_max = float(cfg["q_max_scale"])
        self.miss_growth = float(cfg.get("miss_growth", 1.5))
        self.miss_vel_decay = float(cfg.get("miss_vel_decay", 1.0))
        self.miss_pos_bump = float(cfg.get("miss_pos_std", 25.0)) ** 2
        self.x: np.ndarray | None = None
        self.P: np.ndarray | None = None
        self.scale = 1.0

    @property
    def mean(self):
        return self.x

    @property
    def cov(self):
        return self.P

    def initiate(self, y: np.ndarray) -> None:
        y = np.asarray(y, dtype=np.float64)
        self.x = np.array([y[0], y[1], 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([self.p0_pos, self.p0_pos, self.p0_vel, self.p0_vel])
        self.scale = 1.0

    def predict(self) -> None:
        qv = self.q_vel * self.scale
        q = np.diag([self.q_pos, self.q_pos, qv, qv])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + q
        self.P = 0.5 * (self.P + self.P.T)

    def update(self, y, r_std: float | None = None):
        if y is None:
            self.x[2:] *= self.miss_vel_decay
            self.scale = min(self.scale * self.miss_growth, self.q_max)
            self.P[0, 0] += self.miss_pos_bump
            self.P[1, 1] += self.miss_pos_bump
            self.P = 0.5 * (self.P + self.P.T)
            return

        y = np.asarray(y, dtype=np.float64)
        r = self.r if r_std is None else float(r_std) ** 2
        innov = y - self.x[:2]
        s = self.P[:2, :2] + r * np.eye(2)
        nis = float(innov @ np.linalg.solve(s, innov))

        if nis > _CHI2:
            self.scale = min(max(self.scale, nis / _CHI2), self.q_max)
            bump = self.scale * self.q_vel
            self.P[2, 2] += bump
            self.P[3, 3] += bump
        else:
            self.scale = self.forget * self.scale + (1.0 - self.forget)

        s = self.P[:2, :2] + r * np.eye(2)
        gain = np.linalg.solve(s, self.P[:2, :]).T
        self.x = self.x + gain @ innov
        self.P = self.P - gain @ self.P[:2, :]
        self.P = 0.5 * (self.P + self.P.T)


def rts_smooth(
    state: np.ndarray,
    cov: np.ndarray,
    pred_state: np.ndarray,
    pred_cov: np.ndarray,
) -> np.ndarray:
    """Rauch–Tung–Striebel means for a finished clip.

    ``state[k]`` and ``cov[k]`` are the filtered Gaussian. ``pred_state[k]``
    and ``pred_cov[k]`` are the prediction of frame ``k`` made from frame
    ``k - 1``; the first frame of a run has no prediction. Frames with no
    filtered state stay nan. Returns ``(N, 4)``.
    """
    state = np.asarray(state, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    pred_state = np.asarray(pred_state, dtype=np.float64)
    pred_cov = np.asarray(pred_cov, dtype=np.float64)
    n = len(state)
    out = np.full((n, state.shape[1]), np.nan, dtype=np.float64)
    finite = np.isfinite(state).all(axis=1)
    predicted = np.isfinite(pred_state).all(axis=1)
    start = 0
    while start < n:
        if not finite[start]:
            start += 1
            continue
        end = start
        while end + 1 < n and finite[end + 1] and predicted[end + 1]:
            end += 1
        out[start : end + 1] = _rts_run(
            state[start : end + 1],
            cov[start : end + 1],
            pred_state[start : end + 1],
            pred_cov[start : end + 1],
        )
        start = end + 1
    return out


def _rts_run(
    state: np.ndarray,
    cov: np.ndarray,
    pred_state: np.ndarray,
    pred_cov: np.ndarray,
) -> np.ndarray:
    mean = state.copy()
    cov = cov.copy()
    n = len(mean)
    for k in range(n - 2, -1, -1):
        filtered = cov[k].copy()
        predicted = pred_cov[k + 1]
        gain = np.linalg.solve(predicted, (filtered @ F.T).T).T
        mean[k] = mean[k] + gain @ (mean[k + 1] - pred_state[k + 1])
        updated = filtered + gain @ (cov[k + 1] - predicted) @ gain.T
        updated = 0.5 * (updated + updated.T)
        vals, vecs = np.linalg.eigh(updated)
        cov[k] = (vecs * np.maximum(vals, 0.0)) @ vecs.T
    return mean
