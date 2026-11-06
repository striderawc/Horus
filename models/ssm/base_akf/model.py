"""Constant-velocity SSM with innovation-scaled process noise."""

from __future__ import annotations

import numpy as np

# 95% threshold for a 2-D Gaussian innovation.
_CHI2 = 5.991
_F = np.array(
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
    damps velocity and grows uncertainty so a later gate can reopen.
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
        self.miss_vel_decay = float(cfg.get("miss_vel_decay", 0.0))
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
        self.x = _F @ self.x
        self.P = _F @ self.P @ _F.T + q
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
