"""Adaptive constant-acceleration Kalman filter (learns Q and R online)."""

from __future__ import annotations

from typing import Any

import numpy as np

from horus.track.common import dt_from_cfg, ensure_spd
from horus.track.state import TrackState


class AdaptiveConstantAccelerationKF:
    """CA Kalman filter with Sage-Husa adaptation of ``Q`` and ``R``.

    State is ``[x, y, vx, vy, ax, ay]``. Position measurements ``z = (x, y)``.
    Same public surface as the CV trackers: ``initiate`` / ``predict`` /
    ``update`` / ``gate_window``.
    """

    dim = 6

    def __init__(
        self,
        dt: float,
        *,
        q_pos: float = 1.0,
        q_vel: float = 10.0,
        q_acc: float = 50.0,
        r_meas: float = 5.0,
        init_pos_std: float = 10.0,
        init_vel_std: float = 50.0,
        init_acc_std: float = 100.0,
        forgetting: float = 0.95,
        q_min: float = 1e-4,
        r_min: float = 0.25,
        q_max_scale: float = 100.0,
        r_max_scale: float = 100.0,
    ) -> None:
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        if not (0.0 < forgetting < 1.0):
            raise ValueError(f"forgetting must be in (0, 1), got {forgetting}")
        if q_min <= 0 or r_min <= 0:
            raise ValueError("q_min and r_min must be positive")
        if q_max_scale < 1.0 or r_max_scale < 1.0:
            raise ValueError("q_max_scale and r_max_scale must be >= 1")

        self.dt = float(dt)
        self.q_pos = float(q_pos)
        self.q_vel = float(q_vel)
        self.q_acc = float(q_acc)
        self.r_meas = float(r_meas)
        self.init_pos_std = float(init_pos_std)
        self.init_vel_std = float(init_vel_std)
        self.init_acc_std = float(init_acc_std)
        self.forgetting = float(forgetting)
        self.q_min = float(q_min)
        self.r_min = float(r_min)
        self.q_max_scale = float(q_max_scale)
        self.r_max_scale = float(r_max_scale)

        dt2 = 0.5 * self.dt * self.dt
        self.F = np.array(
            [
                [1.0, 0.0, self.dt, 0.0, dt2, 0.0],
                [0.0, 1.0, 0.0, self.dt, 0.0, dt2],
                [0.0, 0.0, 1.0, 0.0, self.dt, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, self.dt],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        self.Q = np.diag(
            [
                self.q_pos,
                self.q_pos,
                self.q_vel,
                self.q_vel,
                self.q_acc,
                self.q_acc,
            ]
        ).astype(np.float64)
        self.R = (self.r_meas**2) * np.eye(2, dtype=np.float64)

        self._Q0 = self.Q.copy()
        self._R0 = self.R.copy()
        self._q_max = float(np.max(np.diag(self._Q0)) * self.q_max_scale)
        self._r_max = float(np.max(np.diag(self._R0)) * self.r_max_scale)
        self._r_min2 = self.r_min**2
        self._update_count = 0

    def _reset_noise(self) -> None:
        self.Q = self._Q0.copy()
        self.R = self._R0.copy()
        self._update_count = 0

    def _adapt_gain(self) -> float:
        self._update_count += 1
        b = self.forgetting
        k = self._update_count
        return float((1.0 - b) / (1.0 - b**k))

    def initiate(self, z: np.ndarray) -> TrackState:
        """Start a CA track from ``z = (x, y)``; reset learned ``Q`` / ``R``."""
        self._reset_noise()
        z = np.asarray(z, dtype=np.float64).reshape(2)
        mean = np.array(
            [z[0], z[1], 0.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        cov = np.diag(
            [
                self.init_pos_std**2,
                self.init_pos_std**2,
                self.init_vel_std**2,
                self.init_vel_std**2,
                self.init_acc_std**2,
                self.init_acc_std**2,
            ]
        ).astype(np.float64)
        return TrackState(mean=mean, cov=cov)

    def predict(self, state: TrackState) -> TrackState:
        """Propagate with the constant-acceleration motion model."""
        if state.dim != self.dim:
            raise ValueError(
                f"CA filter expects dim={self.dim}, got TrackState dim={state.dim}"
            )
        mean = self.F @ state.mean
        cov = self.F @ state.cov @ self.F.T + self.Q
        cov = 0.5 * (cov + cov.T)
        return TrackState(mean=mean, cov=cov)

    def update(
        self,
        state: TrackState,
        z: np.ndarray | None,
        *,
        r_meas: float | None = None,
    ) -> TrackState:
        """Measurement update with online ``Q`` / ``R`` adaptation.

        When ``r_meas`` is provided (detector uncertainty), that step uses
        ``R = r_meas² I`` and skips adapting ``R``; ``Q`` still adapts.
        """
        if state.dim != self.dim:
            raise ValueError(
                f"CA filter expects dim={self.dim}, got TrackState dim={state.dim}"
            )
        if z is None:
            return TrackState(mean=state.mean.copy(), cov=state.cov.copy())

        z = np.asarray(z, dtype=np.float64).reshape(2)
        x_prior = state.mean
        p_prior = state.cov
        innov = z - self.H @ x_prior
        hph = self.H @ p_prior @ self.H.T

        d = self._adapt_gain()

        if r_meas is not None:
            R_step = (float(r_meas) ** 2) * np.eye(2, dtype=np.float64)
        else:
            r_emp = np.outer(innov, innov) - hph
            self.R = ensure_spd(
                (1.0 - d) * self.R + d * r_emp,
                min_eig=self._r_min2,
                max_eig=self._r_max,
            )
            R_step = self.R

        s = hph + R_step
        k_gain = p_prior @ self.H.T @ np.linalg.inv(s)
        x_post = x_prior + k_gain @ innov
        eye = np.eye(self.dim, dtype=np.float64)
        p_post = (eye - k_gain @ self.H) @ p_prior
        p_post = 0.5 * (p_post + p_post.T)

        q_emp = (
            k_gain @ np.outer(innov, innov) @ k_gain.T
            + p_post
            - self.F @ p_post @ self.F.T
        )
        self.Q = ensure_spd(
            (1.0 - d) * self.Q + d * q_emp,
            min_eig=self.q_min,
            max_eig=self._q_max,
        )

        if r_meas is not None:
            self.r_meas = float(r_meas)
        else:
            self.r_meas = float(np.sqrt(0.5 * (self.R[0, 0] + self.R[1, 1])))
        return TrackState(mean=x_post, cov=p_post)

    def gate_window(
        self, state: TrackState, n_sigma: float = 3.0
    ) -> tuple[float, float, float, float]:
        """Axis-aligned search box from position mean ± ``n_sigma`` std."""
        from horus.track.gate import gate_window as _gate_window

        return _gate_window(state, n_sigma=n_sigma)


def build_ca_akf(cfg: dict[str, Any]) -> AdaptiveConstantAccelerationKF:
    """Build a CA adaptive KF from a loaded Horus config (``tracker:``)."""
    tcfg = cfg.get("tracker") or {}
    return AdaptiveConstantAccelerationKF(
        dt_from_cfg(cfg),
        q_pos=float(tcfg.get("q_pos", 1.0)),
        q_vel=float(tcfg.get("q_vel", 10.0)),
        q_acc=float(tcfg.get("q_acc", 50.0)),
        r_meas=float(tcfg.get("r_meas", 5.0)),
        init_pos_std=float(tcfg.get("init_pos_std", 10.0)),
        init_vel_std=float(tcfg.get("init_vel_std", 50.0)),
        init_acc_std=float(tcfg.get("init_acc_std", 100.0)),
        forgetting=float(tcfg.get("akf_forgetting", 0.95)),
        q_min=float(tcfg.get("akf_q_min", 1e-4)),
        r_min=float(tcfg.get("akf_r_min", 0.25)),
        q_max_scale=float(tcfg.get("akf_q_max_scale", 100.0)),
        r_max_scale=float(tcfg.get("akf_r_max_scale", 100.0)),
    )
