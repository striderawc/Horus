"""Adaptive constant-velocity Kalman filter (learns Q and R online)."""

from __future__ import annotations

from typing import Any

import numpy as np

from horus.track.common import dt_from_cfg, ensure_spd
from horus.track.kf import ConstantVelocityKF
from horus.track.state import TrackState


class AdaptiveConstantVelocityKF(ConstantVelocityKF):
    """CV Kalman filter with Sage-Husa style adaptation of ``Q`` and ``R``.

    On each measurement update, innovations drive a forgetting-factor average of
    the measurement-noise covariance ``R`` and process-noise covariance ``Q``.
    Eigenvalue floors/ceilings keep both SPD. Same public API as
    ``ConstantVelocityKF`` (``initiate`` / ``predict`` / ``update`` /
    ``gate_window``).
    """

    def __init__(
        self,
        dt: float,
        *,
        q_pos: float = 1.0,
        q_vel: float = 10.0,
        r_meas: float = 5.0,
        init_pos_std: float = 10.0,
        init_vel_std: float = 50.0,
        forgetting: float = 0.95,
        q_min: float = 1e-4,
        r_min: float = 0.25,
        q_max_scale: float = 100.0,
        r_max_scale: float = 100.0,
    ) -> None:
        super().__init__(
            dt,
            q_pos=q_pos,
            q_vel=q_vel,
            r_meas=r_meas,
            init_pos_std=init_pos_std,
            init_vel_std=init_vel_std,
        )
        if not (0.0 < forgetting < 1.0):
            raise ValueError(f"forgetting must be in (0, 1), got {forgetting}")
        if q_min <= 0 or r_min <= 0:
            raise ValueError("q_min and r_min must be positive")
        if q_max_scale < 1.0 or r_max_scale < 1.0:
            raise ValueError("q_max_scale and r_max_scale must be >= 1")

        self.forgetting = float(forgetting)
        self.q_min = float(q_min)
        self.r_min = float(r_min)
        self.q_max_scale = float(q_max_scale)
        self.r_max_scale = float(r_max_scale)

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
        """Sage-Husa weight ``d_k = (1-b) / (1-b^k)``."""
        self._update_count += 1
        b = self.forgetting
        k = self._update_count
        return float((1.0 - b) / (1.0 - b**k))

    def initiate(self, z: np.ndarray) -> TrackState:
        """Start a track and reset learned ``Q`` / ``R`` to their priors."""
        self._reset_noise()
        return super().initiate(z)

    def update(
        self,
        state: TrackState,
        z: np.ndarray | None,
        *,
        r_meas: float | None = None,
    ) -> TrackState:
        """Measurement update with online ``Q`` / ``R`` adaptation.

        ``state`` is assumed to be the *predicted* prior from ``predict``.
        ``z=None`` skips both the Kalman update and noise learning.

        When ``r_meas`` is provided (detector uncertainty), that step uses
        ``R = r_meas² I`` and skips adapting ``R``; ``Q`` still adapts.
        """
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
            # --- adapt R from innovation covariance matching ---
            r_emp = np.outer(innov, innov) - hph
            self.R = ensure_spd(
                (1.0 - d) * self.R + d * r_emp,
                min_eig=self._r_min2,
                max_eig=self._r_max,
            )
            R_step = self.R

        # --- standard KF update ---
        s = hph + R_step
        k_gain = p_prior @ self.H.T @ np.linalg.inv(s)
        x_post = x_prior + k_gain @ innov
        i4 = np.eye(4, dtype=np.float64)
        p_post = (i4 - k_gain @ self.H) @ p_prior
        p_post = 0.5 * (p_post + p_post.T)

        # --- adapt Q (Sage-Husa) ---
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

        # Keep r_meas in sync for callers / notebook noise defaults
        if r_meas is not None:
            self.r_meas = float(r_meas)
        else:
            self.r_meas = float(np.sqrt(0.5 * (self.R[0, 0] + self.R[1, 1])))
        return TrackState(mean=x_post, cov=p_post)


def build_akf(cfg: dict[str, Any]) -> AdaptiveConstantVelocityKF:
    """Build an adaptive KF from a loaded Horus config (``tracker:``)."""
    tcfg = cfg.get("tracker") or {}
    return AdaptiveConstantVelocityKF(
        dt_from_cfg(cfg),
        q_pos=float(tcfg.get("q_pos", 1.0)),
        q_vel=float(tcfg.get("q_vel", 10.0)),
        r_meas=float(tcfg.get("r_meas", 5.0)),
        init_pos_std=float(tcfg.get("init_pos_std", 10.0)),
        init_vel_std=float(tcfg.get("init_vel_std", 50.0)),
        forgetting=float(tcfg.get("akf_forgetting", 0.95)),
        q_min=float(tcfg.get("akf_q_min", 1e-4)),
        r_min=float(tcfg.get("akf_r_min", 0.25)),
        q_max_scale=float(tcfg.get("akf_q_max_scale", 100.0)),
        r_max_scale=float(tcfg.get("akf_r_max_scale", 100.0)),
    )
