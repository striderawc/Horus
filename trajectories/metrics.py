"""Shot metrics from one gravity-only flight.

``measure_shot`` reads ``theta`` from ``fit_flight``. Speed is the launch
speed. Clearance is the ball's height above the tape where the flight crosses
the net, before it returns to the ground.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trajectories.ballistic import G, simulate
from trajectories.court import HALF_DOUBLES, HALF_LENGTH, HALF_SINGLES, net_height

# 1 m/s in mph.
MPS_TO_MPH = 2.2369362920544


@dataclass(frozen=True)
class ShotMetrics:
    """One shot. Lengths are meters. ``landing`` is where the ball meets the ground.

    ``net_xyz`` is None when the flight does not cross the net before landing.
    ``net_wide`` is set when that crossing is outside the posts, and then
    ``net_clearance_m`` is None. A negative clearance would hit the tape.
    ``past_net_m`` is positive on the far side of the net from the contact.
    ``baseline_margin_m`` is positive inside the opposite baseline.
    """

    speed_mps: float
    speed_mph: float
    contact_height_m: float
    apex_height_m: float
    apex_xyz: np.ndarray
    net_ball_height_m: float | None
    net_clearance_m: float | None
    net_wide: bool
    net_xyz: np.ndarray | None
    launch_angle_deg: float
    landing_xyz: np.ndarray
    past_net_m: float
    baseline_margin_m: float
    landing: str
    t_apex: float
    t_net: float | None
    t_landing: float


def measure_shot(theta: np.ndarray) -> ShotMetrics:
    """Speed, height, net clearance, and landing for one fitted flight.

    ``theta`` is ``[X0, Y0, Z0, vx0, vy0, vz0]`` at the contact, in court
    meters and meters per second. The ground intersection is the later root
    of ``Z(t) = 0``.
    """
    theta = np.asarray(theta, dtype=np.float64).reshape(6)
    x0, y0, z0, vx, vy, vz = (float(v) for v in theta)
    speed = float(np.linalg.norm(theta[3:]))
    horizontal = float(np.hypot(vx, vy))
    t_landing = _landing_time(z0, vz)
    t_apex = vz / G if vz > 0.0 else 0.0
    if t_apex > t_landing:
        t_apex = 0.0
    apex = _at(theta, t_apex)
    t_net = _net_time(y0, vy, t_landing)
    net_xyz = None
    net_ball = None
    clearance = None
    wide = False
    if t_net is not None:
        net_xyz = _at(theta, t_net)
        net_ball = float(net_xyz[2])
        tape = net_height(float(net_xyz[0]))
        if np.isfinite(tape):
            clearance = net_ball - tape
        else:
            wide = True
    landing_xyz = _at(theta, t_landing)
    side = 1.0 if y0 >= 0.0 else -1.0
    past_net = -side * float(landing_xyz[1])
    return ShotMetrics(
        speed_mps=speed,
        speed_mph=speed * MPS_TO_MPH,
        contact_height_m=z0,
        apex_height_m=float(apex[2]),
        apex_xyz=apex,
        net_ball_height_m=net_ball,
        net_clearance_m=clearance,
        net_wide=wide,
        net_xyz=net_xyz,
        launch_angle_deg=float(np.degrees(np.arctan2(vz, horizontal))),
        landing_xyz=landing_xyz,
        past_net_m=past_net,
        baseline_margin_m=HALF_LENGTH - past_net,
        landing=_landing_label(side, float(landing_xyz[0]), float(landing_xyz[1])),
        t_apex=float(t_apex),
        t_net=t_net,
        t_landing=float(t_landing),
    )


def _at(theta: np.ndarray, t: float) -> np.ndarray:
    pos, _ = simulate(theta, np.array([t], dtype=np.float64))
    return pos[0]


def _landing_time(z0: float, vz: float) -> float:
    """Later root of ``Z(t) = 0``. A ball already on the ground stays at 0."""
    height = max(z0, 0.0)
    disc = vz * vz + 2.0 * G * height
    t = (vz + float(np.sqrt(max(disc, 0.0)))) / G
    if t < 0.0:
        return 0.0
    return float(t)


def _net_time(y0: float, vy: float, t_landing: float) -> float | None:
    """Time the flight crosses ``Y = 0``, if that happens before the landing."""
    if abs(vy) <= 1e-6:
        return None
    t = -y0 / vy
    if t <= 1e-6 or t >= t_landing - 1e-6:
        return None
    return float(t)


def _landing_label(side: float, x: float, y: float) -> str:
    """``side`` is +1 when the contact is on the near side of the net."""
    if y * side >= -1e-4:
        return "short"
    ax, ay = abs(x), abs(y)
    if ay <= HALF_LENGTH and ax <= HALF_SINGLES:
        return "singles"
    if ay <= HALF_LENGTH and ax <= HALF_DOUBLES:
        return "doubles alley"
    if ax <= HALF_DOUBLES:
        return "long"
    return "wide"
