"""Six-parameter ballistic fit in court coordinates.

For one uninterrupted flight, time in seconds from the first sample:

    X(t) = X0 + vx0 t
    Y(t) = Y0 + vy0 t
    Z(t) = Z0 + vz0 t - 0.5 g t^2

``theta = [X0, Y0, Z0, vx0, vy0, vz0]``. The residual is in image pixels
after projection through the camera. ``fit_shot`` pins the bounce to the
ground and keeps the contact height in a playable range. The contact is fit
from that shot's own pixels. An airborne contact starts on the camera ray
through the first measurement, not at the ground intersection of that ray.
When the flight from the bounce to the next hit is passed in, it leaves the
bounce with the vertical velocity reversed. Horizontal velocity keeps its
direction through the bounce and does not speed up. It can slow, so the
continuation stays on the frames that were tracked. That continuation does
not move the contact. ``fit_shots`` starts the next shot at the hit.
A hit may change the horizontal velocity. ``fit_rally`` adds bounce
constraints across several flights.
``quadratic_drag`` is available and stays off unless a caller passes it
as ``acceleration``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from trajectories.camera import Camera
from trajectories.court import net_height

G = 9.80665
FPS = 30.0
# A meter (or a meter per second) of constraint error, counted in pixel units.
CONSTRAINT_PX = 30.0
# Pixels charged per meter the ball sits below the tape.
CLEARANCE_PX = 500.0
# A crossing has to clear the tape by this much, so the margin stays positive.
CLEARANCE_MARGIN = 0.01
# Guess for an endpoint that is not on the court.
START_HEIGHT = 1.0
# A shot leaves the racquet in this height range. Six meters is not a contact.
CONTACT_Z_MIN = 0.2
CONTACT_Z_MAX = 3.0
# Vertical speed after the bounce, as a fraction of the speed coming in.
RESTITUTION = 0.75
# Least horizontal speed kept through a bounce, as a fraction of the speed
# that arrived. The ball does not speed up and it does not turn. It can slow
# by more than a tenth: a hard serve still going at 90% is past the baseline
# before the return, off the frames that were tracked.
BOUNCE_KEEP = 0.05
# Heights tried when placing an airborne contact on the camera ray.
_RAY_HEIGHTS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)


@dataclass
class FlightFit:
    theta: np.ndarray
    uv_hat: np.ndarray
    xyz: np.ndarray
    rmse_px: float
    success: bool
    cost: float
    message: str
    follow_theta: np.ndarray | None = None
    follow_xyz: np.ndarray | None = None
    follow_uv_hat: np.ndarray | None = None
    follow_rmse_px: float | None = None
    hit_xyz: np.ndarray | None = None
    hit_velocity: np.ndarray | None = None


@dataclass
class Segment:
    """One flight. ``t`` is seconds from this flight's own t = 0."""

    uv: np.ndarray
    t: np.ndarray
    weights: np.ndarray | None = None
    bounce_end: bool = False


@dataclass
class RallyFit:
    thetas: list[np.ndarray]
    rmse_px: float
    success: bool


def gravity(pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
    """Acceleration with gravity only. ``pos`` and ``vel`` are unused."""
    return np.array([0.0, 0.0, -G], dtype=np.float64)


def quadratic_drag(pos: np.ndarray, vel: np.ndarray, k: float = 0.02) -> np.ndarray:
    """Gravity plus quadratic drag. Pass this as ``acceleration`` to turn it on.

    ``k`` is in 1/m. The default is a tennis ball in still air and is not used
    by ``fit_flight`` or ``fit_rally`` unless the caller passes this function.
    """
    vel = np.asarray(vel, dtype=np.float64)
    speed = float(np.linalg.norm(vel))
    return gravity(pos, vel) - k * speed * vel


def ballistic(theta: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Court positions for the gravity-only flight. ``t`` is seconds."""
    pos, _ = simulate(theta, t, None)
    return pos


def simulate(
    theta: np.ndarray,
    t: np.ndarray,
    acceleration=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Positions and velocities at ``t``. ``theta`` is the state at t = 0.

    ``acceleration`` is ``None`` for the closed-form gravity model. Any other
    callable is integrated from t = 0.
    """
    theta = np.asarray(theta, dtype=np.float64).reshape(6)
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    if acceleration is None:
        x0, y0, z0, vx, vy, vz = theta
        pos = np.column_stack(
            [x0 + vx * t, y0 + vy * t, z0 + vz * t - 0.5 * G * t * t]
        )
        vel = np.column_stack(
            [np.full(len(t), vx), np.full(len(t), vy), vz - G * t]
        )
        return pos, vel
    return _integrate(theta, t, acceleration)


def fit_flight(
    uv: np.ndarray,
    t: np.ndarray,
    camera: Camera,
    homography: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    bounce_end: bool = False,
    bounce_start: bool = False,
    acceleration=None,
) -> FlightFit:
    """Fit ``theta`` so the projected flight matches ``uv``.

    ``weights`` are measurement standard deviations in pixels. Each residual
    is divided by that std. The initial guess back-projects the first and last
    samples through ``homography``. A bounce endpoint starts at Z = 0; any
    other endpoint starts at a low height.
    """
    uv, t, weights = _finite(uv, t, weights)
    if len(uv) < 4:
        raise ValueError("fit needs at least 4 observations")
    theta0 = initial_theta(
        uv,
        t,
        homography,
        bounce_end=bounce_end,
        bounce_start=bounce_start,
        camera=camera,
    )
    point_w = _image_residual_weights(weights, len(uv))
    solved = least_squares(
        lambda theta: _image_residual(theta, t, uv, camera, point_w, acceleration),
        theta0,
        bounds=_bounds(1),
        method="trf",
        max_nfev=300,
    )
    pos, _ = simulate(solved.x, t, acceleration)
    pred = camera.project(pos)
    return FlightFit(
        theta=solved.x.copy(),
        uv_hat=pred,
        xyz=pos,
        rmse_px=_rmse(pred, uv),
        success=bool(solved.success),
        cost=float(solved.cost),
        message=_message(solved.message),
    )


def fit_shot(
    uv: np.ndarray,
    t: np.ndarray,
    camera: Camera,
    homography: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    contact_z_min: float = CONTACT_Z_MIN,
    contact_z_max: float = CONTACT_Z_MAX,
    acceleration=None,
    follow_uv: np.ndarray | None = None,
    follow_t: np.ndarray | None = None,
    follow_weights: np.ndarray | None = None,
    t_bounce: float | None = None,
    restitution: float = RESTITUTION,
    origin: np.ndarray | None = None,
    t_hit: float | None = None,
) -> FlightFit:
    """Fit one shot that ends on the bounce.

    ``Z(t_bounce) = 0`` fixes ``vz0``. ``t_bounce`` is seconds on the same
    clock as ``t``; it defaults to the last sample. The contact height stays inside
    ``contact_z_min`` and ``contact_z_max``. A flight that crosses the net
    between the posts has to clear the tape.

    ``follow_uv`` is the flight that leaves that bounce and runs to the next
    hit. ``follow_t`` is seconds from the bounce. The shot is fit first, from
    its own pixels. The continuation then starts at that bounce. Vertical
    velocity reverses by ``restitution``. Horizontal velocity stays within a
    small slowdown of the velocity that arrived, and it does not move the
    contact. ``origin``, when set, is the previous hit: this shot starts there.
    ``t_hit`` is seconds from the bounce to the next contact, and is stored
    on the result as ``hit_xyz``.
    """
    if contact_z_min < 0.0 or contact_z_max <= contact_z_min:
        raise ValueError("contact height bounds must satisfy 0 <= min < max")
    if not 0.0 < restitution <= 1.0:
        raise ValueError("restitution must be in (0, 1]")
    uv, t, weights = _finite(uv, t, weights)
    if len(uv) < 4:
        raise ValueError("fit needs at least 4 observations")
    t_end = float(t[-1] if t_bounce is None else t_bounce)
    if t_end + 1e-6 < float(t[-1]):
        raise ValueError("the bounce has to be at or after the last sample")
    if t_end <= 1e-4:
        raise ValueError("the bounce has to be after the first sample")
    follow = _follow_samples(follow_uv, follow_t, follow_weights)
    if t_hit is not None and float(t_hit) <= 1e-4:
        raise ValueError("the next hit has to be after the bounce")
    point_w = _image_residual_weights(weights, len(uv))
    if origin is None:
        theta, solved = _fit_contact(
            uv,
            t,
            t_end,
            camera,
            homography,
            point_w,
            contact_z_min,
            contact_z_max,
            acceleration,
        )
    else:
        theta, solved = _fit_from_origin(
            np.asarray(origin, dtype=np.float64).reshape(3),
            uv,
            t,
            t_end,
            camera,
            homography,
            point_w,
            acceleration,
        )
    pos, _ = simulate(theta, t, acceleration)
    pred = camera.project(pos)
    follow_theta, follow_xyz, follow_uv_hat, follow_rmse, hit = _fit_continuation(
        theta,
        t_end,
        follow,
        None if t_hit is None else float(t_hit),
        camera,
        restitution,
        acceleration,
    )
    return FlightFit(
        theta=theta,
        uv_hat=pred,
        xyz=pos,
        rmse_px=_rmse(pred, uv),
        success=bool(solved.success),
        cost=float(solved.cost),
        message=_message(solved.message),
        follow_theta=follow_theta,
        follow_xyz=follow_xyz,
        follow_uv_hat=follow_uv_hat,
        follow_rmse_px=follow_rmse,
        hit_xyz=None if hit is None else hit[0].copy(),
        hit_velocity=None if hit is None else hit[1].copy(),
    )


@dataclass
class ShotInput:
    """One shot, from the hit to the bounce, plus the flight on to the next hit.

    ``t`` and ``t_bounce`` are seconds from this shot's hit. ``follow_t`` and
    ``t_hit`` are seconds from the bounce. ``t_hit`` is the next shot's
    contact. Set it to carry this shot's hit position into that next shot.
    """

    uv: np.ndarray
    t: np.ndarray
    t_bounce: float
    weights: np.ndarray | None = None
    follow_uv: np.ndarray | None = None
    follow_t: np.ndarray | None = None
    follow_weights: np.ndarray | None = None
    t_hit: float | None = None


def fit_shots(
    shots: list[ShotInput],
    camera: Camera,
    homography: np.ndarray,
    *,
    contact_z_min: float = CONTACT_Z_MIN,
    contact_z_max: float = CONTACT_Z_MAX,
    restitution: float = RESTITUTION,
    acceleration=None,
) -> list[FlightFit]:
    """Fit each shot from its own pixels, then start the next shot at the hit.

    A shot with ``t_hit`` ends at the next shot's contact. That position is
    the next shot's start. The flight already fit is left as it is. The hit
    may change horizontal velocity. Through the bounce, horizontal velocity
    keeps the incoming direction and can slow only down to ``BOUNCE_KEEP``.
    """
    if contact_z_min < 0.0 or contact_z_max <= contact_z_min:
        raise ValueError("contact height bounds must satisfy 0 <= min < max")
    if not 0.0 < restitution <= 1.0:
        raise ValueError("restitution must be in (0, 1]")
    if not shots:
        return []
    prepared = [_prepare_shot(shot) for shot in shots]
    fitted = []
    carried = None
    for shot in prepared:
        kwargs = {
            "t_bounce": shot.t_bounce,
            "restitution": restitution,
            "acceleration": acceleration,
            "contact_z_min": contact_z_min,
            "contact_z_max": contact_z_max,
            "origin": None if carried is None else carried.copy(),
            "t_hit": shot.t_hit,
        }
        if shot.follow_uv is not None:
            kwargs.update(
                follow_uv=shot.follow_uv,
                follow_t=shot.follow_t,
                follow_weights=shot.follow_weights,
            )
        fit = fit_shot(shot.uv, shot.t, camera, homography, shot.weights, **kwargs)
        fitted.append(fit)
        carried = None if fit.hit_xyz is None or shot.t_hit is None else fit.hit_xyz.copy()
    return fitted


def _prepare_shot(shot: ShotInput) -> ShotInput:
    """Drop bad samples and reject a hit that lands before the bounce."""
    uv, t, weights = _finite(shot.uv, shot.t, shot.weights)
    if len(uv) < 4:
        raise ValueError("each shot needs at least 4 observations")
    t_bounce = float(shot.t_bounce)
    if t_bounce + 1e-6 < float(t[-1]):
        raise ValueError("the bounce has to be at or after the last sample")
    if t_bounce <= 1e-4:
        raise ValueError("the bounce has to be after the hit")
    follow_uv = follow_t = follow_w = None
    if shot.follow_uv is not None or shot.follow_t is not None:
        follow = _follow_samples(shot.follow_uv, shot.follow_t, shot.follow_weights)
        if follow is not None:
            follow_uv, follow_t, follow_w = follow
    t_hit = None if shot.t_hit is None else float(shot.t_hit)
    if t_hit is not None and t_hit <= 1e-4:
        raise ValueError("the next hit has to be after the bounce")
    return ShotInput(
        uv=uv,
        t=t,
        t_bounce=t_bounce,
        weights=weights,
        follow_uv=follow_uv,
        follow_t=follow_t,
        follow_weights=follow_w,
        t_hit=t_hit,
    )


def _fit_contact(
    uv, t, t_end, camera, homography, point_w, zmin, zmax, acceleration
):
    """Contact position and velocity from this shot's pixels."""
    theta0 = initial_theta(
        uv, t, homography, bounce_end=True, bounce_start=False, camera=camera
    )
    lower = np.array([-40.0, -40.0, zmin, -90.0, -90.0], dtype=np.float64)
    upper = np.array([40.0, 40.0, zmax, 90.0, 90.0], dtype=np.float64)
    packed0 = np.clip(theta0[:5], lower, upper)

    def residual(packed: np.ndarray) -> np.ndarray:
        theta = _with_bounce(packed, t_end)
        return np.concatenate(
            [
                _image_residual(theta, t, uv, camera, point_w, acceleration),
                _clearance_residual(theta, t_end),
            ]
        )

    solved = least_squares(
        residual, packed0, bounds=(lower, upper), method="trf", max_nfev=300
    )
    return _with_bounce(solved.x, t_end), solved


def _fit_from_origin(origin, uv, t, t_end, camera, homography, point_w, acceleration):
    """Velocity only. The shot starts at ``origin``, the previous hit."""
    end = _ground(uv[-1], homography)
    duration = max(float(t_end), 1e-3)
    guess = np.array(
        [(end[0] - origin[0]) / duration, (end[1] - origin[1]) / duration],
        dtype=np.float64,
    )
    lower = np.array([-90.0, -90.0], dtype=np.float64)
    upper = np.array([90.0, 90.0], dtype=np.float64)
    guess = np.clip(guess, lower, upper)

    def residual(vel: np.ndarray) -> np.ndarray:
        theta = _with_bounce(np.concatenate([origin, vel]), t_end)
        return np.concatenate(
            [
                _image_residual(theta, t, uv, camera, point_w, acceleration),
                _clearance_residual(theta, t_end),
            ]
        )

    solved = least_squares(
        residual, guess, bounds=(lower, upper), method="trf", max_nfev=200
    )
    return _with_bounce(np.concatenate([origin, solved.x]), t_end), solved


def _fit_continuation(theta, t_end, follow, t_hit, camera, restitution, acceleration):
    """Bounce-to-hit flight. The shot's contact stays where it was fit."""
    if follow is None:
        return None, None, None, None, None
    follow_uv, follow_t, follow_w = follow
    point_w = _image_residual_weights(follow_w, len(follow_uv))

    def residual(factor: np.ndarray) -> np.ndarray:
        launched = _bounce_launch(
            theta,
            t_end,
            restitution,
            acceleration,
            vx=float(theta[3]) * float(factor[0]),
            vy=float(theta[4]) * float(factor[0]),
        )
        return np.concatenate(
            [
                _image_residual(
                    launched, follow_t, follow_uv, camera, point_w, acceleration
                ),
                _clearance_residual(launched, float(follow_t[-1])),
                _above_ground(launched, follow_t, acceleration),
            ]
        )

    solved = least_squares(
        residual,
        np.array([0.5]),
        bounds=([BOUNCE_KEEP], [1.0]),
        method="trf",
        max_nfev=40,
    )
    keep = float(solved.x[0])
    launched = _bounce_launch(
        theta,
        t_end,
        restitution,
        acceleration,
        vx=float(theta[3]) * keep,
        vy=float(theta[4]) * keep,
    )
    follow_xyz, _ = simulate(launched, follow_t, acceleration)
    follow_uv_hat = camera.project(follow_xyz)
    follow_rmse = _rmse(follow_uv_hat, follow_uv)
    hit = None
    if t_hit is not None:
        pos, vel = simulate(launched, np.array([t_hit]), acceleration)
        hit = (pos[0], vel[0])
    return launched, follow_xyz, follow_uv_hat, follow_rmse, hit


def _with_bounce(packed: np.ndarray, t_end: float) -> np.ndarray:
    """Five free values, with ``vz`` set so the ball is on the ground at ``t_end``."""
    x0, y0, z0, vx, vy = (float(v) for v in np.asarray(packed, dtype=np.float64).reshape(5))
    vz = -z0 / t_end + 0.5 * G * t_end
    return np.array([x0, y0, z0, vx, vy, vz], dtype=np.float64)


def _bounce_launch(
    theta: np.ndarray,
    t_bounce: float,
    restitution: float,
    acceleration=None,
    vx: float | None = None,
    vy: float | None = None,
) -> np.ndarray:
    """State at the bounce for the flight that continues to the next hit.

    Position is the shot's position at ``t_bounce``. Vertical velocity is
    reversed and scaled by ``restitution``. ``vx`` and ``vy`` override the
    horizontal velocity; omitted, they stay equal to the incoming velocity.
    """
    pos, vel = simulate(theta, np.array([t_bounce], dtype=np.float64), acceleration)
    return np.array(
        [
            pos[0, 0],
            pos[0, 1],
            0.0,
            vel[0, 0] if vx is None else vx,
            vel[0, 1] if vy is None else vy,
            -restitution * vel[0, 2],
        ],
        dtype=np.float64,
    )


def _follow_samples(uv, t, weights):
    """Samples of the bounce-to-hit flight, or None when there are too few."""
    if uv is None and t is None and weights is None:
        return None
    if uv is None or t is None:
        raise ValueError("follow_uv and follow_t are both required")
    uv, t, weights = _finite(uv, t, weights)
    if len(uv) < 2:
        return None
    if float(t[0]) < -1e-6:
        raise ValueError("follow times are seconds from the bounce and cannot be negative")
    if float(t[-1]) <= 1e-4:
        raise ValueError("the hit has to be after the bounce")
    return uv, t, weights


def _above_ground(theta, t, acceleration) -> np.ndarray:
    """Zero while the ball is on or above the court."""
    pos, _ = simulate(theta, t, acceleration)
    below = np.maximum(0.0, -pos[:, 2] - 0.01)
    return below * CONSTRAINT_PX


def _clearance_residual(theta: np.ndarray, t_end: float) -> np.ndarray:
    """Zero unless the ball crosses between the posts below the tape."""
    x0, y0, z0, vx, vy, vz = (float(v) for v in theta)
    y_end = y0 + vy * t_end
    short = 0.0
    if y0 * y_end < 0.0 and abs(vy) > 1e-8:
        t_net = -y0 / vy
        if 1e-6 < t_net < t_end - 1e-6:
            tape = net_height(x0 + vx * t_net)
            if np.isfinite(tape):
                z_net = z0 + vz * t_net - 0.5 * G * t_net * t_net
                short = max(0.0, tape + CLEARANCE_MARGIN - z_net)
    return np.array([short * CLEARANCE_PX], dtype=np.float64)


def fit_rally(
    segments: list[Segment],
    camera: Camera,
    homography: np.ndarray,
    *,
    restitution: float = 0.75,
    acceleration=None,
) -> RallyFit:
    """Fit one ``theta`` per flight.

    A segment with ``bounce_end`` shares its end position with the next
    flight's start, forces both heights to 0, keeps horizontal velocity
    continuous, and reflects vertical velocity by ``restitution``.
    """
    if not segments:
        raise ValueError("fit_rally needs at least one flight")
    packed = []
    cleaned = []
    for i, segment in enumerate(segments):
        uv, t, weights = _finite(segment.uv, segment.t, segment.weights)
        if len(uv) < 4:
            raise ValueError("each flight needs at least 4 observations")
        cleaned.append(Segment(uv, t, weights, segment.bounce_end))
        bounce_start = i > 0 and segments[i - 1].bounce_end
        packed.append(
            initial_theta(
                uv,
                t,
                homography,
                bounce_end=segment.bounce_end,
                bounce_start=bounce_start,
                camera=camera,
            )
        )
    theta0 = np.concatenate(packed)

    def residual(packed_theta: np.ndarray) -> np.ndarray:
        chunks = []
        ends = []
        starts = []
        for i, segment in enumerate(cleaned):
            theta = packed_theta[i * 6 : (i + 1) * 6]
            point_w = _image_residual_weights(segment.weights, len(segment.uv))
            chunks.append(
                _image_residual(theta, segment.t, segment.uv, camera, point_w, acceleration)
            )
            t_end = np.array([float(segment.t[-1])], dtype=np.float64)
            pos_end, vel_end = simulate(theta, t_end, acceleration)
            pos0, vel0 = simulate(theta, np.array([0.0]), acceleration)
            ends.append((pos_end[0], vel_end[0]))
            starts.append((pos0[0], vel0[0]))
        extra = []
        for i, segment in enumerate(cleaned[:-1]):
            if not segment.bounce_end:
                continue
            pos_a, vel_a = ends[i]
            pos_b, vel_b = starts[i + 1]
            extra.extend((pos_a[:2] - pos_b[:2]) * CONSTRAINT_PX)
            extra.append(pos_a[2] * CONSTRAINT_PX)
            extra.append(pos_b[2] * CONSTRAINT_PX)
            extra.append((vel_b[0] - vel_a[0]) * CONSTRAINT_PX)
            extra.append((vel_b[1] - vel_a[1]) * CONSTRAINT_PX)
            extra.append((vel_b[2] + restitution * vel_a[2]) * CONSTRAINT_PX)
        if extra:
            chunks.append(np.asarray(extra, dtype=np.float64))
        return np.concatenate(chunks)

    solved = least_squares(
        residual,
        theta0,
        bounds=_bounds(len(cleaned)),
        method="trf",
        max_nfev=400,
    )
    thetas = [solved.x[i * 6 : (i + 1) * 6].copy() for i in range(len(cleaned))]
    errors = []
    for theta, segment in zip(thetas, cleaned):
        pos, _ = simulate(theta, segment.t, acceleration)
        errors.append(np.sum((camera.project(pos) - segment.uv) ** 2, axis=1))
    rmse = float(np.sqrt(np.mean(np.concatenate(errors))))
    return RallyFit(thetas=thetas, rmse_px=rmse, success=bool(solved.success))


def initial_theta(
    uv: np.ndarray,
    t: np.ndarray,
    homography: np.ndarray,
    *,
    bounce_end: bool,
    bounce_start: bool = False,
    camera: Camera | None = None,
) -> np.ndarray:
    """First and last samples, then a finite difference.

    A bounce endpoint is the ground intersection of its pixel. An airborne
    endpoint is the point on the camera ray through that pixel. The ground
    intersection of a serve pixel is near the net, and lifting it a meter
    is not the contact.
    """
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    if bounce_start or camera is None:
        xy0 = _ground(uv[0], homography)
        z0 = 0.0 if bounce_start else START_HEIGHT
    else:
        xy0, z0 = _airborne(uv, t, camera, homography, bounce_end)
    if bounce_end or camera is None:
        xy1 = _ground(uv[-1], homography)
        z1 = 0.0 if bounce_end else START_HEIGHT
    else:
        landed = _on_ray(camera, uv[-1], START_HEIGHT)
        if landed is None:
            xy1 = _ground(uv[-1], homography)
            z1 = START_HEIGHT
        else:
            xy1 = landed[:2]
            z1 = START_HEIGHT
    duration = float(t[-1] - t[0])
    if duration <= 1e-6:
        duration = 1e-3
    vx = (xy1[0] - xy0[0]) / duration
    vy = (xy1[1] - xy0[1]) / duration
    vz = (z1 - z0) / duration + 0.5 * G * duration
    theta = np.array([xy0[0], xy0[1], z0, vx, vy, vz], dtype=np.float64)
    lower, upper = _bounds(1)
    return np.clip(theta, lower, upper)


def _airborne(uv, t, camera, homography, bounce_end: bool):
    """Court ``(x, y)`` and height of the first sample, on its camera ray.

    The height is the one whose straight flight to the last sample projects
    closest to the measurements.
    """
    if bounce_end:
        end = np.array([*_ground(uv[-1], homography), 0.0], dtype=np.float64)
    else:
        end = _on_ray(camera, uv[-1], START_HEIGHT)
        if end is None:
            end = np.array([*_ground(uv[-1], homography), START_HEIGHT], dtype=np.float64)
    duration = max(float(t[-1] - t[0]), 1e-3)
    best = None
    for height in _RAY_HEIGHTS:
        start = _on_ray(camera, uv[0], float(height))
        if start is None:
            continue
        vx = (end[0] - start[0]) / duration
        vy = (end[1] - start[1]) / duration
        vz = (end[2] - float(height)) / duration + 0.5 * G * duration
        theta = np.array([start[0], start[1], float(height), vx, vy, vz], dtype=np.float64)
        pos, _ = simulate(theta, t)
        err = _rmse(camera.project(pos), uv)
        if best is None or err < best[0]:
            best = (err, start[:2].copy(), float(height))
    if best is None:
        return _ground(uv[0], homography), START_HEIGHT
    return best[1], best[2]


def _on_ray(camera: Camera, uv: np.ndarray, z: float) -> np.ndarray | None:
    """Court point where the pixel's ray crosses height ``z``. None behind the camera."""
    pix = np.asarray(uv, dtype=np.float64).reshape(2)
    direction = camera.rotation().T @ np.linalg.solve(
        camera.K(), np.array([pix[0], pix[1], 1.0], dtype=np.float64)
    )
    direction = direction.copy()
    direction[1] *= -1.0
    if abs(float(direction[2])) < 1e-8:
        return None
    origin = camera.center()
    scale = (float(z) - float(origin[2])) / float(direction[2])
    if scale <= 0.0:
        return None
    point = origin + scale * direction
    if not np.isfinite(point).all():
        return None
    return point


def _ground(uv: np.ndarray, homography: np.ndarray) -> np.ndarray:
    point = np.asarray(uv, dtype=np.float64).reshape(1, 1, 2)
    xy = cv2.perspectiveTransform(point, np.linalg.inv(homography)).reshape(2)
    if not np.isfinite(xy).all():
        return np.zeros(2, dtype=np.float64)
    return xy


def _image_residual(theta, t, uv, camera, point_w, acceleration) -> np.ndarray:
    pos, _ = simulate(theta, t, acceleration)
    err = (camera.project(pos) - uv) * point_w[:, None]
    bad = np.repeat(camera.depth(pos) <= 0.05, 2)
    flat = err.ravel()
    flat[bad | ~np.isfinite(flat)] = 1.0e3
    return flat


def _point_weights(weights: np.ndarray) -> np.ndarray:
    sigma = np.asarray(weights, dtype=np.float64).reshape(-1)
    finite = np.isfinite(sigma) & (sigma > 0)
    fill = float(np.median(sigma[finite])) if np.any(finite) else 1.0
    sigma = np.where(finite, sigma, fill)
    return 1.0 / sigma


def _finite(uv, t, weights):
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    if len(uv) != len(t):
        raise ValueError("uv and t must be the same length")
    keep = np.isfinite(uv).all(axis=1) & np.isfinite(t)
    uv = uv[keep]
    t = t[keep]
    if weights is None:
        point_w_src = None
    else:
        point_w_src = np.asarray(weights, dtype=np.float64).reshape(-1)[keep]
    return uv, t, point_w_src


def _image_residual_weights(weights, n: int) -> np.ndarray:
    if weights is None:
        return np.ones(n, dtype=np.float64)
    return _point_weights(weights)


def _bounds(n_flights: int) -> tuple[np.ndarray, np.ndarray]:
    lower = np.array([-40.0, -40.0, 0.0, -90.0, -90.0, -90.0], dtype=np.float64)
    upper = np.array([40.0, 40.0, 12.0, 90.0, 90.0, 90.0], dtype=np.float64)
    if n_flights == 1:
        return lower, upper
    return np.tile(lower, n_flights), np.tile(upper, n_flights)


def _rmse(pred: np.ndarray, uv: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((pred - uv) ** 2, axis=1))))


def _message(message) -> str:
    if isinstance(message, bytes):
        return message.decode()
    return str(message)


def _integrate(theta: np.ndarray, t: np.ndarray, acceleration):
    """Integrate from t = 0. Samples must be at or after 0."""
    order = np.argsort(t, kind="mergesort")
    pos = theta[:3].copy()
    vel = theta[3:].copy()
    cursor = 0.0
    out_p = np.empty((len(t), 3), dtype=np.float64)
    out_v = np.empty((len(t), 3), dtype=np.float64)
    for idx in order:
        target = float(t[idx])
        if target < -1e-9:
            raise ValueError("drag integration expects times at or after 0")
        target = max(target, 0.0)
        while cursor < target - 1e-12:
            dt = min(0.005, target - cursor)
            pos, vel = _rk4(pos, vel, dt, acceleration)
            cursor += dt
        out_p[idx] = pos
        out_v[idx] = vel
    return out_p, out_v


def _rk4(pos, vel, dt, acceleration):
    def acc(p, v):
        return np.asarray(acceleration(p, v), dtype=np.float64)

    k1_v, k1_a = vel, acc(pos, vel)
    k2_v = vel + 0.5 * dt * k1_a
    k2_a = acc(pos + 0.5 * dt * k1_v, k2_v)
    k3_v = vel + 0.5 * dt * k2_a
    k3_a = acc(pos + 0.5 * dt * k2_v, k3_v)
    k4_v = vel + dt * k3_a
    k4_a = acc(pos + dt * k3_v, k4_v)
    pos2 = pos + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
    vel2 = vel + (dt / 6.0) * (k1_a + 2.0 * k2_a + 2.0 * k3_a + k4_a)
    return pos2, vel2
