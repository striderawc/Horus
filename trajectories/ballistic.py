"""Six-parameter ballistic fit in court coordinates.

``theta = [X0, Y0, Z0, vx0, vy0, vz0]`` at the contact. ``FlightRules`` is
the model: drag, whether the bounce is on the ground, contact height, and
how the ball leaves a bounce. ``fit_shots`` stores those rules on the fit.
Playing the fit back uses that stored model, so the printed path is the one
that was fit.

``drag_k = 0`` is the closed-form parabola

    X(t) = X0 + vx0 t
    Y(t) = Y0 + vy0 t
    Z(t) = Z0 + vz0 t - 0.5 g t^2

A positive ``drag_k`` integrates gravity plus ``-k |v| v`` on every
component. The residual is image pixels after projection through the camera.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np
from scipy.optimize import least_squares, root_scalar

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
# Least horizontal speed kept through a bounce, as a fraction of the speed
# that arrived. The ball does not speed up and it does not turn. It can slow
# by more than a tenth: a hard serve still going at 90% is past the baseline
# before the return, off the frames that were tracked.
BOUNCE_KEEP = 0.05
# Heights tried when placing an airborne contact on the camera ray.
_RAY_HEIGHTS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
# Pixels charged per meter of a landing or contact-height miss. Large enough
# that the flight stays on the court before it chases the image.
HEIGHT_PX = 5000.0
# Pixels charged per unit of horizontal speed kept outside the bounce band.
# A keep 0.3 outside the band costs 600 px, so it moves unless the image
# residual is larger than that.
BOUNCE_BAND_PX = 2000.0


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
    rules: FlightRules | None = None

    def integrate(self, t, *, follow: bool = False):
        """Position and velocity of the model that was fit.

        ``follow`` is the flight that leaves the bounce. The acceleration is
        the one stored on this fit.
        """
        theta = self.follow_theta if follow else self.theta
        if theta is None:
            raise ValueError("this fit has no bounce-to-hit flight")
        acceleration = None if self.rules is None else self.rules.acceleration()
        return simulate(theta, np.asarray(t, dtype=np.float64), acceleration)

    def path(self, t, *, follow: bool = False) -> np.ndarray:
        """Court positions of the model that was fit."""
        pos, _ = self.integrate(t, follow=follow)
        return pos


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
    """Gravity plus quadratic drag on every velocity component.

    ``k`` is in 1/m. The drag term is ``-k |v| v``. ``FlightRules.drag_k``
    is the value a shot fit uses.
    """
    vel = np.asarray(vel, dtype=np.float64)
    speed = float(np.linalg.norm(vel))
    return gravity(pos, vel) - k * speed * vel


@dataclass(frozen=True)
class FlightRules:
    """The trajectory model. Change this, refit, and the printout follows.

    ``drag_k`` is quadratic drag in 1/m. ``0`` is the closed-form gravity
    flight. ``pin_bounce_z`` solves vertical speed so the ball is on the
    ground at the bounce. Set both contact heights, or set both to ``None``
    to leave contact height free. ``contact_z_max`` is the serve. Every later
    hit is capped at ``hit_z_max``. ``clear_net`` makes a crossing between the
    posts clear the tape. ``lock_bounce_heading`` keeps the horizontal
    direction through the bounce. The speed scale may sit in
    ``[bounce_keep, 1]``, and a scale outside ``[bounce_keep_low,
    bounce_keep_high]`` is penalized. Set both band edges to ``None`` to
    turn that penalty off.
    """

    drag_k: float = 0.02
    pin_bounce_z: bool = True
    contact_z_min: float | None = CONTACT_Z_MIN
    contact_z_max: float | None = CONTACT_Z_MAX
    hit_z_max: float | None = 2.0
    clear_net: bool = True
    stay_above_court: bool = True
    lock_bounce_heading: bool = True
    bounce_keep: float = BOUNCE_KEEP
    bounce_keep_low: float | None = 0.40
    bounce_keep_high: float | None = 0.60

    def acceleration(self):
        """``None`` is the closed-form parabola. Otherwise gravity plus drag."""
        if self.drag_k <= 0.0:
            return None
        k = float(self.drag_k)
        return lambda pos, vel, k=k: quadratic_drag(pos, vel, k=k)

    def contact_window(self) -> tuple[float, float] | None:
        """Contact height limits, or ``None`` when height is free."""
        if self.contact_z_min is None and self.contact_z_max is None:
            return None
        if self.contact_z_min is None or self.contact_z_max is None:
            raise ValueError("set both contact heights, or set both to None")
        zmin, zmax = float(self.contact_z_min), float(self.contact_z_max)
        if zmin < 0.0 or zmax <= zmin:
            raise ValueError("contact height bounds must satisfy 0 <= min < max")
        return zmin, zmax

    def validate(self) -> None:
        if self.drag_k < 0.0:
            raise ValueError("drag_k must be >= 0")
        if not 0.0 <= self.bounce_keep <= 1.0:
            raise ValueError("bounce_keep must be between 0 and 1")
        self.contact_window()
        _hit_limits(self)
        low, high = self.bounce_keep_low, self.bounce_keep_high
        if (low is None) != (high is None):
            raise ValueError("set both bounce keep band edges, or set both to None")
        if low is not None and not 0.0 <= float(low) <= float(high) <= 1.0:
            raise ValueError("bounce keep band must satisfy 0 <= low <= high <= 1")


DEFAULT_RULES = FlightRules()


def ballistic(
    theta: np.ndarray,
    t: np.ndarray,
    acceleration=None,
) -> np.ndarray:
    """Court positions. ``acceleration=None`` is the gravity parabola."""
    pos, _ = simulate(theta, t, acceleration)
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
    rules: FlightRules | None = None,
    follow_uv: np.ndarray | None = None,
    follow_t: np.ndarray | None = None,
    follow_weights: np.ndarray | None = None,
    t_bounce: float | None = None,
    origin: np.ndarray | None = None,
    t_hit: float | None = None,
    bounce_xy: np.ndarray | None = None,
) -> FlightFit:
    """Fit one shot. ``rules`` is the model, and it is stored on the result.

    ``t_bounce`` is seconds on the same clock as ``t``. It defaults to the
    last sample. ``follow_uv`` is the flight that leaves the bounce and runs
    to the next hit, with ``follow_t`` in seconds from the bounce. The shot
    is fit from its own pixels. The continuation starts at that bounce and
    does not move the contact. ``origin``, when set, is the previous hit.
    ``t_hit`` is seconds from the bounce to the next contact.
    ``bounce_xy`` is the court point of the bounce from the ground homography.
    With ``pin_bounce_z`` it is fixed: the launch velocity is the one that
    arrives there.
    """
    rules = DEFAULT_RULES if rules is None else rules
    rules.validate()
    bounce_xy = _bounce_xy(bounce_xy, rules.pin_bounce_z)
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
            uv, t, t_end, camera, homography, point_w, rules, bounce_xy
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
            rules,
            bounce_xy,
        )
    pos, _ = simulate(theta, t, rules.acceleration())
    pred = camera.project(pos)
    follow_theta, follow_xyz, follow_uv_hat, follow_rmse, hit = _fit_continuation(
        theta,
        t_end,
        follow,
        None if t_hit is None else float(t_hit),
        camera,
        rules,
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
        rules=rules,
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
    bounce_xy: np.ndarray | None = None


def fit_shots(
    shots: list[ShotInput],
    camera: Camera,
    homography: np.ndarray,
    *,
    rules: FlightRules | None = None,
) -> list[FlightFit]:
    """Fit each shot with ``rules``, then start the next shot at the hit.

    The same rules are stored on every fit. A shot with ``t_hit`` ends at the
    next shot's contact, and that position is the next shot's start.
    """
    rules = DEFAULT_RULES if rules is None else rules
    rules.validate()
    if not shots:
        return []
    prepared = [_prepare_shot(shot) for shot in shots]
    fitted = []
    carried = None
    for shot in prepared:
        kwargs = {
            "t_bounce": shot.t_bounce,
            "rules": rules if len(fitted) == 0 else _return_rules(rules),
            "origin": None if carried is None else carried.copy(),
            "t_hit": shot.t_hit,
            "bounce_xy": shot.bounce_xy,
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
        bounce_xy=None if shot.bounce_xy is None else np.asarray(shot.bounce_xy, dtype=np.float64).reshape(2).copy(),
    )


def _fit_contact(uv, t, t_end, camera, homography, point_w, rules: FlightRules, bounce_xy):
    """Contact position and velocity from this shot's pixels."""
    theta0 = initial_theta(
        uv,
        t,
        homography,
        bounce_end=rules.pin_bounce_z,
        bounce_start=False,
        camera=camera,
    )
    window = rules.contact_window()
    zmin, zmax = (0.0, 12.0) if window is None else window
    acceleration = rules.acceleration()
    if bounce_xy is not None:
        lower = np.array([-40.0, -40.0, zmin], dtype=np.float64)
        upper = np.array([40.0, 40.0, zmax], dtype=np.float64)
        packed0 = np.clip(theta0[:3], lower, upper)

        def residual(packed: np.ndarray) -> np.ndarray:
            theta = _launch_to_bounce(packed, t_end, bounce_xy, acceleration)
            return _shot_residual(theta, t, t_end, uv, camera, point_w, rules)

        solved = least_squares(
            residual, packed0, bounds=(lower, upper), method="trf", max_nfev=80
        )
        return _launch_to_bounce(solved.x, t_end, bounce_xy, acceleration), solved
    if rules.pin_bounce_z:
        lower = np.array([-40.0, -40.0, zmin, -90.0, -90.0], dtype=np.float64)
        upper = np.array([40.0, 40.0, zmax, 90.0, 90.0], dtype=np.float64)
        packed0 = np.clip(theta0[:5], lower, upper)

        def residual(packed: np.ndarray) -> np.ndarray:
            theta = _with_bounce(packed, t_end, acceleration)
            return _shot_residual(theta, t, t_end, uv, camera, point_w, rules)

        solved = least_squares(
            residual, packed0, bounds=(lower, upper), method="trf", max_nfev=300
        )
        return _with_bounce(solved.x, t_end, acceleration), solved
    lower = np.array([-40.0, -40.0, zmin, -90.0, -90.0, -90.0], dtype=np.float64)
    upper = np.array([40.0, 40.0, zmax, 90.0, 90.0, 90.0], dtype=np.float64)
    packed0 = np.clip(theta0, lower, upper)

    def residual(theta: np.ndarray) -> np.ndarray:
        return _shot_residual(theta, t, t_end, uv, camera, point_w, rules)

    solved = least_squares(
        residual, packed0, bounds=(lower, upper), method="trf", max_nfev=300
    )
    return solved.x.copy(), solved


def _fit_from_origin(
    origin, uv, t, t_end, camera, homography, point_w, rules: FlightRules, bounce_xy
):
    """Velocity only. The shot starts at ``origin``, the previous hit."""
    acceleration = rules.acceleration()
    if bounce_xy is not None:
        theta = _launch_to_bounce(origin, t_end, bounce_xy, acceleration)
        return theta, _direct_solve()
    end = _ground(uv[-1], homography)
    duration = max(float(t_end), 1e-3)
    horizontal = np.array(
        [(end[0] - origin[0]) / duration, (end[1] - origin[1]) / duration],
        dtype=np.float64,
    )
    if rules.pin_bounce_z:
        lower = np.array([-90.0, -90.0], dtype=np.float64)
        upper = np.array([90.0, 90.0], dtype=np.float64)
        guess = np.clip(horizontal, lower, upper)

        def residual(vel: np.ndarray) -> np.ndarray:
            theta = _with_bounce(np.concatenate([origin, vel]), t_end, acceleration)
            return _shot_residual(theta, t, t_end, uv, camera, point_w, rules)

        solved = least_squares(
            residual, guess, bounds=(lower, upper), method="trf", max_nfev=200
        )
        return _with_bounce(np.concatenate([origin, solved.x]), t_end, acceleration), solved
    lower = np.array([-90.0, -90.0, -90.0], dtype=np.float64)
    upper = np.array([90.0, 90.0, 90.0], dtype=np.float64)
    vz = -float(origin[2]) / duration + 0.5 * G * duration
    guess = np.clip(np.array([horizontal[0], horizontal[1], vz]), lower, upper)

    def residual(vel: np.ndarray) -> np.ndarray:
        theta = np.concatenate([origin, vel])
        return _shot_residual(theta, t, t_end, uv, camera, point_w, rules)

    solved = least_squares(
        residual, guess, bounds=(lower, upper), method="trf", max_nfev=200
    )
    return np.concatenate([origin, solved.x]), solved


def _fit_continuation(theta, t_end, follow, t_hit, camera, rules: FlightRules):
    """Bounce-to-hit flight. The shot's contact stays where it was fit."""
    if follow is None:
        return None, None, None, None, None
    follow_uv, follow_t, follow_w = follow
    point_w = _image_residual_weights(follow_w, len(follow_uv))
    acceleration = rules.acceleration()
    _, incoming = simulate(theta, np.array([t_end], dtype=np.float64), acceleration)
    arrived_vx = float(incoming[0, 0])
    arrived_vy = float(incoming[0, 1])
    t_contact = float(follow_t[-1] if t_hit is None else t_hit)
    window = _hit_limits(rules)
    if acceleration is None and window is not None:
        vz_lo, vz_hi = _bounce_vz_bounds(t_contact, window[0], window[1])
        t_air = max(t_contact, float(follow_t[-1]))
        if t_air > t_contact + 1e-6:
            vz_lo = min(vz_hi, max(vz_lo, 0.5 * G * t_air))
    else:
        vz_lo, vz_hi = 0.0, 90.0
    vz_guess = float(np.clip(abs(incoming[0, 2]), vz_lo, vz_hi))
    if acceleration is not None and window is not None:
        mid = 0.5 * (window[0] + window[1])
        vz_guess = mid / max(t_contact, 1e-3) + 0.5 * G * t_contact
        vz_guess = float(np.clip(vz_guess, vz_lo, vz_hi))
    ground_times = _time_grid(max(t_contact, float(follow_t[-1])), follow_t)

    def launch(packed: np.ndarray) -> np.ndarray:
        if rules.lock_bounce_heading:
            scale = float(packed[0])
            vz = float(packed[1])
            vx, vy = arrived_vx * scale, arrived_vy * scale
        else:
            vx, vy, vz = (float(v) for v in packed)
        return _bounce_launch(
            theta,
            t_end,
            acceleration,
            vx=vx,
            vy=vy,
            vz=vz,
            on_ground=rules.pin_bounce_z,
        )

    def residual(packed: np.ndarray) -> np.ndarray:
        launched = launch(packed)
        parts = [
            _image_residual(
                launched, follow_t, follow_uv, camera, point_w, acceleration
            )
        ]
        if rules.clear_net:
            parts.append(_clearance_residual(launched, t_contact, acceleration))
        if rules.stay_above_court:
            parts.append(_above_ground(launched, ground_times, acceleration, HEIGHT_PX))
        if acceleration is not None and window is not None:
            pos, _ = simulate(launched, np.array([t_contact]), acceleration)
            parts.append(_height_window(float(pos[0, 2]), window[0], window[1]))
        parts.append(
            _bounce_band(float(launched[3]), float(launched[4]), arrived_vx, arrived_vy, rules)
        )
        return np.concatenate(parts)

    if rules.lock_bounce_heading:
        guess = np.array([0.5, vz_guess])
        bounds = ([rules.bounce_keep, vz_lo], [1.0, vz_hi])
    else:
        guess = np.array([arrived_vx * 0.5, arrived_vy * 0.5, vz_guess])
        bounds = ([-90.0, -90.0, vz_lo], [90.0, 90.0, vz_hi])
    solved = least_squares(residual, guess, bounds=bounds, method="trf", max_nfev=80)
    launched = launch(solved.x)
    follow_xyz, _ = simulate(launched, follow_t, acceleration)
    follow_uv_hat = camera.project(follow_xyz)
    follow_rmse = _rmse(follow_uv_hat, follow_uv)
    hit = None
    if t_hit is not None:
        pos, vel = simulate(launched, np.array([t_hit]), acceleration)
        hit = (pos[0], vel[0])
    return launched, follow_xyz, follow_uv_hat, follow_rmse, hit


def _hit_limits(rules: FlightRules) -> tuple[float, float] | None:
    """Height of a hit after the serve. ``None`` when that cap is off and contact is free."""
    if rules.hit_z_max is None:
        return rules.contact_window()
    zmin = 0.0 if rules.contact_z_min is None else float(rules.contact_z_min)
    zmax = float(rules.hit_z_max)
    if zmax <= zmin:
        raise ValueError("hit_z_max must be above the contact minimum")
    return zmin, zmax


def _return_rules(rules: FlightRules) -> FlightRules:
    """Same model, with the serve cap replaced by the later-hit cap."""
    limits = _hit_limits(rules)
    if limits is None:
        return rules
    return replace(rules, contact_z_min=limits[0], contact_z_max=limits[1])


def _bounce_xy(bounce_xy, pin: bool) -> np.ndarray | None:
    """Ground point of the bounce. Ignored unless the bounce is on the court."""
    if bounce_xy is None or not pin:
        return None
    point = np.asarray(bounce_xy, dtype=np.float64).reshape(2)
    if not np.isfinite(point).all():
        return None
    return point.copy()


class _DirectSolve:
    success = True
    cost = 0.0
    message = "bounce position fixed"


def _direct_solve() -> _DirectSolve:
    return _DirectSolve()


def _launch_to_bounce(contact, t_end, bounce_xy, acceleration) -> np.ndarray:
    """Launch velocity whose flight is at ``bounce_xy`` on the ground at ``t_end``.

    With no drag the velocity is the parabola that joins the two points. With
    drag, that parabola is the start of a small solve for the same landing.
    """
    x0, y0, z0 = (float(v) for v in np.asarray(contact, dtype=np.float64).reshape(3))
    xb, yb = (float(v) for v in bounce_xy)
    t_end = max(float(t_end), 1e-3)
    vx = (xb - x0) / t_end
    vy = (yb - y0) / t_end
    vz = -z0 / t_end + 0.5 * G * t_end
    if acceleration is None:
        return np.array([x0, y0, z0, vx, vy, vz], dtype=np.float64)
    guess = np.clip(np.array([vx, vy, vz], dtype=np.float64), -90.0, 90.0)
    target = np.array([xb, yb, 0.0], dtype=np.float64)

    def miss(vel: np.ndarray) -> np.ndarray:
        state = np.array([x0, y0, z0, vel[0], vel[1], vel[2]], dtype=np.float64)
        pos, _ = simulate(state, np.array([t_end]), acceleration)
        return pos[0] - target

    solved = least_squares(
        miss,
        guess,
        bounds=([-90.0, -90.0, -90.0], [90.0, 90.0, 90.0]),
        method="trf",
        xtol=1e-8,
        ftol=1e-8,
        max_nfev=40,
    )
    return np.array([x0, y0, z0, *solved.x], dtype=np.float64)


def _bounce_band(vx: float, vy: float, arrived_vx: float, arrived_vy: float, rules: FlightRules) -> np.ndarray:
    """Zero while the kept horizontal speed is inside the band."""
    if rules.bounce_keep_low is None or rules.bounce_keep_high is None:
        return np.zeros(0, dtype=np.float64)
    speed_in = float(np.hypot(arrived_vx, arrived_vy))
    if speed_in < 1e-3:
        return np.zeros(1, dtype=np.float64)
    frac = float(np.hypot(vx, vy)) / speed_in
    gap = max(0.0, float(rules.bounce_keep_low) - frac)
    gap += max(0.0, frac - float(rules.bounce_keep_high))
    return np.array([gap * BOUNCE_BAND_PX], dtype=np.float64)


def _bounce_vz_bounds(t_contact: float, z_min: float, z_max: float) -> tuple[float, float]:
    """Upward speeds that leave the hit inside ``[z_min, z_max]``.

    The continuation starts on the ground. Under gravity,
    ``Z(t) = vz t - 0.5 g t^2``, so a hit still above the court means the
    whole flight from the bounce to that hit stays above the court.
    """
    t_contact = max(float(t_contact), 1e-3)
    low = z_min / t_contact + 0.5 * G * t_contact
    high = z_max / t_contact + 0.5 * G * t_contact
    low = float(np.clip(low, 0.0, 90.0))
    high = float(np.clip(high, low, 90.0))
    return low, high


def _shot_residual(theta, t, t_end, uv, camera, point_w, rules: FlightRules) -> np.ndarray:
    """Pixels, plus whichever rules are turned on."""
    acceleration = rules.acceleration()
    parts = [_image_residual(theta, t, uv, camera, point_w, acceleration)]
    if rules.clear_net:
        parts.append(_clearance_residual(theta, t_end, acceleration))
    if rules.stay_above_court and acceleration is not None:
        parts.append(_above_ground(theta, _time_grid(t_end, t), acceleration, HEIGHT_PX))
    if rules.pin_bounce_z and acceleration is not None:
        landed, _ = simulate(theta, np.array([float(t_end)]), acceleration)
        parts.append(np.array([float(landed[0, 2]) * HEIGHT_PX], dtype=np.float64))
    return np.concatenate(parts)


def _height_window(z: float, z_min: float, z_max: float) -> np.ndarray:
    """Zero while ``z`` is inside the contact range."""
    return np.array(
        [
            max(0.0, float(z_min) - z) * HEIGHT_PX,
            max(0.0, z - float(z_max)) * HEIGHT_PX,
        ],
        dtype=np.float64,
    )


def _time_grid(t_end: float, extra=None) -> np.ndarray:
    """Times from 0 through ``t_end``, plus any sample times in that span."""
    t_end = max(float(t_end), 1e-3)
    step = 0.02
    grid = np.linspace(0.0, t_end, max(2, int(np.ceil(t_end / step)) + 1))
    if extra is None:
        return grid
    extra = np.asarray(extra, dtype=np.float64).ravel()
    extra = extra[(extra >= -1e-9) & (extra <= t_end + 1e-9)]
    if len(extra) == 0:
        return grid
    return np.unique(np.concatenate([grid, np.clip(extra, 0.0, t_end)]))


def _with_bounce(packed: np.ndarray, t_end: float, acceleration=None) -> np.ndarray:
    """Five free values, with ``vz`` set so the ball is on the ground at ``t_end``."""
    x0, y0, z0, vx, vy = (float(v) for v in np.asarray(packed, dtype=np.float64).reshape(5))
    if acceleration is None:
        vz = -z0 / t_end + 0.5 * G * t_end
    else:
        vz = _vz_landing(x0, y0, z0, vx, vy, t_end, acceleration)
    return np.array([x0, y0, z0, vx, vy, vz], dtype=np.float64)


def _vz_landing(x0, y0, z0, vx, vy, t_end, acceleration) -> float:
    """Launch ``vz`` whose integrated flight meets the ground at ``t_end``.

    Drag makes the ball land sooner than the gravity guess, so the root is
    searched around that guess. Higher ``vz`` is still higher at ``t_end``.
    """
    t_end = max(float(t_end), 1e-3)
    guess = float(np.clip(-z0 / t_end + 0.5 * G * t_end, -90.0, 90.0))

    def height(vz: float) -> float:
        state = np.array([x0, y0, z0, vx, vy, float(vz)], dtype=np.float64)
        pos, _ = simulate(state, np.array([t_end]), acceleration)
        return float(pos[0, 2])

    h_guess = height(guess)
    if abs(h_guess) < 1e-4:
        return guess
    lo, hi = guess, guess
    h_lo, h_hi = h_guess, h_guess
    step = 2.0
    if h_guess < 0.0:
        while h_hi <= 0.0 and hi < 90.0:
            hi = min(90.0, hi + step)
            h_hi = height(hi)
            step *= 1.6
    else:
        while h_lo >= 0.0 and lo > -90.0:
            lo = max(-90.0, lo - step)
            h_lo = height(lo)
            step *= 1.6
    if h_lo * h_hi > 0.0:
        return lo if abs(h_lo) <= abs(h_hi) else hi
    if lo > hi:
        lo, hi = hi, lo
    root = root_scalar(height, bracket=(lo, hi), xtol=1e-3)
    return float(root.root)


def _bounce_launch(
    theta: np.ndarray,
    t_bounce: float,
    acceleration=None,
    vx: float | None = None,
    vy: float | None = None,
    vz: float = 0.0,
    on_ground: bool = True,
) -> np.ndarray:
    """State at the bounce for the flight that continues to the next hit.

    Position is the shot's position at ``t_bounce``. ``on_ground`` puts that
    position on the court. ``vx`` and ``vy`` override the horizontal velocity.
    ``vz`` is the upward speed of the continuation.
    """
    pos, vel = simulate(theta, np.array([t_bounce], dtype=np.float64), acceleration)
    return np.array(
        [
            pos[0, 0],
            pos[0, 1],
            0.0 if on_ground else pos[0, 2],
            vel[0, 0] if vx is None else vx,
            vel[0, 1] if vy is None else vy,
            vz,
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


def _above_ground(theta, t, acceleration, weight: float = CONSTRAINT_PX) -> np.ndarray:
    """Zero while the ball is on or above the court."""
    pos, _ = simulate(theta, t, acceleration)
    below = np.maximum(0.0, -pos[:, 2])
    return below * weight


def _clearance_residual(theta: np.ndarray, t_end: float, acceleration=None) -> np.ndarray:
    """Zero unless the ball crosses between the posts below the tape."""
    if acceleration is None:
        return _clearance_gravity(theta, t_end)
    t_end = float(t_end)
    if t_end <= 1e-4:
        return np.zeros(1, dtype=np.float64)
    pos, _ = simulate(theta, _time_grid(t_end), acceleration)
    y = pos[:, 1]
    short = 0.0
    for i in range(len(pos) - 1):
        if y[i] * y[i + 1] >= 0.0:
            continue
        span = float(y[i] - y[i + 1])
        if abs(span) < 1e-8:
            continue
        frac = float(np.clip(y[i] / span, 0.0, 1.0))
        point = pos[i] + frac * (pos[i + 1] - pos[i])
        tape = net_height(float(point[0]))
        if np.isfinite(tape):
            short = max(short, tape + CLEARANCE_MARGIN - float(point[2]))
    return np.array([short * CLEARANCE_PX], dtype=np.float64)


def _clearance_gravity(theta: np.ndarray, t_end: float) -> np.ndarray:
    """Net clearance for the closed-form flight, where ``vy`` is constant."""
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
    acceleration=None,
) -> RallyFit:
    """Fit one ``theta`` per flight.

    A segment with ``bounce_end`` shares its end position with the next
    flight's start, forces both heights to 0, and keeps horizontal velocity
    continuous. Vertical velocity is not tied across the bounce.
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
