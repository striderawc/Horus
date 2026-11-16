"""Shot and bounce cuts on a track.

The track is positions in one plane. Callers pass court ``(X, Y)`` meters
from the smoothed image track, so a pixel of motion near the camera is not
a bigger event than the same motion at the far baseline. Velocity is the
frame-to-frame difference of those positions. Court Y points the same way
as pixel y, toward the near baseline. Court X points to the right.

A rise in ``|Δvy|`` through ``event_change`` is an event, and that frame has
to sit within ``x_zero_frames`` of a court-X velocity near zero. The change
that classifies it is the median velocity in a short window before that
frame against the median in the same-length window after it. A court-X
change that clears ``x_change`` is a hit. When ``require_y_flip`` is set, a
court-Y sign flip is a hit too. If neither of those happens, the court-Y
change decides: at or above ``hit_y_change`` it is a hit, and at or above
``bounce_y_change`` it is a bounce. A frame labeled ``status == 2`` is also
a bounce, and it replaces a kinematic bounce within ``min_gap`` frames so
the two are not both kept.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Per frame, in the same units as the track. For court meters this is meters
# per frame. Notebooks quote the same cuts in meters per second.
EVENT_CHANGE = 0.20
HIT_Y_CHANGE = 0.25
BOUNCE_Y_CHANGE = 0.25
X_CHANGE = 0.35
REQUIRE_Y_FLIP = True
# An event must fall within this many frames of a court-X velocity that is
# within X_ZERO of zero, or of a court-X sign change.
X_ZERO_FRAMES = 4
X_ZERO = 0.0
WINDOW = 3
REARM = 0.13
MIN_GAP = 8


@dataclass(frozen=True)
class Flight:
    """Inclusive frame indices into the track. ``end`` is the bounce when one closes it."""

    start: int
    end: int
    ends_on_bounce: bool


def kinematics(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Velocity and the size of its frame-to-frame change.

    ``velocity[i]`` is ``xy[i] - xy[i - 1]``, in the units of ``xy`` per frame.
    ``change[i]`` is ``||velocity[i] - velocity[i - 1]||``. Both stay nan
    until the samples they need are finite.
    """
    return _kinematics(xy)


def velocity_shift(
    velocity: np.ndarray, index: int, window: int = WINDOW
) -> tuple[float, float, bool] | None:
    """Change from the window before ``index`` to the window after it.

    Each side is the median of the finite velocities in that window.
    Returns ``(|Δvx|, |Δvy|, y_sign_flip)`` in the units of ``velocity``.
    The sign flips when the two court-Y medians have opposite signs.
    Returns ``None`` when either window has no finite velocity.
    """
    if window < 1:
        raise ValueError("window must be at least 1")
    velocity = np.asarray(velocity, dtype=np.float64)
    before = velocity[max(0, index - window) : index]
    after = velocity[index : index + window]
    before = before[np.isfinite(before).all(axis=1)] if len(before) else before
    after = after[np.isfinite(after).all(axis=1)] if len(after) else after
    if len(before) == 0 or len(after) == 0:
        return None
    b = np.median(before, axis=0)
    a = np.median(after, axis=0)
    return abs(float(a[0] - b[0])), abs(float(a[1] - b[1])), float(b[1] * a[1]) < 0.0


def detect_cuts(
    xy: np.ndarray,
    status: np.ndarray | None = None,
    *,
    event_change: float = EVENT_CHANGE,
    hit_y_change: float = HIT_Y_CHANGE,
    bounce_y_change: float = BOUNCE_Y_CHANGE,
    x_change: float = X_CHANGE,
    require_y_flip: bool = REQUIRE_Y_FLIP,
    x_zero_frames: int = X_ZERO_FRAMES,
    x_zero: float = X_ZERO,
    window: int = WINDOW,
    rearm: float = REARM,
    min_gap: int = MIN_GAP,
) -> tuple[list[int], list[int]]:
    """Shots and bounces from one pass over the velocity changes.

    A rising edge of ``|Δvy|`` through ``event_change`` is an event, and it
    has to fall within ``x_zero_frames`` of a near-zero court-X velocity.
    ``velocity_shift`` compares the windows around that frame. A court-X
    change that clears ``x_change`` is a shot. When ``require_y_flip`` is
    set, a court-Y sign flip is a shot too. Otherwise the court-Y change
    decides: ``hit_y_change`` is a shot and ``bounce_y_change`` is a bounce.
    """
    velocity, _ = _kinematics(xy)
    y_change = _abs_component_change(velocity, 1)
    shots: list[int] = []
    bounces: list[int] = []
    for index in _rising_edges(y_change, event_change, rearm, min_gap):
        x_near = _x_near_zero(velocity, index, x_zero_frames, x_zero)
        shift = velocity_shift(velocity, index, window)
        if shift is None:
            continue
        dx, dy, flipped = shift
        if dx >= x_change or (require_y_flip and flipped):
            shots.append(index)
        elif dy >= hit_y_change and x_near:
            shots.append(index)
        elif dy >= bounce_y_change:
            bounces.append(index)
    if status is not None:
        bounces = _labeled_bounces(bounces, shots, status, min_gap)
    return shots, bounces


def iter_flights(n: int, shots: list[int], bounces: list[int]) -> list[Flight]:
    """Uninterrupted flights. A bounce ends the flight that reaches it."""
    bounce_set = {int(i) for i in bounces}
    shot_set = {int(i) for i in shots}
    starts = sorted(shot_set | bounce_set)
    if not starts:
        if n < 2:
            return []
        return [Flight(0, n - 1, False)]

    flights: list[Flight] = []
    if starts[0] > 1:
        ends = starts[0] in bounce_set and starts[0] not in shot_set
        end = starts[0] if ends else starts[0] - 1
        flights.append(Flight(0, end, ends))

    for i, start in enumerate(starts):
        nxt = starts[i + 1] if i + 1 < len(starts) else None
        if nxt is None:
            if start <= n - 1:
                flights.append(Flight(start, n - 1, False))
            continue
        if nxt in bounce_set and nxt not in shot_set:
            flights.append(Flight(start, nxt, True))
        elif nxt - 1 >= start:
            flights.append(Flight(start, nxt - 1, False))
    return flights


def select_flight(
    xy: np.ndarray,
    shots: list[int],
    bounces: list[int],
    *,
    min_points: int = 6,
) -> Flight:
    """One flight. Prefer a flight that ends on a bounce and has enough samples."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)

    def finite_count(flight: Flight) -> int:
        segment = xy[flight.start : flight.end + 1]
        return int(np.isfinite(segment).all(axis=1).sum())

    usable = [
        flight
        for flight in iter_flights(len(xy), shots, bounces)
        if finite_count(flight) >= min_points
    ]
    if not usable:
        raise RuntimeError("no flight with enough tracked points")
    ending = [flight for flight in usable if flight.ends_on_bounce]
    pool = ending or usable
    return max(pool, key=finite_count)


def _abs_component_change(velocity: np.ndarray, component: int) -> np.ndarray:
    """``|velocity[i, component] - velocity[i - 1, component]|``. Nan until both are finite."""
    out = np.full(len(velocity), np.nan)
    for i in range(1, len(velocity)):
        cur = velocity[i, component]
        prev = velocity[i - 1, component]
        if np.isfinite(cur) and np.isfinite(prev):
            out[i] = abs(float(cur - prev))
    return out


def _x_near_zero(velocity: np.ndarray, index: int, frames: int, x_zero: float) -> bool:
    """Court-X velocity within ``frames`` of ``index`` is near zero or changes sign.

    ``x_zero`` is the absolute speed that still counts as zero, in the units
    of ``velocity``. A sign change between two adjacent samples in the span
    counts as passing through zero.
    """
    if frames < 0:
        raise ValueError("x_zero_frames must be at least 0")
    lo = max(0, index - frames)
    hi = min(len(velocity), index + frames + 1)
    vx = velocity[lo:hi, 0]
    finite = vx[np.isfinite(vx)]
    if finite.size and np.any(np.abs(finite) <= x_zero):
        return True
    for i in range(lo, hi - 1):
        a = velocity[i, 0]
        b = velocity[i + 1, 0]
        if np.isfinite(a) and np.isfinite(b) and float(a * b) < 0.0:
            return True
    return False


def _rising_edges(
    change: np.ndarray, level: float, rearm: float, min_gap: int
) -> list[int]:
    """Frames where ``change`` rises through ``level``, after it has gone quiet."""
    found: list[int] = []
    armed = True
    last: int | None = None
    fell = False
    for i, cur in enumerate(change):
        if not np.isfinite(cur):
            continue
        if last is not None and fell and i - last >= min_gap:
            armed = True
        prev = change[i - 1] if i else np.nan
        gap_ok = last is None or i - last >= min_gap
        crossed = (not np.isfinite(prev) or prev < level) and cur >= level
        if armed and gap_ok and crossed:
            found.append(i)
            armed = False
            fell = False
            last = i
        if cur < rearm:
            fell = True
    return found


def _labeled_bounces(
    bounces: list[int], shots: list[int], status: np.ndarray, min_gap: int
) -> list[int]:
    """Dataset ``status == 2`` frames. A label replaces a nearby kinematic bounce."""
    shot_set = {int(i) for i in shots}
    labeled = [
        i
        for i, code in enumerate(np.asarray(status))
        if int(code) == 2 and i not in shot_set
    ]
    found = list(bounces)
    for label in labeled:
        found = [bounce for bounce in found if abs(bounce - label) >= min_gap]
    for label in labeled:
        near_shot = any(abs(label - shot) < min_gap for shot in shot_set)
        near_bounce = any(abs(label - bounce) < min_gap for bounce in found)
        if not near_shot and not near_bounce:
            found.append(label)
    found.sort()
    return found


def _kinematics(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    n = len(xy)
    velocity = np.full((n, 2), np.nan)
    change = np.full(n, np.nan)
    for i in range(1, n):
        if np.isfinite(xy[i]).all() and np.isfinite(xy[i - 1]).all():
            velocity[i] = xy[i] - xy[i - 1]
    for i in range(1, n):
        if np.isfinite(velocity[i]).all() and np.isfinite(velocity[i - 1]).all():
            change[i] = float(np.linalg.norm(velocity[i] - velocity[i - 1]))
    return velocity, change
