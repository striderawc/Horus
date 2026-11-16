"""ITF court coordinates and the game-1 landmark file.

Origin is the court center on the ground. Axes match the image: +X is
horizontal, to the right, and +Y is vertical, downward, toward the near
baseline. +Z is up. Distances are meters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

LENGTH = 23.77
DOUBLES_WIDTH = 10.97
SINGLES_WIDTH = 8.23
SERVICE_FROM_NET = 6.40
NET_HEIGHT_CENTER = 0.914
NET_HEIGHT_POST = 1.07
# Posts stand this far outside each doubles sideline.
POST_OUTSIDE = 0.914

HALF_LENGTH = LENGTH / 2.0
HALF_DOUBLES = DOUBLES_WIDTH / 2.0
HALF_SINGLES = SINGLES_WIDTH / 2.0
POST_X = HALF_DOUBLES + POST_OUTSIDE

# Lengths checked after the homography is inverted.
EXPECTED_LENGTHS = {
    "baseline_length": LENGTH,
    "doubles_width": DOUBLES_WIDTH,
    "service_depth": SERVICE_FROM_NET,
}

# Pairs whose catalog distance is the baseline-to-baseline length.
COURT_LENGTH_PAIRS = (
    ("doubles_near_left", "doubles_far_left"),
    ("doubles_near_right", "doubles_far_right"),
    ("singles_near_left", "singles_far_left"),
    ("singles_near_right", "singles_far_right"),
    ("baseline_near", "baseline_far"),
)
DOUBLES_WIDTH_PAIRS = (
    ("doubles_near_left", "doubles_near_right"),
    ("doubles_far_left", "doubles_far_right"),
)
# Opposite service lines are two depths apart. Half the span is the depth.
SERVICE_SPAN_PAIRS = (
    ("service_near_left", "service_far_left"),
    ("service_near_right", "service_far_right"),
    ("service_near", "service_far"),
)


def _xyz(x: float, y: float, z: float = 0.0) -> np.ndarray:
    return np.array([x, y, z], dtype=np.float64)


def catalog() -> dict[str, np.ndarray]:
    """Named court points. ``left``/``right`` are the sidelines, ``near``/``far`` the depth."""
    hl, hd, hs, s = HALF_LENGTH, HALF_DOUBLES, HALF_SINGLES, SERVICE_FROM_NET
    points = {
        "doubles_near_left": _xyz(-hd, hl),
        "doubles_near_right": _xyz(hd, hl),
        "doubles_far_left": _xyz(-hd, -hl),
        "doubles_far_right": _xyz(hd, -hl),
        "singles_near_left": _xyz(-hs, hl),
        "singles_near_right": _xyz(hs, hl),
        "singles_far_left": _xyz(-hs, -hl),
        "singles_far_right": _xyz(hs, -hl),
        "service_near_left": _xyz(-hs, s),
        "service_near_right": _xyz(hs, s),
        "service_far_left": _xyz(-hs, -s),
        "service_far_right": _xyz(hs, -s),
        "baseline_near": _xyz(0.0, hl),
        "baseline_far": _xyz(0.0, -hl),
        "service_near": _xyz(0.0, s),
        "service_far": _xyz(0.0, -s),
        "net_post_left": _xyz(-POST_X, 0.0, NET_HEIGHT_POST),
        "net_post_right": _xyz(POST_X, 0.0, NET_HEIGHT_POST),
    }
    return points


def segments() -> list[tuple[np.ndarray, np.ndarray]]:
    """Doubles, singles, service, and net lines, as endpoint pairs in meters."""
    hl, hd, hs, s = HALF_LENGTH, HALF_DOUBLES, HALF_SINGLES, SERVICE_FROM_NET
    doubles = [(-hd, hl), (hd, hl), (hd, -hl), (-hd, -hl)]
    lines = []
    for i in range(4):
        a, b = doubles[i], doubles[(i + 1) % 4]
        lines.append((_xyz(*a), _xyz(*b)))
    lines.append((_xyz(-hs, hl), _xyz(-hs, -hl)))
    lines.append((_xyz(hs, hl), _xyz(hs, -hl)))
    lines.append((_xyz(-hs, s), _xyz(hs, s)))
    lines.append((_xyz(-hs, -s), _xyz(hs, -s)))
    lines.append((_xyz(0.0, s), _xyz(0.0, -s)))
    lines.append((_xyz(-hd, 0.0), _xyz(hd, 0.0)))
    lines.append((_xyz(-POST_X, 0.0), _xyz(-POST_X, 0.0, NET_HEIGHT_POST)))
    lines.append((_xyz(POST_X, 0.0), _xyz(POST_X, 0.0, NET_HEIGHT_POST)))
    lines.append((_xyz(-POST_X, 0.0, NET_HEIGHT_POST), _xyz(0.0, 0.0, NET_HEIGHT_CENTER)))
    lines.append((_xyz(0.0, 0.0, NET_HEIGHT_CENTER), _xyz(POST_X, 0.0, NET_HEIGHT_POST)))
    return lines


def net_height(x: float) -> float:
    """Tape height at court ``X``. Nan wide of the posts.

    The tape is 0.914 m at the center and 1.07 m at each post, linear in ``|X|``.
    """
    span = abs(float(x))
    if not np.isfinite(span) or span > POST_X:
        return float("nan")
    return NET_HEIGHT_CENTER + (NET_HEIGHT_POST - NET_HEIGHT_CENTER) * (span / POST_X)


@dataclass(frozen=True)
class Landmark:
    name: str
    world: np.ndarray
    image: np.ndarray | None


def _image(value) -> np.ndarray | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"image pixel must be [x, y] or null, got {value!r}")
    if value[0] is None or value[1] is None:
        return None
    return np.array([float(value[0]), float(value[1])], dtype=np.float64)


def load_landmarks(path: str | Path) -> tuple[list[Landmark], dict]:
    """Load landmarks. ``image`` is None until the pixels are filled in.

    ``meta`` has ``frame`` (path under the dataset, or None) and ``image_size``
    as ``(width, height)``.
    """
    path = Path(path)
    with path.open() as f:
        raw = yaml.safe_load(f)
    size = raw.get("image_size") or [1280, 720]
    meta = {"frame": raw.get("frame"), "image_size": (int(size[0]), int(size[1]))}
    landmarks = []
    for name, item in raw["landmarks"].items():
        world = np.asarray(item["world"], dtype=np.float64).reshape(3)
        landmarks.append(Landmark(str(name), world, _image(item.get("image"))))
    return landmarks, meta


def marked(landmarks: list[Landmark]) -> list[Landmark]:
    """Landmarks that already have image pixels."""
    return [item for item in landmarks if item.image is not None]
