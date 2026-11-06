"""Scores for connected foreground blobs from a frame-difference mask."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

MIN_AREA = 4
# Ratios at or above ASPECT_KEEP are compact enough (a short streak still counts).
ASPECT_KEEP = 0.40
ASPECT_ZERO = 0.15
# Circularity at or above this is round enough for a small, slightly blurred ball.
CIRC_KEEP = 0.50
CIRC_ZERO = 0.15
TEMPORAL_SIGMA_FLOOR = 8.0
# OpenCV hue is 0–180. 30 is yellow; this footage also records the ball as a pale highlight near 125.
HUE_YELLOW = 30.0
HUE_HIGHLIGHT = 125.0


@dataclass
class Candidate:
    xy: np.ndarray
    bbox: tuple[int, int, int, int]
    area: float
    perimeter: float
    hull_area: float
    mean_hsv: np.ndarray
    contour: np.ndarray


def candidates(mask: np.ndarray, bgr: np.ndarray, min_area: int = MIN_AREA) -> list[Candidate]:
    """External contours on a binary mask, skipping specks smaller than ``min_area``."""
    binary = mask if mask.dtype == np.uint8 else (mask > 0).astype(np.uint8) * 255
    found, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    out: list[Candidate] = []
    for contour in found:
        if len(contour) < 3:
            continue
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        moments = cv2.moments(contour)
        if moments["m00"] <= 0.0 or hull_area <= 0.0:
            continue
        xy = np.array(
            [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
            dtype=np.float64,
        )
        filled = np.zeros(binary.shape, np.uint8)
        cv2.drawContours(filled, [contour], -1, 255, thickness=-1)
        mean_hsv = np.array(cv2.mean(hsv, mask=filled)[:3], dtype=np.float64)
        out.append(
            Candidate(
                xy=xy,
                bbox=cv2.boundingRect(contour),
                area=area,
                perimeter=float(cv2.arcLength(contour, True)),
                hull_area=hull_area,
                mean_hsv=mean_hsv,
                contour=contour,
            )
        )
    return out


def score(
    candidate: Candidate,
    pred: np.ndarray | None,
    cov: np.ndarray | None,
    history: list[np.ndarray],
) -> dict[str, float]:
    """Six cues in ``[0, 1]`` and their geometric mean ``s``.

    ``pred`` / ``cov`` are the SSM predicted position and covariance (``None`` before
    the filter has a state). ``history`` is accepted centroids, oldest first.
    """
    parts = {
        "s_ssm": _ssm(candidate.xy, pred, cov),
        "s_aspect": _hinge(_aspect_ratio(candidate.bbox), ASPECT_KEEP, ASPECT_ZERO),
        "s_circularity": _hinge(_circularity(candidate.area, candidate.perimeter), CIRC_KEEP, CIRC_ZERO),
        "s_solidity": _solidity(candidate.area, candidate.hull_area),
        "s_appearance": _appearance(candidate.mean_hsv),
        "s_temporal": _temporal(candidate.xy, history),
    }
    values = list(parts.values())
    if any(v <= 0.0 for v in values):
        combined = 0.0
    else:
        combined = math.exp(sum(math.log(v) for v in values) / len(values))
    parts["s"] = float(combined)
    return parts


def _hinge(value: float, keep: float, zero: float) -> float:
    """1 at or above ``keep``, 0 at or below ``zero``."""
    if value >= keep:
        return 1.0
    if value <= zero:
        return 0.0
    return (value - zero) / (keep - zero)


def _aspect_ratio(bbox: tuple[int, int, int, int]) -> float:
    _, _, w, h = bbox
    long_side = max(w, h)
    if long_side <= 0:
        return 0.0
    return min(w, h) / long_side


def _circularity(area: float, perimeter: float) -> float:
    if perimeter <= 0.0:
        return 0.0
    return min(1.0, 4.0 * math.pi * area / (perimeter * perimeter))


def _solidity(area: float, hull_area: float) -> float:
    if hull_area <= 0.0:
        return 0.0
    return min(1.0, area / hull_area)


def _appearance(mean_hsv: np.ndarray) -> float:
    hue, _sat, value = (float(v) for v in mean_hsv)
    color = max(
        math.exp(-0.5 * ((hue - HUE_YELLOW) / 18.0) ** 2),
        math.exp(-0.5 * ((hue - HUE_HIGHLIGHT) / 15.0) ** 2),
    )
    bright = min(1.0, max(0.0, (value - 150.0) / 90.0))
    return min(1.0, 0.5 * (bright + color))


def _ssm(xy: np.ndarray, pred: np.ndarray | None, cov: np.ndarray | None) -> float:
    if pred is None or cov is None:
        return 1.0
    innov = np.asarray(xy, dtype=np.float64) - np.asarray(pred, dtype=np.float64)
    position = np.array(cov, dtype=np.float64, copy=True)[:2, :2]
    position[0, 0] += 1e-6
    position[1, 1] += 1e-6
    distance2 = float(innov @ np.linalg.solve(position, innov))
    return math.exp(-0.5 * max(distance2, 0.0))


def _temporal(xy: np.ndarray, history: list[np.ndarray]) -> float:
    if len(history) < 2:
        return 1.0
    previous = np.asarray(history[-2], dtype=np.float64)
    last = np.asarray(history[-1], dtype=np.float64)
    step = last - previous
    residual = float(np.linalg.norm(np.asarray(xy, dtype=np.float64) - (last + step)))
    sigma = max(float(np.linalg.norm(step)), TEMPORAL_SIGMA_FLOOR)
    return math.exp(-0.5 * (residual / sigma) ** 2)
