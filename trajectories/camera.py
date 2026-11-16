"""Ground-plane homography and a pinhole camera for one frame."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from trajectories.court import (
    COURT_LENGTH_PAIRS,
    DOUBLES_WIDTH_PAIRS,
    EXPECTED_LENGTHS,
    SERVICE_SPAN_PAIRS,
    Landmark,
    segments,
)

RMS_MAX_M = 0.20
LENGTH_TOL = 0.03


class CourtGeometryError(RuntimeError):
    """The reconstructed court does not match the real dimensions."""

    def __init__(self, report: "CourtReport") -> None:
        self.report = report
        super().__init__("court geometry is not close enough to the real court\n" + str(report))


@dataclass
class CourtReport:
    rms_m: float
    point_error_m: dict[str, float]
    lengths_m: dict[str, float | None]
    expected_m: dict[str, float]

    def __str__(self) -> str:
        lines = [f"court RMS {self.rms_m:.3f} m"]
        for name, err in self.point_error_m.items():
            lines.append(f"  {name}: {err:.3f} m")
        for key, expected in self.expected_m.items():
            label = key.replace("_", " ")
            got = self.lengths_m.get(key)
            if got is None:
                lines.append(f"  {label}: not measured (expected {expected:.2f} m)")
            else:
                lines.append(f"  {label}: {got:.3f} m (expected {expected:.2f} m)")
        return "\n".join(lines)


@dataclass
class Camera:
    """Pinhole camera.

    ``rvec`` and ``tvec`` are the OpenCV pose of the object frame used
    internally, which is court meters with Y negated. ``project``, ``depth``,
    and ``center`` take and return court meters.
    """

    focal: float
    principal: tuple[float, float]
    rvec: np.ndarray
    tvec: np.ndarray

    def K(self) -> np.ndarray:
        f = float(self.focal)
        cx, cy = self.principal
        return np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    def rotation(self) -> np.ndarray:
        R, _ = cv2.Rodrigues(np.asarray(self.rvec, dtype=np.float64).reshape(3, 1))
        return R

    def center(self) -> np.ndarray:
        """Camera position in court meters."""
        t = np.asarray(self.tvec, dtype=np.float64).reshape(3)
        center = -self.rotation().T @ t
        center = center.copy()
        center[1] *= -1.0
        return center

    def depth(self, xyz: np.ndarray) -> np.ndarray:
        pts = _object_points(xyz)
        Xc = (self.rotation() @ pts.T) + np.asarray(self.tvec, dtype=np.float64).reshape(3, 1)
        return Xc[2]

    def project(self, xyz: np.ndarray) -> np.ndarray:
        """Court meters to image pixels. ``xyz`` is (3,) or (N, 3)."""
        img, _ = cv2.projectPoints(
            _object_points(xyz),
            np.asarray(self.rvec, dtype=np.float64).reshape(3, 1),
            np.asarray(self.tvec, dtype=np.float64).reshape(3, 1),
            self.K(),
            None,
        )
        return img.reshape(-1, 2)


def court_lines() -> list[tuple[np.ndarray, np.ndarray]]:
    """Doubles, singles, service, and net segments in court meters."""
    return segments()


def estimate_homography(world_xy: np.ndarray, image_uv: np.ndarray) -> np.ndarray:
    """Homography from ground-plane meters ``(X, Y)`` to image pixels."""
    world_xy = np.asarray(world_xy, dtype=np.float64).reshape(-1, 2)
    image_uv = np.asarray(image_uv, dtype=np.float64).reshape(-1, 2)
    if len(world_xy) < 4:
        raise ValueError("homography needs at least 4 ground points")
    if len(world_xy) != len(image_uv):
        raise ValueError("world points and image points must be the same length")
    H, _ = cv2.findHomography(world_xy, image_uv, method=0)
    if H is None:
        raise ValueError("homography failed")
    return H


def court_from_image(uv: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Image pixels to court ``(X, Y)`` meters, through the ground-plane homography.

    A pixel that is not finite stays nan. This is the ground point that
    projects to that pixel, so a ball in the air lands off its true ``(X, Y)``.
    """
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    out = np.full((len(uv), 2), np.nan, dtype=np.float64)
    keep = np.isfinite(uv).all(axis=1)
    if not np.any(keep):
        return out
    pts = uv[keep].reshape(-1, 1, 2)
    xy = cv2.perspectiveTransform(pts, np.linalg.inv(homography)).reshape(-1, 2)
    out[keep] = xy
    return out


def estimate_camera(
    world_xyz: np.ndarray,
    image_uv: np.ndarray,
    image_size: tuple[int, int],
    homography: np.ndarray | None = None,
) -> Camera:
    """Pinhole camera for the airborne ball.

    The principal point is the image center. Focal length starts from the
    ground-plane homography (rotation columns have to stay orthonormal) and
    ``solvePnP`` supplies the pose. Landmark reprojection then refines the
    pose. Focal length is refined only when a point is off the ground, which
    is what makes the net posts informative. Ground points alone still fix
    the homography.
    """
    world_xyz = np.asarray(world_xyz, dtype=np.float64).reshape(-1, 3)
    image_uv = np.asarray(image_uv, dtype=np.float64).reshape(-1, 2)
    if len(world_xyz) != len(image_uv):
        raise ValueError("world points and image points must be the same length")
    width, height = int(image_size[0]), int(image_size[1])
    principal = (width / 2.0, height / 2.0)
    ground = np.abs(world_xyz[:, 2]) <= 1e-8
    if int(ground.sum()) < 4:
        raise ValueError("camera estimate needs at least 4 ground points")
    H = homography
    if H is None:
        H = estimate_homography(world_xyz[ground, :2], image_uv[ground])
    focal = _focal_from_homography(H, principal)
    if focal is None:
        focal = float(max(width, height))
    K = _matrix(focal, principal)
    rvec, tvec = _pose(world_xyz, image_uv, K, ground)
    elevated = bool(np.any(~ground))
    return _refine(focal, principal, rvec, tvec, world_xyz, image_uv, refine_focal=elevated)


def validate_court(
    landmarks: list[Landmark],
    homography: np.ndarray,
    *,
    rms_max: float = RMS_MAX_M,
    length_tol: float = LENGTH_TOL,
) -> CourtReport:
    """Map ground points back to the court and compare them with the catalog.

    Raises ``CourtGeometryError`` when the RMS error is above ``rms_max``
    meters or a measured length is off by more than ``length_tol`` (fraction).
    """
    ground = [
        item
        for item in landmarks
        if item.image is not None and abs(float(item.world[2])) <= 1e-8
    ]
    if len(ground) < 4:
        raise CourtGeometryError(
            CourtReport(
                rms_m=float("inf"),
                point_error_m={},
                lengths_m={key: None for key in EXPECTED_LENGTHS},
                expected_m=dict(EXPECTED_LENGTHS),
            )
        )
    pixels = np.array([item.image for item in ground], dtype=np.float64).reshape(-1, 1, 2)
    try:
        back = cv2.perspectiveTransform(pixels, np.linalg.inv(homography)).reshape(-1, 2)
    except np.linalg.LinAlgError as exc:
        raise CourtGeometryError(
            CourtReport(
                rms_m=float("inf"),
                point_error_m={},
                lengths_m={key: None for key in EXPECTED_LENGTHS},
                expected_m=dict(EXPECTED_LENGTHS),
            )
        ) from exc
    world = np.array([item.world[:2] for item in ground], dtype=np.float64)
    error = np.linalg.norm(back - world, axis=1)
    rms = float(np.sqrt(np.mean(error**2)))
    point_error = {item.name: float(error[i]) for i, item in enumerate(ground)}
    reconstructed = {item.name: back[i] for i, item in enumerate(ground)}
    lengths = {
        "baseline_length": _mean_pair(reconstructed, COURT_LENGTH_PAIRS),
        "doubles_width": _mean_pair(reconstructed, DOUBLES_WIDTH_PAIRS),
        "service_depth": _mean_pair(reconstructed, SERVICE_SPAN_PAIRS, scale=0.5),
    }
    report = CourtReport(rms, point_error, lengths, dict(EXPECTED_LENGTHS))
    reasons = []
    if not np.isfinite(rms) or rms > rms_max:
        reasons.append(f"RMS {rms:.3f} m is above {rms_max:.2f} m")
    for key, expected in EXPECTED_LENGTHS.items():
        got = lengths[key]
        if got is None:
            continue
        if abs(got - expected) / expected > length_tol:
            reasons.append(
                f"{key.replace('_', ' ')} {got:.3f} m is more than {length_tol:.0%} off {expected:.2f} m"
            )
    if reasons:
        raise CourtGeometryError(report)
    return report


def draw_court(
    image: np.ndarray,
    camera: Camera,
    color: tuple[int, int, int] = (0, 220, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw the projected court boundaries on a BGR frame."""
    out = image.copy()
    for start, end in court_lines():
        pa = camera.project(start)[0]
        pb = camera.project(end)[0]
        if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
            continue
        cv2.line(out, _pixel(pa), _pixel(pb), color, thickness, cv2.LINE_AA)
    return out


def reprojection_rmse(camera: Camera, world_xyz: np.ndarray, image_uv: np.ndarray) -> float:
    pred = camera.project(world_xyz)
    delta = pred - np.asarray(image_uv, dtype=np.float64).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(delta**2, axis=1))))


def _mean_pair(
    points: dict[str, np.ndarray],
    pairs: tuple[tuple[str, str], ...],
    scale: float = 1.0,
) -> float | None:
    distances = []
    for a, b in pairs:
        if a not in points or b not in points:
            continue
        distances.append(float(np.linalg.norm(points[a] - points[b])) * scale)
    if not distances:
        return None
    return float(np.mean(distances))


def _pixel(point: np.ndarray) -> tuple[int, int]:
    return int(round(float(point[0]))), int(round(float(point[1])))


def _matrix(focal: float, principal: tuple[float, float]) -> np.ndarray:
    cx, cy = principal
    return np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _focal_from_homography(H: np.ndarray, principal: tuple[float, float]) -> float | None:
    """Focal length that makes the homography's first two columns a rotation.

    Image coordinates are shifted onto the principal point, so K is diag(f, f, 1).
    """
    cx, cy = principal
    shift = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    centered = shift @ H
    estimates = []
    for sign in (1.0, -1.0):
        h1 = sign * centered[:, 0]
        h2 = sign * centered[:, 1]
        dot_den = h1[2] * h2[2]
        if abs(dot_den) > 1e-9:
            f2 = -(h1[0] * h2[0] + h1[1] * h2[1]) / dot_den
            if f2 > 1.0e4:
                estimates.append(float(np.sqrt(f2)))
        len_den = h2[2] ** 2 - h1[2] ** 2
        if abs(len_den) > 1e-9:
            f2 = ((h1[0] ** 2 + h1[1] ** 2) - (h2[0] ** 2 + h2[1] ** 2)) / len_den
            if f2 > 1.0e4:
                estimates.append(float(np.sqrt(f2)))
    if not estimates:
        return None
    return float(np.median(estimates))


def _pose(
    world_xyz: np.ndarray,
    image_uv: np.ndarray,
    K: np.ndarray,
    ground: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    world_cv = _object_points(world_xyz)
    obj = np.ascontiguousarray(world_cv[ground], dtype=np.float64)
    img = np.ascontiguousarray(image_uv[ground], dtype=np.float64)
    ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, img, K, None, flags=cv2.SOLVEPNP_IPPE)
    pose = _upright_pose(rvecs if ok else [], tvecs if ok else [], obj, img, K)
    if pose is None:
        ok, rvec, tvec = cv2.solvePnP(world_cv, image_uv, K, None, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            raise RuntimeError("camera pose failed")
        pose = (rvec.reshape(3), tvec.reshape(3))
    rvec, tvec = pose
    if np.any(~ground):
        ok, rvec, tvec = cv2.solvePnP(
            world_cv,
            image_uv,
            K,
            None,
            rvec.reshape(3, 1),
            tvec.reshape(3, 1),
            True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if ok:
            pose = (rvec.reshape(3), tvec.reshape(3))
    return pose


def _object_points(xyz: np.ndarray) -> np.ndarray:
    """Court meters with Y negated, for OpenCV.

    World +Y points the same way as pixel y, toward the near baseline, and
    +Z points up. OpenCV's camera has +Y down and +Z along the optical axis.
    A camera on the near side then looks along world -Y. Image-right,
    image-down, and that optical axis are a left-handed triad, so a pose
    solved directly in court meters is the reflection that puts the camera
    under the court and draws the net posts downward. Negating Y makes object
    +Y point toward the far baseline, and the pose is a real rotation.
    ``Camera.center`` negates Y again on the way back to court meters.
    """
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3).copy()
    pts[:, 1] *= -1.0
    return pts


def _upright_pose(rvecs, tvecs, obj, img, K) -> tuple[np.ndarray, np.ndarray] | None:
    """The above-court pose with the smaller reprojection error.

    A planar target has two poses. Both can sit above the court once Y has
    been negated, and only one of them matches the image.
    """
    best = None
    best_err = np.inf
    img = np.asarray(img, dtype=np.float64).reshape(-1, 2)
    for rvec, tvec in zip(rvecs, tvecs):
        R, _ = cv2.Rodrigues(rvec)
        t = np.asarray(tvec, dtype=np.float64).reshape(3)
        center = -R.T @ t
        depth = (R @ obj.T) + t.reshape(3, 1)
        if center[2] <= 0.5 or np.any(depth[2] <= 0.05):
            continue
        pred, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
        err = float(np.mean(np.sum((pred.reshape(-1, 2) - img) ** 2, axis=1)))
        if err < best_err:
            best = (np.asarray(rvec, dtype=np.float64).reshape(3), t)
            best_err = err
    return best


def _refine(
    focal: float,
    principal: tuple[float, float],
    rvec: np.ndarray,
    tvec: np.ndarray,
    world_xyz: np.ndarray,
    image_uv: np.ndarray,
    *,
    refine_focal: bool,
) -> Camera:
    seed = Camera(float(focal), principal, np.asarray(rvec, dtype=np.float64), np.asarray(tvec, dtype=np.float64))

    def residual(packed: np.ndarray) -> np.ndarray:
        if refine_focal:
            camera = Camera(float(packed[0]), principal, packed[1:4], packed[4:7])
        else:
            camera = Camera(float(focal), principal, packed[:3], packed[3:6])
        err = (camera.project(world_xyz) - image_uv).ravel()
        bad = np.repeat(camera.depth(world_xyz) <= 0.05, 2)
        err = err.copy()
        err[bad | ~np.isfinite(err)] = 1.0e3
        return err

    if refine_focal:
        packed = np.array([focal, *seed.rvec.reshape(3), *seed.tvec.reshape(3)], dtype=np.float64)
        lower = np.array([100.0, *[-np.inf] * 6])
        upper = np.array([2.0e4, *[np.inf] * 6])
        solved = least_squares(residual, packed, bounds=(lower, upper), method="trf")
        camera = Camera(
            float(solved.x[0]), principal, solved.x[1:4].copy(), solved.x[4:7].copy()
        )
    else:
        packed = np.array([*seed.rvec.reshape(3), *seed.tvec.reshape(3)], dtype=np.float64)
        solved = least_squares(residual, packed, method="trf")
        camera = Camera(float(focal), principal, solved.x[:3].copy(), solved.x[3:6].copy())
    if camera.center()[2] <= 0.5 or np.any(camera.depth(world_xyz) <= 0.05):
        return seed
    return camera
