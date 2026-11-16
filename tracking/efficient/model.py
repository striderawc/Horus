"""CNN crop detector plus a constant-velocity AKF.

``EfficientTracker.find_positions`` walks one clip. Each frame is a short
sequence you can change in place:

1. Signed grayscale differences.
2. No track yet: tile the frame (or take a baseline peak) and initiate.
3. Track running: predict, then grow one crop on that prediction. With
   ``start_at_uncertainty`` the first side covers the predicted gate;
   otherwise every search starts at ``min_side``. A miss still steps up
   the ladder. If the hybrid baseline is on, a high predicted std or a
   failed crop ladder falls back to the full-frame baseline before a
   miss update.
4. Clip the heatmap sigma and update. A miss is ``update(None)``.

The uncertainty gate is stored on the step for display. A peak is kept
wherever the crop found it. Each step also keeps the posterior state and
covariance, and the predicted state and covariance from before the update,
so a backward pass can smooth the finished track.

``Rules`` holds the knobs. Baseline hooks stay off, so the default path is
the crop ladder. The full baseline network is loaded either way.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from data.crop_dataloader import SRC_H, SRC_W, _crop
from models.baseline_cnn.model import BaselineCNN
from models.detection.cnn_detector.model import CNNDetector
from tracking.ssm.base_akf.model import SSM
from tracking.ssm.base_akf.track import gate

ROOT = Path(__file__).resolve().parents[2]

# 4-state constant-velocity AKF with a 2-D measurement. A matmul is 2*m*n*k,
# the same convention as FlopCounterMode. A 2x2 solve is (2/3)*n^3 + 2*n^2*k.
# predict: scale mul, F@x (32), F@P (128), that@F.T (128), add Q (16), symmetrize (32).
_SSM_PREDICT = 337
# initiate writes the 4-state and its four diagonal variances. No matmul.
_SSM_INITIATE = 8
# update(y): r^2, innovation, two solves (1 and 4 rhs), gain, covariance downdate, symmetrize.
_SSM_UPDATE = 204
# update(None): keep velocity (or damp by miss_vel_decay), grow two position variances, symmetrize.
_SSM_UPDATE_MISS = 37


@dataclass
class Rules:
    """Detector and baseline knobs. Edit these, or pass your own instance."""

    # Peak test on the CNN heatmap.
    conf_min: float = 0.60
    peak_frac: float = 0.30
    # Sides are min_side, then side_step apart, up to max_side. A frame is
    # a miss only after max_side also has no ball.
    min_side: int = 48
    side_step: int = 48
    max_side: int = 720
    multiple: int = 4
    search_batch: int = 16
    # False: every tracked frame starts at min_side and grows.
    # True: the first side is the smallest rung that covers the predicted
    # uncertainty frame, then grows from there on a miss.
    start_at_uncertainty: bool = False
    # Heatmap sigma (source pixels) is clipped into this range and passed to the AKF.
    r_min: float = 2.0
    r_max: float = 40.0
    # Off by default. Any of these turns on the hybrid baseline path (needs rgbs):
    # start from the baseline, ask it when predicted position std exceeds
    # uncertainty_px, and/or ask it when the crop ladder finds no peak.
    use_baseline_to_start: bool = False
    uncertainty_px: float | None = None
    use_baseline_on_miss: bool = False


@dataclass
class Meas:
    """One detector reading. ``xy`` is None when nothing fired."""

    xy: np.ndarray | None
    spread: float
    spread_x: float
    spread_y: float
    conf: float
    heat: np.ndarray | None = None
    origin: tuple[int, int] | None = None
    side: int | None = None
    source: str | None = None


def _side_limit(rules: Rules) -> int:
    return max(rules.min_side, min(rules.max_side, max(SRC_W, SRC_H)))


def snap_side(span: float, rules: Rules) -> int:
    """Smallest ``side_step`` rung that covers ``span``.

    Rungs are ``min_side``, then ``side_step`` apart, up to ``max_side``.
    A span below ``min_side`` starts at ``min_side``.
    """
    limit = _side_limit(rules)
    step = max(rules.side_step, rules.multiple)
    side = rules.min_side
    while side < span and side < limit:
        nxt = min(limit, side + step)
        if nxt == side:
            break
        side = nxt
    return side


def crop_sides(rules: Rules, span: float = 0.0) -> list[int]:
    """Ladder up to ``max_side``, on the ``side_step`` rungs.

    ``span`` is the predicted uncertainty frame. It is used only when
    ``start_at_uncertainty`` is set; otherwise the ladder starts at
    ``min_side``. A zero span is the full ladder either way, which is
    what the benchmark flop table counts.
    """
    if not rules.start_at_uncertainty:
        span = 0.0
    limit = _side_limit(rules)
    side = snap_side(span, rules)
    sides = [side]
    step = max(rules.side_step, rules.multiple)
    while side < limit:
        side = min(limit, side + step)
        if side != sides[-1]:
            sides.append(side)
    return sides


def measurement_std(spread: float, rules: Rules) -> float:
    """Heatmap sigma in source pixels, clipped for the AKF."""
    if not np.isfinite(spread):
        return rules.r_max
    return float(np.clip(spread, rules.r_min, rules.r_max))


def spike_filter(ssm: SSM, meas: Meas) -> None:
    """Extra uncertainty after predict and before update.

    Misses already grow ``P`` inside ``update(None)``. A bump here belongs to
    this update; whatever is left widens the next gate.
    """
    return


def _miss() -> Meas:
    return Meas(None, float("nan"), float("nan"), float("nan"), float("nan"))


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return path


def _load_run(folder: str | Path) -> tuple[Path, dict, dict]:
    run_dir = _resolve(folder)
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    with (run_dir / "config.yaml").open() as f:
        run_cfg = yaml.safe_load(f)
    checkpoint = next(run_dir.glob("*.pt"), None)
    if checkpoint is None:
        raise FileNotFoundError(f"no checkpoint in {run_dir}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    return checkpoint, run_cfg, state


class EfficientTracker:
    """Loaded CNN detector, loaded baseline, and one AKF clip runner."""

    def __init__(
        self,
        cnn_dir: str | Path = "models/detection/cnn_detector/saved_models/cnn_detector",
        baseline_dir: str | Path = "models/baseline_cnn/saved_models/baseline_cnn_train1",
        ssm_config: str | Path = "configs/base_akf.yaml",
        rules: Rules | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.rules = rules or Rules()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with _resolve(ssm_config).open() as f:
            ssm_file = yaml.safe_load(f)
        self.ssm_cfg = dict(ssm_file["tracker"])
        self.gate_sigma = float(self.ssm_cfg["gate_sigma"])

        self.cnn_checkpoint, cnn_cfg, cnn_state = _load_run(cnn_dir)
        self.num_diffs = int(cnn_cfg["model"]["num_diffs"])
        self.cnn = CNNDetector(in_channels=self.num_diffs).to(self.device).eval()
        self.cnn.load_state_dict(cnn_state["model"])

        self.baseline_checkpoint, base_cfg, base_state = _load_run(baseline_dir)
        self.num_frames = int(base_cfg["model"]["num_frames"])
        self.channels_per_frame = int(base_cfg["model"]["channels_per_frame"])
        self.out_w = int(base_cfg["image"]["width"])
        self.out_h = int(base_cfg["image"]["height"])
        self.baseline = BaselineCNN(self.num_frames, self.channels_per_frame).to(self.device).eval()
        self.baseline.load_state_dict(base_state["model"])

        # Every detector forward, including a search that does not lock.
        self.forward_counts: dict[int, int] = {}
        self._frame_forwards: list[tuple[int, int]] = []
        self._baseline_forwards = 0
        self._ssm_flops = 0

    def find_positions(
        self,
        grays: list[np.ndarray],
        rgbs: list[np.ndarray] | None = None,
        keep_heat: bool | set[int] = False,
        start_xy: np.ndarray | None = None,
        start_index: int = 0,
    ) -> list[dict | None]:
        """Return one step per frame. ``None`` until the filter locks.

        ``grays`` are source-resolution frames. ``rgbs`` are model-resolution,
        channel-first frames, required only when a baseline hook is on.
        ``keep_heat`` stores the winning CNN heatmap (True, or a set of indices).
        ``start_xy`` seeds the filter at ``start_index`` and skips the search.
        """
        if rgbs is not None and len(rgbs) != len(grays):
            raise ValueError("rgbs and grays must be the same length")

        ssm = SSM(self.ssm_cfg)
        steps: list[dict | None] = []
        for i in range(len(grays)):
            store = _wants_heat(keep_heat, i)

            if ssm.mean is None and start_xy is not None:
                if i < start_index:
                    steps.append(None)
                    continue
                if i == start_index: # seed the filter with the start_xy
                    steps.append(self._seed(ssm, start_xy))
                    continue

            if ssm.mean is None and not self._opening_ready(i): # if the filter is not initialized and the opening is not ready, skip
                steps.append(None)
                continue

            self._begin_frame()
            t0 = time.perf_counter()

            if ssm.mean is None:
                meas = self._opening(grays, rgbs, i, store) # get the opening measurement
                if meas.xy is None:
                    self._sync()
                    steps.append(None)
                    continue
                ssm.initiate(meas.xy) # initialize the filter with the opening measurement
                self._ssm_flops += _SSM_INITIATE
                box = gate(ssm.mean, ssm.cov, self.gate_sigma)
                self._sync()
                steps.append(
                    self._record(
                        ssm,
                        None,
                        box,
                        meas.xy,
                        meas,
                        measurement_std(meas.spread, self.rules),
                        _ms(t0),
                    )
                )
                continue

            ssm.predict()
            self._ssm_flops += _SSM_PREDICT
            # Snapshot before update. The miss path and the innovation bump
            # both write these arrays in place, and the smoother needs the
            # predicted Gaussian intact.
            pred_state = ssm.mean.copy()
            pred_cov = ssm.cov.copy()
            pred = pred_state[:2].copy()
            box = gate(ssm.mean, ssm.cov, self.gate_sigma)
            # The gate is the uncertainty frame. start_at_uncertainty uses
            # it as the first crop; otherwise the ladder starts at min_side.
            # Hybrid baseline: high predicted std, or a crop ladder that
            # never finds a peak, both fall back to baseline.
            meas = self._measure_tracked(grays, rgbs, i, ssm, pred, box, store)
            used, r_std = self._apply(ssm, meas)
            self._sync()
            steps.append(
                self._record(
                    ssm, pred, box, used, meas, r_std, _ms(t0), pred_state, pred_cov
                )
            )
        return steps

    def needs_rgbs(self) -> bool:
        """True when any hybrid baseline hook is on (callers must pass rgbs)."""
        return self._uses_baseline()

    def differences(self, grays: list[np.ndarray], i: int) -> list[np.ndarray] | None:
        """``num_diffs`` signed frames ending at ``i``. None before the window fills."""
        if i < self.num_diffs:
            return None
        window = grays[i - self.num_diffs : i + 1]
        return [
            (window[k].astype(np.float32) - window[k - 1].astype(np.float32)) / 255.0
            for k in range(1, len(window))
        ]

    def search_frame(self, diffs: list[np.ndarray], store: bool) -> Meas:
        """Best peak on ``max_side`` tiles covering the whole frame."""
        origins = self._search_origins(self.rules.max_side, None)
        best: Meas | None = None
        best_conf = -1.0
        low_conf = float("nan")
        batch = self.rules.search_batch
        for start in range(0, len(origins), batch):
            chunk = origins[start : start + batch]
            heats = self._forward([self._stack(diffs, x0, y0, self.rules.max_side) for x0, y0 in chunk])
            for (x0, y0), heat in zip(chunk, heats):
                meas = self._place(heat, x0, y0, self.rules.max_side, store)
                if meas.xy is None:
                    if not np.isfinite(low_conf) or meas.conf > low_conf:
                        low_conf = float(meas.conf)
                    continue
                if meas.conf <= best_conf:
                    continue
                best = meas
                best_conf = meas.conf
        if best is not None:
            return best
        missed = _miss()
        missed.conf = low_conf
        missed.source = "cnn"
        return missed

    def grow_crop(
        self,
        diffs: list[np.ndarray],
        pred: np.ndarray,
        store: bool,
        span: float,
    ) -> Meas:
        """Grow one crop around ``pred`` until a peak appears or the crop is maxed.

        ``span`` is the predicted uncertainty frame. It sets the first side
        only when ``start_at_uncertainty`` is on. A crop with no peak is
        not a miss; the side still steps up.
        """
        missed = _miss()
        for side in crop_sides(self.rules, span):
            x0 = int(round(float(pred[0]) - side / 2))
            y0 = int(round(float(pred[1]) - side / 2))
            heat = self._forward([self._stack(diffs, x0, y0, side)])[0]
            meas = self._place(heat, x0, y0, side, store)
            if meas.xy is None:
                if not np.isfinite(missed.conf) or meas.conf >= missed.conf:
                    missed = meas
                continue
            return meas
        return missed

    def baseline_position(self, rgbs: list[np.ndarray] | None, i: int) -> Meas:
        """Baseline heatmap argmax in source pixels.

        The position is the argmax, mapped the same way as the baseline runner.
        Spread is the peak blob, scaled into source pixels. Below ``conf_min``
        the reading is a miss.
        """
        if rgbs is None:
            raise ValueError(
                "pass rgbs when use_baseline_to_start, uncertainty_px, "
                "or use_baseline_on_miss is set"
            )
        if i < self.num_frames - 1:
            return _miss()

        window = rgbs[i - self.num_frames + 1 : i + 1]
        x = torch.from_numpy(np.concatenate(window, axis=0)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            heat_t = self.baseline(x)[0, 0]
        self._baseline_forwards += 1
        heat = heat_t.detach().cpu().numpy()
        meas = self._read_peak(heat)
        meas.source = "baseline"
        if meas.xy is None:
            return meas

        height, width = heat_t.shape
        flat = int(heat_t.argmax().item())
        sx = SRC_W / width
        sy = SRC_H / height
        meas.xy = np.array([(flat % width) * sx, (flat // width) * sy], dtype=np.float64)
        meas.spread_x *= sx
        meas.spread_y *= sy
        meas.spread = math.sqrt(max(0.0, 0.5 * (meas.spread_x**2 + meas.spread_y**2)))
        return meas

    def _opening(
        self,
        grays: list[np.ndarray],
        rgbs: list[np.ndarray] | None,
        i: int,
        store: bool,
    ) -> Meas:
        if self.rules.use_baseline_to_start:
            return self.baseline_position(rgbs, i)
        diffs = self.differences(grays, i)
        if diffs is None:
            return _miss()
        return self.search_frame(diffs, store)

    def _opening_ready(self, i: int) -> bool:
        if self.rules.use_baseline_to_start:
            return i >= self.num_frames - 1
        return i >= self.num_diffs

    def _uses_baseline(self) -> bool:
        return (
            self.rules.use_baseline_to_start
            or self.rules.use_baseline_on_miss
            or self.rules.uncertainty_px is not None
        )

    def _uncertain(self, ssm: SSM) -> bool:
        limit = self.rules.uncertainty_px
        if limit is None:
            return False
        std = np.sqrt(np.maximum(np.diag(ssm.cov)[:2], 0.0))
        return float(std[0]) > limit or float(std[1]) > limit

    def _measure_tracked(
        self,
        grays: list[np.ndarray],
        rgbs: list[np.ndarray] | None,
        i: int,
        ssm: SSM,
        pred: np.ndarray,
        box: tuple[float, float, float, float],
        store: bool,
    ) -> Meas:
        """Crop around the prediction, with hybrid baseline as a fallback.

        High predicted std skips the crop ladder and goes straight to baseline.
        Otherwise grow from ``min_side``, or from the uncertainty frame
        ``box`` when ``start_at_uncertainty`` is on. If that never finds a
        peak and hybrid is on, ask the baseline before a miss.
        """
        if self._uncertain(ssm):
            return self.baseline_position(rgbs, i)

        diffs = self.differences(grays, i)
        span = max(box[2] - box[0], box[3] - box[1])
        meas = _miss() if diffs is None else self.grow_crop(diffs, pred, store, span)
        if meas.xy is not None:
            return meas
        if self._uses_baseline():
            return self.baseline_position(rgbs, i)
        return meas

    def _apply(self, ssm: SSM, meas: Meas) -> tuple[np.ndarray | None, float]:
        if meas.xy is None:
            spike_filter(ssm, meas)
            ssm.update(None)
            self._ssm_flops += _SSM_UPDATE_MISS
            return None, float("nan")
        r_std = measurement_std(meas.spread, self.rules)
        spike_filter(ssm, meas)
        ssm.update(meas.xy, r_std)
        self._ssm_flops += _SSM_UPDATE
        return meas.xy, r_std

    def _seed(self, ssm: SSM, xy: np.ndarray) -> dict:
        self._begin_frame()
        t0 = time.perf_counter()
        y = np.asarray(xy, dtype=np.float64)
        ssm.initiate(y)
        self._ssm_flops += _SSM_INITIATE
        box = gate(ssm.mean, ssm.cov, self.gate_sigma)
        meas = _miss()
        meas.source = "start"
        self._sync()
        return self._record(ssm, None, box, y, meas, float("nan"), _ms(t0))

    def _read_peak(self, heat: np.ndarray) -> Meas:
        """Weighted centroid of pixels at or above ``peak_frac`` of the peak."""
        conf = float(heat.max()) if heat.size else 0.0
        if conf < self.rules.conf_min:
            meas = _miss()
            meas.conf = conf
            return meas
        ys, xs = np.nonzero(heat >= self.rules.peak_frac * conf)
        weight = heat[ys, xs].astype(np.float64)
        weight /= weight.sum()
        mx = float(xs @ weight)
        my = float(ys @ weight)
        var_x = float(((xs - mx) ** 2) @ weight)
        var_y = float(((ys - my) ** 2) @ weight)
        return Meas(
            np.array([mx, my], dtype=np.float64),
            math.sqrt(max(0.0, 0.5 * (var_x + var_y))),
            math.sqrt(max(var_x, 0.0)),
            math.sqrt(max(var_y, 0.0)),
            conf,
        )

    def _place(self, heat: np.ndarray, x0: int, y0: int, side: int, store: bool) -> Meas:
        meas = self._read_peak(heat)
        meas.source = "cnn"
        meas.side = side
        if meas.xy is not None:
            meas.xy = meas.xy + np.array([x0, y0], dtype=np.float64)
        if store:
            meas.heat = heat
            meas.origin = (x0, y0)
        return meas

    def _stack(self, diffs: list[np.ndarray], x0: int, y0: int, side: int) -> np.ndarray:
        return np.stack([_crop(d, x0, y0, side, side) for d in diffs], axis=0)

    def _forward(self, crops: list[np.ndarray]) -> np.ndarray:
        side = int(crops[0].shape[-1])
        n = len(crops)
        self.forward_counts[side] = self.forward_counts.get(side, 0) + n
        self._frame_forwards.append((side, n))
        batch = torch.from_numpy(np.stack(crops)).to(self.device)
        with torch.inference_mode():
            return self.cnn(batch)[:, 0].detach().cpu().numpy()

    def _search_origins(
        self,
        side: int,
        box: tuple[float, float, float, float] | None,
    ) -> list[tuple[int, int]]:
        if box is None:
            box = (0.0, 0.0, float(SRC_W), float(SRC_H))
        xs = self._axis_origins(box[0], box[2], side, SRC_W)
        ys = self._axis_origins(box[1], box[3], side, SRC_H)
        return [(x, y) for y in ys for x in xs]

    def _axis_origins(self, lo: float, hi: float, side: int, frame: int) -> list[int]:
        lo_i = max(0, int(math.floor(lo)))
        hi_i = min(frame, int(math.ceil(hi)))
        last = max(0, frame - side)
        if hi_i <= lo_i:
            return [0]
        if hi_i - lo_i <= side:
            origin = lo_i + (hi_i - lo_i - side) // 2
            return [min(max(0, origin), last)]
        stride = max(self.rules.multiple, side // 2)
        first = min(max(0, lo_i), last)
        end = min(max(0, hi_i - side), last)
        if end < first:
            first, end = end, first
        origins = list(range(first, end + 1, stride))
        if origins[-1] != end:
            origins.append(end)
        return origins

    def _record(
        self,
        ssm: SSM,
        pred: np.ndarray | None,
        box: tuple[float, float, float, float],
        z: np.ndarray | None,
        meas: Meas,
        r_std: float,
        ms: float,
        pred_state: np.ndarray | None = None,
        pred_cov: np.ndarray | None = None,
    ) -> dict:
        state = ssm.mean.copy()
        post = state[:2].copy()
        # ``box`` was built from the predicted covariance. Recover that std, and
        # also store the posterior position std after this update.
        pred_std = np.array(
            [
                (box[2] - box[0]) / (2.0 * self.gate_sigma),
                (box[3] - box[1]) / (2.0 * self.gate_sigma),
            ],
            dtype=np.float64,
        )
        pos_std = np.sqrt(np.maximum(np.diag(ssm.cov)[:2], 0.0)).copy()
        return {
            "xy": post.copy(),
            "pred": None if pred is None else np.asarray(pred, dtype=np.float64).copy(),
            "post": post,
            "vel": state[2:].copy(),
            "state": state,
            "cov": ssm.cov.copy(),
            "pred_state": None
            if pred_state is None
            else np.asarray(pred_state, dtype=np.float64).copy(),
            "pred_cov": None
            if pred_cov is None
            else np.asarray(pred_cov, dtype=np.float64).copy(),
            "pred_std": pred_std,
            "pos_std": pos_std,
            "box": box,
            "z": None if z is None else np.asarray(z, dtype=np.float64).copy(),
            "spread": float(meas.spread),
            "spread_x": float(meas.spread_x),
            "spread_y": float(meas.spread_y),
            "r_std": float(r_std),
            "conf": float(meas.conf),
            "scale": float(ssm.scale),
            "heat": meas.heat,
            "origin": meas.origin,
            "side": meas.side,
            "source": meas.source,
            "used": z is not None,
            "ms": ms,
            "forwards": list(self._frame_forwards),
            "baseline_forwards": self._baseline_forwards,
            "ssm_flops": self._ssm_flops,
        }

    def _begin_frame(self) -> None:
        self._frame_forwards = []
        self._baseline_forwards = 0
        self._ssm_flops = 0
        self._sync()

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize()


def _wants_heat(keep_heat: bool | set[int], i: int) -> bool:
    if isinstance(keep_heat, bool):
        return keep_heat
    return i in keep_heat


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0
