# Horus

Neuromorphic tennis ball tracker. The goal is to predict ball position over time from:

1. **RGB** video frames (baseline)
2. **Frame differencing**
3. **V2E-simulated event streams** (event frames and voxel grids)

Tracks feed later stages (court geometry → bounce location and ball speed). Models include heatmap CNNs and an event-frame SNN.

## Pipeline stages

1. **Tracking (`horus/track`)** — SSM stage-1 filters over ball kinematics (`cv_kf` Kalman, `cv_akf` adaptive CV, `ca_akf` adaptive CA with acceleration in the state, `cv_ekf` / `ca_ekf` EKFs with quadratic drag / optional gravity, `cv_pf` particle filter). Filters expose `gate_window` / `snap_square_window` for uncertainty-sized detector crops and accept per-step `r_meas` from the detector.
2. **Detection (`horus/models`)** — heatmap CNN / SNN localizers. Full-frame by default; enable `crop.enabled` (or `--crop`) for jittered bucket crops (64/128/256) with negatives and a `log_σ` head used as KF measurement noise.

```bash
python scripts/smoke_kf.py
python scripts/smoke_kf.py --plot   # optional matplotlib path plot
python scripts/smoke_track_detect.py --tracker cv_kf
```

Config knobs live under `tracker:` and `crop:` in [`configs/data.yaml`](configs/data.yaml).

## Dataset

Uses the [TrackNet](https://nol.cs.nctu.edu.tw/ythuang/epapers/ICIP18_TrackNet.pdf) tennis dataset (place under `Dataset/`, gitignored):

```text
Dataset/
  game1/Clip1/0000.jpg … Label.csv
  …
  game10/…
```

Frames are 1280×720 @ 30 fps. Each `Label.csv` has `file name`, `visibility`, `x-coordinate`, `y-coordinate`, `status`.

Default splits (by game, no clip leakage):

| Split | Games |
|-------|-------|
| train | game1–game7 |
| val   | game8–game9 |
| test  | game10 |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Config lives in [`configs/data.yaml`](configs/data.yaml).

## Modalities

### RGB & frame differencing

Loaded on-the-fly by `BallTrackDataset` (no convert step required):

```bash
python scripts/smoke_dataloader.py --modality rgb
python scripts/smoke_dataloader.py --modality diff
```

### Events (V2E)

Horus **wraps** [SensorsINI/v2e](https://github.com/SensorsINI/v2e) as an external CLI — it is not vendored into this package.

```bash
# one-time install (recommended under third_party/, gitignored)
git clone https://github.com/SensorsINI/v2e.git third_party/v2e
cd third_party/v2e
# follow upstream README: same venv as Horus + PyTorch, then:
pip install -e .
cd ../..
```

**Compatibility notes (this repo’s `.venv`):**

- Use the Horus venv so `v2e` is on `PATH` (`.venv/bin/v2e`).
- Interactive file dialogs need `python3-tk`; CLI use with `-i` does **not**. This tree patches `v2ecore/v2e_utils.py` so missing tkinter does not block CLI runs.
- TrackNet `Label.csv` in clip folders is ignored by the patched image-folder reader (images only).
- v2e HDF5 layout is dataset `events` with shape `(N, 4)` uint32 `[t_us, x, y, p]`; Horus converts `t` to seconds when loading.
- Wrapper forces `--overwrite` and `--unique_output_folder false` so outputs land in `data/processed/events/<game>/<clip>/events.h5`.
- Optional: `sudo apt install python3-tk` if you want v2e’s GUI file picker.

```bash
# convert one clip (smoke)
python scripts/convert_v2e.py --clip game1/Clip1 --force

# convert all clips (skips existing events.h5 unless --force)
python scripts/convert_v2e.py

# bin raw events → event frames + voxels
python scripts/precompute_events.py --clip game1/Clip1
python scripts/precompute_events.py

python scripts/smoke_dataloader.py --modality event_frame
python scripts/smoke_dataloader.py --modality voxel
```

Until v2e is installed, you can smoke-test the event → frame/voxel → dataloader path with a synthetic generator (not for training experiments):

```bash
python scripts/synthesize_events.py --clip game1/Clip1
python scripts/precompute_events.py --clip game1/Clip1
python scripts/smoke_dataloader.py --modality event_frame
```

Processed layout:

```text
data/processed/events/<game>/<clip>/
  events.h5          # raw (t, x, y, p) from v2e
  event_frames.h5    # polarity images per frame
  voxels.h5          # voxel grids per frame
  v2e_run.yaml       # conversion settings used
```

## Dataloader sample

Each sample includes:

- `input` — modality tensor (RGB / diff / event_frame / voxel)
- `heatmap` — Gaussian target `(1, H, W)`
- `coords` — `(x, y)` in training resolution
- `visibility`, `status`, plus clip metadata

Label is taken from the **last** frame of each length-`N` window (`sequence.length` in config, fixed at **3** for the heatmap models).

## Heatmap CNNs (RGB / frame-diff)

Same 2D encoder–decoder (`BallHeatmapCNN`); only input channels change:

| Modality | Input | Checkpoint |
|----------|-------|------------|
| `rgb` | 3 RGB frames → `(9, H, W)` | `checkpoints/rgb_cnn.pt` |
| `diff` | 2 grayscale diffs → `(2, H, W)` | `checkpoints/diff_cnn.pt` |
| `event_frame` | 3 polarity frames → `(6, H, W)` | `checkpoints/event_frame_cnn.pt` |

`event_frame` needs precomputed `data/processed/events/<game>/<clip>/event_frames.h5` (run `convert_v2e` + `precompute_events`). Clips without those files are skipped; train **and** val games need coverage.

Training hyperparams live under `train:` in [`configs/data.yaml`](configs/data.yaml) (`epochs`, `lr`, `batch_size`, `num_workers`). CLI flags override them. Loss is Euclidean pixel distance / image diagonal (CNN: hard-argmax heatmap; SNN: denormalized sigmoid `(x, y)`); `val_px` stays in raw pixels.

Every train run writes TensorBoard logs under `runs/<model>/<utc-timestamp>/` (loss, Acc@5/10, mean_px, grad norm, LR; SNN also logs per-layer activity). View with:

```bash
tensorboard --logdir runs
```

```bash
python scripts/train_cnn.py --modality rgb
python scripts/train_cnn.py --modality diff
python scripts/train_cnn.py --modality event_frame
# shortcuts:
python scripts/train_rgb.py
python scripts/train_diff.py
python scripts/train_event.py
# override config:
python scripts/train_diff.py --epochs 20 --batch-size 8 --lr 1e-3
# disable TB for a quick smoke run:
python scripts/train_rgb.py --no-tensorboard --epochs 1
```

## Event-frame SNN

`BallHeatmapSNN` uses the same `event_frame` samples as the CNN (`(6, H, W)` by default). It unrolls them as polarity steps (2 channels each) through a **shallow 2-LIF encoder at full resolution** (no stride / pooling), then a **1×1 head → sigmoid** heatmap `(1, H, W)`. Train loss is **MSE** against a small disk of 1s around the ball (`snn.circle_radius`); `val_px` / Acc@k still come from the heatmap argmax. Leak, threshold, width, optional `steps`, and `repeat_frame` live under `snn:` in [`configs/data.yaml`](configs/data.yaml). Checkpoint: `checkpoints/event_frame_snn.pt` (retrain after architecture changes).

```bash
python scripts/train_snn.py
# overfit ablation: final lif2 is ReLU (no LIF) before the head
python scripts/train_snn.py --overfit --drop-last-lif
# same event frame tiled across timesteps (e,e,…,e); adjust length with --steps
python scripts/train_snn.py --overfit --repeat-frame --steps 5
```

## Evaluating models

Bench any heatmap model on accuracy (`mean_px`, Acc@5/10), FLOPs, and latency. Same metrics for every architecture so results are comparable.

```bash
pip install 'horus[eval]'   # fvcore for FLOPs (optional but recommended)

python scripts/eval_models.py \
  --checkpoint checkpoints/rgb_cnn.pt \
  --arch rgb_cnn \
  --split test

python scripts/eval_models.py \
  --checkpoint checkpoints/diff_cnn.pt \
  --arch diff_cnn \
  --split test

python scripts/eval_models.py \
  --checkpoint checkpoints/event_frame_cnn.pt \
  --arch event_frame_cnn \
  --split test

python scripts/eval_models.py \
  --checkpoint checkpoints/event_frame_snn.pt \
  --arch event_frame_snn \
  --split test
```

Compare multiple models with repeated `--model` specs (`name=...,ckpt=...,arch=...,modality=...`). The table reports best accuracy (lowest `mean_px`) and best efficiency (`acc@5 / GFLOPs`, `acc@5 / latency_ms`). JSON is written under `results/eval/`.

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/index_dataset.py` | Print clip / split / sample counts |
| `scripts/smoke_dataloader.py` | Load one batch; print tensor shapes |
| `scripts/train_cnn.py` | Train RGB / frame-diff / event-frame heatmap CNN |
| `scripts/train_rgb.py` | Shortcut → `train_cnn --modality rgb` |
| `scripts/train_diff.py` | Shortcut → `train_cnn --modality diff` |
| `scripts/train_event.py` | Shortcut → `train_cnn --modality event_frame` |
| `scripts/train_snn.py` | Train event-frame coord SNN |
| `scripts/eval_models.py` | Accuracy / FLOPs / latency bench |
| `scripts/convert_v2e.py` | Run v2e → `events.h5` (`--clip`, `--force`, `--limit`) |
| `scripts/precompute_events.py` | Bin events → frames + voxels (same flags; skips existing) |
| `scripts/synthesize_events.py` | Dev-only synthetic `events.h5` without v2e |

Batch conversion is resume-friendly: existing outputs are skipped unless you pass `--force`.

## Project status

| Step | Status |
|------|--------|
| 1 Learn | — |
| 2 Collect / prepare dataset | **in progress** |
| 3 Train RGB CNN, Diff CNN, V2E SNN | RGB / Diff / event-frame CNN and event-frame SNN ready |
| 4 Metrics (bounce, speed) | later |
| 5 App | later |
