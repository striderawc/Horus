# Horus

Neuromorphic tennis ball tracker. The goal is to predict ball position over time from:

1. **RGB** video frames (baseline)
2. **Frame differencing**
3. **V2E-simulated event streams** (event frames and voxel grids)

Tracks feed later stages (court geometry → bounce location and ball speed). Models will include a CNN baseline and an SNN; this repo phase focuses on the **dataset pipeline**.

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

Label is taken from the **last** frame of each length-`N` window (`sequence.length` in config, default 3).

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/index_dataset.py` | Print clip / split / sample counts |
| `scripts/smoke_dataloader.py` | Load one batch; print tensor shapes |
| `scripts/convert_v2e.py` | Run v2e → `events.h5` (`--clip`, `--force`, `--limit`) |
| `scripts/precompute_events.py` | Bin events → frames + voxels (same flags; skips existing) |
| `scripts/synthesize_events.py` | Dev-only synthetic `events.h5` without v2e |

Batch conversion is resume-friendly: existing outputs are skipped unless you pass `--force`.

## Project status

| Step | Status |
|------|--------|
| 1 Learn | — |
| 2 Collect / prepare dataset | **in progress** |
| 3 Train RGB CNN, Diff CNN, V2E SNN | later |
| 4 Metrics (bounce, speed) | later |
| 5 App | later |
