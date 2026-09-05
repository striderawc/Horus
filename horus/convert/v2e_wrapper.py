"""Thin wrapper around the SensorsINI/v2e CLI."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml


def resolve_v2e_executable(name: str = "v2e") -> str:
    """Return path to v2e on PATH, or raise with install hints."""
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(
        f"'{name}' not found on PATH. Install SensorsINI/v2e first:\n"
        "  git clone https://github.com/SensorsINI/v2e.git third_party/v2e\n"
        "  cd third_party/v2e && pip install -e .\n"
        "See README.md for details."
    )


def build_v2e_command(
    input_dir: Path,
    output_folder: Path,
    cfg: dict[str, Any],
    h5_name: str = "events.h5",
) -> list[str]:
    """Build the v2e CLI argv from ``configs/data.yaml`` ``v2e`` section."""
    v2e_cfg = cfg.get("v2e", {})
    exe = resolve_v2e_executable(v2e_cfg.get("executable", "v2e"))
    cmd: list[str] = [
        exe,
        "-i",
        str(input_dir),
        "--output_folder",
        str(output_folder),
        # Keep writes in the exact folder Horus expects (not output1, output2, …)
        "--unique_output_folder",
        "false",
        "--overwrite",
        "--input_frame_rate",
        str(v2e_cfg.get("input_frame_rate", 30)),
        "--dvs_h5",
        h5_name,
        "--no_preview",
    ]
    if v2e_cfg.get("disable_slomo", True):
        cmd.append("--disable_slomo")
    if v2e_cfg.get("skip_video_output", True):
        cmd.append("--skip_video_output")

    ow = v2e_cfg.get("output_width")
    oh = v2e_cfg.get("output_height")
    if ow:
        cmd.extend(["--output_width", str(int(ow))])
    if oh:
        cmd.extend(["--output_height", str(int(oh))])

    dvs_params = v2e_cfg.get("dvs_params")
    if dvs_params:
        cmd.extend(["--dvs_params", str(dvs_params)])

    extra = v2e_cfg.get("extra_args") or []
    cmd.extend(str(a) for a in extra)
    return cmd


def _normalize_events_h5(output_dir: Path, events_path: Path) -> Path:
    """Ensure ``events.h5`` exists at the expected path after a v2e run."""
    if events_path.is_file():
        return events_path
    candidates = list(output_dir.rglob("*.h5"))
    if not candidates:
        raise FileNotFoundError(f"No .h5 found under {output_dir}")
    preferred = [c for c in candidates if c.name == "events.h5"]
    src = preferred[0] if preferred else candidates[0]
    if src.resolve() != events_path.resolve():
        if events_path.exists():
            events_path.unlink()
        src.replace(events_path)
    return events_path


def run_v2e_on_clip(
    clip_dir: Path,
    output_dir: Path,
    cfg: dict[str, Any],
    *,
    force: bool = False,
) -> Path:
    """Convert one clip folder of jpgs to ``events.h5``. Returns path to HDF5."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "events.h5"
    sidecar = output_dir / "v2e_run.yaml"

    if events_path.is_file() and not force:
        return events_path

    cmd = build_v2e_command(Path(clip_dir), output_dir, cfg, h5_name="events.h5")
    sidecar.write_text(
        yaml.safe_dump(
            {
                "command": cmd,
                "clip_dir": str(clip_dir),
                "output_dir": str(output_dir),
                "v2e": cfg.get("v2e", {}),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        # Older forks used --skip_video_files instead of --skip_video_output
        err = (result.stderr or "") + (result.stdout or "")
        if "skip_video_output" in err and "--skip_video_files" not in cmd:
            alt = [a if a != "--skip_video_output" else "--skip_video_files" for a in cmd]
            result = subprocess.run(alt, check=False, capture_output=True, text=True)
            cmd = alt
        if result.returncode != 0:
            raise RuntimeError(
                "v2e failed with code "
                f"{result.returncode}\nCMD: {' '.join(cmd)}\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )

    try:
        return _normalize_events_h5(output_dir, events_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"v2e finished but no .h5 found under {output_dir}. "
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        ) from exc
