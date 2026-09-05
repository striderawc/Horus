"""Walk TrackNet folders and build clip / sample indices."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from horus.data.labels import FrameLabel, load_labels


_GAME_RE = re.compile(r"^game(\d+)$", re.IGNORECASE)
_CLIP_RE = re.compile(r"^Clip(\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class ClipInfo:
    """One rally clip under ``Dataset/gameN/ClipM``."""

    game: str
    clip: str
    path: Path
    frames: tuple[Path, ...]
    labels: tuple[FrameLabel, ...]

    @property
    def key(self) -> str:
        return f"{self.game}/{self.clip}"

    @property
    def num_frames(self) -> int:
        return len(self.frames)


@dataclass
class SampleRef:
    """Index into a clip for a sequence ending at ``end_idx`` (inclusive)."""

    clip_key: str
    end_idx: int
    game: str
    clip: str


def _natural_frame_sort(paths: list[Path]) -> list[Path]:
    def key(p: Path) -> tuple[int, str]:
        stem = p.stem
        return (int(stem) if stem.isdigit() else 10**9, stem)

    return sorted(paths, key=key)


def index_clips(dataset_root: str | Path) -> list[ClipInfo]:
    """Discover all clips with matching jpg frames and Label.csv."""
    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    clips: list[ClipInfo] = []
    game_dirs = sorted(
        (p for p in root.iterdir() if p.is_dir() and _GAME_RE.match(p.name)),
        key=lambda p: int(_GAME_RE.match(p.name).group(1)),  # type: ignore[union-attr]
    )
    for game_dir in game_dirs:
        clip_dirs = sorted(
            (p for p in game_dir.iterdir() if p.is_dir() and _CLIP_RE.match(p.name)),
            key=lambda p: int(_CLIP_RE.match(p.name).group(1)),  # type: ignore[union-attr]
        )
        for clip_dir in clip_dirs:
            label_path = clip_dir / "Label.csv"
            if not label_path.is_file():
                continue
            frames = _natural_frame_sort(list(clip_dir.glob("*.jpg")))
            if not frames:
                continue
            labels = load_labels(label_path)
            # Align by filename when possible
            label_by_name = {lb.frame: lb for lb in labels}
            aligned: list[FrameLabel] = []
            for fp in frames:
                lb = label_by_name.get(fp.name)
                if lb is None:
                    aligned.append(
                        FrameLabel(
                            frame=fp.name,
                            visibility=0,
                            x=0.0,
                            y=0.0,
                            status=-1,
                        )
                    )
                else:
                    aligned.append(lb)
            clips.append(
                ClipInfo(
                    game=game_dir.name,
                    clip=clip_dir.name,
                    path=clip_dir,
                    frames=tuple(frames),
                    labels=tuple(aligned),
                )
            )
    return clips


def split_clips(
    clips: list[ClipInfo],
    splits: dict[str, list[str]],
) -> dict[str, list[ClipInfo]]:
    """Partition clips by game name lists in config ``splits``."""
    by_game = {c.game: [] for c in clips}  # type: ignore[var-annotated]
    for c in clips:
        by_game.setdefault(c.game, []).append(c)

    out: dict[str, list[ClipInfo]] = {}
    for split_name, games in splits.items():
        selected: list[ClipInfo] = []
        for g in games:
            selected.extend(by_game.get(g, []))
        out[split_name] = selected
    return out


def build_sample_refs(
    clips: list[ClipInfo],
    sequence_length: int,
) -> list[SampleRef]:
    """Emit one sample per valid end frame (needs ``sequence_length`` frames)."""
    if sequence_length < 1:
        raise ValueError("sequence_length must be >= 1")
    refs: list[SampleRef] = []
    for clip in clips:
        n = clip.num_frames
        for end_idx in range(sequence_length - 1, n):
            refs.append(
                SampleRef(
                    clip_key=clip.key,
                    end_idx=end_idx,
                    game=clip.game,
                    clip=clip.clip,
                )
            )
    return refs


def clips_by_key(clips: list[ClipInfo]) -> dict[str, ClipInfo]:
    return {c.key: c for c in clips}
