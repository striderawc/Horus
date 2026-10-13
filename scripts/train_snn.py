#!/usr/bin/env python3
"""Train BallHeatmapSNN on event frames → full-res heatmap (circle MSE)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from horus.config import load_config, repo_root, resolve_checkpoint_path
from horus.data.dataset import build_dataloader
from horus.eval.snn_activity import format_activity_report, measure_activity
from horus.models import BallHeatmapSNN
from horus.train import (
    evaluate,
    log_epoch_metrics,
    log_hparams,
    log_loader_heatmaps,
    open_run_writer,
    train_one_epoch,
)

ARCH = "event_frame_snn"
CKPT_NAME = "event_frame_snn.pt"
CKPT_OVERFIT = "event_frame_snn_overfit.pt"
CKPT_OVERFIT_NO_LAST = "event_frame_snn_overfit_nolastlif.pt"
CKPT_NO_LAST = "event_frame_snn_nolastlif.pt"
CKPT_REPEAT = "event_frame_snn_repeat.pt"
LOSS = "focal_presence"


def _subset_loader(
    loader: DataLoader,
    max_samples: int,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    crop_enabled: bool = False,
) -> DataLoader:
    """First ``max_samples`` indices of ``loader.dataset``."""
    from horus.data.dataset import BallTrackDataset

    n = min(int(max_samples), len(loader.dataset))
    if n < 1:
        raise ValueError("max_samples must be >= 1")
    subset = Subset(loader.dataset, list(range(n)))
    # Crop mode needs homogeneous bucket sizes inside each batch.
    base_ds = loader.dataset
    while isinstance(base_ds, Subset):
        base_ds = base_ds.dataset
    if crop_enabled and isinstance(base_ds, BallTrackDataset):
        # Sample only indices that share one bucket (first bucket).
        buckets = base_ds.crop_buckets
        b0 = buckets[0]
        same = [i for i in range(n) if buckets[i % len(buckets)] == b0]
        if not same:
            same = list(range(n))
        return DataLoader(
            Subset(loader.dataset, same[:n]),
            batch_size=min(batch_size, len(same[:n])),
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False,
        )
    return DataLoader(
        subset,
        batch_size=min(batch_size, n),
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )


def _resolve_steps(args: argparse.Namespace, snn_cfg: dict, seq_len: int) -> int:
    if args.steps is not None:
        return int(args.steps)
    cfg_steps = snn_cfg.get("steps", None)
    if cfg_steps is not None:
        return int(cfg_steps)
    return int(seq_len)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Use only the first N train samples (sanity / debug)",
    )
    p.add_argument(
        "--overfit",
        action="store_true",
        help=(
            "Memorization check: train+eval on the same tiny subset "
            "(default --max-samples 8). Writes checkpoints/event_frame_snn_overfit.pt"
        ),
    )
    p.add_argument(
        "--drop-last-lif",
        action="store_true",
        help=(
            "Replace final lif2 with ReLU (continuous readout). "
            "Useful with --overfit to test whether the last LIF blocks memorization."
        ),
    )
    p.add_argument(
        "--repeat-frame",
        action="store_true",
        help=(
            "Ablation: tile one event frame across all timesteps (e,e,…,e) "
            "instead of a temporal window. Overrides snn.repeat_frame."
        ),
    )
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help="SNN temporal unroll length (overrides snn.steps / sequence.length)",
    )
    p.add_argument(
        "--circle-radius",
        type=float,
        default=None,
        help="Pixel radius of the binary target disk (overrides snn.circle_radius)",
    )
    p.add_argument(
        "--checkpoint-name",
        type=str,
        default=None,
        help="Checkpoint .pt filename (overrides train.checkpoint_name in config)",
    )
    p.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="TensorBoard root (default: train.log_dir from config)",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=None,
        help="Log train step scalars every N optimizer steps",
    )
    p.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard")
    p.add_argument(
        "--crop",
        action="store_true",
        help="Enable jittered bucket crops (crop.enabled in config)",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.crop:
        cfg.setdefault("crop", {})["enabled"] = True
    crop_cfg = cfg.get("crop") or {}
    crop_enabled = bool(crop_cfg.get("enabled", False))
    dcfg = cfg.get("detector") or {}
    focal_alpha = float(dcfg.get("focal_alpha", 2.0))
    focal_beta = float(dcfg.get("focal_beta", 4.0))
    presence_weight = float(dcfg.get("presence_weight", 1.0))
    presence_thresh = float(dcfg.get("presence_thresh", 0.5))

    snn_cfg = cfg.get("snn", {})
    steps = _resolve_steps(args, snn_cfg, int(cfg["sequence"]["length"]))
    if steps < 1:
        raise SystemExit("--steps / snn.steps must be >= 1")
    repeat_frame = bool(args.repeat_frame or snn_cfg.get("repeat_frame", False))
    circle_radius = (
        float(args.circle_radius)
        if args.circle_radius is not None
        else float(snn_cfg.get("circle_radius", 3.0))
    )

    tcfg = cfg.setdefault("train", {})
    dl_cfg = cfg.get("dataloader", {})
    epochs = args.epochs if args.epochs is not None else int(tcfg.get("epochs", 20))
    lr = args.lr if args.lr is not None else float(tcfg.get("lr", 1e-3))
    bs = (
        args.batch_size
        if args.batch_size is not None
        else int(tcfg.get("batch_size", dl_cfg.get("batch_size", 4)))
    )
    nw = (
        args.num_workers
        if args.num_workers is not None
        else int(tcfg.get("num_workers", dl_cfg.get("num_workers", 0)))
    )
    log_every = (
        args.log_every if args.log_every is not None else int(tcfg.get("log_every", 50))
    )
    beta = float(snn_cfg.get("beta", 0.9))
    threshold = float(snn_cfg.get("threshold", 1.0))
    base = int(snn_cfg.get("base", 16))

    max_samples = args.max_samples
    if args.overfit and max_samples is None:
        max_samples = 8

    if args.checkpoint_name or tcfg.get("checkpoint_name"):
        default_ckpt = CKPT_NAME
    elif args.overfit and args.drop_last_lif:
        default_ckpt = CKPT_OVERFIT_NO_LAST
    elif args.overfit:
        default_ckpt = CKPT_OVERFIT
    elif args.drop_last_lif:
        default_ckpt = CKPT_NO_LAST
    elif repeat_frame:
        default_ckpt = CKPT_REPEAT
    else:
        default_ckpt = CKPT_NAME
    ckpt_path = resolve_checkpoint_path(
        tcfg, default_ckpt, name_override=args.checkpoint_name
    )

    modality = "event_frame"
    train_loader = build_dataloader(
        cfg,
        "train",
        modality,
        batch_size=bs,
        num_workers=nw,
        repeat_frame=repeat_frame,
        input_steps=steps,
    )
    val_loader = build_dataloader(
        cfg,
        "val",
        modality,
        batch_size=bs,
        num_workers=nw,
        shuffle=False,
        repeat_frame=repeat_frame,
        input_steps=steps,
    )

    if max_samples is not None:
        train_loader = _subset_loader(
            train_loader,
            max_samples,
            batch_size=bs,
            num_workers=nw,
            shuffle=True,
            crop_enabled=crop_enabled,
        )
        if args.overfit:
            # Same indices, no shuffle — watch train_loss / px collapse on this set.
            val_loader = _subset_loader(
                train_loader,
                len(train_loader.dataset),
                batch_size=bs,
                num_workers=nw,
                shuffle=False,
                crop_enabled=crop_enabled,
            )
        else:
            val_loader = _subset_loader(
                val_loader,
                max_samples,
                batch_size=bs,
                num_workers=nw,
                shuffle=False,
                crop_enabled=crop_enabled,
            )

    in_ch = 2 * steps
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BallHeatmapSNN(
        in_ch=2,
        steps=steps,
        base=base,
        beta=beta,
        threshold=threshold,
        drop_last_lif=bool(args.drop_last_lif),
        predict_presence=True,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    run_bits = [ARCH]
    if args.overfit:
        run_bits.append("overfit")
    if args.drop_last_lif:
        run_bits.append("nolastlif")
    if repeat_frame:
        run_bits.append(f"repeat{steps}")
    if crop_enabled:
        run_bits.append("crop")
    writer = open_run_writer(
        run_name="_".join(run_bits),
        repo=repo_root(),
        log_dir=args.log_dir,
        disabled=args.no_tensorboard,
        default_root=tcfg.get("log_dir", "runs"),
    )

    mode = "overfit" if args.overfit else ("subset" if max_samples else "full")
    print(
        f"arch={ARCH} modality={modality} in_ch={in_ch} device={device} mode={mode} "
        f"drop_last_lif={bool(args.drop_last_lif)} repeat_frame={repeat_frame} "
        f"steps={steps} base={base} beta={beta} threshold={threshold} "
        f"circle_radius={circle_radius} crop={crop_enabled} "
        f"epochs={epochs} lr={lr} batch_size={bs} loss={LOSS} "
        f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
        f"ckpt={ckpt_path.name}"
    )
    best_px = float("inf")
    activity_batches = 1 if args.overfit else 20
    global_step = 0
    for epoch in range(1, epochs + 1):
        train_stats, global_step = train_one_epoch(
            model,
            train_loader,
            opt,
            device,
            writer=writer,
            global_step=global_step,
            log_every=log_every,
            loss=LOSS,
            circle_radius=circle_radius,
            focal_alpha=focal_alpha,
            focal_beta=focal_beta,
            presence_weight=presence_weight,
        )
        val = evaluate(
            model,
            val_loader,
            device,
            loss=LOSS,
            circle_radius=circle_radius,
            focal_alpha=focal_alpha,
            focal_beta=focal_beta,
            presence_weight=presence_weight,
            presence_thresh=presence_thresh,
        )
        print(
            f"epoch {epoch:03d}  train_loss={train_stats.loss:.5f}  "
            f"grad_norm={train_stats.grad_norm:.4f}  "
            f"val_loss={val.loss:.5f}  val_px={val.mean_px:.2f}  "
            f"acc@5={val.acc_at_5:.3f}  acc@10={val.acc_at_10:.3f}"
        )
        activity = measure_activity(
            model, val_loader, device, max_batches=activity_batches
        )
        print(format_activity_report(activity))
        if writer is not None:
            activity_scalars = {
                f"activity/{layer}/{key}": float(value)
                for layer, metrics in activity.items()
                for key, value in metrics.items()
            }
            log_epoch_metrics(
                writer, epoch, train_stats, val, lr=lr, extra=activity_scalars
            )
            # GT | pred | |err| for the first sample (watch peak drift in Images).
            log_loader_heatmaps(
                writer,
                model,
                val_loader,
                device,
                epoch,
                circle_radius=circle_radius,
            )
        if val.mean_px < best_px:
            best_px = val.mean_px
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_px": val.mean_px,
                    "val_loss": val.loss,
                    "acc_at_5": val.acc_at_5,
                    "acc_at_10": val.acc_at_10,
                    "modality": modality,
                    "in_ch": in_ch,
                    "steps": steps,
                    "arch": ARCH,
                    "overfit": bool(args.overfit),
                    "drop_last_lif": bool(args.drop_last_lif),
                    "repeat_frame": repeat_frame,
                    "max_samples": max_samples,
                    "circle_radius": circle_radius,
                    "loss": LOSS,
                    "crop": crop_enabled,
                    "predict_presence": True,
                },
                ckpt_path,
            )
            print(f"  saved {ckpt_path} (best val_px={best_px:.2f})")

    if writer is not None:
        log_hparams(
            writer,
            {
                "arch": ARCH,
                "modality": modality,
                "epochs": epochs,
                "lr": lr,
                "batch_size": bs,
                "beta": beta,
                "threshold": threshold,
                "base": base,
                "steps": steps,
                "circle_radius": circle_radius,
                "overfit": bool(args.overfit),
                "drop_last_lif": bool(args.drop_last_lif),
                "repeat_frame": repeat_frame,
                "max_samples": max_samples if max_samples is not None else -1,
                "loss": LOSS,
            },
            {
                "hparam/best_val_px": best_px if best_px < float("inf") else float("nan"),
            },
        )
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
