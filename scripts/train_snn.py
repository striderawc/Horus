#!/usr/bin/env python3
"""Train BallHeatmapSNN on event frames → Gaussian heatmap."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from horus.config import load_config, repo_root
from horus.data.dataset import build_dataloader
from horus.eval.metrics import input_channels
from horus.eval.snn_activity import format_activity_report, measure_activity
from horus.models import BallHeatmapSNN
from horus.train import (
    evaluate,
    log_epoch_metrics,
    log_hparams,
    open_run_writer,
    train_one_epoch,
)

ARCH = "event_frame_snn"
CKPT_NAME = "event_frame_snn.pt"
CKPT_OVERFIT = "event_frame_snn_overfit.pt"


def _subset_loader(
    loader: DataLoader,
    max_samples: int,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    """First ``max_samples`` indices of ``loader.dataset``."""
    n = min(int(max_samples), len(loader.dataset))
    if n < 1:
        raise ValueError("max_samples must be >= 1")
    return DataLoader(
        Subset(loader.dataset, list(range(n))),
        batch_size=min(batch_size, n),
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--pos-weight", type=float, default=None, help="Peak pixel loss weight")
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
    args = p.parse_args()

    cfg = load_config(args.config)
    steps = int(cfg["sequence"]["length"])
    assert steps == 3, "SNN expects sequence.length == 3"

    tcfg = cfg.setdefault("train", {})
    snn_cfg = cfg.get("snn", {})
    dl_cfg = cfg.get("dataloader", {})
    epochs = args.epochs if args.epochs is not None else int(tcfg.get("epochs", 20))
    lr = args.lr if args.lr is not None else float(tcfg.get("lr", 1e-3))
    pos_weight = (
        args.pos_weight if args.pos_weight is not None else float(tcfg.get("pos_weight", 100.0))
    )
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

    ckpt_dir = Path(tcfg.get("checkpoint_dir", "checkpoints"))
    if not ckpt_dir.is_absolute():
        ckpt_dir = repo_root() / ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / (CKPT_OVERFIT if args.overfit else CKPT_NAME)

    modality = "event_frame"
    train_loader = build_dataloader(cfg, "train", modality, batch_size=bs, num_workers=nw)
    val_loader = build_dataloader(
        cfg, "val", modality, batch_size=bs, num_workers=nw, shuffle=False
    )

    if max_samples is not None:
        train_loader = _subset_loader(
            train_loader, max_samples, batch_size=bs, num_workers=nw, shuffle=True
        )
        if args.overfit:
            # Same indices, no shuffle — watch train_loss / px collapse on this set.
            val_loader = _subset_loader(
                train_loader,
                len(train_loader.dataset),
                batch_size=bs,
                num_workers=nw,
                shuffle=False,
            )
        else:
            val_loader = _subset_loader(
                val_loader, max_samples, batch_size=bs, num_workers=nw, shuffle=False
            )

    in_ch = input_channels(modality, cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BallHeatmapSNN(
        in_ch=2,
        steps=steps,
        base=base,
        beta=beta,
        threshold=threshold,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    writer = open_run_writer(
        run_name=f"{ARCH}_overfit" if args.overfit else ARCH,
        repo=repo_root(),
        log_dir=args.log_dir,
        disabled=args.no_tensorboard,
        default_root=tcfg.get("log_dir", "runs"),
    )

    mode = "overfit" if args.overfit else ("subset" if max_samples else "full")
    print(
        f"arch={ARCH} modality={modality} in_ch={in_ch} device={device} mode={mode} "
        f"steps={steps} base={base} beta={beta} threshold={threshold} "
        f"epochs={epochs} lr={lr} batch_size={bs} pos_weight={pos_weight} "
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
            pos_weight,
            writer=writer,
            global_step=global_step,
            log_every=log_every,
        )
        val = evaluate(model, val_loader, device, pos_weight)
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
                    "pos_weight": pos_weight,
                    "modality": modality,
                    "in_ch": in_ch,
                    "arch": ARCH,
                    "overfit": bool(args.overfit),
                    "max_samples": max_samples,
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
                "pos_weight": pos_weight,
                "beta": beta,
                "threshold": threshold,
                "base": base,
                "overfit": bool(args.overfit),
                "max_samples": max_samples if max_samples is not None else -1,
            },
            {
                "hparam/best_val_px": best_px if best_px < float("inf") else float("nan"),
            },
        )
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
