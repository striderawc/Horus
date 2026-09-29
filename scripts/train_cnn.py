#!/usr/bin/env python3
"""Train BallHeatmapCNN on RGB / frame-diff / event-frame → Gaussian heatmap."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from horus.config import load_config, repo_root
from horus.data.dataset import Modality, build_dataloader
from horus.eval.metrics import input_channels
from horus.models import BallHeatmapCNN
from horus.train import (
    evaluate,
    log_epoch_metrics,
    log_hparams,
    open_run_writer,
    train_one_epoch,
)

CKPT_NAME = {
    "rgb": "rgb_cnn.pt",
    "diff": "diff_cnn.pt",
    "event_frame": "event_frame_cnn.pt",
}

RUN_NAME = {
    "rgb": "rgb_cnn",
    "diff": "diff_cnn",
    "event_frame": "event_frame_cnn",
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument(
        "--modality",
        choices=["rgb", "diff", "event_frame"],
        default="rgb",
        help="rgb=9ch, diff=2ch, event_frame=6ch (needs event_frames.h5)",
    )
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--pos-weight", type=float, default=None, help="Peak pixel loss weight")
    p.add_argument("--num-workers", type=int, default=None)
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

    modality: Modality = args.modality  # type: ignore[assignment]
    cfg = load_config(args.config)
    assert int(cfg["sequence"]["length"]) == 3, "CNN expects sequence.length == 3"

    tcfg = cfg.setdefault("train", {})
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

    ckpt_dir = Path(tcfg.get("checkpoint_dir", "checkpoints"))
    if not ckpt_dir.is_absolute():
        ckpt_dir = repo_root() / ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / CKPT_NAME[modality]

    train_loader = build_dataloader(
        cfg, "train", modality, batch_size=bs, num_workers=nw
    )
    val_loader = build_dataloader(
        cfg, "val", modality, batch_size=bs, num_workers=nw, shuffle=False
    )

    in_ch = input_channels(modality, cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BallHeatmapCNN(in_ch=in_ch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    writer = open_run_writer(
        run_name=RUN_NAME[modality],
        repo=repo_root(),
        log_dir=args.log_dir,
        disabled=args.no_tensorboard,
        default_root=tcfg.get("log_dir", "runs"),
    )

    print(
        f"modality={modality} in_ch={in_ch} device={device} "
        f"epochs={epochs} lr={lr} batch_size={bs} pos_weight={pos_weight} "
        f"train={len(train_loader.dataset)} val={len(val_loader.dataset)}"
    )
    best_px = float("inf")
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
        if writer is not None:
            log_epoch_metrics(writer, epoch, train_stats, val, lr=lr)
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
                },
                ckpt_path,
            )
            print(f"  saved {ckpt_path} (best val_px={best_px:.2f})")

    if writer is not None:
        log_hparams(
            writer,
            {
                "modality": modality,
                "arch": RUN_NAME[modality],
                "epochs": epochs,
                "lr": lr,
                "batch_size": bs,
                "pos_weight": pos_weight,
                "in_ch": in_ch,
            },
            {
                "hparam/best_val_px": best_px if best_px < float("inf") else float("nan"),
            },
        )
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
