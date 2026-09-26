#!/usr/bin/env python3
"""Train BallHeatmapSNN on event frames → Gaussian heatmap."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from horus.config import load_config, repo_root
from horus.data.dataset import build_dataloader
from horus.eval.metrics import input_channels
from horus.models import BallHeatmapSNN
from horus.train import evaluate, train_one_epoch

ARCH = "event_frame_snn"
CKPT_NAME = "event_frame_snn.pt"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--pos-weight", type=float, default=None, help="Peak pixel loss weight")
    p.add_argument("--num-workers", type=int, default=None)
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
    beta = float(snn_cfg.get("beta", 0.9))
    threshold = float(snn_cfg.get("threshold", 1.0))
    base = int(snn_cfg.get("base", 16))

    ckpt_dir = Path(tcfg.get("checkpoint_dir", "checkpoints"))
    if not ckpt_dir.is_absolute():
        ckpt_dir = repo_root() / ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / CKPT_NAME

    modality = "event_frame"
    train_loader = build_dataloader(cfg, "train", modality, batch_size=bs, num_workers=nw)
    val_loader = build_dataloader(
        cfg, "val", modality, batch_size=bs, num_workers=nw, shuffle=False
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

    print(
        f"arch={ARCH} modality={modality} in_ch={in_ch} device={device} "
        f"steps={steps} base={base} beta={beta} threshold={threshold} "
        f"epochs={epochs} lr={lr} batch_size={bs} pos_weight={pos_weight} "
        f"train={len(train_loader.dataset)} val={len(val_loader.dataset)}"
    )
    best_px = float("inf")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, opt, device, pos_weight)
        val_loss, val_px = evaluate(model, val_loader, device, pos_weight)
        print(
            f"epoch {epoch:03d}  train_loss={train_loss:.5f}  "
            f"val_loss={val_loss:.5f}  val_px={val_px:.2f}"
        )
        if val_px < best_px:
            best_px = val_px
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_px": val_px,
                    "val_loss": val_loss,
                    "pos_weight": pos_weight,
                    "modality": modality,
                    "in_ch": in_ch,
                    "arch": ARCH,
                },
                ckpt_path,
            )
            print(f"  saved {ckpt_path} (best val_px={best_px:.2f})")


if __name__ == "__main__":
    main()
