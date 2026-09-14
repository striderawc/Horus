#!/usr/bin/env python3
"""Train BallHeatmapCNN on RGB (3-frame concat → Gaussian heatmap)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from horus.config import load_config, repo_root
from horus.data.dataset import build_dataloader
from horus.models import BallHeatmapCNN, heatmap_to_coords, weighted_heatmap_mse


@torch.no_grad()
def evaluate(
    model: BallHeatmapCNN,
    loader,
    device: torch.device,
    pos_weight: float,
) -> tuple[float, float]:
    model.eval()
    total_loss, n_batches = 0.0, 0
    dist_sum, dist_n = 0.0, 0
    for batch in loader:
        x = batch["input"].to(device)
        y = batch["heatmap"].to(device)
        pred = model(x)
        total_loss += weighted_heatmap_mse(pred, y, pos_weight).item()
        n_batches += 1
        visible = batch["visible"].to(device) > 0.5
        if visible.any():
            pred_xy = heatmap_to_coords(pred[visible])
            gt_xy = batch["coords"].to(device)[visible]
            dist_sum += (pred_xy - gt_xy).norm(dim=1).sum().item()
            dist_n += int(visible.sum())
    loss = total_loss / max(n_batches, 1)
    px = dist_sum / max(dist_n, 1)
    return loss, px


def train_one_epoch(
    model: BallHeatmapCNN,
    loader,
    opt: torch.optim.Optimizer,
    device: torch.device,
    pos_weight: float,
) -> float:
    model.train()
    total, n = 0.0, 0
    for batch in tqdm(loader, desc="train", leave=False):
        x = batch["input"].to(device)
        y = batch["heatmap"].to(device)
        opt.zero_grad(set_to_none=True)
        loss = weighted_heatmap_mse(model(x), y, pos_weight)
        loss.backward()
        opt.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


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
    assert int(cfg["sequence"]["length"]) == 3, "RGB CNN expects sequence.length == 3"

    tcfg = cfg.setdefault("train", {})
    epochs = args.epochs if args.epochs is not None else int(tcfg.get("epochs", 20))
    lr = args.lr if args.lr is not None else float(tcfg.get("lr", 1e-3))
    pos_weight = (
        args.pos_weight if args.pos_weight is not None else float(tcfg.get("pos_weight", 100.0))
    )
    ckpt_dir = Path(tcfg.get("checkpoint_dir", "checkpoints"))
    if not ckpt_dir.is_absolute():
        ckpt_dir = repo_root() / ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "rgb_cnn.pt"

    bs = args.batch_size
    nw = args.num_workers
    train_loader = build_dataloader(cfg, "train", "rgb", batch_size=bs, num_workers=nw)
    val_loader = build_dataloader(
        cfg, "val", "rgb", batch_size=bs, num_workers=nw, shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BallHeatmapCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    print(
        f"device={device} epochs={epochs} lr={lr} pos_weight={pos_weight} "
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
                },
                ckpt_path,
            )
            print(f"  saved {ckpt_path} (best val_px={best_px:.2f})")


if __name__ == "__main__":
    main()
