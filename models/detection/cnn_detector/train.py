"""Minimal CNNDetector training loop."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.crop_dataloader import get_loader
from models.detection.cnn_detector.model import CNNDetector


MODEL_DIR = Path(__file__).resolve().parent


def focal_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """CenterNet focal loss for a Gaussian ball heatmap.

    The peak pixel of each visible frame is the positive. Every other pixel is a
    negative, down-weighted near the ball by (1 - target)^beta and down-weighted
    again when the prediction is already close to zero.
    """
    pred = pred.clamp(1e-6, 1.0 - 1e-6)
    # A grid-sampled Gaussian rarely lands on exactly 1, so the brightest pixel is the positive.
    peak = target.amax(dim=(2, 3), keepdim=True)
    pos = target.eq(peak) & peak.gt(0)

    pos_term = -((1 - pred) ** alpha) * pred.log()
    neg_term = -((1 - target) ** beta) * (pred ** alpha) * (1 - pred).log()
    neg_term = neg_term.masked_fill(pos, 0)

    return (pos_term[pos].sum() + neg_term.sum()) / pos.sum().clamp(min=1)


def train(model: torch.nn.Module, data, cfg: dict) -> None:
    tcfg = cfg["train"]
    name = str(tcfg["name"])
    epochs = int(tcfg["epochs"])
    lr = float(tcfg["lr"])
    alpha = float(tcfg.get("focal_alpha", 2.0))
    beta = float(tcfg.get("focal_beta", 4.0))

    out_dir = MODEL_DIR / "saved_models" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "config.yaml").open("w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    writer = SummaryWriter(str(out_dir))

    step = 0
    model.train()
    for epoch in range(epochs):
        total = 0.0
        n = 0
        for batch in data:
            x = batch["input"].to(device)
            y = batch["heatmap"].to(device)
            pred = model(x)
            loss = focal_loss(pred, y, alpha, beta)

            opt.zero_grad()
            loss.backward()
            opt.step()

            step += 1
            total += float(loss.item())
            n += 1
            writer.add_scalar("train/loss", loss.item(), step)

        avg = total / max(n, 1)
        writer.add_scalar("train/epoch_loss", avg, epoch)
        print(f"epoch {epoch + 1}/{epochs}  loss={avg:.6f}  step={step}")

    ckpt_path = out_dir / f"{name}.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg, "step": step}, ckpt_path)
    writer.close()
    print(f"saved {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train CNNDetector")
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "cnn_detector.yaml")
    args = p.parse_args()

    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    ds = Path(cfg["paths"]["dataset"])
    if not ds.is_absolute():
        cfg["paths"]["dataset"] = str(ROOT / ds)

    model = CNNDetector(in_channels=int(cfg["model"]["num_diffs"]))
    loader = get_loader(cfg, "train")
    train(model, loader, cfg)
