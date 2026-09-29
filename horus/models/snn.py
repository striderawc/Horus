"""Convolutional LIF network: event frames → Gaussian ball heatmap.

Architecture mirrors ``BallHeatmapCNN`` (encoder–decoder, same widths) but
replaces ReLU with LIF and unrolls one polarity frame per timestep.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Width of the fast-sigmoid surrogate. Wider than a step, narrow enough to
# stay local to the threshold (same default snnTorch uses).
_SURROGATE_SLOPE = 25.0

# Spatial stages matching BallHeatmapCNN; each has two LIFs (_a / _b).
STAGE_NAMES = ("down1", "down2", "down3", "bottleneck", "up3", "up2", "up1")
LAYER_NAMES = tuple(f"{stage}_{ab}" for stage in STAGE_NAMES for ab in ("a", "b"))


class _Spike(torch.autograd.Function):
    """Heaviside spike in the forward pass, fast-sigmoid gradient backward."""

    @staticmethod
    def forward(ctx, membrane: torch.Tensor, threshold: float, slope: float) -> torch.Tensor:
        ctx.save_for_backward(membrane)
        ctx.threshold = threshold
        ctx.slope = slope
        return (membrane >= threshold).to(membrane.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (membrane,) = ctx.saved_tensors
        centered = membrane - ctx.threshold
        surrogate = 1.0 / (1.0 + ctx.slope * centered.abs()).pow(2)
        return grad_output * surrogate, None, None


class LIF(nn.Module):
    """Leaky integrate-and-fire step with subtractive reset.

    Returns ``(spike, next_membrane, pre_reset_membrane)``. The spike is
    detached on the reset path so the surrogate is applied once.
    """

    def __init__(self, beta: float = 0.9, threshold: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("beta", torch.tensor(float(beta)))
        self.register_buffer("threshold", torch.tensor(float(threshold)))

    def forward(
        self,
        current: torch.Tensor,
        membrane: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        v = self.beta * membrane + current
        spike = _Spike.apply(v, float(self.threshold), _SURROGATE_SLOPE)
        v_next = v - spike.detach() * self.threshold
        return spike, v_next, v


class LifBlock(nn.Module):
    """Conv→BN→LIF→Conv→BN→LIF — SNN counterpart of CNN ``_block``."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        beta: float = 0.9,
        threshold: float = 1.0,
    ) -> None:
        super().__init__()
        self.out_ch = out_ch
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.lif_a = LIF(beta, threshold)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.lif_b = LIF(beta, threshold)

    def forward(
        self,
        x: torch.Tensor,
        v_a: torch.Tensor,
        v_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(spike_b, v_a, v_b, spike_a, mem_a, mem_b)``."""
        current = self.bn1(self.conv1(x))
        s_a, v_a, m_a = self.lif_a(current, v_a)
        current = self.bn2(self.conv2(s_a))
        s_b, v_b, m_b = self.lif_b(current, v_b)
        return s_b, v_a, v_b, s_a, m_a, m_b


@dataclass
class LayerActivity:
    """Per-LIF spike and pre-reset membrane stacks over time.

    ``spikes`` / ``membrane`` shapes: ``(B, T, C, H, W)``.
    """

    spikes: torch.Tensor
    membrane: torch.Tensor


@dataclass
class SNNActivityTrace:
    """Activity from all LIF layers for one forward pass."""

    layers: dict[str, LayerActivity]

    def items(self) -> list[tuple[str, LayerActivity]]:
        return [(name, self.layers[name]) for name in LAYER_NAMES if name in self.layers]

    def __getitem__(self, name: str) -> LayerActivity:
        return self.layers[name]


class BallHeatmapSNN(nn.Module):
    """CNN-matched conv-LIF encoder–decoder on polarity frames → heatmap.

    Input is the stacked event-frame window ``(B, steps * in_ch, H, W)``
    (default 3 steps × 2 polarity channels). Each step runs the full
    encoder–decoder with LIF activations; membranes carry across time. The
    head reads the last pre-reset membrane of ``up1_b``.
    """

    def __init__(
        self,
        in_ch: int = 2,
        steps: int = 3,
        base: int = 16,
        beta: float = 0.9,
        threshold: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.steps = steps
        self.base = base
        b, b2, b4 = base, base * 2, base * 4
        # Channel width at each LIF (both _a and _b of a stage share width).
        self._lif_channels = {
            "down1_a": b,
            "down1_b": b,
            "down2_a": b2,
            "down2_b": b2,
            "down3_a": b4,
            "down3_b": b4,
            "bottleneck_a": b4,
            "bottleneck_b": b4,
            "up3_a": b2,
            "up3_b": b2,
            "up2_a": b,
            "up2_b": b,
            "up1_a": b,
            "up1_b": b,
        }
        self.down1 = LifBlock(in_ch, b, beta, threshold)
        self.down2 = LifBlock(b, b2, beta, threshold)
        self.down3 = LifBlock(b2, b4, beta, threshold)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = LifBlock(b4, b4, beta, threshold)
        self.up3 = LifBlock(b4, b2, beta, threshold)
        self.up2 = LifBlock(b2, b, beta, threshold)
        self.up1 = LifBlock(b, b, beta, threshold)
        self.head = nn.Conv2d(b, 1, 1)

    def _spatial_sizes(self, height: int, width: int) -> dict[str, tuple[int, int]]:
        """Native (H, W) for each LIF given input resolution."""
        h, w = height, width
        h2, w2 = h // 2, w // 2
        h4, w4 = h // 4, w // 4
        h8, w8 = h // 8, w // 8
        return {
            "down1_a": (h, w),
            "down1_b": (h, w),
            "down2_a": (h2, w2),
            "down2_b": (h2, w2),
            "down3_a": (h4, w4),
            "down3_b": (h4, w4),
            "bottleneck_a": (h8, w8),
            "bottleneck_b": (h8, w8),
            "up3_a": (h4, w4),
            "up3_b": (h4, w4),
            "up2_a": (h2, w2),
            "up2_b": (h2, w2),
            "up1_a": (h, w),
            "up1_b": (h, w),
        }

    def _init_membranes(
        self,
        ref: torch.Tensor,
        batch: int,
        height: int,
        width: int,
    ) -> dict[str, torch.Tensor]:
        sizes = self._spatial_sizes(height, width)
        return {
            name: ref.new_zeros(batch, self._lif_channels[name], *sizes[name])
            for name in LAYER_NAMES
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        heatmap, _ = self._forward_impl(x, collect=False)
        return heatmap

    def forward_with_activity(self, x: torch.Tensor) -> tuple[torch.Tensor, SNNActivityTrace]:
        """Heatmap plus per-layer spike/membrane stacks over time."""
        heatmap, trace = self._forward_impl(x, collect=True)
        assert trace is not None
        return heatmap, trace

    def _run_block(
        self,
        block: LifBlock,
        stage: str,
        x: torch.Tensor,
        membranes: dict[str, torch.Tensor],
        collected: dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one LifBlock; update membranes; optionally record activity.

        Returns ``(spike_b, pre_reset_mem_b)``.
        """
        key_a, key_b = f"{stage}_a", f"{stage}_b"
        s_b, v_a, v_b, s_a, m_a, m_b = block(x, membranes[key_a], membranes[key_b])
        membranes[key_a] = v_a
        membranes[key_b] = v_b
        if collected is not None:
            collected[key_a][0].append(s_a)
            collected[key_a][1].append(m_a)
            collected[key_b][0].append(s_b)
            collected[key_b][1].append(m_b)
        return s_b, m_b

    def _forward_impl(
        self,
        x: torch.Tensor,
        collect: bool,
    ) -> tuple[torch.Tensor, SNNActivityTrace | None]:
        # x: (B, steps * in_ch, H, W)
        b, c, h, w = x.shape
        expected = self.steps * self.in_ch
        if c != expected:
            raise ValueError(f"expected {expected} input channels, got {c}")
        x = torch.log1p(x.clamp_min(0))
        x = x.view(b, self.steps, self.in_ch, h, w)

        membranes = self._init_membranes(x, b, h, w)
        collected: dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]] | None = None
        if collect:
            collected = {name: ([], []) for name in LAYER_NAMES}

        membrane_out: torch.Tensor | None = None
        for t in range(self.steps):
            s, _ = self._run_block(self.down1, "down1", x[:, t], membranes, collected)
            s, _ = self._run_block(self.down2, "down2", self.pool(s), membranes, collected)
            # Encoder spatial sizes for upsample targets (no skip concat).
            size_d2 = s.shape[-2:]
            s, _ = self._run_block(self.down3, "down3", self.pool(s), membranes, collected)
            size_d3 = s.shape[-2:]
            s, _ = self._run_block(
                self.bottleneck, "bottleneck", self.pool(s), membranes, collected
            )
            s = F.interpolate(s, size=size_d3, mode="bilinear", align_corners=False)
            s, _ = self._run_block(self.up3, "up3", s, membranes, collected)
            s = F.interpolate(s, size=size_d2, mode="bilinear", align_corners=False)
            s, _ = self._run_block(self.up2, "up2", s, membranes, collected)
            s = F.interpolate(s, size=(h, w), mode="bilinear", align_corners=False)
            _, membrane_out = self._run_block(self.up1, "up1", s, membranes, collected)

        assert membrane_out is not None
        heatmap = torch.sigmoid(self.head(membrane_out))

        if not collect:
            return heatmap, None

        assert collected is not None

        def _stack(ts: list[torch.Tensor]) -> torch.Tensor:
            return torch.stack(ts, dim=1)

        layers = {
            name: LayerActivity(
                spikes=_stack(collected[name][0]),
                membrane=_stack(collected[name][1]),
            )
            for name in LAYER_NAMES
        }
        return heatmap, SNNActivityTrace(layers=layers)
