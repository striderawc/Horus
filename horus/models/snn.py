"""Convolutional LIF network: event frames → full-resolution ball heatmap.

Shallow 2-LIF encoder at native resolution (no stride / pool), then a 1×1
head + sigmoid → ``(B, 1, H, W)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

# Width of the fast-sigmoid surrogate. Wider than a step, narrow enough to
# stay local to the threshold (same default snnTorch uses).
_SURROGATE_SLOPE = 25.0

# Activity / logging order for the two LIF layers.
LAYER_NAMES = ("lif1", "lif2")
# Alias kept for notebook / export compatibility (one name per LIF).
STAGE_NAMES = LAYER_NAMES


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
    """Shallow 2-LIF conv encoder on polarity frames → full-res heatmap.

    Input is the stacked event-frame window ``(B, steps * in_ch, H, W)``
    (default 3 steps × 2 polarity channels). Each step runs:

    ``conv1→BN→LIF1`` → ``conv2→BN→LIF2`` (both stride-1, full res) →
    ``1×1 head → sigmoid`` ``(B, 1, H, W)``.

    With ``drop_last_lif=True``, ``lif2`` is replaced by ReLU so the head
    reads continuous features — useful as an overfit / ablation check.
    """

    def __init__(
        self,
        in_ch: int = 2,
        steps: int = 3,
        base: int = 16,
        beta: float = 0.9,
        threshold: float = 1.0,
        drop_last_lif: bool = False,
        *,
        predict_uncertainty: bool = False,
    ) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.steps = steps
        self.base = base
        self.drop_last_lif = bool(drop_last_lif)
        self.predict_uncertainty = bool(predict_uncertainty)
        self.lif_layer_names = ("lif1",) if self.drop_last_lif else LAYER_NAMES

        hidden = base * 2
        self.conv1 = nn.Conv2d(in_ch, base, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base)
        self.lif1 = LIF(beta, threshold)
        self.conv2 = nn.Conv2d(base, hidden, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden)
        if self.drop_last_lif:
            self.lif2 = None
            self.act2 = nn.ReLU(inplace=True)
        else:
            self.lif2 = LIF(beta, threshold)
            self.act2 = None
        self.head = nn.Conv2d(hidden, 1, kernel_size=1)
        if self.predict_uncertainty:
            self.sigma_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(hidden, 1),
            )
            nn.init.constant_(self.sigma_head[-1].bias, 1.6)
        else:
            self.sigma_head = None

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        heat, log_sigma, _ = self._forward_impl(x, collect=False)
        if self.sigma_head is None:
            return heat
        assert log_sigma is not None
        return heat, log_sigma

    def forward_with_activity(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, SNNActivityTrace] | tuple[
        torch.Tensor, torch.Tensor, SNNActivityTrace
    ]:
        """Heatmap ``(B, 1, H, W)`` plus per-layer spike/membrane stacks."""
        heat, log_sigma, trace = self._forward_impl(x, collect=True)
        assert trace is not None
        if self.sigma_head is None:
            return heat, trace
        assert log_sigma is not None
        return heat, log_sigma, trace

    def _forward_impl(
        self,
        x: torch.Tensor,
        collect: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, SNNActivityTrace | None]:
        # x: (B, steps * in_ch, H, W)
        b, c, h, w = x.shape
        expected = self.steps * self.in_ch
        if c != expected:
            raise ValueError(f"expected {expected} input channels, got {c}")
        x = torch.log1p(x.clamp_min(0))
        x = x.view(b, self.steps, self.in_ch, h, w)

        hidden = self.conv2.out_channels
        v1 = x.new_zeros(b, self.conv1.out_channels, h, w)
        v2 = x.new_zeros(b, hidden, h, w)

        collected: dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]] | None = None
        if collect:
            collected = {name: ([], []) for name in self.lif_layer_names}

        head_features: torch.Tensor | None = None
        for t in range(self.steps):
            current = self.bn1(self.conv1(x[:, t]))
            spike, v1, m1 = self.lif1(current, v1)
            if collected is not None:
                collected["lif1"][0].append(spike.detach().cpu())
                collected["lif1"][1].append(m1.detach().cpu())

            current = self.bn2(self.conv2(spike))
            if self.drop_last_lif:
                assert self.act2 is not None
                head_features = self.act2(current)
            else:
                assert self.lif2 is not None
                spike2, v2, m2 = self.lif2(current, v2)
                head_features = m2
                if collected is not None:
                    collected["lif2"][0].append(spike2.detach().cpu())
                    collected["lif2"][1].append(m2.detach().cpu())

        assert head_features is not None
        heat = torch.sigmoid(self.head(head_features))
        log_sigma = None
        if self.sigma_head is not None:
            log_sigma = self.sigma_head(head_features).squeeze(-1)

        if not collect:
            return heat, log_sigma, None

        assert collected is not None

        def _stack(ts: list[torch.Tensor]) -> torch.Tensor:
            return torch.stack(ts, dim=1)

        layers = {
            name: LayerActivity(
                spikes=_stack(collected[name][0]),
                membrane=_stack(collected[name][1]),
            )
            for name in self.lif_layer_names
        }
        return heat, log_sigma, SNNActivityTrace(layers=layers)
