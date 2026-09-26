"""Convolutional LIF network: event frames → Gaussian ball heatmap."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Width of the fast-sigmoid surrogate. Wider than a step, narrow enough to
# stay local to the threshold (same default snnTorch uses).
_SURROGATE_SLOPE = 25.0


def _conv_out_size(conv: nn.Conv2d, height: int, width: int) -> tuple[int, int]:
    """Spatial size of ``conv`` applied to ``(height, width)``."""

    def _side(size: int, axis: int) -> int:
        pad = int(conv.padding[axis])
        kernel = int(conv.kernel_size[axis])
        stride = int(conv.stride[axis])
        return (size + 2 * pad - kernel) // stride + 1

    return _side(height, 0), _side(width, 1)


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


class BallHeatmapSNN(nn.Module):
    """Shallow conv-LIF encoder on polarity frames → 1-channel heatmap.

    Input is the stacked event-frame window ``(B, steps * in_ch, H, W)``
    (default 3 steps × 2 polarity channels). Hidden layers spike; the head
    reads the last pre-reset membrane so the output matches the CNN heatmap.
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
        hidden = base * 2
        self.conv1 = nn.Conv2d(in_ch, base, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base)
        self.conv2 = nn.Conv2d(base, hidden, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden)
        self.conv3 = nn.Conv2d(hidden, hidden, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(hidden)
        self.lif1 = LIF(beta, threshold)
        self.lif2 = LIF(beta, threshold)
        self.lif3 = LIF(beta, threshold)
        self.head = nn.Conv2d(hidden, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, steps * in_ch, H, W)
        b, c, h, w = x.shape
        expected = self.steps * self.in_ch
        if c != expected:
            raise ValueError(f"expected {expected} input channels, got {c}")
        x = torch.log1p(x.clamp_min(0))
        x = x.view(b, self.steps, self.in_ch, h, w)

        # conv2 is stride 2, so layers 2–3 live at half resolution.
        h2, w2 = _conv_out_size(self.conv2, h, w)
        hidden = self.conv2.out_channels
        v1 = x.new_zeros(b, self.conv1.out_channels, h, w)
        v2 = x.new_zeros(b, hidden, h2, w2)
        v3 = x.new_zeros(b, hidden, h2, w2)
        membrane = v3
        for t in range(self.steps):
            current = self.bn1(self.conv1(x[:, t]))
            spike, v1, _ = self.lif1(current, v1)
            current = self.bn2(self.conv2(spike))
            spike, v2, _ = self.lif2(current, v2)
            current = self.bn3(self.conv3(spike))
            _, v3, membrane = self.lif3(current, v3)

        assert membrane is not None
        up = F.interpolate(membrane, size=(h, w), mode="bilinear", align_corners=False)
        return torch.sigmoid(self.head(up))
