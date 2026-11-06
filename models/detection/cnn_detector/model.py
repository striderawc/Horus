import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNDetector(nn.Module):
    """Small U-Net: signed frame-difference crops → ball heatmap.

    Input (B, in_channels, H, W), any H and W at least 4.
    Output is a single-channel heatmap (B, 1, H, W) at the same spatial size.
    """

    def __init__(self, in_channels: int = 2):
        super().__init__()
        self.enc1 = self._conv_block(in_channels, 16, n=2)
        self.enc2 = self._conv_block(16, 32, n=2)
        self.bottleneck = self._conv_block(32, 64, n=2)
        self.pool = nn.MaxPool2d(2)

        self.dec2 = self._conv_block(64 + 32, 32, n=2)
        self.dec1 = self._conv_block(32 + 16, 16, n=2)
        self.head = nn.Conv2d(16, 1, kernel_size=1)

    @staticmethod
    def _conv_block(in_ch: int, out_ch: int, n: int = 2):
        layers = []
        for i in range(n):
            layers.extend(
                [
                    nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                ]
            )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))

        d2 = self.dec2(torch.cat([F.interpolate(b, size=e2.shape[-2:], mode="nearest"), e2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="nearest"), e1], dim=1))

        return torch.sigmoid(self.head(d1))
