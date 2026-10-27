import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineCNN(nn.Module):
    """TrackNet-style U-Net: stacked frames → full-resolution ball heatmap.

    Input (B, num_frames * channels_per_frame, H, W)
    Output is a single-channel heatmap (B, 1, H, W)
    """

    def __init__(self, num_frames: int = 3, channels_per_frame: int = 3):
        super().__init__()
        in_ch = num_frames * channels_per_frame

        # Encoder (VGG-like depths, same as TrackNet)
        self.enc1 = self._conv_block(in_ch, 64, n=2)
        self.enc2 = self._conv_block(64, 128, n=2)
        self.enc3 = self._conv_block(128, 256, n=3)
        self.bottleneck = self._conv_block(256, 512, n=3)
        self.pool = nn.MaxPool2d(2)

        # Decoder with U-Net skip connections
        self.dec3 = self._conv_block(512 + 256, 256, n=3)
        self.dec2 = self._conv_block(256 + 128, 128, n=2)
        self.dec1 = self._conv_block(128 + 64, 64, n=2)

        self.head = nn.Conv2d(64, 1, kernel_size=1)

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
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([F.interpolate(b, size=e3.shape[-2:], mode="nearest"), e3], dim=1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, size=e2.shape[-2:], mode="nearest"), e2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="nearest"), e1], dim=1))

        return torch.sigmoid(self.head(d1))
