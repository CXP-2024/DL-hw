import torch
import torch.nn as nn
from torch import Tensor


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block: learns channel-wise attention weights."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = channels // reduction
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y


class BasicBlock(nn.Module):
    """Pre-activation style residual block with SE attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        se_reduction: int = 16,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels, se_reduction)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class CustomModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # YOUR CODE BEGIN.

        # === Stem: two 3x3 convolutions + max pool ===
        # Designed for 64x64 input (no aggressive 7x7 downsampling like ImageNet ResNet)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x64 -> 32x32
        )

        # === 4 residual stages (ResNet-18 layout: [2, 2, 2, 2] blocks) ===
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)   # 32x32
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)  # 16x16
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2) # 8x8
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2) # 4x4

        # === Classifier head ===
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 4x4 -> 1x1
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(512, 200)

        # === Weight initialization ===
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual block so that
        # each block initially behaves like an identity mapping.
        # This trick improves convergence (~0.2-0.3% acc gain).
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

        # YOUR CODE END.

    def _make_layer(
        self, in_channels: int, out_channels: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers: list[nn.Module] = [
            BasicBlock(in_channels, out_channels, stride, downsample)
        ]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # YOUR CODE BEGIN.

        x = self.stem(x)       # (B, 64, 32, 32)
        x = self.layer1(x)     # (B, 64, 32, 32)
        x = self.layer2(x)     # (B, 128, 16, 16)
        x = self.layer3(x)     # (B, 256, 8, 8)
        x = self.layer4(x)     # (B, 512, 4, 4)
        x = self.avgpool(x)    # (B, 512, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 512)
        x = self.dropout(x)
        x = self.fc(x)         # (B, 200)
        return x

        # YOUR CODE END.
