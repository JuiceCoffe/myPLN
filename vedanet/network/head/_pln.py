import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import layer

__all__ = ["PLNHead"]


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ExpandConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, kernel_size=3, padding=8, dilation=8),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, kernel_size=3, padding=16, dilation=16),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.layers(x)


class PLNHead(nn.Module):
    def __init__(self, num_classes=20, grid_size=14, in_channels=512, hidden_channels=1536):
        super().__init__()
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.feature_size = 1 + 2 + (2 * grid_size) + num_classes
        self.out_channels = self.feature_size * 4

        # The original PLN head expects a 1536-channel feature map before the branches.
        # ResNet18 produces 512 channels, so we only add this adapter and keep the rest aligned.
        self.shared = nn.Sequential(
            BasicConv2d(in_channels, hidden_channels, kernel_size=1, stride=1),
            BasicConv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            BasicConv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
        )
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=False),
                
                # 1. 降维到一个用于注意力的语义维度（比如 256 或 512，避免 1536 太大跑不动）
                nn.Conv2d(hidden_channels, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=False),
                
                # 2. 空洞卷积在语义特征上操作
                ExpandConv(256), # 注意把 ExpandConv 里的通道也改成 256
                
                # 3. 自注意力层在语义特征上操作
                layer.DETRImageSelfAttention(
                    embed_dim=256,
                    num_heads=8,
                    dim_feedforward=1024,
                    dropout=0.1,
                ),
                
                # 4. 关键：加一个最后的 1x1 卷积，把特征转为预测所需的通道数 (204)
                nn.Conv2d(256, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
            )
            for _ in range(4)
        ])

    def _activate(self, out):
        chunks = []
        for idx in range(4):
            start = idx * self.feature_size
            point = out[:, start:start + self.feature_size, ...]

            conf = torch.sigmoid(point[:, 0:1, ...])
            offset = torch.sigmoid(point[:, 1:3, ...])
            link_x = F.softmax(point[:, 3:3 + self.grid_size, ...], dim=1)
            link_y = F.softmax(point[:, 3 + self.grid_size:3 + (2 * self.grid_size), ...], dim=1)
            cls = F.softmax(point[:, 3 + (2 * self.grid_size):, ...], dim=1)
            chunks.append(torch.cat((conf, offset, link_x, link_y, cls), dim=1))

        return torch.cat(chunks, dim=1)

    def forward(self, x):
        shared = self.shared(x)
        return [self._activate(branch(shared)) for branch in self.branches]
