import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["PLNHead"]


def _conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class PLNBranch(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            _conv_bn_relu(in_channels, in_channels, 3, 1, 1),
            _conv_bn_relu(in_channels, in_channels // 2, 3, 1, 1),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.layers(x)


class PLNHead(nn.Module):
    def __init__(self, num_classes=20, grid_size=14):
        super().__init__()
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.feature_size = 1 + 2 + (2 * grid_size) + num_classes
        self.out_channels = self.feature_size * 4

        self.shared = nn.Sequential(
            _conv_bn_relu(512, 512, 3, 1, 1),
            _conv_bn_relu(512, 512, 3, 1, 1),
        )
        self.branches = nn.ModuleList([PLNBranch(512, self.out_channels) for _ in range(4)])

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
