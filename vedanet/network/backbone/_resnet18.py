import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

class ResNet18Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        self.backbone = resnet_fpn_backbone('resnet18', pretrained=True)
        
        # torchvision FPN 的默认输出通道数是 256
        fpn_channels = 256 
        
        # 1. 定义可学习的卷积下采样层
        # P4 -> P5: 缩小 2倍 (1个步长为2的卷积)
        self.p4_down = self._make_downsample_block(in_channels=fpn_channels, num_downs=1)
        # P3 -> P5: 缩小 4倍 (2个步长为2的卷积)
        self.p3_down = self._make_downsample_block(in_channels=fpn_channels, num_downs=2)
        # P2 -> P5: 缩小 8倍 (3个步长为2的卷积)
        self.p2_down = self._make_downsample_block(in_channels=fpn_channels, num_downs=3)
        

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(1024, 3328, kernel_size=1, bias=False),
            nn.BatchNorm2d(3328),
            nn.GELU()
        )

    def _make_downsample_block(self, in_channels, num_downs):
        """
        辅助函数：构建连续的步长为2的卷积块，用于倍数下采样。
        包含了 Conv + BN + GELU，确保下采样过程也有非线性特征学习能力。
        """
        layers = []
        for _ in range(num_downs):
            layers.extend([
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.GELU()
            ])
        return nn.Sequential(*layers)

    def forward(self, x):
        # 1. 获取官方 FPN 字典
        features = self.backbone(x)
        
        # 2. 提取出需要的 4 个尺度 (忽略 'pool' 层)
        p2 = features['0']
        p3 = features['1']
        p4 = features['2']
        p5 = features['3']
        

        p4_down = self.p4_down(p4)
        
        p3_down = self.p3_down(p3)
        
        p2_down = self.p2_down(p2)
        
        # 5. 拼接并降维/升维
        out = torch.cat([p5, p4_down, p3_down, p2_down], dim=1)
        out = self.fusion_conv(out)
        
        # 最终输出张量: [B, 3328, target_H, target_W]
        return out