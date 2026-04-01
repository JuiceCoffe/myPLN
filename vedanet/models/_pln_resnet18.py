import torch
import torch.nn as nn

from ._lightnet import Lightnet
from .. import loss
from ..network import backbone
from ..network import head

__all__ = ["PLNResnet18"]


class PLNResnet18(Lightnet):
    def __init__(
        self,
        num_classes=20,
        weights_file=None,
        train_flag=1,
        clear=False,
        backbone_pretrained=False,
        point_weight=1.0,
        coord_weight=2.0,
        link_weight=0.5,
        class_weight=0.5,
        grid_size=14,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.nloss = 4
        self.train_flag = train_flag

        self.backbone = backbone.ResNet18Backbone(pretrained=False)
        self.head = head.PLNHead(num_classes=num_classes, grid_size=grid_size)

        # ---------------- 核心修改区：纯可学习的阶梯式下采样 ----------------
        # 为了代码整洁，你可以先在类外部或者顶部定义一个辅助函数
        def conv3x3_bn_relu(in_planes, out_planes, stride):
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_planes),
                nn.ReLU(inplace=True)
            )

        # x_4 (64通道) 降 8倍 -> 经历3次 2倍降采样
        self.downsample_4 = nn.Sequential(
            conv3x3_bn_relu(64, 128, stride=2),   # -> x_8
            conv3x3_bn_relu(128, 256, stride=2),  # -> x_16
            conv3x3_bn_relu(256, 512, stride=2)   # -> x_32
        )

        # x_8 (128通道) 降 4倍 -> 经历2次 2倍降采样
        self.downsample_8 = nn.Sequential(
            conv3x3_bn_relu(128, 256, stride=2),  # -> x_16
            conv3x3_bn_relu(256, 512, stride=2)   # -> x_32
        )

        # x_16 (256通道) 降 2倍 -> 经历1次 2倍降采样
        self.downsample_16 = nn.Sequential(
            conv3x3_bn_relu(256, 512, stride=2)   # -> x_32
        )

        # 融合层保持不变
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(512 * 4, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        # --------------------------------------------------------------

        self.loss = [
            loss.PLNLoss(
                branch_idx=i,
                num_classes=num_classes,
                grid_size=grid_size,
                point_weight=point_weight,
                coord_weight=coord_weight,
                link_weight=link_weight,
                class_weight=class_weight,
            )
            for i in range(self.nloss)
        ]

        if weights_file is not None:
            self.load_weights(weights_file, clear)
        else:
            self.init_weights()
            if backbone_pretrained:
                self.backbone.load_pretrained()

    def _forward(self, x):
        # 1. 获取主干网络的字典输出
        features = self.backbone(x)
        
        x_4 = features[4]    # (B, 64, H/4, W/4)
        x_8 = features[8]    # (B, 128, H/8, W/8)
        x_16 = features[16]  # (B, 256, H/16, W/16)
        x_32 = features[32]  # (B, 512, H/32, W/32)

        # 2. 将浅层特征统一卷积下采样到 1/32，并映射到 512 通道
        feat_4_down = self.downsample_4(x_4)
        feat_8_down = self.downsample_8(x_8)
        feat_16_down = self.downsample_16(x_16)
        
        # 3. 在通道维度拼接 (Concat) 所有特征
        # 此时形状为 (B, 512*4, H/32, W/32)
        fused_feat = torch.cat([feat_4_down, feat_8_down, feat_16_down, x_32], dim=1)
        
        # 4. 通过 1x1 卷积降维回 512 通道，并输入给 Head
        out = self.fusion_conv(fused_feat)

        # out = features[32] 

        return self.head(out)
        
    def modules_recurse(self, mod=None):
        if mod is None:
            mod = self

        for module in mod.children():
            if isinstance(module, (nn.ModuleList, nn.Sequential)):
                yield from self.modules_recurse(module)
            else:
                yield module
