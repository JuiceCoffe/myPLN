import torch
import torch.nn as nn

from ._lightnet import Lightnet
from .. import loss
from ..network import backbone
from ..network import head
from ..network import layer

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

        # Learned step-wise downsampling to align shallow features to stride 32.
        def conv3x3_bn_relu(in_planes, out_planes, stride):
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_planes),
                nn.ReLU(inplace=True),
            )

        # x4 -> x32 through three stride-2 stages.
        self.downsample_4 = nn.Sequential(
            conv3x3_bn_relu(64, 128, stride=2),
            conv3x3_bn_relu(128, 256, stride=2),
            conv3x3_bn_relu(256, 512, stride=2),
        )

        # x8 -> x32 through two stride-2 stages.
        self.downsample_8 = nn.Sequential(
            conv3x3_bn_relu(128, 256, stride=2),
            conv3x3_bn_relu(256, 512, stride=2),
        )

        # x16 -> x32 through one stride-2 stage.
        self.downsample_16 = nn.Sequential(
            conv3x3_bn_relu(256, 512, stride=2),
        )

        # Fuse the concatenated multi-scale features back to 512 channels.
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(512 * 4, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.backbone_attention = layer.DETRImageSelfAttention(
            embed_dim=512,
            num_heads=8,
            dim_feedforward=2048,
            dropout=0.1,
        )

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
        # Gather the multi-scale backbone outputs.
        features = self.backbone(x)

        x_4 = features[4]
        x_8 = features[8]
        x_16 = features[16]
        x_32 = features[32]

        # Downsample shallow features so every branch reaches stride 32.
        feat_4_down = self.downsample_4(x_4)
        feat_8_down = self.downsample_8(x_8)
        feat_16_down = self.downsample_16(x_16)

        # Concatenate all aligned features on the channel dimension.
        fused_feat = torch.cat([feat_4_down, feat_8_down, feat_16_down, x_32], dim=1)

        # Reduce back to 512 channels, then refine with DETR-style attention.
        out = self.fusion_conv(fused_feat)
        out = self.backbone_attention(out)
        return self.head(out)

    def modules_recurse(self, mod=None):
        if mod is None:
            mod = self

        for module in mod.children():
            if isinstance(module, (nn.ModuleList, nn.Sequential)):
                yield from self.modules_recurse(module)
            else:
                yield module
