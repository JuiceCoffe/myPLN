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
        return self.head(self.backbone(x))

    def modules_recurse(self, mod=None):
        if mod is None:
            mod = self

        for module in mod.children():
            if isinstance(module, (nn.ModuleList, nn.Sequential)):
                yield from self.modules_recurse(module)
            else:
                yield module
