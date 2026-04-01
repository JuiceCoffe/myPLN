from collections import OrderedDict

import torch.nn as nn
from torch.hub import load_state_dict_from_url

__all__ = ["ResNet18Backbone"]


MODEL_URLS = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
}


def _conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = _conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet18Backbone(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.inplanes = 64

        self.stem = nn.Sequential(
            OrderedDict([
                ("conv1", nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)),
                ("bn1", nn.BatchNorm2d(64)),
                ("relu", nn.ReLU(inplace=True)),
                ("maxpool", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ])
        )
        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)

        if pretrained:
            self.load_pretrained()

    def _make_layer(self, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = [BasicBlock(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def load_pretrained(self):
        state_dict = load_state_dict_from_url(MODEL_URLS["resnet18"], progress=True, map_location="cpu")
        filtered = {}
        
        for k, v in state_dict.items():
            # 丢弃全连接层
            if k.startswith("fc."):
                continue
            
            # 修复名称不匹配的问题
            if k.startswith("conv1."):
                k = "stem." + k
            elif k.startswith("bn1."):
                k = "stem." + k
                
            filtered[k] = v

        load_result = self.load_state_dict(filtered, strict=False)
        
        print("="*20,"预训练权重检查","="*20)
        if load_result.missing_keys:
            print(f"Warning: 缺失了以下权重: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            print(f"Warning: 发现了未预期的权重: {load_result.unexpected_keys}")


    def forward(self, x):
        # 创建一个字典用于保存多尺度特征
        features = {}
        
        x = self.stem(x)     # 经过 stem，下采样率变为 4
        
        x = self.layer1(x)   # 经过 layer1，下采样率依然是 4
        features[4] = x      # 保存下采样率为 4 的特征，shape: (B, 64, H/4, W/4)
        
        x = self.layer2(x)   # 经过 layer2，下采样率变为 8
        features[8] = x      # 保存下采样率为 8 的特征，shape: (B, 128, H/8, W/8)
        
        x = self.layer3(x)   # 经过 layer3，下采样率变为 16
        features[16] = x     # 保存下采样率为 16 的特征，shape: (B, 256, H/16, W/16)
        
        x = self.layer4(x)   # 经过 layer4，下采样率变为 32
        features[32] = x     # 保存下采样率为 32 的特征，shape: (B, 512, H/32, W/32)
        
        return features