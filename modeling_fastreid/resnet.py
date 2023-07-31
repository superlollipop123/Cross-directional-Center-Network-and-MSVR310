# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import logging
import math

import torch
from torch import nn
# from torch.utils import model_zoo
from modeling_fastreid.layers import IBN,SELayer,Non_local,get_norm

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        return input

logger = logging.getLogger(__name__)
model_urls = {
    18: 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    34: 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    50: 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    101: 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    152: 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, bn_norm, num_splits, with_ibn=False, with_se=False,
                 stride=1, downsample=None, reduction=16):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = get_norm(bn_norm, planes, num_splits)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = get_norm(bn_norm, planes, num_splits)
        self.relu = nn.ReLU(inplace=True)
        if with_se:
            self.se = SELayer(planes, reduction)
        else:
            self.se = Identity()
        self.downsample = downsample
        self.stride = stride

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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, bn_norm, num_splits, with_ibn=False, with_se=False,
                 stride=1, downsample=None, reduction=16):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if with_ibn:
            self.bn1 = IBN(planes, bn_norm, num_splits)
        else:
            self.bn1 = get_norm(bn_norm, planes, num_splits)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = get_norm(bn_norm, planes, num_splits)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = get_norm(bn_norm, planes * self.expansion, num_splits)
        self.relu = nn.ReLU(inplace=True)
        if with_se:
            self.se = SELayer(planes * self.expansion, reduction)
        else:
            self.se = nn.Sequential()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, do_conv3=True, multi_out=False):
        residual = x

        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)

        out2 = self.conv2(out1)
        out2 = self.bn2(out2)
        out2 = self.relu(out2)

        if not do_conv3:
            return out2

        out3 = self.conv3(out2)
        out3 = self.bn3(out3)
        out3 = self.se(out3)

        if self.downsample is not None:
            residual = self.downsample(x)

        out3 += residual
        out = self.relu(out3)

        if multi_out:
            return [out1, out]

        return out


class ResNet(nn.Module):
    def __init__(self, last_stride, bn_norm, num_splits, with_ibn, with_se, with_nl, block, layers, non_layers):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = get_norm(bn_norm, 64, num_splits)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], 1, bn_norm, num_splits, with_ibn, with_se)
        self.layer2 = self._make_layer(block, 128, layers[1], 2, bn_norm, num_splits, with_ibn, with_se)
        self.layer3 = self._make_layer(block, 256, layers[2], 2, bn_norm, num_splits, with_ibn, with_se)
        self.layer4 = self._make_layer(block, 512, layers[3], last_stride, bn_norm, num_splits, with_se=with_se)

        self.random_init()

        if with_nl:
            self._build_nonlocal(layers, non_layers, bn_norm, num_splits)
        else:
            self.NL_1_idx = self.NL_2_idx = self.NL_3_idx = self.NL_4_idx = []

    def _make_layer(self, block, planes, blocks, stride=1, bn_norm="BN", num_splits=1, with_ibn=False, with_se=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                get_norm(bn_norm, planes * block.expansion, num_splits),
            )

        layers = []
        if planes == 512:
            with_ibn = False
        layers.append(block(self.inplanes, planes, bn_norm, num_splits, with_ibn, with_se, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, bn_norm, num_splits, with_ibn, with_se))

        return nn.Sequential(*layers)

    def _build_nonlocal(self, layers, non_layers, bn_norm, num_splits):
        self.NL_1 = nn.ModuleList(
            [Non_local(256, bn_norm, num_splits) for _ in range(non_layers[0])])
        self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])
        self.NL_2 = nn.ModuleList(
            [Non_local(512, bn_norm, num_splits) for _ in range(non_layers[1])])
        self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
        self.NL_3 = nn.ModuleList(
            [Non_local(1024, bn_norm, num_splits) for _ in range(non_layers[2])])
        self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
        self.NL_4 = nn.ModuleList(
            [Non_local(2048, bn_norm, num_splits) for _ in range(non_layers[3])])
        self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])])

    def forward(self, x):
        # import pdb; pdb.set_trace()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        forward_feat_list = []
        backward_feat_list = []

        for i in range(len(self.layer1)):
            x = self.layer1[i](x)

        # Layer 2
        for i in range(len(self.layer2)):
            x = self.layer2[i](x, multi_out=True)
            # forward_feat_list.append(x[0])
            backward_feat_list.append(x[1])
            x = x[1]

        # Layer 3
        for i in range(len(self.layer3)):
            x = self.layer3[i](x, multi_out=True)
            # forward_feat_list.append(x[0])
            backward_feat_list.append(x[1])
            x = x[1]

        # Layer 4
        for i in range(len(self.layer4)):
            x = self.layer4[i](x, multi_out=True)
            # forward_feat_list.append(x[0])
            backward_feat_list.append(x[1])
            x = x[1]

        # x = self.layer4[-1](x, False)
        feat_list = [*forward_feat_list, *backward_feat_list]

        return x, feat_list

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, 0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def build_resnet_backbone():
    """
    Create a ResNet instance from config.
    Returns:
        ResNet: a :class:`ResNet` instance.
    """

    # fmt: off
    pretrain_path = r"E:\HRCN\HRCN-master\resnet50_ibn_a-d9d0bb7b.pth"


    num_blocks_per_stage = {34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}
    nl_layers_per_stage = {34: [0, 2, 3, 0], 50: [0, 2, 3, 0], 101: [0, 2, 9, 0]}
    block = {34: BasicBlock, 50: Bottleneck, 101: Bottleneck}
    model = ResNet(1, "BN", 1, True, False, False, block[50],
                   num_blocks_per_stage[50], nl_layers_per_stage[50])

    state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))  # ibn-net
    # Remove module in name
    new_state_dict = {}
    for k in state_dict:
        new_k = k
        if new_k in model.state_dict() and (model.state_dict()[new_k].shape == state_dict[k].shape):
            new_state_dict[new_k] = state_dict[k]
    state_dict = new_state_dict
    logger.info("Loading pretrained model from {}".format(pretrain_path))
    model.load_state_dict(state_dict, strict=False)
    # if incompatible.missing_keys:
    #     print("missing keys:")
    #     print(incompatible.missing_keys)
    #     print()

    return model