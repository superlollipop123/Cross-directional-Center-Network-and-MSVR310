from __future__ import absolute_import

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict
from torch.autograd import Variable
from torch.nn import init
from torch.optim import lr_scheduler
import torch.utils.model_zoo as model_zoo
# resnet: only use RGB
# resnet2modal: combine thermal and RGB
# resnet3modal: combine th, RGB and ni

__all__ = ['resnet', 'resnet_p6',
           'resnet2modal','resnet2modal_PFnet',
           'resnet3modal', 'resnet3modal_AHU', 'resnet3modal_part6', 'ResNet3modal_PFnet']
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class ResNet3modal_PFnet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth=50, pretrained=True, num_classes=0):
        super(ResNet3modal_PFnet, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        # self.cut_at_pooling = cut_at_pooling
        # self.num_features = num_features
        self.num_classes = num_classes
        self.rgb_SA = SpatialAttention()
        self.th_SA = SpatialAttention()
        self.ni_SA = SpatialAttention()
        # Construct base (pretrained) resnet
        if depth not in ResNet3modal_PFnet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base1 = ResNet3modal_PFnet.__factory[depth](pretrained=False)
        self.base2 = ResNet3modal_PFnet.__factory[depth](pretrained=False)
        self.base3 = ResNet3modal_PFnet.__factory[depth](pretrained=False)
        
        if self.pretrained == True:
            trained_state_dict = torch.load('F:\\myproject_mmdataset\\pretrain_file\\resnet50-19c8e357.pth')
            self.base1.load_state_dict(trained_state_dict)
            # import pdb; pdb.set_trace()
            self.base2.load_state_dict(trained_state_dict)
            self.base3.load_state_dict(trained_state_dict)
            print('pretrained model loading...')

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        # fully connected layer
        self.fc_rgb_top = self._fc_layer(2048, 256)
        self.fc_th_top = self._fc_layer(2048, 256)
        self.fc_ni_top = self._fc_layer(2048, 256)
        self.fc_rgb_bottom = self._fc_layer(2048, 256)
        self.fc_th_bottom = self._fc_layer(2048, 256)
        self.fc_ni_bottom = self._fc_layer(2048, 256)
        self.fc_fusion_top1 = self._fc_layer(2048, 256)
        self.fc_fusion_bottom1 = self._fc_layer(2048, 256)
        self.fc_fusion_top2 = self._fc_layer(2048, 256)
        self.fc_fusion_bottom2 = self._fc_layer(2048, 256)

        # identity classification layer
        self.classifier_all = nn.Linear(2560, num_classes)
        self.classifier_rgb_top = nn.Linear(256, num_classes)
        self.classifier_th_top = nn.Linear(256, num_classes)
        self.classifier_ni_top = nn.Linear(256, num_classes)
        self.classifier_rgb_bottom = nn.Linear(256, num_classes)
        self.classifier_th_bottom = nn.Linear(256, num_classes)
        self.classifier_ni_bottom = nn.Linear(256, num_classes)
        self.classifier_fusion_top1 = nn.Linear(256, num_classes)
        self.classifier_fusion_bottom1 = nn.Linear(256, num_classes)
        self.classifier_fusion_top2 = nn.Linear(256, num_classes)
        self.classifier_fusion_bottom2 = nn.Linear(256, num_classes)

        if not self.pretrained:
            self.reset_params()

    def _fc_layer(self, in_channel, out_channel):
        layers = []
        layers.append(
            nn.Sequential(
                nn.Linear(in_channel, out_channel),
                nn.BatchNorm1d(out_channel),
                nn.ReLU()
            )
        )
        return nn.Sequential(*layers)

    def forward(self, x1, x2, x3):
        # feature fusion
        for name1, module1 in self.base1._modules.items():
            if name1 == 'avgpool':
                break
            if name1 == 'conv1':
                x1 = module1(x1)
            if name1 == 'layer1':
                x1 = module1(x1)
            if name1 == 'layer2':
                x1 = module1(x1)
            if name1 == 'layer3':
                x1 = module1(x1)
                # print('11')
                x1 = self.rgb_SA(x1) * x1
            if name1 == 'layer4':
                x1 = module1(x1)
                x1 = self.rgb_SA(x1) * x1

        for name2, module2 in self.base2._modules.items():
            if name2 == 'avgpool':
                break
            if name2 == 'conv1':
                x2 = module2(x2)
            if name2 == 'layer1':
                x2 = module2(x2)
            if name2 == 'layer2':
                x2 = module2(x2)
            if name2 == 'layer3':
                x2 = module2(x2)
                x2 = self.th_SA(x2) * x2
            if name2 == 'layer4':
                x2 = module2(x2)
                x2 = self.th_SA(x2) * x2

        for name3, module3 in self.base3._modules.items():
            if name3 == 'avgpool':
                break
            if name3 == 'conv1':
                x3 = module3(x3)
            if name3 == 'layer1':
                x3 = module3(x3)
            if name3 == 'layer2':
                x3 = module3(x3)
            if name3 == 'layer3':
                x3 = module3(x3)
                x3 = self.ni_SA(x3) * x3
            if name3 == 'layer4':
                x3 = module3(x3)
                x3 = self.ni_SA(x3) * x3

        sum_x12 = x1 + x2
        sum_x13 = x1 + x3

        rgb_top, rgb_bottom = x1.chunk(2, dim=2)
        th_top, th_bottom = x2.chunk(2, dim=2)
        ni_top, ni_bottom = x3.chunk(2, dim=2)
        sum_top1, sum_bottom1 = sum_x12.chunk(2, dim=2)
        sum_top2, sum_bottom2 = sum_x13.chunk(2, dim=2)

        # global avgpool layer
        # modal feature
        rgb_top = self.global_avgpool(rgb_top)
        rgb_bottom = self.global_avgpool(rgb_bottom)
        th_top = self.global_avgpool(th_top)
        th_bottom = self.global_avgpool(th_bottom)
        ni_top = self.global_avgpool(ni_top)
        ni_bottom = self.global_avgpool(ni_bottom)
        # fusion feature
        sum_top1 = self.global_avgpool(sum_top1)
        sum_bottom1 = self.global_avgpool(sum_bottom1)
        sum_top2 = self.global_avgpool(sum_top2)
        sum_bottom2 = self.global_avgpool(sum_bottom2)

        # FC layer
        batch = x1.shape[0]
        # modal feature
        rgb_top = self.fc_rgb_top(rgb_top.view(batch, -1))
        rgb_bottom = self.fc_rgb_bottom(rgb_bottom.view(batch, -1))
        th_top = self.fc_th_top(th_top.view(batch, -1))
        th_bottom = self.fc_th_bottom(th_bottom.view(batch, -1))
        ni_top = self.fc_ni_top(ni_top.view(batch, -1))
        ni_bottom = self.fc_ni_bottom(ni_bottom.view(batch, -1))
        # fusion feature
        sum_top1 = self.fc_fusion_top1(sum_top1.view(batch, -1))
        sum_bottom1 = self.fc_fusion_bottom1(sum_bottom1.view(batch, -1))
        sum_top2 = self.fc_fusion_top2(sum_top2.view(batch, -1))
        sum_bottom2 = self.fc_fusion_bottom2(sum_bottom2.view(batch, -1))
        final = torch.cat([th_top, th_bottom, sum_top1, sum_bottom1,
                           rgb_top, rgb_bottom, sum_top2, sum_bottom2,
                           ni_top, ni_bottom], dim=1)

        # identity classification layer
        # modal feature
        y_all = self.classifier_all(final)
        y_rgb_top = self.classifier_rgb_top(rgb_top)
        y_th_top = self.classifier_th_top(th_top)
        y_ni_top = self.classifier_ni_top(ni_top)
        y_rgb_bottom = self.classifier_rgb_top(rgb_bottom)
        y_th_bottom = self.classifier_th_bottom(th_bottom)
        y_ni_bottom = self.classifier_ni_bottom(ni_bottom)
        # fusion feature
        y_top1 = self.classifier_fusion_top1(sum_top1)
        y_bottom1 = self.classifier_fusion_bottom1(sum_bottom1)
        y_top2 = self.classifier_fusion_top2(sum_top2)
        y_bottom2 = self.classifier_fusion_bottom2(sum_bottom2)
        return y_all, y_rgb_top, y_rgb_bottom, \
            y_th_top, y_th_bottom, \
            y_ni_top, y_ni_bottom, \
            y_top1, y_bottom1, \
            y_top2, y_bottom2, \
            final

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)


def resnet3modal_PFnet(**kwargs):
    model = ResNet3modal_PFnet(**kwargs)
    # pretrained_state_dict = model_zoo.load_url(model_urls['resnet50'])
    # now_state_dict = model.state_dict()
    # pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in now_state_dict}
    # now_state_dict.update(pretrained_state_dict)
    # model.load_state_dict(now_state_dict)
    return model


class ResNet2modal_PFnet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0):
        super(ResNet2modal_PFnet, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling
        self.num_features = num_features
        self.num_classes = num_classes
        self.rgb_SA = SpatialAttention()
        self.th_SA = SpatialAttention()
        # Construct base (pretrained) resnet
        if depth not in ResNet3modal_PFnet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base1 = ResNet3modal_PFnet.__factory[depth](pretrained=pretrained)
        self.base2 = ResNet3modal_PFnet.__factory[depth](pretrained=pretrained)

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        # fully connected layer
        self.fc_rgb_top = self._fc_layer(2048, 256)
        self.fc_th_top = self._fc_layer(2048, 256)
        self.fc_rgb_bottom = self._fc_layer(2048, 256)
        self.fc_th_bottom = self._fc_layer(2048, 256)
        self.fc_fusion_top1 = self._fc_layer(2048, 256)
        self.fc_fusion_bottom1 = self._fc_layer(2048, 256)

        # identity classification layer
        self.classifier_all = nn.Linear(2560, num_classes)
        self.classifier_rgb_top = nn.Linear(256, num_classes)
        self.classifier_th_top = nn.Linear(256, num_classes)
        self.classifier_rgb_bottom = nn.Linear(256, num_classes)
        self.classifier_th_bottom = nn.Linear(256, num_classes)
        self.classifier_fusion_top1 = nn.Linear(256, num_classes)
        self.classifier_fusion_bottom1 = nn.Linear(256, num_classes)

        if not self.pretrained:
            self.reset_params()
        else:
            trained_state_dict = torch.load('F:\\myproject_mmdataset\\pretrain_file\\resnet50-19c8e357.pth')
            self.base1.load_state_dict(trained_state_dict)
            self.base2.load_state_dict(trained_state_dict)
            print('pretrained model loading...')

    def _fc_layer(self, in_channel, out_channel):
        layers = []
        layers.append(
            nn.Sequential(
                nn.Linear(in_channel, out_channel),
                nn.BatchNorm1d(out_channel),
                nn.ReLU()
            )
        )
        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        # feature fusion
        for name1, module1 in self.base1._modules.items():
            if name1 == 'avgpool':
                break
            if name1 == 'conv1':
                x1 = module1(x1)
            if name1 == 'layer1':
                x1 = module1(x1)
            if name1 == 'layer2':
                x1 = module1(x1)
            if name1 == 'layer3':
                x1 = module1(x1)
                # print('11')
                x1 = self.rgb_SA(x1) * x1
            if name1 == 'layer4':
                x1 = module1(x1)
                x1 = self.rgb_SA(x1) * x1

        for name2, module2 in self.base2._modules.items():
            if name2 == 'avgpool':
                break
            if name2 == 'conv1':
                x2 = module2(x2)
            if name2 == 'layer1':
                x2 = module2(x2)
            if name2 == 'layer2':
                x2 = module2(x2)
            if name2 == 'layer3':
                x2 = module2(x2)
                x2 = self.th_SA(x2) * x2
            if name2 == 'layer4':
                x2 = module2(x2)
                x2 = self.th_SA(x2) * x2


        sum_x12 = x1 + x2

        rgb_top, rgb_bottom = x1.chunk(2, dim=2)
        th_top, th_bottom = x2.chunk(2, dim=2)
        sum_top1, sum_bottom1 = sum_x12.chunk(2, dim=2)

        # global avgpool layer
        # modal feature
        rgb_top = self.global_avgpool(rgb_top)
        rgb_bottom = self.global_avgpool(rgb_bottom)
        th_top = self.global_avgpool(th_top)
        th_bottom = self.global_avgpool(th_bottom)
        # fusion feature
        sum_top1 = self.global_avgpool(sum_top1)
        sum_bottom1 = self.global_avgpool(sum_bottom1)

        # FC layer
        batch = x1.shape[0]
        # modal feature
        rgb_top = self.fc_rgb_top(rgb_top.view(batch, -1))
        rgb_bottom = self.fc_rgb_bottom(rgb_bottom.view(batch, -1))
        th_top = self.fc_th_top(th_top.view(batch, -1))
        th_bottom = self.fc_th_bottom(th_bottom.view(batch, -1))
        # fusion feature
        sum_top1 = self.fc_fusion_top1(sum_top1.view(batch, -1))
        sum_bottom1 = self.fc_fusion_bottom1(sum_bottom1.view(batch, -1))
        final = torch.cat([th_top, th_bottom, sum_top1, sum_bottom1,
                           rgb_top, rgb_bottom], dim=1)

        # identity classification layer
        # modal feature
        y_all = self.classifier_all(final)
        y_rgb_top = self.classifier_rgb_top(rgb_top)
        y_th_top = self.classifier_th_top(th_top)
        y_rgb_bottom = self.classifier_rgb_top(rgb_bottom)
        y_th_bottom = self.classifier_th_bottom(th_bottom)
        # fusion feature
        y_top1 = self.classifier_fusion_top1(sum_top1)
        y_bottom1 = self.classifier_fusion_bottom1(sum_bottom1)
        return y_all, y_rgb_top, y_rgb_bottom, \
            y_th_top, y_th_bottom, \
            y_top1, y_bottom1

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)


def resnet2modal_PFnet(**kwargs):
    model = ResNet2modal_PFnet(50, **kwargs)
    pretrained_state_dict = model_zoo.load_url(model_urls['resnet50'])
    now_state_dict = model.state_dict()
    pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in now_state_dict}
    now_state_dict.update(pretrained_state_dict)
    model.load_state_dict(now_state_dict)
    return model


class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0):
        super(ResNet, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = ResNet.__factory[depth](pretrained=pretrained)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = self.base.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal(self.feat.weight, mode='fan_out')
                init.constant(self.feat.bias, 0)
                init.constant(self.feat_bn.weight, 1)
                init.constant(self.feat_bn.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes)
                init.normal(self.classifier.weight, std=0.001)
                init.constant(self.classifier.bias, 0)

        if not self.pretrained:
            self.reset_params()

    def forward(self, x):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

        if self.cut_at_pooling:
            return x

        if self.has_embedding:
            x = self.feat(x)
            x = self.feat_bn(x)
        if self.norm:
            x = F.normalize(x)
        elif self.has_embedding:
            x = F.relu(x)
        if self.dropout > 0:
            x = self.drop(x)
        if self.num_classes > 0:
            x = self.classifier(x)
        # print(x.shape)
        return x

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)


def resnet(**kwargs):
    model = ResNet(50, **kwargs)
    pretrained_state_dict = model_zoo.load_url(model_urls['resnet50'])
    now_state_dict = model.state_dict()
    pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in now_state_dict}
    now_state_dict.update(pretrained_state_dict)
    model.load_state_dict(now_state_dict)

    return model


class ResNet_p6(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0):
        super(ResNet_p6, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling
        self.num_features = num_features
        self.num_classes = num_classes

        # Construct base (pretrained) resnet
        if depth not in ResNet3modal_part6.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base1 = ResNet_p6.__factory[depth](pretrained=pretrained)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc_rgb_p1 = self._fc_layer(2048, 128)
        self.fc_rgb_p2 = self._fc_layer(2048, 128)
        self.fc_rgb_p3 = self._fc_layer(2048, 128)
        self.fc_rgb_p4 = self._fc_layer(2048, 128)
        self.fc_rgb_p5 = self._fc_layer(2048, 128)
        self.fc_rgb_p6 = self._fc_layer(2048, 128)

        # identity classification layer
        self.classifier_rgb_p1 = nn.Linear(128, num_classes)
        self.classifier_rgb_p2 = nn.Linear(128, num_classes)
        self.classifier_rgb_p3 = nn.Linear(128, num_classes)
        self.classifier_rgb_p4 = nn.Linear(128, num_classes)
        self.classifier_rgb_p5 = nn.Linear(128, num_classes)
        self.classifier_rgb_p6 = nn.Linear(128, num_classes)
        self.classifier_all = nn.Linear(768, num_classes)

        if not self.pretrained:
            self.reset_params()

    def _fc_layer(self, in_channel, out_channel):
        layers = []
        layers.append(
            nn.Sequential(
                nn.Linear(in_channel, out_channel),
                nn.BatchNorm1d(out_channel),
                nn.ReLU()
            )
        )
        return nn.Sequential(*layers)

    def forward(self, x1):
        # feature fusion
        for name1, module1 in self.base1._modules.items():
            if name1 == 'avgpool':
                break
            if name1 == 'conv1':
                x1 = module1(x1)
            if name1 == 'layer1':
                x1 = module1(x1)
            if name1 == 'layer2':
                x1 = module1(x1)
            if name1 == 'layer3':
                x1 = module1(x1)
            if name1 == 'layer4':
                x1 = module1(x1)

        rgb_p1, rgb_p2, rgb_p3, rgb_p4, rgb_p5, rgb_p6 = x1.chunk(6, dim=2)

        # global avgpool layer
        # modal feature
        rgb_p1 = self.global_avgpool(rgb_p1)
        rgb_p2 = self.global_avgpool(rgb_p2)
        rgb_p3 = self.global_avgpool(rgb_p3)
        rgb_p4 = self.global_avgpool(rgb_p4)
        rgb_p5 = self.global_avgpool(rgb_p5)
        rgb_p6 = self.global_avgpool(rgb_p6)

        # FC layer
        batch = x1.shape[0]
        # modal feature
        rgb_p1 = self.fc_rgb_p1(rgb_p1.view(batch, -1))
        rgb_p2 = self.fc_rgb_p2(rgb_p2.view(batch, -1))
        rgb_p3 = self.fc_rgb_p3(rgb_p3.view(batch, -1))
        rgb_p4 = self.fc_rgb_p4(rgb_p4.view(batch, -1))
        rgb_p5 = self.fc_rgb_p5(rgb_p5.view(batch, -1))
        rgb_p6 = self.fc_rgb_p6(rgb_p6.view(batch, -1))
        final = torch.cat([rgb_p1, rgb_p2, rgb_p3, rgb_p4, rgb_p5, rgb_p6], dim=1)

        # identity classification layer
        # modal feature
        final = self.classifier_all(final)
        rgb_p1 = self.classifier_rgb_p1(rgb_p1)
        rgb_p2 = self.classifier_rgb_p2(rgb_p2)
        rgb_p3 = self.classifier_rgb_p3(rgb_p3)
        rgb_p4 = self.classifier_rgb_p4(rgb_p4)
        rgb_p5 = self.classifier_rgb_p5(rgb_p5)
        rgb_p6 = self.classifier_rgb_p6(rgb_p6)

        # fusion feature

        return final, rgb_p1, rgb_p2, rgb_p3, rgb_p4, rgb_p5, rgb_p6


    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)


def resnet_p6(**kwargs):
    model = ResNet_p6(50, **kwargs)
    pretrained_state_dict = model_zoo.load_url(model_urls['resnet50'])
    now_state_dict = model.state_dict()
    pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in now_state_dict}
    now_state_dict.update(pretrained_state_dict)
    model.load_state_dict(now_state_dict)

    return model


class ResNet2modal(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0):
        super(ResNet2modal, self).__init__()

        self.part = 2
        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling
        self.num_features = num_features
        self.num_classes = num_classes

        # Construct base (pretrained) resnet
        if depth not in ResNet2modal.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base1 = ResNet2modal.__factory[depth](pretrained=pretrained)
        self.base2 = ResNet2modal.__factory[depth](pretrained=pretrained)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        # fully connected layer
        self.fc_rgb = self._fc_layer(2048, 512)
        self.fc_th = self._fc_layer(2048, 512)
        # identity classification layer
        self.classifier_all = nn.Linear(1024, num_classes)
        self.classifier_rgb = nn.Linear(512, num_classes)
        self.classifier_th = nn.Linear(512, num_classes)

        if not self.pretrained:
            self.reset_params()

    def _fc_layer(self, in_channel, out_channel):
        layers = []
        layers.append(
            nn.Sequential(
                nn.Linear(in_channel, out_channel),
                nn.BatchNorm1d(out_channel),
                nn.ReLU()
            )
        )
        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        for name1, module1 in self.base1._modules.items():
            if name1 == 'avgpool':
                break
            if name1 == 'conv1':
                x1 = module1(x1)
            if name1 == 'layer1':
                x1 = module1(x1)
            if name1 == 'layer2':
                x1 = module1(x1)
            if name1 == 'layer3':
                x1 = module1(x1)
            if name1 == 'layer4':
                x1 = module1(x1)

        for name2, module2 in self.base2._modules.items():
            if name2 == 'avgpool':
                break
            if name2 == 'conv1':
                x2 = module2(x2)
            if name2 == 'layer1':
                x2 = module2(x2)
            if name2 == 'layer2':
                x2 = module2(x2)
            if name2 == 'layer3':
                x2 = module2(x2)
            if name2 == 'layer4':
                x2 = module2(x2)

        # global avgpool layer
        rgb = self.global_avgpool(x1)
        th = self.global_avgpool(x2)

        # FC layer
        batch = rgb.size(0)
        rgb = self.fc_rgb(rgb.view(batch, -1))
        th = self.fc_th(th.view(batch, -1))

        final = torch.cat([rgb, th], dim=1)

        # identity classification layer
        y_all = self.classifier_all(final)
        y_rgb = self.classifier_rgb(rgb)
        y_th = self.classifier_th(th)
        return y_all, y_rgb, y_th

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)


def resnet2modal(**kwargs):
    return ResNet2modal(50, **kwargs)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ResNet3modal(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0):
        super(ResNet3modal, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling
        self.num_features = num_features
        self.num_classes = num_classes

        # Construct base (pretrained) resnet
        if depth not in ResNet3modal.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base1 = ResNet3modal.__factory[depth](pretrained=pretrained)
        self.base2 = ResNet3modal.__factory[depth](pretrained=pretrained)
        self.base3 = ResNet3modal.__factory[depth](pretrained=pretrained)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        # fully connected layer
        self.fc_rgb = self._fc_layer(2048, 512)
        self.fc_th = self._fc_layer(2048, 512)
        self.fc_ni = self._fc_layer(2048, 512)
        # identity classification layer
        self.classifier_all = nn.Linear(1536, num_classes)
        self.classifier_rgb = nn.Linear(512, num_classes)
        self.classifier_th = nn.Linear(512, num_classes)
        self.classifier_ni = nn.Linear(512, num_classes)

        if not self.pretrained:
            self.reset_params()

    def _fc_layer(self, in_channel, out_channel):
        layers = []
        layers.append(
            nn.Sequential(
                nn.Linear(in_channel, out_channel),
                nn.BatchNorm1d(out_channel),
                nn.ReLU()
            )
        )
        return nn.Sequential(*layers)

    def forward(self, x1, x2, x3):
        # feature fusion
        for name1, module1 in self.base1._modules.items():
            if name1 == 'avgpool':
                break
            if name1 == 'conv1':
                x1 = module1(x1)
            if name1 == 'layer1':
                x1 = module1(x1)
            if name1 == 'layer2':
                x1 = module1(x1)
            if name1 == 'layer3':
                x1 = module1(x1)
            if name1 == 'layer4':
                x1 = module1(x1)

        for name2, module2 in self.base2._modules.items():
            if name2 == 'avgpool':
                break
            if name2 == 'conv1':
                x2 = module2(x2)
            if name2 == 'layer1':
                x2 = module2(x2)
            if name2 == 'layer2':
                x2 = module2(x2)
            if name2 == 'layer3':
                x2 = module2(x2)
            if name2 == 'layer4':
                x2 = module2(x2)

        for name3, module3 in self.base3._modules.items():
            if name3 == 'avgpool':
                break
            if name3 == 'conv1':
                x3 = module3(x3)
            if name3 == 'layer1':
                x3 = module3(x3)
            if name3 == 'layer2':
                x3 = module3(x3)
            if name3 == 'layer3':
                x3 = module3(x3)
            if name3 == 'layer4':
                x3 = module3(x3)

        # global avgpool layer
        rgb = self.global_avgpool(x1)
        th = self.global_avgpool(x2)
        ni = self.global_avgpool(x3)

        # FC layer
        batch = rgb.size(0)
        rgb = self.fc_rgb(rgb.view(batch, -1))
        th = self.fc_th(th.view(batch, -1))
        ni = self.fc_ni(ni.view(batch, -1))

        final = torch.cat([th, rgb, ni], dim=1)

        # identity classification layer
        y_all = self.classifier_all(final)
        y_rgb = self.classifier_rgb(rgb)
        y_th = self.classifier_th(th)
        y_ni = self.classifier_ni(ni)
        return y_all, y_rgb, y_th, y_ni

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)


def resnet3modal(**kwargs):
    model = ResNet3modal(50, **kwargs)
    pretrained_state_dict = model_zoo.load_url(model_urls['resnet50'])
    now_state_dict = model.state_dict()
    pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in now_state_dict}
    now_state_dict.update(pretrained_state_dict)
    model.load_state_dict(now_state_dict)
    return model


class ResNet3modal_AHU(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0):
        super(ResNet3modal_AHU, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling
        self.num_features = num_features
        self.num_classes = num_classes
        self.rgb_SA = SpatialAttention()
        self.th_SA = SpatialAttention()
        self.ni_SA = SpatialAttention()
        # Construct base (pretrained) resnet
        if depth not in ResNet3modal_AHU.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base1 = ResNet3modal_AHU.__factory[depth](pretrained=pretrained)
        self.base2 = ResNet3modal_AHU.__factory[depth](pretrained=pretrained)
        self.base3 = ResNet3modal_AHU.__factory[depth](pretrained=pretrained)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        # fully connected layer
        self.fc_rgb_top = self._fc_layer(2048, 256)
        self.fc_th_top = self._fc_layer(2048, 256)
        self.fc_ni_top = self._fc_layer(2048, 256)
        self.fc_rgb_bottom = self._fc_layer(2048, 256)
        self.fc_th_bottom = self._fc_layer(2048, 256)
        self.fc_ni_bottom = self._fc_layer(2048, 256)
        self.fc_fusion_top1 = self._fc_layer(2048, 256)
        self.fc_fusion_bottom1 = self._fc_layer(2048, 256)
        self.fc_fusion_top2 = self._fc_layer(2048, 256)
        self.fc_fusion_bottom2 = self._fc_layer(2048, 256)

        # identity classification layer
        self.classifier_all = nn.Linear(2560, num_classes)
        self.classifier_rgb_top = nn.Linear(256, num_classes)
        self.classifier_th_top = nn.Linear(256, num_classes)
        self.classifier_ni_top = nn.Linear(256, num_classes)
        self.classifier_rgb_bottom = nn.Linear(256, num_classes)
        self.classifier_th_bottom = nn.Linear(256, num_classes)
        self.classifier_ni_bottom = nn.Linear(256, num_classes)
        self.classifier_fusion_top1 = nn.Linear(256, num_classes)
        self.classifier_fusion_bottom1 = nn.Linear(256, num_classes)
        self.classifier_fusion_top2 = nn.Linear(256, num_classes)
        self.classifier_fusion_bottom2 = nn.Linear(256, num_classes)

        if not self.pretrained:
            self.reset_params()

    def _fc_layer(self, in_channel, out_channel):
        layers = []
        layers.append(
            nn.Sequential(
                nn.Linear(in_channel, out_channel),
                nn.BatchNorm1d(out_channel),
                nn.ReLU()
            )
        )
        return nn.Sequential(*layers)

    def forward(self, x1, x2, x3):
        # feature fusion
        for name1, module1 in self.base1._modules.items():
            if name1 == 'avgpool':
                break
            if name1 == 'conv1':
                x1 = module1(x1)
            if name1 == 'layer1':
                x1 = module1(x1)
            if name1 == 'layer2':
                x1 = module1(x1)
            if name1 == 'layer3':
                x1 = module1(x1)
                # print('11')
                x1 = self.rgb_SA(x1) * x1
            if name1 == 'layer4':
                x1 = module1(x1)
                x1 = self.rgb_SA(x1) * x1

        for name2, module2 in self.base2._modules.items():
            if name2 == 'avgpool':
                break
            if name2 == 'conv1':
                x2 = module2(x2)
            if name2 == 'layer1':
                x2 = module2(x2)
            if name2 == 'layer2':
                x2 = module2(x2)
            if name2 == 'layer3':
                x2 = module2(x2)
                x2 = self.th_SA(x2) * x2
            if name2 == 'layer4':
                x2 = module2(x2)
                x2 = self.th_SA(x2) * x2

        for name3, module3 in self.base3._modules.items():
            if name3 == 'avgpool':
                break
            if name3 == 'conv1':
                x3 = module3(x3)
            if name3 == 'layer1':
                x3 = module3(x3)
            if name3 == 'layer2':
                x3 = module3(x3)
            if name3 == 'layer3':
                x3 = module3(x3)
                x3 = self.ni_SA(x3) * x3
            if name3 == 'layer4':
                x3 = module3(x3)
                x3 = self.ni_SA(x3) * x3

        sum_x12 = x1 + x2
        sum_x13 = x1 + x3

        rgb_top, rgb_bottom = x1.chunk(2, dim=2)
        th_top, th_bottom = x2.chunk(2, dim=2)
        ni_top, ni_bottom = x3.chunk(2, dim=2)
        sum_top1, sum_bottom1 = sum_x12.chunk(2, dim=2)
        sum_top2, sum_bottom2 = sum_x13.chunk(2, dim=2)

        # global avgpool layer
        # modal feature
        rgb_top = self.global_avgpool(rgb_top)
        rgb_bottom = self.global_avgpool(rgb_bottom)
        th_top = self.global_avgpool(th_top)
        th_bottom = self.global_avgpool(th_bottom)
        ni_top = self.global_avgpool(ni_top)
        ni_bottom = self.global_avgpool(ni_bottom)
        # fusion feature
        sum_top1 = self.global_avgpool(sum_top1)
        sum_bottom1 = self.global_avgpool(sum_bottom1)
        sum_top2 = self.global_avgpool(sum_top2)
        sum_bottom2 = self.global_avgpool(sum_bottom2)

        # FC layer
        batch = x1.shape[0]
        # modal feature
        rgb_top = self.fc_rgb_top(rgb_top.view(batch, -1))
        rgb_bottom = self.fc_rgb_bottom(rgb_bottom.view(batch, -1))
        th_top = self.fc_th_top(th_top.view(batch, -1))
        th_bottom = self.fc_th_bottom(th_bottom.view(batch, -1))
        ni_top = self.fc_ni_top(ni_top.view(batch, -1))
        ni_bottom = self.fc_ni_bottom(ni_bottom.view(batch, -1))
        # fusion feature
        sum_top1 = self.fc_fusion_top1(sum_top1.view(batch, -1))
        sum_bottom1 = self.fc_fusion_bottom1(sum_bottom1.view(batch, -1))
        sum_top2 = self.fc_fusion_top2(sum_top2.view(batch, -1))
        sum_bottom2 = self.fc_fusion_bottom2(sum_bottom2.view(batch, -1))
        final = torch.cat([th_top, th_bottom, sum_top1, sum_bottom1,
                           rgb_top, rgb_bottom, sum_top2, sum_bottom2,
                           ni_top, ni_bottom], dim=1)

        # identity classification layer
        # modal feature
        y_all = self.classifier_all(final)
        y_rgb_top = self.classifier_rgb_top(rgb_top)
        y_th_top = self.classifier_th_top(th_top)
        y_ni_top = self.classifier_ni_top(ni_top)
        y_rgb_bottom = self.classifier_rgb_top(rgb_bottom)
        y_th_bottom = self.classifier_th_bottom(th_bottom)
        y_ni_bottom = self.classifier_ni_bottom(ni_bottom)
        # fusion feature
        y_top1 = self.classifier_fusion_top1(sum_top1)
        y_bottom1 = self.classifier_fusion_bottom1(sum_bottom1)
        y_top2 = self.classifier_fusion_top2(sum_top2)
        y_bottom2 = self.classifier_fusion_bottom2(sum_bottom2)
        return y_all, y_rgb_top, y_rgb_bottom, \
            y_th_top, y_th_bottom, \
            y_ni_top, y_ni_bottom, \
            y_top1, y_bottom1, \
            y_top2, y_bottom2

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)


def resnet3modal_AHU(**kwargs):
    model = ResNet3modal_AHU(50, **kwargs)
    pretrained_state_dict = model_zoo.load_url(model_urls['resnet50'])
    now_state_dict = model.state_dict()
    pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in now_state_dict}
    now_state_dict.update(pretrained_state_dict)
    model.load_state_dict(now_state_dict)
    return model


class ResNet3modal_attentionOnly(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0):
        super(ResNet3modal_attentionOnly, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling
        self.num_features = num_features
        self.num_classes = num_classes
        self.rgb_SA = SpatialAttention()
        self.th_SA = SpatialAttention()
        self.ni_SA = SpatialAttention()
        # Construct base (pretrained) resnet
        if depth not in ResNet3modal_attentionOnly.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base1 = ResNet3modal_attentionOnly.__factory[depth](pretrained=pretrained)
        self.base2 = ResNet3modal_attentionOnly.__factory[depth](pretrained=pretrained)
        self.base3 = ResNet3modal_attentionOnly.__factory[depth](pretrained=pretrained)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        # fully connected layer
        self.fc_rgb = self._fc_layer(2048, 512)
        self.fc_th = self._fc_layer(2048, 512)
        self.fc_ni = self._fc_layer(2048, 512)
        self.fc_fusion1 = self._fc_layer(2048, 512)
        self.fc_fusion2 = self._fc_layer(2048, 512)

        # identity classification layer
        self.classifier_all = nn.Linear(2560, num_classes)
        self.classifier_rgb = nn.Linear(512, num_classes)
        self.classifier_th = nn.Linear(512, num_classes)
        self.classifier_ni = nn.Linear(512, num_classes)
        self.classifier_fusion1 = nn.Linear(512, num_classes)
        self.classifier_fusion2 = nn.Linear(512, num_classes)

        if not self.pretrained:
            self.reset_params()

    def _fc_layer(self, in_channel, out_channel):
        layers = []
        layers.append(
            nn.Sequential(
                nn.Linear(in_channel, out_channel),
                nn.BatchNorm1d(out_channel),
                nn.ReLU()
            )
        )
        return nn.Sequential(*layers)

    def forward(self, x1, x2, x3):
        # feature fusion
        for name1, module1 in self.base1._modules.items():
            if name1 == 'avgpool':
                break
            if name1 == 'conv1':
                x1 = module1(x1)
            if name1 == 'layer1':
                x1 = module1(x1)
            if name1 == 'layer2':
                x1 = module1(x1)
            if name1 == 'layer3':
                x1 = module1(x1)
                # print('11')
                x1 = self.rgb_SA(x1) * x1
            if name1 == 'layer4':
                x1 = module1(x1)
                x1 = self.rgb_SA(x1) * x1

        for name2, module2 in self.base2._modules.items():
            if name2 == 'avgpool':
                break
            if name2 == 'conv1':
                x2 = module2(x2)
            if name2 == 'layer1':
                x2 = module2(x2)
            if name2 == 'layer2':
                x2 = module2(x2)
            if name2 == 'layer3':
                x2 = module2(x2)
                x2 = self.rgb_SA(x2) * x2
            if name2 == 'layer4':
                x2 = module2(x2)
                x2 = self.rgb_SA(x2) * x2

        for name3, module3 in self.base3._modules.items():
            if name3 == 'avgpool':
                break
            if name3 == 'conv1':
                x3 = module3(x3)
            if name3 == 'layer1':
                x3 = module3(x3)
            if name3 == 'layer2':
                x3 = module3(x3)
            if name3 == 'layer3':
                x3 = module3(x3)
                x3 = self.rgb_SA(x3) * x3
            if name3 == 'layer4':
                x3 = module3(x3)
                x3 = self.rgb_SA(x3) * x3

        sum1 = x1 + x2
        sum2 = x1 + x3

        rgb = x1
        th = x2
        ni = x3

        # global avgpool layer
        # modal feature
        rgb = self.global_avgpool(rgb)
        th = self.global_avgpool(th)
        ni = self.global_avgpool(ni)
        # fusion feature
        sum1 = self.global_avgpool(sum1)
        sum2 = self.global_avgpool(sum2)

        # FC layer
        batch = x1.shape[0]
        # modal feature
        rgb = self.fc_rgb(rgb.view(batch, -1))
        th = self.fc_th(th.view(batch, -1))
        ni = self.fc_ni(ni.view(batch, -1))
        # fusion feature
        sum1 = self.fc_fusion1(sum1.view(batch, -1))
        sum2 = self.fc_fusion2(sum2.view(batch, -1))
        final = torch.cat([th, sum1,
                           rgb, sum2,
                           ni], dim=1)

        # identity classification layer
        # modal feature
        y_all = self.classifier_all(final)
        y_rgb = self.classifier_rgb(rgb)
        y_th = self.classifier_th(th)
        y_ni = self.classifier_ni(ni)
        # fusion feature
        y_top1 = self.classifier_fusion1(sum1)
        y_top2 = self.classifier_fusion2(sum2)
        return y_all, y_rgb, \
            y_th, \
            y_ni, \
            y_top1, \
            y_top2

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)


def resnet3modal_attentionOnly(**kwargs):
    model = ResNet3modal_attentionOnly(50, **kwargs)
    pretrained_state_dict = model_zoo.load_url(model_urls['resnet50'])
    now_state_dict = model.state_dict()
    pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in now_state_dict}
    now_state_dict.update(pretrained_state_dict)
    model.load_state_dict(now_state_dict)
    return model


class ResNet2modal_part6(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0):
        super(ResNet2modal_part6, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling
        self.num_features = num_features
        self.num_classes = num_classes

        # Construct base (pretrained) resnet
        if depth not in ResNet3modal_part6.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base1 = ResNet3modal_part6.__factory[depth](pretrained=pretrained)
        self.base2 = ResNet3modal_part6.__factory[depth](pretrained=pretrained)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc_rgb_p1 = self._fc_layer(2048, 128)
        self.fc_rgb_p2 = self._fc_layer(2048, 128)
        self.fc_rgb_p3 = self._fc_layer(2048, 128)
        self.fc_rgb_p4 = self._fc_layer(2048, 128)
        self.fc_rgb_p5 = self._fc_layer(2048, 128)
        self.fc_rgb_p6 = self._fc_layer(2048, 128)
        self.fc_th_p1 = self._fc_layer(2048, 128)
        self.fc_th_p2 = self._fc_layer(2048, 128)
        self.fc_th_p3 = self._fc_layer(2048, 128)
        self.fc_th_p4 = self._fc_layer(2048, 128)
        self.fc_th_p5 = self._fc_layer(2048, 128)
        self.fc_th_p6 = self._fc_layer(2048, 128)

        # identity classification layer
        self.classifier_rgb_p1 = nn.Linear(128, num_classes)
        self.classifier_rgb_p2 = nn.Linear(128, num_classes)
        self.classifier_rgb_p3 = nn.Linear(128, num_classes)
        self.classifier_rgb_p4 = nn.Linear(128, num_classes)
        self.classifier_rgb_p5 = nn.Linear(128, num_classes)
        self.classifier_rgb_p6 = nn.Linear(128, num_classes)
        self.classifier_th_p1 = nn.Linear(128, num_classes)
        self.classifier_th_p2 = nn.Linear(128, num_classes)
        self.classifier_th_p3 = nn.Linear(128, num_classes)
        self.classifier_th_p4 = nn.Linear(128, num_classes)
        self.classifier_th_p5 = nn.Linear(128, num_classes)
        self.classifier_th_p6 = nn.Linear(128, num_classes)
        self.classifier_all = nn.Linear(1536, num_classes)

        if not self.pretrained:
            self.reset_params()

    def _fc_layer(self, in_channel, out_channel):
        layers = []
        layers.append(
            nn.Sequential(
                nn.Linear(in_channel, out_channel),
                nn.BatchNorm1d(out_channel),
                nn.ReLU()
            )
        )
        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        # feature fusion
        for name1, module1 in self.base1._modules.items():
            if name1 == 'avgpool':
                break
            if name1 == 'conv1':
                x1 = module1(x1)
            if name1 == 'layer1':
                x1 = module1(x1)
            if name1 == 'layer2':
                x1 = module1(x1)
            if name1 == 'layer3':
                x1 = module1(x1)
            if name1 == 'layer4':
                x1 = module1(x1)

        for name2, module2 in self.base2._modules.items():
            if name2 == 'avgpool':
                break
            if name2 == 'conv1':
                x2 = module2(x2)
            if name2 == 'layer1':
                x2 = module2(x2)
            if name2 == 'layer2':
                x2 = module2(x2)
            if name2 == 'layer3':
                x2 = module2(x2)
            if name2 == 'layer4':
                x2 = module2(x2)


        rgb_p1, rgb_p2, rgb_p3, rgb_p4, rgb_p5, rgb_p6 = x1.chunk(6, dim=2)
        th_p1, th_p2, th_p3, th_p4, th_p5, th_p6 = x2.chunk(6, dim=2)

        # global avgpool layer
        # modal feature
        rgb_p1 = self.global_avgpool(rgb_p1)
        rgb_p2 = self.global_avgpool(rgb_p2)
        rgb_p3 = self.global_avgpool(rgb_p3)
        rgb_p4 = self.global_avgpool(rgb_p4)
        rgb_p5 = self.global_avgpool(rgb_p5)
        rgb_p6 = self.global_avgpool(rgb_p6)
        th_p1 = self.global_avgpool(th_p1)
        th_p2 = self.global_avgpool(th_p2)
        th_p3 = self.global_avgpool(th_p3)
        th_p4 = self.global_avgpool(th_p4)
        th_p5 = self.global_avgpool(th_p5)
        th_p6 = self.global_avgpool(th_p6)

        # FC layer
        batch = x1.shape[0]
        # modal feature
        rgb_p1 = self.fc_rgb_p1(rgb_p1.view(batch, -1))
        rgb_p2 = self.fc_rgb_p2(rgb_p2.view(batch, -1))
        rgb_p3 = self.fc_rgb_p3(rgb_p3.view(batch, -1))
        rgb_p4 = self.fc_rgb_p4(rgb_p4.view(batch, -1))
        rgb_p5 = self.fc_rgb_p5(rgb_p5.view(batch, -1))
        rgb_p6 = self.fc_rgb_p6(rgb_p6.view(batch, -1))
        th_p1 = self.fc_th_p1(th_p1.view(batch, -1))
        th_p2 = self.fc_th_p2(th_p2.view(batch, -1))
        th_p3 = self.fc_th_p3(th_p3.view(batch, -1))
        th_p4 = self.fc_th_p4(th_p4.view(batch, -1))
        th_p5 = self.fc_th_p5(th_p5.view(batch, -1))
        th_p6 = self.fc_th_p6(th_p6.view(batch, -1))
        final = torch.cat([rgb_p1, rgb_p2, rgb_p3, rgb_p4, rgb_p5, rgb_p6,
                           th_p1, th_p2, th_p3, th_p4, th_p5, th_p6], dim=1)

        # identity classification layer
        # modal feature
        final = self.classifier_all(final)
        rgb_p1 = self.classifier_rgb_p1(rgb_p1)
        rgb_p2 = self.classifier_rgb_p2(rgb_p2)
        rgb_p3 = self.classifier_rgb_p3(rgb_p3)
        rgb_p4 = self.classifier_rgb_p4(rgb_p4)
        rgb_p5 = self.classifier_rgb_p5(rgb_p5)
        rgb_p6 = self.classifier_rgb_p6(rgb_p6)
        th_p1 = self.classifier_th_p1(th_p1)
        th_p2 = self.classifier_th_p2(th_p2)
        th_p3 = self.classifier_th_p3(th_p3)
        th_p4 = self.classifier_th_p4(th_p4)
        th_p5 = self.classifier_th_p5(th_p5)
        th_p6 = self.classifier_th_p6(th_p6)

        # fusion feature

        return final, rgb_p1, rgb_p2, rgb_p3, rgb_p4, rgb_p5, rgb_p6, th_p1, th_p2, th_p3, th_p4, th_p5, th_p6

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)


def resnet2modal_part6(**kwargs):
    model = ResNet2modal_part6(50, **kwargs)
    pretrained_state_dict = model_zoo.load_url(model_urls['resnet50'])
    now_state_dict = model.state_dict()
    pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in now_state_dict}
    now_state_dict.update(pretrained_state_dict)
    model.load_state_dict(now_state_dict)
    return model


class ResNet3modal_part6(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0):
        super(ResNet3modal_part6, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling
        self.num_features = num_features
        self.num_classes = num_classes

        # Construct base (pretrained) resnet
        if depth not in ResNet3modal_part6.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base1 = ResNet3modal_part6.__factory[depth](pretrained=pretrained)
        self.base2 = ResNet3modal_part6.__factory[depth](pretrained=pretrained)
        self.base3 = ResNet3modal_part6.__factory[depth](pretrained=pretrained)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc_rgb_p1 = self._fc_layer(2048, 128)
        self.fc_rgb_p2 = self._fc_layer(2048, 128)
        self.fc_rgb_p3 = self._fc_layer(2048, 128)
        self.fc_rgb_p4 = self._fc_layer(2048, 128)
        self.fc_rgb_p5 = self._fc_layer(2048, 128)
        self.fc_rgb_p6 = self._fc_layer(2048, 128)
        self.fc_th_p1 = self._fc_layer(2048, 128)
        self.fc_th_p2 = self._fc_layer(2048, 128)
        self.fc_th_p3 = self._fc_layer(2048, 128)
        self.fc_th_p4 = self._fc_layer(2048, 128)
        self.fc_th_p5 = self._fc_layer(2048, 128)
        self.fc_th_p6 = self._fc_layer(2048, 128)
        self.fc_ni_p1 = self._fc_layer(2048, 128)
        self.fc_ni_p2 = self._fc_layer(2048, 128)
        self.fc_ni_p3 = self._fc_layer(2048, 128)
        self.fc_ni_p4 = self._fc_layer(2048, 128)
        self.fc_ni_p5 = self._fc_layer(2048, 128)
        self.fc_ni_p6 = self._fc_layer(2048, 128)

        # identity classification layer
        self.classifier_rgb_p1 = nn.Linear(128, num_classes)
        self.classifier_rgb_p2 = nn.Linear(128, num_classes)
        self.classifier_rgb_p3 = nn.Linear(128, num_classes)
        self.classifier_rgb_p4 = nn.Linear(128, num_classes)
        self.classifier_rgb_p5 = nn.Linear(128, num_classes)
        self.classifier_rgb_p6 = nn.Linear(128, num_classes)
        self.classifier_th_p1 = nn.Linear(128, num_classes)
        self.classifier_th_p2 = nn.Linear(128, num_classes)
        self.classifier_th_p3 = nn.Linear(128, num_classes)
        self.classifier_th_p4 = nn.Linear(128, num_classes)
        self.classifier_th_p5 = nn.Linear(128, num_classes)
        self.classifier_th_p6 = nn.Linear(128, num_classes)
        self.classifier_ni_p1 = nn.Linear(128, num_classes)
        self.classifier_ni_p2 = nn.Linear(128, num_classes)
        self.classifier_ni_p3 = nn.Linear(128, num_classes)
        self.classifier_ni_p4 = nn.Linear(128, num_classes)
        self.classifier_ni_p5 = nn.Linear(128, num_classes)
        self.classifier_ni_p6 = nn.Linear(128, num_classes)
        self.classifier_all = nn.Linear(2304, num_classes)

        if not self.pretrained:
            self.reset_params()

    def _fc_layer(self, in_channel, out_channel):
        layers = []
        layers.append(
            nn.Sequential(
                nn.Linear(in_channel, out_channel),
                nn.BatchNorm1d(out_channel),
                nn.ReLU()
            )
        )
        return nn.Sequential(*layers)

    def forward(self, x1, x2, x3):
        # feature fusion
        for name1, module1 in self.base1._modules.items():
            if name1 == 'avgpool':
                break
            if name1 == 'conv1':
                x1 = module1(x1)
            if name1 == 'layer1':
                x1 = module1(x1)
            if name1 == 'layer2':
                x1 = module1(x1)
            if name1 == 'layer3':
                x1 = module1(x1)
            if name1 == 'layer4':
                x1 = module1(x1)

        for name2, module2 in self.base2._modules.items():
            if name2 == 'avgpool':
                break
            if name2 == 'conv1':
                x2 = module2(x2)
            if name2 == 'layer1':
                x2 = module2(x2)
            if name2 == 'layer2':
                x2 = module2(x2)
            if name2 == 'layer3':
                x2 = module2(x2)
            if name2 == 'layer4':
                x2 = module2(x2)

        for name3, module3 in self.base3._modules.items():
            if name3 == 'avgpool':
                break
            if name3 == 'conv1':
                x3 = module3(x3)
            if name3 == 'layer1':
                x3 = module3(x3)
            if name3 == 'layer2':
                x3 = module3(x3)
            if name3 == 'layer3':
                x3 = module3(x3)
            if name3 == 'layer4':
                x3 = module3(x3)

        rgb_p1, rgb_p2, rgb_p3, rgb_p4, rgb_p5, rgb_p6 = x1.chunk(6, dim=2)
        th_p1, th_p2, th_p3, th_p4, th_p5, th_p6 = x2.chunk(6, dim=2)
        ni_p1, ni_p2, ni_p3, ni_p4, ni_p5, ni_p6 = x3.chunk(6, dim=2)

        # global avgpool layer
        # modal feature
        rgb_p1 = self.global_avgpool(rgb_p1)
        rgb_p2 = self.global_avgpool(rgb_p2)
        rgb_p3 = self.global_avgpool(rgb_p3)
        rgb_p4 = self.global_avgpool(rgb_p4)
        rgb_p5 = self.global_avgpool(rgb_p5)
        rgb_p6 = self.global_avgpool(rgb_p6)
        th_p1 = self.global_avgpool(th_p1)
        th_p2 = self.global_avgpool(th_p2)
        th_p3 = self.global_avgpool(th_p3)
        th_p4 = self.global_avgpool(th_p4)
        th_p5 = self.global_avgpool(th_p5)
        th_p6 = self.global_avgpool(th_p6)
        ni_p1 = self.global_avgpool(ni_p1)
        ni_p2 = self.global_avgpool(ni_p2)
        ni_p3 = self.global_avgpool(ni_p3)
        ni_p4 = self.global_avgpool(ni_p4)
        ni_p5 = self.global_avgpool(ni_p5)
        ni_p6 = self.global_avgpool(ni_p6)

        # FC layer
        batch = x1.shape[0]
        # modal feature
        rgb_p1 = self.fc_rgb_p1(rgb_p1.view(batch, -1))
        rgb_p2 = self.fc_rgb_p2(rgb_p2.view(batch, -1))
        rgb_p3 = self.fc_rgb_p3(rgb_p3.view(batch, -1))
        rgb_p4 = self.fc_rgb_p4(rgb_p4.view(batch, -1))
        rgb_p5 = self.fc_rgb_p5(rgb_p5.view(batch, -1))
        rgb_p6 = self.fc_rgb_p6(rgb_p6.view(batch, -1))
        th_p1 = self.fc_th_p1(th_p1.view(batch, -1))
        th_p2 = self.fc_th_p2(th_p2.view(batch, -1))
        th_p3 = self.fc_th_p3(th_p3.view(batch, -1))
        th_p4 = self.fc_th_p4(th_p4.view(batch, -1))
        th_p5 = self.fc_th_p5(th_p5.view(batch, -1))
        th_p6 = self.fc_th_p6(th_p6.view(batch, -1))
        ni_p1 = self.fc_ni_p1(ni_p1.view(batch, -1))
        ni_p2 = self.fc_ni_p2(ni_p2.view(batch, -1))
        ni_p3 = self.fc_ni_p3(ni_p3.view(batch, -1))
        ni_p4 = self.fc_ni_p4(ni_p4.view(batch, -1))
        ni_p5 = self.fc_ni_p5(ni_p5.view(batch, -1))
        ni_p6 = self.fc_ni_p6(ni_p6.view(batch, -1))
        final = torch.cat([rgb_p1, rgb_p2, rgb_p3, rgb_p4, rgb_p5, rgb_p6, th_p1, th_p2, th_p3, th_p4, th_p5, th_p6,
                           ni_p1, ni_p2, ni_p3, ni_p4, ni_p5, ni_p6], dim=1)

        # identity classification layer
        # modal feature
        final = self.classifier_all(final)
        rgb_p1 = self.classifier_rgb_p1(rgb_p1)
        rgb_p2 = self.classifier_rgb_p2(rgb_p2)
        rgb_p3 = self.classifier_rgb_p3(rgb_p3)
        rgb_p4 = self.classifier_rgb_p4(rgb_p4)
        rgb_p5 = self.classifier_rgb_p5(rgb_p5)
        rgb_p6 = self.classifier_rgb_p6(rgb_p6)
        th_p1 = self.classifier_th_p1(th_p1)
        th_p2 = self.classifier_th_p2(th_p2)
        th_p3 = self.classifier_th_p3(th_p3)
        th_p4 = self.classifier_th_p4(th_p4)
        th_p5 = self.classifier_th_p5(th_p5)
        th_p6 = self.classifier_th_p6(th_p6)
        ni_p1 = self.classifier_ni_p1(ni_p1)
        ni_p2 = self.classifier_ni_p2(ni_p2)
        ni_p3 = self.classifier_ni_p3(ni_p3)
        ni_p4 = self.classifier_ni_p4(ni_p4)
        ni_p5 = self.classifier_ni_p5(ni_p5)
        ni_p6 = self.classifier_ni_p6(ni_p6)

        # fusion feature

        return final, rgb_p1, rgb_p2, rgb_p3, rgb_p4, rgb_p5, rgb_p6, th_p1, th_p2, th_p3, th_p4, th_p5, th_p6, \
            ni_p1, ni_p2, ni_p3, ni_p4, ni_p5, ni_p6

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)


def resnet3modal_part6(**kwargs):
    model = ResNet3modal_part6(50, **kwargs)
    pretrained_state_dict = model_zoo.load_url(model_urls['resnet50'])
    now_state_dict = model.state_dict()
    pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in now_state_dict}
    now_state_dict.update(pretrained_state_dict)
    model.load_state_dict(now_state_dict)
    return model


if __name__ == '__main__':
    model = resnet3modal_PFnet(num_classes=155)