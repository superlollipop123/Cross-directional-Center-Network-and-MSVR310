# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import pdb
import torch
from torch import nn

from .backbones.resnet import ResNet, BasicBlock, Bottleneck 
from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck 
from .backbones.squeezenet import SqueezeNet,Fire
from .backbones.densenet import _DenseLayer, _DenseBlock, _Transition, DenseNet
from .backbones.mobilenet import ConvBNReLU, InvertedResidual, MobileNetV2
from .backbones.inception import Inception3, BasicConv2d

import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.nn.functional as F

from modeling.baseline_zxp import ModalNorm

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice):
        super(Baseline, self).__init__()
        print("basic model: ", model_name)
        self.model_name = model_name
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride, 
                               block=BasicBlock, 
                               layers=[2, 2, 2, 2])
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50':
            self.base  = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck, 
                               layers=[3, 4, 23, 3])
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride, 
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])
            
        elif model_name == 'se_resnet50':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 6, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
            self.modalnorm = ModalNorm(512)
        elif model_name == 'se_resnet101':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 23, 3], 
                              groups=1,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet152':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)  
        elif model_name == 'se_resnext50':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 6, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnext101':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 23, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'senet154':
            self.base = SENet(block=SEBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=64, 
                              reduction=16,
                              dropout_p=0.2, 
                              last_stride=last_stride)
        elif model_name == 'squeezenet':
             self.in_planes = 512
             self.base = SqueezeNet()
        elif model_name == 'densenet':
             self.in_planes = 1024
             self.base = DenseNet()
        elif model_name == 'mobilenet':
             self.in_planes = 1280
             self.base = MobileNetV2()
             self.modalnorm = ModalNorm(160)
            #  model_path = r"C:\Users\zxp\.torch\models\mobilenet_v2-b0353104.pth"
        elif model_name == 'inception':
             self.in_planes = 2048
             self.base = Inception3()
             self.modalnorm = ModalNorm(768)
        else:
            raise NotImplementedError

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')
            
        # self.conv1 = nn.Conv2d(self.in_planes, num_classes, kernel_size=1, bias=False)
        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes    
        self.neck = neck
        self.neck_feat = neck_feat
        

        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x):

        if self.model_name == 'se_resnet50':
            x = self.base.layer0(x)
            x = self.base.layer1(x)
            x = self.base.layer2(x)
            x, _ = self.modalnorm(x)
            x = self.base.layer3(x)
            x = self.base.layer4(x)
            global_feat = self.gap(x)
        elif self.model_name == 'mobilenet':
            layers = self.base.features
            for i in range(len(layers)):
                # print()
                # print(layers[i])
                # print()
                if i == len(layers) - 3:
                    x,_ = self.modalnorm(x)
                x = layers[i](x)
            global_feat = self.gap(x)
            # assert 0
        elif self.model_name == 'inception':
            # N x 3 x 299 x 299
            x = self.base.Conv2d_1a_3x3(x)
            # N x 32 x 149 x 149
            x = self.base.Conv2d_2a_3x3(x)
            # N x 32 x 147 x 147
            x = self.base.Conv2d_2b_3x3(x)
            # N x 64 x 147 x 147
            x = F.max_pool2d(x, kernel_size=3, stride=2)
            # N x 64 x 73 x 73
            x = self.base.Conv2d_3b_1x1(x)
            # N x 80 x 73 x 73
            x = self.base.Conv2d_4a_3x3(x)
            # N x 192 x 71 x 71
            x = F.max_pool2d(x, kernel_size=3, stride=2)
            # N x 192 x 35 x 35
            x = self.base.Mixed_5b(x)
            # N x 256 x 35 x 35
            x = self.base.Mixed_5c(x)
            # N x 288 x 35 x 35
            x = self.base.Mixed_5d(x)
            # N x 288 x 35 x 35
            x = self.base.Mixed_6a(x)
            # N x 768 x 17 x 17
            x, _ = self.modalnorm(x)
            x = self.base.Mixed_6b(x)
            # N x 768 x 17 x 17
            x = self.base.Mixed_6c(x)
            # N x 768 x 17 x 17
            x = self.base.Mixed_6d(x)
            # N x 768 x 17 x 17
            x = self.base.Mixed_6e(x)
            # N x 768 x 17 x 17
            # N x 768 x 17 x 17
            x = self.base.Mixed_7a(x)
            # N x 1280 x 8 x 8
            x = self.base.Mixed_7b(x)
            # N x 2048 x 8 x 8
            x = self.base.Mixed_7c(x)

            global_feat = self.gap(x)
        # global_feat = self.gap(self.base(x))
        global_feat = global_feat.view(global_feat.shape[0], -1)

  
        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)  # normalize for angular softmax
            
        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat  # global feature for triplet loss
        else:
            return global_feat, feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

class MyNet(nn.Module):
    
    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice, branches=3):
        super(MyNet, self).__init__()

        self.branches = branches
        if self.branches == 3:
            self.net_r = Baseline(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.net_n = Baseline(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.net_t = Baseline(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
        elif self.branches == 2:
            self.net_r = Baseline(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.net_n = Baseline(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)

    def forward(self, x):
        
        if self.training:
            score_r, gf_r = self.net_r(x[0])
            score_n, gf_n = self.net_n(x[1])
            if self.branches == 3:
                score_t, gf_t = self.net_t(x[2])
                return [score_r, score_n, score_t], [gf_r, gf_n, gf_t], None, None
            else:
                return [score_r, score_n], [gf_r, gf_n], None, None

        else:
            gf_r, bnf_r = self.net_r(x[0])
            gf_n, bnf_n = self.net_n(x[1])
            if self.branches == 3:
                gf_t, bnf_t = self.net_t(x[2])
                return [gf_r, gf_n, gf_t], None, [bnf_r, bnf_n, bnf_t], None
            elif self.branches == 2:
                return [gf_r, gf_n], None, [bnf_r, bnf_n], None
    
    def load_param(self, trained_path):
        self.net_r.load_param(trained_path)
        self.net_n.load_param(trained_path)
        if self.branches == 3:
            self.net_t.load_param(trained_path)