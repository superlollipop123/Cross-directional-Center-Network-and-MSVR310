# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
from functools import reduce
import operator
import pdb
import torch
from torch import nn


from modeling.backbones.resnet import ResNet
# from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck 
# from .backbones.squeezenet import SqueezeNet,Fire
# from .backbones.densenet import _DenseLayer, _DenseBlock, _Transition, DenseNet
# from .backbones.mobilenet import ConvBNReLU, InvertedResidual, MobileNetV2
# from .backbones.inception import Inception3, BasicConv2d


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

def conv3x3(in_plane, out_plane, stride=1, bn=True,relu=True):
    layer = []
    if bn:
        bias_flag = False
    else:
        bias_flag = True
    layer.append(nn.Conv2d(in_channels=in_plane, out_channels=out_plane, kernel_size=(3, 3), 
                     stride=stride, padding=1, bias=bias_flag))
    if bn:
        layer.append(nn.BatchNorm2d(out_plane))
    if relu:
        layer.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layer)

def conv1x1(in_plane, out_plane, bn=False, activation=False):
    layer = []
    if bn:
        bias_flag = False
    else:
        bias_flag = True
    layer.append(nn.Conv2d(in_channels=in_plane, out_channels=out_plane, kernel_size=(1, 1), bias=bias_flag))
    if bn:
        layer.append(nn.BatchNorm2d(out_plane))
    if activation == 'relu':
        layer.append(nn.ReLU(inplace=True))
    elif activation == 'sigmoid':
        layer.append(nn.Sigmoid())
    elif activation == 'leakyrelu':
        layer.append(nn.LeakyReLU(inplace=True))
    return nn.Sequential(*layer)

class ChannelAttModule(nn.Module):
    def __init__(self, in_planes, out_plane, reduction=4):
        super(ChannelAttModule, self).__init__()
        self.downsample = nn.Sequential(
        conv3x3(in_plane=in_planes, out_plane=in_planes),
        conv1x1(in_plane=in_planes, out_plane=out_plane),
        conv3x3(in_plane=out_plane, out_plane=out_plane)
        )
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv1x1(in_plane=out_plane, out_plane=out_plane // reduction, activation='relu'),
            conv1x1(in_plane=out_plane // reduction, out_plane=out_plane),
            nn.Sigmoid()
         )

    def forward(self, x):
        feat = self.downsample(x)
        att = self.attention(feat)
        return feat*att + feat

class ASU(nn.Module):
    def __init__(self, in_dim=512):
        super(ASU, self).__init__()
        self.conv1 = nn.Sequential(
            conv3x3(in_plane=in_dim, out_plane=in_dim//4, stride=1, bn=True, relu=True),
            conv3x3(in_plane=in_dim//4, out_plane=in_dim//16, stride=1, bn=True, relu=True)
        )
        self.GAP = nn.AdaptiveAvgPool2d(output_size=1)
        self.GMP = nn.AdaptiveMaxPool2d(output_size=1)
        self.conv_alpha = conv1x1(in_plane=in_dim//8, out_plane=1, bn=False, activation='sigmoid')
        # self.conv_beta = conv1x1(in_plane=in_dim//8, out_plane=1, bn=False, activation='sigmoid')

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([self.GAP(x), self.GMP(x)], dim=1)
        alpha = self.conv_alpha(x)
        # beta = self.conv_beta(x)
        return alpha#, beta
        
class Non_local_att(nn.Module):
    def __init__(self, in_dim, reduce_rate=2) -> None:
        super(Non_local_att, self).__init__()
        self.conv_key = conv1x1(in_plane=in_dim, out_plane=in_dim//reduce_rate)
        self.conv_prod = conv1x1(in_plane=in_dim, out_plane=in_dim//reduce_rate)
        self.conv_value = conv1x1(in_plane=in_dim, out_plane=in_dim) # keep dim
        self.softmax = nn.Softmax(dim=-1)
        self.param = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        B, C, H, W = x.shape

        f_key = self.conv_key(x).view([B, -1, H*W]).permute([0, 2, 1])
        f_prod = self.conv_prod(x).view([B, -1, H*W])
        f_value = self.conv_value(x).view([B, -1, H*W])

        energy = torch.bmm(f_key, f_prod)
        similarity = self.softmax(energy)

        f_value = torch.bmm(f_value, similarity.permute([0, 2, 1])).view([B, -1, H, W])
        final_feat = x + f_value * self.param

        return final_feat

class CrossModalFusing(nn.Module):

    def __init__(self, in_dim1, in_dim2, reduce_rate=4) -> None:
        super(CrossModalFusing, self).__init__()
        self.conv_key = conv1x1(in_plane=in_dim1, out_plane=in_dim1//reduce_rate, activation='relu')
        self.conv_prod = conv1x1(in_plane=in_dim2, out_plane=in_dim1//reduce_rate, activation='relu')
        # self.conv_value = conv1x1(in_plane=in_dim1, out_plane=out_dim)
        # self.softmax = nn.Softmax(dim=-1)
        # self.param = nn.Parameter(torch.zeros(1))

        self.att_conv_1 = conv1x1(in_plane=16*32*2, out_plane=1, activation='sigmoid')
        self.att_conv_2 = conv1x1(in_plane=16*32*2, out_plane=1, activation='sigmoid')
    
    def forward(self, x1, x2):
        B, _, H, W = x1.shape

        f_key = self.conv_key(x1).view([B, -1, H*W]).permute([0, 2, 1])
        f_prod = self.conv_prod(x2).view([B, -1, H*W])
        # f_value = self.conv_value(x1).view([B, -1, H*W])

        energy = torch.bmm(f_key, f_prod)
        # similarity = self.softmax(energy)
        energy_t = energy.permute([0, 2, 1])

        energy = energy.view([B, -1, H, W])
        energy_t = energy_t.view([B, -1, H, W])
        relation = torch.cat([energy, energy_t], dim=1)

        att_map_1 = self.att_conv_1(relation)
        att_map_2 = self.att_conv_2(relation)

        feat_1 = x1 * att_map_1 + x1
        feat_2 = x2 * att_map_2 + x2
        # f_value = torch.bmm(f_value, similarity.permute([0, 2, 1])).view([B, -1, H, W])
        # f_value = f_value * self.param
        
        return feat_1, feat_2, att_map_1, att_map_2

class ModalNorm(nn.Module):
    def __init__(self):
        super(ModalNorm, self).__init__()
        
        # self.gama = nn.Parameter(torch.ones(1))
        # self. beta = nn.Parameter(torch.zeros(1))
        # self.sigmoid = nn.Sigmoid()

        self.conv_gamma = ASU(512)
        self.conv_beta = ASU(512)

    def forward(self, x):
        
        # target_mean = self.sigmoid(self.sigma)
        # var = ((X - mean) ** 2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        cur_mean = x.mean(dim=[1, 2, 3], keepdim=True)
        cur_var = ((x - cur_mean) ** 2).mean(dim=[1, 2, 3], keepdim=True)
        x_hat = (x - cur_mean) / torch.sqrt(cur_var + 1e-6)
        
        gamma = self.conv_gamma(x)
        beta = self.conv_beta(x)

        f = gamma * x_hat + beta
        return f, x

class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice):
        super(Baseline, self).__init__()
        if model_name == 'resnet18':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride, 
                            #    block=BasicBlock, 
                               layers=[2, 2, 2, 2])
        elif model_name == 'resnet50':
            self.base  = ResNet(last_stride=last_stride,
                            #    block=Bottleneck,
                               layers=[3, 4, 6, 3])
        else:
            raise Exception('unsupported model')

        if pretrain_choice == 'imagenet':
            self.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        # self.conv1 = nn.Conv2d(self.in_planes, num_classes, kernel_size=1, bias=False)
        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        # self.scModule = ASU(512)
        self.modalnorm = ModalNorm()

        if self.neck == 'no':
            # pass
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)
    
    def forward_shallow(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.maxpool(x)
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        return x

    def forward(self, x):

        # pdb.set_trace()
        f, _ = self.modalnorm(x)
        # f = x
        f = self.base.layer3(f)
        f = self.base.layer4(f)
        global_feat= self.gap(f) # B, Channel, 1, 1
        # f = self.conv1(f) # B, Classnum, H, W
        # f = self.conv1(self.base(x))
        # global_feat1 = self.gap(self.base(x))  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'bnneck':
            feat = self.bottleneck(global_feat) # normalize for angular softmax
        else:
            feat = global_feat
        score = self.classifier(feat)

        return global_feat, feat, score, 0, x.mean(dim=[2, 3])

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        net_params_keys = self.state_dict().keys()
        for key in net_params_keys:
            if 'num_batches_tracked' in key:
                continue
            if key[5:] not in param_dict:
                continue
            self.state_dict()[key].copy_(param_dict[key[5:]])

class MainNet(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride=1, model_path=None, neck="bnneck", neck_feat="after", model_name="resnet50", pretrain_choice=False, branches=3):
        super(MainNet, self).__init__()
        
        self.branches = branches
        for i in range(branches):
            self.__setattr__('branch_'+str(i), Baseline(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice))
 
    def forward(self, inputs):
    

        # operation for branches
        bra_mid_feats = []
        for i in range(self.branches):
            f = self.__getattr__('branch_'+str(i)).forward_shallow(inputs[i])
            # w = self.ASU(f)
            # f = w * f
            # f = self.__getattr__('nl_att_'+str(i))(f)
            bra_mid_feats.append(f)

        
        # get weightes
        weight_list = []     
        # ds_fused_feat = self.ds_fuse_branch(torch.cat(fused_f_list, dim=1))

        # stage 2---------------------------------------------------------------------------------
        gf_list = [] # saving global features for final representation and loss computing
        bn_f_list = []
        pred_list = []
        mid_list = []
        for i in range(self.branches):
            global_feat, feat, score, w, mid_feat = self.__getattr__('branch_'+str(i)).forward(bra_mid_feats[i]) # stage 2 for branches
            gf_list.append(global_feat)
            bn_f_list.append(feat)
            pred_list.append(score)
            weight_list.append(w)
            mid_list.append(mid_feat)
        # ----------------------------------------------------------------------------------------------------

        if self.training:
            print("only used for feature extraction")
            return
            # return  pred_list, gf_list, weight_list, bn_f_list
            # return  branch_cls_scores, cls_score4, None, fs, gf_list, weight_list
        else:
            # return gf_list, weight_list, bn_f_list
            return gf_list, weight_list, bn_f_list, mid_list
            # return normed_x, ori_x
            
    def load_param(self, trained_path, use_gpu=False):
        # param_dict = torch.load(trained_path)
        if use_gpu:
            param_dict = torch.load(trained_path)
        else:
            param_dict = torch.load(trained_path, map_location="cpu")
        # pdb.set_trace()
        for i in param_dict:
            # if 'classifier' in i:
            #     continue
            self.state_dict()[i].copy_(param_dict[i])