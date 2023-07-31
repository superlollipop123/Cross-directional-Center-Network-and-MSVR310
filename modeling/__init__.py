# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

# from .baseline1 import Baseline6
from .baseline_zxp import MainNet
from .baseline_zxp2 import MainNet as MainNet_bs
from .osnet import osnet_for_msvr300
from .myosnet import osnet_for_msvr300 as osnet_for_msvr300_2
from .PFNet import resnet3modal_PFnet
from .encoder_decoder import AutoEncoder
from .baseNet import MyNet as BaseNet
from modeling_fastreid.hrcn import MyNet as myHRCN

def build_model(cfg, num_classes):
    # if cfg.MODEL.NAME == 'resnet50':
    #     model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT)
   # model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE)
    # model = Baseline6(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE)
    if cfg.JUST_BASELINE == "no":
        model = MainNet(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE, cfg.MODEL.BRANCHES)
    elif cfg.JUST_BASELINE == "HAMNET":
        print("build HAMNET ...")
        model = MainNet_bs(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE, cfg.MODEL.BRANCHES)
    elif cfg.JUST_BASELINE == 'osnet':
        print('use osnet as baseline !')
        # model = osnet_for_msvr300(num_classes)
        model = osnet_for_msvr300_2(num_classes)
    elif cfg.JUST_BASELINE == 'pfnet':
        print('use PFNet !')
        model = resnet3modal_PFnet(num_classes=num_classes)
    elif cfg.JUST_BASELINE == 'AE':
        print("use autoencoder")
        model = AutoEncoder()
    elif cfg.JUST_BASELINE == 'basenet':
        print("use basenet")
        model = BaseNet(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE, cfg.MODEL.BRANCHES)
    elif cfg.JUST_BASELINE == 'hrcn':
        print("use hrcn !!!")
        model = myHRCN(num_classes)
    else:
        raise Exception("model for JUST_BASELINE is not supported")
   # model = Baseline5(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE)
    return model
