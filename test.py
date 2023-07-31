# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
from os import mkdir

import torch
import pdb
from torch.backends import cudnn

sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.inference import inference
from modeling import build_model
# from outputs.mm_ca.baseline_zxp import MainNet as Test_Model
# from outputs.no_fusing.baseline_zxp import MainNet as Test_Model
# from outputs.attmap_convac.baseline_zxp import MainNet as Test_Model
# from outputs.nonl_convac.baseline_zxp import MainNet as Test_Model
# from outputs.bn_after.baseline_zxp import MainNet as Test_Model
# from outputs.hc_loss_6_nopretrain.baseline_zxp import MainNet as Test_Model
# from outputs.test_new.baseline_zxp import MainNet as Test_Model
# from outputs.bs_cls_hc_mmic.baseline_zxp import MainNet as Test_Model
# from outputs_2.ours.baseline_zxp import MainNet as Test_Model
# from outputs.ALNU_CdC.baseline_zxp import MainNet as Test_Model
from modeling.osnet import osnet_for_msvr300
from modeling.PFNet import resnet3modal_PFnet
from modeling.baseline_zxp2 import MainNet as hamnet
# from modeling.baseline_zxp2 import MainNet as Test_Model
# from outputs.CdC_lam03alpha06_ALNU.baseline_zxp import MainNet as Test_Model
# from outputs.veri_mm_test.baseline_zxp import MainNet as Test_Model
from outputs.veri_mm_gan_mmtri.baseline_zxp import MainNet as Test_Model
from utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument(
        "--config_file", default="softmax_triplet.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    # logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        # os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join([str(i) for i in cfg.GPU])
    cudnn.benchmark = True

    train_loader, val_loader, num_query, num_classes = make_data_loader(cfg)
    model = Test_Model(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE, cfg.MODEL.BRANCHES)
    # model = hamnet(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE, cfg.MODEL.BRANCHES)
    # model = osnet_for_msvr300(num_classes)
    # model = resnet3modal_PFnet(num_classes=num_classes)
    # model = build_model(cfg, num_classes)
    logger.info('load trained parameters from {}'.format(cfg.TEST.WEIGHT))
    model.load_param(cfg.TEST.WEIGHT)
    # pdb.set_trace()
    
    # param_dict = torch.load(cfg.TEST.WEIGHT)
    #     # pdb.set_trace()
    # for i in param_dict:
    #     if 'classifier' in i:
    #         continue
    #     model.state_dict()[i].copy_(param_dict[i])

    inference(cfg, model, val_loader, num_query)
    # inference(cfg, model, train_loader, num_query)


if __name__ == '__main__':
    main()
