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
import torch.nn as nn
import numpy as np
import pdb
from torch.backends import cudnn
import pickle

sys.path.append('.')
from config import cfg
from data import make_data_loader
from ignite.engine import Engine
from utils.reid_metric import R1_mAP
from utils.mytools import modal_rand_missing
# from engine.save_feat import save_feat
from modeling import build_model
# from outputs.mm_ca.baseline_zxp import MainNet as Test_Model
# from outputs.no_fusing.baseline_zxp import MainNet as Test_Model
# from outputs.attmap_convac.baseline_zxp import MainNet as Test_Model
# from outputs.nonl_convac.baseline_zxp import MainNet as Test_Model
# from outputs.bn_after.baseline_zxp import MainNet as Test_Model
# from outputs.hc_loss_6.baseline_zxp import MainNet as Test_Model
# from outputs.test_new.baseline_zxp import MainNet as Test_Model
# from outputs.test_new_2.baseline_zxp import MainNet as Test_Model
# from modeling.baseline_zxp import MainNet as Test_Model
# from outputs.bs_cls_hc_mmic.baseline_zxp import MainNet as Test_Model
# from modeling.baseline_zxp import MainNet as Test_Model
# from outputs.bs_cls_hc_mmic_SC.baseline_zxp import MainNet as Test_Model
# from outputs.all_rdmiss.baseline_zxp import MainNet as Test_Model
# from outputs_2.baseline_cls_2.baseline_zxp import MainNet as Test_Model
# from outputs_2.baseline_cls.baseline_zxp import MainNet as Test_Model
# from modeling.baseline_zxp import MainNet as Test_Model
from utils.logger import setup_logger

from collections import defaultdict
import importlib

# __MODEL_NAME = "baseline_cls"
__MODEL_NAME = "pfnet_2"

# lib = importlib.import_module(".{}.baseline_zxp".format(__MODEL_NAME), package="outputs")
# Test_Model = lib.MainNet


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline save_feat")
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
    # model = Test_Model(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, False, cfg.MODEL.BRANCHES)
    model = build_model(cfg, num_classes)
    # logger.info('load trained parameters from {}'.format(cfg.TEST.WEIGHT))
    # weights = "outputs/baseline_cls_hc_2/resnet50_model_1200.pth" # bs_cls_hc_sc
    # weights = "outputs/bs_cls_hc_mmic_SC/resnet50_model_800.pth" # bs_cls_hc_mmic_sc
    # weights = "outputs/all_rdmiss/resnet50_model_800.pth" # train for rand missing
    # weights = "outputs/all+rmloss/resnet50_model_800.pth" # train with rand missing loss
    # weights = "outputs/all+rmloss_m005_2/resnet50_model_800.pth" # train with rand missing loss
    # weights = "outputs/all+rmloss_m001/resnet50_model_800.pth" # train with rand missing loss
    # weights = "outputs_2/baseline_cls_2/resnet50_model_1200.pth" # train with rand missing loss
    # weights = "outputs/test3/resnet50_model_800.pth"
    # weights = "outputs/test_alu/resnet50_model_1200.pth"
    weights = "outputs_2/{}/resnet50_model_800.pth".format(__MODEL_NAME)
    # model.load_param(cfg.TEST.WEIGHT)
    # model.load_param(weights)
    param_dict = torch.load(weights)
    for i in param_dict:
        if 'classifier' in i:
            continue
        model.state_dict()[i].copy_(param_dict[i])
    print('paramters loaded:'+weights)
    # pdb.set_trace()

    save_dict = defaultdict(list)

    print("get features for val set")
    save_feat(cfg, model, val_loader, save_dict)
    # print("get features for train set")
    # save_feat(cfg, model, train_loader, save_dict)

    for k in save_dict.keys():
        if 'f' in k:
            print(save_dict[k][0].shape)
            save_dict[k] = np.asarray(torch.cat(save_dict[k], dim=0))
        if 'weights' in k and save_dict['weights'][0] is not None:
            save_dict[k] = np.asarray(torch.cat(save_dict[k], dim=0))
    # output_file = os.path.join('v', 'tSNE', 'bs_hc_mmic_sc_test.pkl')
    output_file = os.path.join('v', 'tSNE', '{}_test.pkl'.format(__MODEL_NAME))
    with open(output_file, 'wb') as f:
        pickle.dump(save_dict, f)
    print('feats saved')

    print('down')

def save_feat(cfg, model, data_loader, save_dict):
    device = cfg.MODEL.DEVICE
    # save_dict = {'feats': [], 'ids': [], 'camids': [], 'sceneids': [], 'paths': []}
    
    evaluator = evaluator_savefeat(model, device=device, data_saver = save_dict)
    evaluator.run(data_loader)
    print('get features')


def evaluator_savefeat(model, device=None, data_saver=None):

    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data1, data2, data3, pids, camids, sceneids, img_path = batch
            # imgs, mask = modal_rand_missing([data1, data2, data3], prob=0.1)
            # data1, data2, data3 = imgs
            data1 = data1.to(device) if torch.cuda.device_count() >= 1 else data1
            data2 = data2.to(device) if torch.cuda.device_count() >= 1 else data2
            data3 = data3.to(device) if torch.cuda.device_count() >= 1 else data3
            # g_feats, bn_f_list = model([data1, data2, data3])
            # g_feats, weights, bn_f_list, mid_list = model([data1, data2, data3])
            # outputs= model([data1, data2, data3])
            outputs= model(data1, data2, data3)
            feat = outputs[-1]
            # import pdb; pdb.set_trace()
            # feat = torch.cat(f_list, dim=1)
            # feat = torch.cat(g_feats, dim=1)
            # data_saver['gf_feat'].append(feat.detach().cpu())
            # data_saver['bn_f_feat'].append(torch.cat(f2_list, dim=1).detach().cpu())
            data_saver['bn_f_feat'].append(feat.detach().cpu())
            # import pdb;pdb.set_trace()
            # data_saver['mid_feat'].append(torch.cat(mid_list, dim=1).detach().cpu())
            # if weights[0] is not None and isinstance(weights[0], torch.Tensor):
            #     data_saver['weights'].append(torch.cat(weights, dim=1).squeeze().detach().cpu())
            # else:
            #     data_saver['weights'].append(None)
            # # data_saver['share_gfs'].append(torch.cat(share_gfs, dim=1).detach().cpu())
            # # data_saver['share_bnfs'].append(torch.cat(share_bnfs, dim=1).detach().cpu())
            data_saver['ids'].extend(np.asarray(pids))
            data_saver['camids'].extend(np.asarray(camids))
            data_saver['sceneids'].extend(np.asarray(sceneids))
            data_saver['paths'].extend(np.asarray(img_path))

            return feat, pids, camids, sceneids, img_path

    engine = Engine(_inference)

    # for name, metric in metrics.items():
    #     metric.attach(engine, name)

    return engine

if __name__ == '__main__':
    main()
    # from v.tSNE.feat_tSNE import tSNE_of_features
    # tSNE_of_features(__MODEL_NAME)
