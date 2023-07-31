# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
import torch
from torch import rand
from torch.backends import cudnn
import shutil


sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.trainer import do_train, do_train_with_center
# from engine.trainer_osnet import do_train as do_train_with_osnet
from engine.trainer_bs import do_train as do_train_with_hamnet
from engine.trainer_pfnet import do_train_in_pfnet
from engine.trainer_hrcn import do_train_with_hrcn
from engine.trainer_ae import do_train as do_train_with_AE
from modeling import build_model
from layers import make_loss, make_loss_with_center
from solver import make_optimizer, make_optimizer_with_center, WarmupMultiStepLR

from utils.logger import setup_logger

# set random seed
import random
import numpy as np


def seed_torch(seed=42):
    # https://blog.csdn.net/qq_41645987/article/details/107592810
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # 为当前CPU 设置随机种子
    torch.cuda.manual_seed(seed) # 为当前的GPU 设置随机种子
    torch.cuda.manual_seed_all(seed) # 当使用多块GPU 时，均设置随机种子
    torch.backends.cudnn.deterministic = True # 设置每次返回的卷积算法是一致的
    torch.backends.cudnn.benchmark = False  # cuDNN使用的非确定性算法自动寻找最适合当前配置的高效算法，设置为False 则每次的算法一致
    torch.backends.cudnn.enabled = True # pytorch 使用CUDANN 加速，即使用GPU加速

def train(cfg):
    # prepare dataset
    train_loader, val_loader, num_query, num_classes = make_data_loader(cfg)
    #pdb.set_trace()
    # prepare model
    model = build_model(cfg, num_classes)
    #pdb.set_trace()
    if cfg.MODEL.IF_WITH_CENTER == 'no':
        print('Train without center loss, the loss type is', cfg.MODEL.METRIC_LOSS_TYPE)
        optimizer = make_optimizer(cfg, model)
        # scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
        #                               cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)

        loss_func = make_loss(cfg, num_classes)     # modified by gu
        #pdb.set_trace()
        # Add for using self trained model by cfg.NAME in cfg.OUTPUTS # by zhu
        if cfg.MODEL.PRETRAIN_CHOICE == 'self':
            output_path = os.path.join(cfg.OUTPUT_ROOT, cfg.NAME)
            files = os.listdir(output_path)
            files = filter(lambda f: f.endswith("pth"), files)
            start_epoch = max([int(f.split(".")[0].split("_")[-1]) for f in files])
            print('Start epoch:', start_epoch)
            path_to_model = os.path.join(output_path, "resnet50_model_{}.pth".format(start_epoch))
            path_to_optimizer = os.path.join(output_path, "resnet50_optimizer_{}.pth".format(start_epoch))
            print('Path to the checkpoint of optimizer:', path_to_optimizer)
            model.load_state_dict(torch.load(path_to_model))
            optimizer.load_state_dict(torch.load(path_to_optimizer))
            scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                          cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD, start_epoch)
        elif cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
            start_epoch = 0
            scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                          cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
        else:
            print('Only support pretrain_choice for imagenet and self, but got {}'.format(cfg.MODEL.PRETRAIN_CHOICE))

        arguments = {}

        # 用just baseline指定想用的模型和trainer
        if cfg.JUST_BASELINE in ['no', 'osnet', 'basenet']:
            do_train(
                cfg,
                model,
                train_loader,
                val_loader,
                optimizer,
                scheduler,      # modify for using self trained model
                loss_func,
                num_query,
                start_epoch     # add for using self trained model
            )
        elif cfg.JUST_BASELINE == 'HAMNET':
            print("using HAMNET")
            do_train_with_hamnet(
                cfg,
                model,
                train_loader,
                val_loader,
                optimizer,
                scheduler,      # modify for using self trained model
                loss_func,
                num_query,
                start_epoch     # add for using self trained model
            )
        elif cfg.JUST_BASELINE == 'pfnet':
            print("using pfnet")
            do_train_in_pfnet(
                cfg,
                model,
                train_loader,
                val_loader,
                optimizer,
                scheduler,      # modify for using self trained model
                loss_func,
                num_query,
                start_epoch     # add for using self trained model
            )
        elif cfg.JUST_BASELINE == 'AE':
            print("train an encoder-decoder")
            do_train_with_AE(
                cfg,
                model,
                train_loader,
                val_loader,
                optimizer,
                scheduler,      # modify for using self trained model
                loss_func,
                num_query,
                start_epoch     # add for using self trained model
            )
        elif cfg.JUST_BASELINE == 'hrcn':
            print("train HRCN")
            do_train_with_hrcn(
                cfg,
                model,
                train_loader,
                val_loader,
                optimizer,
                scheduler,      # modify for using self trained model
                loss_func,
                num_query,
                start_epoch     # add for using self trained model
            )
        else:
            raise NotImplementedError
        #optimizer.swap_swa_sgd()
    elif cfg.MODEL.IF_WITH_CENTER == 'yes': # 我的实验里面没用到，这里指的是额外使用center loss
        print('Train with center loss, the loss type is', cfg.MODEL.METRIC_LOSS_TYPE)
        loss_func, center_criterion = make_loss_with_center(cfg, num_classes)  # modified by gu
        optimizer, optimizer_center = make_optimizer_with_center(cfg, model, center_criterion)
        # scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
        #                               cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)

        arguments = {}

        # Add for using self trained model
        if cfg.MODEL.PRETRAIN_CHOICE == 'self':
            start_epoch = eval(cfg.MODEL.PRETRAIN_PATH.split('/')[-1].split('.')[0].split('_')[-1])
            print('Start epoch:', start_epoch)
            path_to_optimizer = cfg.MODEL.PRETRAIN_PATH.replace('model', 'optimizer')
            print('Path to the checkpoint of optimizer:', path_to_optimizer)
            path_to_optimizer_center = cfg.MODEL.PRETRAIN_PATH.replace('model', 'optimizer_center')
            print('Path to the checkpoint of optimizer_center:', path_to_optimizer_center)
            model.load_state_dict(torch.load(cfg.MODEL.PRETRAIN_PATH))
            optimizer.load_state_dict(torch.load(path_to_optimizer))
            optimizer_center.load_state_dict(torch.load(path_to_optimizer_center))
            scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                          cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD, start_epoch)
        elif cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
            start_epoch = 0
            scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                          cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
        else:
            print('Only support pretrain_choice for imagenet and self, but got {}'.format(cfg.MODEL.PRETRAIN_CHOICE))

        do_train_with_center(
            cfg,
            model,
            center_criterion,
            train_loader,
            val_loader,
            optimizer,
            optimizer_center,
            scheduler,      # modify for using self trained model
            loss_func,
            num_query,
            start_epoch     # add for using self trained model
        )
    else:
        print("Unsupported value for cfg.MODEL.IF_WITH_CENTER {}, only support yes or no!\n".format(cfg.MODEL.IF_WITH_CENTER))


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
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

    output_dir = os.path.join(cfg.OUTPUT_ROOT, cfg.NAME)
    cfg.OUTPUT_DIR = output_dir
    cfg.freeze()
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    # save model files
    for f in cfg.SAVE_LIST:
        file_save_path = os.path.join(output_dir, f.split('/')[-1])
        if os.path.exists(file_save_path):
            os.chmod(file_save_path, 0o700)
        shutil.copyfile(f, file_save_path)
        os.chmod(file_save_path, 0o400) # read only
        logger.info('saving file: {}'.format(f))

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    #指定gpu
    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join([str(i) for i in cfg.GPU])
    # cudnn.benchmark = True
    seed_torch()
    train(cfg)


if __name__ == '__main__':
    main()
