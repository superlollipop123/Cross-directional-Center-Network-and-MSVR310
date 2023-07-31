# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import logging

import torch
import torch.nn as nn
from ignite.engine import Engine

import numpy as np

from utils.reid_metric import R1_mAP, Weighted_R1_mAP, Param_R1_mAP, R1_mAP_reranking

# 大概流程跟trainer里面差不多，就不单独写啦

weight_list = [[], [], []]
mean_list = [[], [], []]
std_list = [[], [], []]
info_list = [[], [], []]

MEAN=[0.485, 0.456, 0.406]
STD=[0.229, 0.224, 0.225]
def recover_img(feat):  # feat [3, H, W]
    img = feat.permute([1, 2, 0]).numpy()
    # pmin = np.min(img)
    # pmax = np.max(img)
    # img = (img - pmin)/(pmax - pmin + 0.0001)
    for i in range(3):
        img[:, :, i] = img[:, :, i] * STD[i]
        img[:, :, i] = img[:, :, i] + MEAN[i]
    img = (img * 255).astype(np.uint8)
    img = np.clip(img, 0, 255)

    return img[:,:,::-1] # BGR

def get_brightness(feat):
    img = recover_img(img)
    # 0.3 R + 0.6 G + 0.1 B
    pixel_brightness = 0.1 * img[:, :, 0] + 0.6 * img[:, :, 1] + 0.3 * img[:, :, 2]
    h, w = pixel_brightness.shape
    return np.sum(pixel_brightness)/(h*w)

def getImgEntropy(feat):
    img = recover_img(feat)
    # convert to gray image
    img = np.sum(img, axis=-1)/3
    img = img.astype(np.uint8)
    img = np.clip(img, 0, 255)

    h, w = img.shape
    f = [0 for _ in range(256)]
    for i in range(h):
        for j in range(w):
            val = img[i][j]
            f[val] = float(f[val] + 1)
    res = 0
    for i in range(256):
        f[i] = float(f[i]/(h*w))
        if f[i] > 0:
            res -= f[i] * np.log2(f[i])
    return res
    

def create_evaluator(model, metrics, return_ctler, device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
        return_ctler: function to decide content returned
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data1, data2, data3, pids, camids, sceneids, img_path = batch
            data1 = data1.to(device) if torch.cuda.device_count() >= 1 else data1
            data2 = data2.to(device) if torch.cuda.device_count() >= 1 else data2
            data3 = data3.to(device) if torch.cuda.device_count() >= 1 else data3
            # gf_list, weight_list, bn_f_list, fc_list, sf_list, sbnf_list = model([data1, data2, data3])
            # g_feats, weights, bn_f_list, _ = model([data1, data2, data3])
            # final_f, gf_l, bf_l = model([data1, data2, data3])
            g_feats, weights, bn_f_list, _ = model([data1, data2, data3])
            # feats = model(data1, data2, data3)[-1]
            # _, g_feats, _ = model([data1, data2, data3])

            # result = return_ctler(g_feats, weights)
            result = return_ctler(g_feats, weights, bn_f_list)
            # result = return_ctler(bf_l, final_f, gf_l)
            # result = feat
            if isinstance(result, tuple):
                return  (*result, pids, camids, sceneids, img_path)
            else:
                return result, pids, camids, sceneids, img_path

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine

def inference(
        cfg,
        model,
        val_loader,
        num_query
):
    device = cfg.MODEL.DEVICE

    logger = logging.getLogger("reid_baseline.inference")
    logger.info("Enter inferencing")
    
    # r1_map_metric = {'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}
    # w_r1_map_metric = {'w_r1_mAP': Weighted_R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}
    metrics_r1map = {'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}
    metrics_wr1map = {'w_r1_mAP': Weighted_R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}
    metrics_r1map_rerank = {'rerank_r1_mAP': R1_mAP_reranking(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}
    # metrics_pr1map = {'p_r1_mAP': Param_R1_mAP(num_query, max_rank=50, weight_param=p_weights, feat_norm=cfg.TEST.FEAT_NORM)}

    evaluator_gf = create_evaluator(model, metrics=metrics_r1map, device=device, \
        return_ctler=lambda  g_fs, weights, bn_f_list: torch.cat(g_fs, dim=1))
    evaluator_gf_re = create_evaluator(model, metrics=metrics_r1map_rerank, device=device, \
        return_ctler=lambda  g_fs, weights, bn_f_list: torch.cat(g_fs, dim=1))
    # evaluator_gf_nofuse = create_evaluator(model, metrics=metrics_r1map, device=device, \
    #     return_ctler=lambda  g_fs, weights, bn_f_list: torch.cat(g_fs[:3], dim=1))
    # w_evaluator_gf = create_evaluator(model, metrics=metrics_wr1map, device=device, \
    #     return_ctler=lambda  g_fs, weights, bn_f_list: (g_fs, weights))
    # w_evaluator_gf_nofuse = create_evaluator(model, metrics=metrics_wr1map, device=device, \
    #     return_ctler=lambda  g_fs, weights, bn_f_list: (g_fs[:3], weights[:3]))
    evaluator_bnf = create_evaluator(model, metrics=metrics_r1map, device=device, \
        return_ctler=lambda  g_fs, weights, bn_f_list: torch.cat(bn_f_list, dim=1))
    evaluator_bnf_re = create_evaluator(model, metrics=metrics_r1map_rerank, device=device, \
        return_ctler=lambda  g_fs, weights, bn_f_list: torch.cat(bn_f_list, dim=1))
    # evaluator_bnf_nofuse = create_evaluator(model, metrics=metrics_r1map, device=device, \
    #     return_ctler=lambda  g_fs, weights, bn_f_list: torch.cat(bn_f_list[:3], dim=1))
    # w_evaluator_bnf = create_evaluator(model, metrics=metrics_wr1map, device=device, \
    #     return_ctler=lambda  g_fs, weights, bn_f_list: (bn_f_list, weights))
    # w_evaluator_bnf_nofuse = create_evaluator(model, metrics=metrics_wr1map, device=device, \
    #     return_ctler=lambda  g_fs, weights, bn_f_list: (bn_f_list[:3], weights[:3]))
    evaluator_rgb = create_evaluator(model, metrics=metrics_r1map, device=device, \
        return_ctler=lambda  g_fs, weights, bn_f_list: g_fs[0])
    evaluator_ni = create_evaluator(model, metrics=metrics_r1map, device=device, \
        return_ctler=lambda  g_fs, weights, bn_f_list: g_fs[1])
    evaluator_t = create_evaluator(model, metrics=metrics_r1map, device=device, \
        return_ctler=lambda  g_fs, weights, bn_f_list: g_fs[2])
    evaluator_bn_rgb = create_evaluator(model, metrics=metrics_r1map, device=device, \
        return_ctler=lambda  g_fs, weights, bn_f_list: bn_f_list[0])
    evaluator_bn_ni = create_evaluator(model, metrics=metrics_r1map, device=device, \
        return_ctler=lambda  g_fs, weights, bn_f_list: bn_f_list[1])
    evaluator_bn_t = create_evaluator(model, metrics=metrics_r1map, device=device, \
        return_ctler=lambda  g_fs, weights, bn_f_list: bn_f_list[2])
    evaluator_bn_r_n = create_evaluator(model, metrics=metrics_r1map, device=device, \
        return_ctler=lambda  g_fs, weights, bn_f_list: torch.cat([bn_f_list[0], bn_f_list[1]], dim=1))
    evaluator_bn_r_t = create_evaluator(model, metrics=metrics_r1map, device=device, \
        return_ctler=lambda  g_fs, weights, bn_f_list: torch.cat([bn_f_list[0], bn_f_list[2]], dim=1))
    evaluator_bn_n_t = create_evaluator(model, metrics=metrics_r1map, device=device, \
        return_ctler=lambda  g_fs, weights, bn_f_list: torch.cat([bn_f_list[1], bn_f_list[2]], dim=1))
    # p_evaluator_gf = create_evaluator(model, metrics=metrics_pr1map, device=device, \
    #     return_ctler=lambda  g_fs, weights, bn_f_list: g_fs)
    # p_evaluator_bnf = create_evaluator(model, metrics=metrics_pr1map, device=device, \
    #     return_ctler=lambda  g_fs, weights, bn_f_list: bn_f_list)

    # if cfg.TEST.RE_RANKING == 'no':
    #     print("Create evaluator")
    #     evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)},
    #                                           )
    # elif cfg.TEST.RE_RANKING == 'yes':
    #     print("Create evaluator for reranking")
    #     evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP_reranking(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)},
    #                                           )
    # else:
    #     print("Unsupported re_ranking config. Only support for no or yes, but got {}.".format(cfg.TEST.RE_RANKING))

    def __print_info(info, evaluator, metric_name):
            # logger.info(' ')
            # logger.info(info)
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics[metric_name]
            # logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            # logger.info("mAP: {:.1%}".format(mAP))
            # for r in [1, 5, 10]:
            #     logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            print('info:{:<30} mAP: {:.1%} r-1: {:.1%} r-5: {:.1%} r-10: {:.1%}'.format(info, mAP, cmc[0], cmc[4], cmc[9]))
            
    # __print_info('0. gf concat', evaluator_gf, 'r1_mAP')
    __print_info('0. gf concat, rerank', evaluator_gf_re, 'rerank_r1_mAP')
    # __print_info('1. bnf concat', evaluator_bnf, 'r1_mAP')
    __print_info('1. bnf concat, rerank', evaluator_bnf_re, 'rerank_r1_mAP')
    # __print_info('2. rgb', evaluator_bn_rgb, 'r1_mAP')
    # __print_info('3. nir', evaluator_bn_ni, 'r1_mAP')
    # __print_info('4. tir', evaluator_bn_t, 'r1_mAP')
    # __print_info('5. r+n', evaluator_bn_r_n, 'r1_mAP')
    # __print_info('6. r+t', evaluator_bn_r_t, 'r1_mAP')
    # __print_info('7. n+t', evaluator_bn_n_t, 'r1_mAP')


    # for i in range(3):
    #     w_arr = np.array(weight_list[i])
    #     mean_arr = np.array(mean_list[i])
    #     std_arr = np.array(std_list[i])

    #     def scale_norm(x):
    #         x_min = np.min(x)
    #         x_max = np.max(x)
    #         return (x - x_min) / (x_max - x_min + 1e-12)

    #     w_arr = scale_norm(w_arr)
    #     mean_arr = scale_norm(mean_arr)
    #     std_arr = scale_norm(std_arr)
        
    #     print(np.cov(w_arr, mean_arr))
    #     print(np.cov(w_arr, std_arr))
    #     print('----------'*3)
    # import pickle
    # with open('train_info.pkl', 'wb') as f:
    #     pickle.dump(info_list, f)