# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging
import operator
import torch
import torch.nn as nn
import numpy as np
from layers.triplet_loss import TripletLoss
from layers.weighted_dist_cp import WeightedTripletLoss, ParamTripletLoss
import torch.nn.functional as F
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage

from utils.reid_metric import R1_mAP, Weighted_R1_mAP, Param_R1_mAP
from utils.mytools import ResultSaver

from functools import reduce
from utils.mytools import LOSS_ZERO

from layers.triplet_loss import euclidean_dist

from layers import fastreid_circle_loss
from utils.mytools import modal_rand_missing

def normalize(feat, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      feat: pytorch Variable
    Returns:
      feat: pytorch Variable, same shape as input
    """
    feat = 1. * feat / (torch.norm(feat, 2, axis, keepdim=True).expand_as(feat) + 1e-12)
    return feat

def hetero_loss(f_list, label, margin=0.1):
    uni_label_num = len(label.unique())
    chunk_f_list = [f.chunk(uni_label_num, 0) for f in f_list]
    l = len(f_list)

    dist = 0
    dist_func = nn.MSELoss(reduction='sum')
    center_list = []
    for l_idx in range(uni_label_num):
        for f in chunk_f_list:
            center_list.append(torch.mean(f[l_idx], 0))
        for i in range(l):
            for j in range(i+1, l):
                dist += max(0, dist_func(center_list[i], center_list[j]) - margin)
        
    return 2*dist/(l*(l-1))

def MMIC_LOSS(f_list, label, margin):
    uni_label_num = len(label.unique())
    chunk_f_list = [f.chunk(uni_label_num, 0) for f in f_list]
    l = len(f_list)

    dist = 0
    for i in range(l):
        f = [chunk_f_list[n][i] for n in range(l)]
        dist += MultiModalIdConsistLoss(f, margin=margin)
    return dist

def MultiModalIdConsistLoss(f_list, margin=0.1):
    # feat should belong to same class
    N_f = f_list[0].shape[0]
    N_m = len(f_list)

    center_list = []
    for i in range(N_f):
        center_f = 0
        for n in range(N_m):
            center_f += f_list[n][i]
        center_f = center_f/N_m
        center_list.append(center_f)
    
    l = len(center_list)
    assert l >= 2
    dist_func = nn.MSELoss(reduction='sum')
    dist = 0
    for i in range(l):
        for j in range(i+1, l):
            dist += max(0, dist_func(center_list[i], center_list[j]) - margin)
    
    return 2*dist/(l*(l-1))

# from light-reid
def kl_div_loss(logits_s, logits_t, mini=1e-8):
    '''
    :param logits_s: student score
    :param logits_t: teacher score as target
    :param mini: for number stable
    :return:
    '''
    logits_t = logits_t.detach()
    prob1 = F.softmax(logits_s, dim=1)
    prob2 = F.softmax(logits_t, dim=1)
    loss = torch.sum(prob2 * torch.log(mini + prob2 / (prob1 + mini)), 1) + \
            torch.sum(prob1 * torch.log(mini + prob1 / (prob2 + mini)), 1)
    return loss.mean()


def ClassCompactLoss(feat, alpha=1.5, margin=0.2, feat_norm=False):
    # feat should belong to same class
    if feat_norm:
        feat = 1. * feat / (torch.norm(feat, 2, axis=-1, keepdim=True).expand_as(feat) + 1e-12)
    B = feat.shape[0]
    assert B > 1
    center = torch.mean(feat, dim=0)
    dist = torch.pow(feat - center, 2)
    dist = torch.sum(dist)/(B - 1)
    # dist = torch.sum(torch.pow(torch.std(feat, dim=0), 2))
    dist = max(dist - margin, 0)
    loss = torch.exp(alpha*dist) - 1
    return loss, center

def CenterMarginLoss(f_list, label, margin, engine):
    uni_label = label.unique()
    uni_label_num = len(uni_label)
    chunk_f_list = [f.chunk(uni_label_num, 0) for f in f_list]
    l = len(f_list)
    
    all_feat = torch.cat(f_list, dim=0)
    all_label = torch.cat([label]*l)

    final_loss = 0
    for i in range(uni_label_num):
        feat = torch.cat([chunk_f_list[j][i] for j in range(l)], dim=0)
        center_feat = torch.mean(feat, dim=0, keepdim=True)
        dist_mat = euclidean_dist(center_feat, all_feat).squeeze()
        is_pos = all_label.eq(uni_label[i])
        is_neg = 1 - is_pos

        max_pos_dist = torch.max(dist_mat[is_pos])
        min_neg_dist = torch.min(dist_mat[is_neg])

        if engine.state.epoch > 805:
            import pdb; pdb.set_trace()
        
        final_loss += max(0, margin + max_pos_dist - min_neg_dist)
    return final_loss

def rand_missing_loss(feats, margin=0.05, missing_rate=0.3):
    N = feats[0].shape[0]
    modalities = len(feats)
    rand = torch.rand(size=(N, modalities)).cuda()
    miss_mask = rand < missing_rate

    mask_sum = torch.sum(miss_mask, dim=1)
    keep_mask = mask_sum.eq(modalities)
    miss_mask = miss_mask - keep_mask.unsqueeze(dim=1)
    mask = 1 - miss_mask
    mask_sum = torch.sum(mask, dim=1).unsqueeze(dim=1)
    
    feats = torch.stack(feats, dim=1) # (B, 3, 2048)
    keep_feats = feats * mask.unsqueeze(dim=2).float()

    c1 = torch.sum(feats, dim=1) / modalities
    c2 = torch.sum(keep_feats, dim=1) / mask_sum.float()

    loss = nn.MSELoss(reduction='mean')(c1, c2)
    loss = max(0, loss - margin)

    return loss

# def rand_missing_loss(feats, label, margin=0.05, missing_rate=0.3):
#     N = feats[0].shape[0]
#     modalities = len(feats)
#     rand = torch.rand(size=(N, modalities)).cuda()
#     miss_mask = rand < missing_rate

#     mask_sum = torch.sum(miss_mask, dim=1)
#     keep_mask = mask_sum.eq(modalities)
#     miss_mask = miss_mask - keep_mask.unsqueeze(dim=1)
#     mask = 1 - miss_mask
#     mask = mask.unsqueeze(dim=2).float() # (B, 3, 1)
#     feats = torch.stack(feats, dim=1) # (B, 3, 2048)
#     keep_feats = feats * mask

#     dist_func = nn.MSELoss(reduction='mean')
#     id_num = len(label.unique())
#     id_samples = N / id_num
#     id_feats = feats.chunk(id_num, dim=0)
#     id_keep_feats = keep_feats.chunk(id_num, dim=0)
#     id_mask = mask.chunk(id_num, dim=0)
#     loss = 0
#     for i in range(id_num):
#         # missing consistency in samples
#         c = id_feats[i].sum(dim=0) / id_samples
#         c_m = id_keep_feats[i].sum(dim=0) / id_mask[i].sum(dim=0, keepdim=True)
#         loss += max(0, dist_func(c, c_m) - margin)

#         # missing consistency in modal
#         # c = id_feats[i].sum(dim=0) / id_samples
#         # c_m = id_keep_feats[i].sum(dim=0) / id_mask[i].sum(dim=0, keepdim=True)
#         # loss += max(0, dist_func(c, c_m) - margin)

#     return loss

def r_trans(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.item()
    else:
        return tensor

def create_supervised_trainer(model, optimizer, loss_fn,
                              device=None):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    Loss = nn.L1Loss()

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        # optimizer_ptrip.zero_grad()

        img1, img2, img3, target, _, _, _ = batch
        # imgs, mask = modal_rand_missing([img1, img2, img3], prob=0.1)
        # img1, img2, img3 = imgs
        img1 = img1.to(device) if torch.cuda.device_count() >= 1 else img1
        img2 = img2.to(device) if torch.cuda.device_count() >= 1 else img2
        img3 = img3.to(device) if torch.cuda.device_count() >= 1 else img3

        target = target.to(device) if torch.cuda.device_count() >= 1 else target
        B = target.shape[0]

        re_img1, re_img2, re_img3 = model([img1, img2, img3])
        re_loss_1 = Loss(re_img1, img1)
        re_loss_2 = Loss(re_img2, img2)
        re_loss_3 = Loss(re_img3, img3)
        

        
        # make sure LOSS_ZERO is in front of Tensor
        # or you can just remove all LOSS_ZERO() from loss_list to make sure only Tensor contained in it.
        # loss_list.sort(key=lambda feat: not isinstance(feat, LOSS_ZERO))
        total_loss = re_loss_1 + re_loss_2 + re_loss_3
        total_loss.backward()
        optimizer.step()
        
        # optimizer_ptrip.step()
        # compute acc
        
        return r_trans(total_loss), r_trans(re_loss_1), r_trans(re_loss_2), r_trans(re_loss_3)

    # return Engine(_update), p_trip_loss.weights
    return Engine(_update)

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
            re_img1, re_img2, re_img3 = model([data1, data2, data3])
            # result = return_ctler(g_feats, weights)
            # result = return_ctler(g_feats, weights, bn_f_list)
            # if isinstance(result, tuple):
            #     return  (*result, pids, camids, sceneids, img_path)
            # else:
            #     return result, pids, camids, sceneids, img_path

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine

def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        num_query,
        start_epoch
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS
    #pdb.set_trace()

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")
    # trainer, p_weights = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)

    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=10, require_empty=False, save_as_state_dict=False)

    timer = Timer(average=True)
    R_Saver = ResultSaver()

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model.state_dict(),
                                                                     'optimizer': optimizer.state_dict()})
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda feat: feat[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda feat: feat[1]).attach(trainer, 'loss1')
    RunningAverage(output_transform=lambda feat: feat[2]).attach(trainer, 'loss2')
    RunningAverage(output_transform=lambda feat: feat[3]).attach(trainer, 'loss3')

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1

        if iter % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f},  Base Lr: {:.2e}"
                        .format(engine.state.epoch, iter, len(train_loader),
                                engine.state.metrics['avg_loss'],
                                scheduler.get_lr()[0]))

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        g = lambda name: engine.state.metrics[name]
        logger.info('Loss=> loss1:{:.3f} loss2:{:.3f} loss3:{:.3f}'.format(g('loss1'), g('loss2'), g('loss3')))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:

            def __print_info(info, evaluator, metric_name):
                logger.info(' ')
                logger.info(info)
                evaluator.run(val_loader)
                cmc, mAP = evaluator.state.metrics[metric_name]
                logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                R_Saver.add(info, (engine.state.epoch, mAP, cmc[0], cmc[4], cmc[9]))
            

            # __print_info('05. all bn feat concat', evaluator_bnf, 'r1_mAP')
            # __print_info('06. bn rgb feat', evaluator_bn_rgb, 'r1_mAP')
            # __print_info('07. bn nir feat', evaluator_bn_ni, 'r1_mAP')
            # __print_info('08. bn tir feat', evaluator_bn_t, 'r1_mAP')

            R_Saver.saveResults(cfg.OUTPUT_DIR)

    trainer.run(train_loader, max_epochs=epochs)
    # R_Saver.saveAsCSV(cfg.OUTPUT_DIR)