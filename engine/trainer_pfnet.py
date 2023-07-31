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
                dist += max(margin, dist_func(center_list[i], center_list[j]))
        
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
            dist += max(margin, dist_func(center_list[i], center_list[j]))
    
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

    trip_loss = TripletLoss(margin=0.3)
    # w_trip_loss = WeightedTripletLoss(margin=0.3)
    # circle_loss = CircleLoss(m=0.25, gamma=256)
    # p_trip_loss = ParamTripletLoss(3, 0.3)
    # optimizer_ptrip = torch.optim.SGD(p_trip_loss.parameters(), lr=0.5)
    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        # optimizer_ptrip.zero_grad()

        img1, img2, img3, target, _, _, _ = batch
        img1 = img1.to(device) if torch.cuda.device_count() >= 1 else img1
        img2 = img2.to(device) if torch.cuda.device_count() >= 1 else img2
        img3 = img3.to(device) if torch.cuda.device_count() >= 1 else img3

        target = target.to(device) if torch.cuda.device_count() >= 1 else target
        B = target.shape[0]
        outputs  = model(img1, img2, img3)
        preds = outputs[:-1]
        feat = outputs[-1]
        # import pdb; pdb.set_trace()
        cls_loss = 0
        for pred in preds:
            cls_loss += F.cross_entropy(pred, target)


        score = sum(preds)

        # cls_loss_0 = F.cross_entropy(pred_list[0], target) #+ trip_loss(gf_list[0], target)[0]
        # cls_loss_1 = F.cross_entropy(pred_list[1], target) #+ trip_loss(gf_list[1], target)[0]
        # cls_loss_2 = F.cross_entropy(pred_list[2], target) #+ trip_loss(gf_list[2], target)[0]
        # cls_loss_3 = F.cross_entropy(pred_list[3], target) #+ trip_loss(gf_list[3], target)[0]
        cls_loss_0 = 0
        cls_loss_1 = 0
        cls_loss_2 = 0
        cls_loss_3 = cls_loss

        # loss_hsc = 0.001*(B-reduce(operator.mul, [normalize(s) for s in pred_list], 1).sum()) # aaai 2020
        loss_hsc = 0

        # w_loss = w_trip_loss(gf_list, weight_list, target)[0]
        # w_loss = trip_loss(torch.cat(gf_list, dim=1), target)[0]
        w_loss = trip_loss(feat, target)[0]
        # w_loss = 0
        # w_loss = trip_loss(gf_list[0], target)[0] + trip_loss(gf_list[1], target)[0] + trip_loss(gf_list[2], target)[0]

        # p_loss = p_trip_loss(gf_list, target)[0]
        p_loss = 0

        # hc_loss = 0.1 * hetero_loss(gf_list, target, 0.0)
        # hc_loss = 0.1*hetero_loss(gf_list, target, 0.1)
        hc_loss = 0

        # mmic_loss = 0.1 * MMIC_LOSS(gf_list[0:3], target, 0.0)
        mmic_loss = 0

        # cm_loss = CenterMarginLoss(gf_list, target, 0.2, engine)
        cm_loss = 0

        # mutual learning
        # ml_loss  = kl_div_loss(pred_list[0], pred_list[1]) + kl_div_loss(pred_list[0], pred_list[2])
        # ml_loss += kl_div_loss(pred_list[1], pred_list[0]) + kl_div_loss(pred_list[1], pred_list[2])
        # ml_loss += kl_div_loss(pred_list[2], pred_list[0]) + kl_div_loss(pred_list[2], pred_list[1])
        # ml_loss = 10 * ml_loss/3
        ml_loss = 0

        # circle loss
        # cir_rgb = fastreid_circle_loss(gf_list[0], target, 0.25, 256)
        # cir_ir = fastreid_circle_loss(gf_list[1], target, 0.25, 256)
        # cir_t = fastreid_circle_loss(gf_list[2], target, 0.25, 256)
        # cir_loss = cir_rgb + cir_ir + cir_t
        cir_loss = 0
        # print(cir_loss)
        # loss_list = [loss_msid, loss_hsc, loss_caid, loss_fb, w_loss]
        loss_list = [cls_loss_0, cls_loss_1, cls_loss_2, cls_loss_3, loss_hsc, hc_loss, w_loss, mmic_loss, p_loss, ml_loss, cm_loss, cir_loss]
        # make sure LOSS_ZERO is in front of Tensor
        # or you can just remove all LOSS_ZERO() from loss_list to make sure only Tensor contained in it.
        # loss_list.sort(key=lambda feat: not isinstance(feat, LOSS_ZERO))
        total_loss = sum(loss_list)
        total_loss.backward()
        optimizer.step()
        
        # optimizer_ptrip.step()
        # compute acc
        get_acc = lambda score, target: (score.max(1)[1] == target).float().mean()
        acc = get_acc(score, target)
        # branch_acc = [get_acc(b_score, target) for b_score in pred_list]
        branch_acc = [0, 0, 0, acc]
        return r_trans(total_loss), r_trans(acc), r_trans(cls_loss_0), r_trans(loss_hsc), r_trans(cls_loss_1), r_trans(cls_loss_2), r_trans(cls_loss_3), r_trans(w_loss), r_trans(hc_loss),\
               branch_acc[0], branch_acc[1], branch_acc[2], branch_acc[3], r_trans(mmic_loss), r_trans(p_loss), r_trans(ml_loss), r_trans(cm_loss), r_trans(cir_loss)

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
            outputs = model(data1, data2, data3)
            # result = return_ctler(g_feats, weights)
            result = outputs[-1]
            if isinstance(result, tuple):
                return  (*result, pids, camids, sceneids, img_path)
            else:
                return result, pids, camids, sceneids, img_path

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine

def do_train_in_pfnet(
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
    logger.info('trainer for pfnet')
    # trainer, p_weights = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)

    metrics_r1map = {'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}
    metrics_wr1map = {'w_r1_mAP': Weighted_R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}
    # metrics_pr1map = {'p_r1_mAP': Param_R1_mAP(num_query, max_rank=50, weight_param=p_weights, feat_norm=cfg.TEST.FEAT_NORM)}

    evaluator_gf = create_evaluator(model, metrics=metrics_r1map, device=device, \
        return_ctler=lambda  g_fs, weights, bn_f_list: torch.cat(g_fs, dim=1))
    evaluator_gf_nofuse = create_evaluator(model, metrics=metrics_r1map, device=device, \
        return_ctler=lambda  g_fs, weights, bn_f_list: torch.cat(g_fs[:3], dim=1))
    w_evaluator_gf = create_evaluator(model, metrics=metrics_wr1map, device=device, \
        return_ctler=lambda  g_fs, weights, bn_f_list: (g_fs, weights))
    w_evaluator_gf_nofuse = create_evaluator(model, metrics=metrics_wr1map, device=device, \
        return_ctler=lambda  g_fs, weights, bn_f_list: (g_fs[:3], weights[:3]))
    evaluator_bnf = create_evaluator(model, metrics=metrics_r1map, device=device, \
        return_ctler=lambda  g_fs, weights, bn_f_list: torch.cat(bn_f_list, dim=1))
    evaluator_bnf_nofuse = create_evaluator(model, metrics=metrics_r1map, device=device, \
        return_ctler=lambda  g_fs, weights, bn_f_list: torch.cat(bn_f_list[:3], dim=1))
    w_evaluator_bnf = create_evaluator(model, metrics=metrics_wr1map, device=device, \
        return_ctler=lambda  g_fs, weights, bn_f_list: (bn_f_list, weights))
    w_evaluator_bnf_nofuse = create_evaluator(model, metrics=metrics_wr1map, device=device, \
        return_ctler=lambda  g_fs, weights, bn_f_list: (bn_f_list[:3], weights[:3]))
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
    # p_evaluator_gf = create_evaluator(model, metrics=metrics_pr1map, device=device, \
    #     return_ctler=lambda  g_fs, weights, bn_f_list: g_fs)
    # p_evaluator_bnf = create_evaluator(model, metrics=metrics_pr1map, device=device, \
    #     return_ctler=lambda  g_fs, weights, bn_f_list: bn_f_list)
    evaluator_bn_rn = create_evaluator(model, metrics=metrics_r1map, device=device, \
        return_ctler=lambda  g_fs, weights, bn_f_list: torch([bn_f_list[0], bn_f_list[1]], dim=1))
    evaluator_bn_rt = create_evaluator(model, metrics=metrics_r1map, device=device, \
        return_ctler=lambda  g_fs, weights, bn_f_list: torch([bn_f_list[0], bn_f_list[2]], dim=1))
    evaluator_bn_nt = create_evaluator(model, metrics=metrics_r1map, device=device, \
        return_ctler=lambda  g_fs, weights, bn_f_list: torch([bn_f_list[1], bn_f_list[2]], dim=1))

    
  
    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=10, require_empty=False, save_as_state_dict=False)

    timer = Timer(average=True)
    R_Saver = ResultSaver()

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model.state_dict(),
                                                                     'optimizer': optimizer.state_dict()})
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda feat: feat[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda feat: feat[1]).attach(trainer, 'avg_acc')
    RunningAverage(output_transform=lambda feat: feat[2]).attach(trainer, 'cls_0')
    RunningAverage(output_transform=lambda feat: feat[3]).attach(trainer, 'loss_hsc')
    RunningAverage(output_transform=lambda feat: feat[4]).attach(trainer, 'cls_1')
    RunningAverage(output_transform=lambda feat: feat[5]).attach(trainer, 'cls_2')
    RunningAverage(output_transform=lambda feat: feat[6]).attach(trainer, 'cls_3')
    RunningAverage(output_transform=lambda feat: feat[7]).attach(trainer, 'loss_t')
    RunningAverage(output_transform=lambda feat: feat[8]).attach(trainer, 'loss_hc')
    RunningAverage(output_transform=lambda feat: feat[9]).attach(trainer, 'acc_0')
    RunningAverage(output_transform=lambda feat: feat[10]).attach(trainer, 'acc_1')
    RunningAverage(output_transform=lambda feat: feat[11]).attach(trainer, 'acc_2')
    RunningAverage(output_transform=lambda feat: feat[12]).attach(trainer, 'acc_fuse')
    RunningAverage(output_transform=lambda feat: feat[13]).attach(trainer, 'mmic_loss')
    RunningAverage(output_transform=lambda feat: feat[14]).attach(trainer, 'ptrip_loss')
    RunningAverage(output_transform=lambda feat: feat[15]).attach(trainer, 'ml_loss')
    RunningAverage(output_transform=lambda feat: feat[16]).attach(trainer, 'cm_loss')
    RunningAverage(output_transform=lambda feat: feat[17]).attach(trainer, 'cir_loss')

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
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, iter, len(train_loader),
                                engine.state.metrics['avg_loss'], engine.state.metrics['avg_acc'],
                                scheduler.get_lr()[0]))

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        g = lambda name: engine.state.metrics[name]
        logger.info('Loss=> cls_0:{:.3f} cls_1:{:.3f} cls_2:{:.3f} cls_3:{:.3f} hsc:{:.3f} w:{:.3f} hc:{:.3f} mmic:{:.3f}, pw:{:.3f}, ml:{:.3f}, cm:{:.3f}, cir:{:.3f}'.format(g('cls_0'), g('cls_1'), g('cls_2'), \
            g('cls_3'), g('loss_hsc'), g('loss_t'), g('loss_hc'), g('mmic_loss'), g('ptrip_loss'), g('ml_loss'), g('cm_loss'), g('cir_loss')))
        logger.info('Acc=> rgb:{:.3f} nir:{:.3f} tir:{:.3f} fuse:{:.3f} '.format(g('acc_0'), g('acc_1'), g('acc_2'), g('acc_fuse')))
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
            
            # __print_info('1. using final feature for test', evaluator, 'r1_mAP')
            # __print_info('1. all global feat concat', evaluator_gf, 'r1_mAP')
            # __print_info('2. rgb feat', evaluator_rgb, 'r1_mAP')
            # __print_info('3. nir feat', evaluator_ni, 'r1_mAP')
            # __print_info('4. tir feat', evaluator_t, 'r1_mAP')
            # # __print_info('2. [1] - fusing feat', evaluator_gf_nofuse, 'r1_mAP')
            # __print_info('3. [1] + weighted metric', w_evaluator_gf, 'w_r1_mAP')
            # # __print_info('4. [2] + weighted metric', w_evaluator_gf_nofuse, 'w_r1_mAP')
            __print_info('5. all bn feat concat', evaluator_bnf, 'r1_mAP')
            # __print_info('6. bn rgb feat', evaluator_bn_rgb, 'r1_mAP')
            # __print_info('7. bn nir feat', evaluator_bn_ni, 'r1_mAP')
            # __print_info('8. bn tir feat', evaluator_bn_t, 'r1_mAP')
            # __print_info('9. bn r_n feat', evaluator_bn_rn, 'r1_mAP')
            # __print_info('10. bn r_t feat', evaluator_bn_rt, 'r1_mAP')
            # __print_info('11. bn n_t feat', evaluator_bn_nt, 'r1_mAP')

            # __print_info('6. [5] - fusing feat', evaluator_bnf_nofuse, 'r1_mAP')
            # __print_info('7. [5] + weighted metirc', w_evaluator_bnf, 'w_r1_mAP')
            # __print_info('8. [6] + weighted metric', w_evaluator_bnf_nofuse, 'w_r1_mAP')
            # __print_info('9. p_weighted, gf', p_evaluator_gf, 'p_r1_mAP')
            # __print_info('10. p_weighted, bnf', p_evaluator_bnf, 'p_r1_mAP')
            # normed_p_weights = p_weights.detach().cpu().numpy()
            # # min_p = min(normed_p_weights)
            # # max_p = max(normed_p_weights)
            # # normed_p_weights = (normed_p_weights - min_p)/(max_p - min_p + 1e-12)
            # normed_p_weights = np.exp(normed_p_weights)
            # normed_p_weights = normed_p_weights / np.sum(normed_p_weights)
            # logger.info('parameters in p_trip:\n {}'.format(normed_p_weights.reshape((3, 3))))
            R_Saver.saveResults(cfg.OUTPUT_DIR)

    trainer.run(train_loader, max_epochs=epochs)
    # R_Saver.saveAsCSV(cfg.OUTPUT_DIR)