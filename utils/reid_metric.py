# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from layers.triplet_loss import normalize
import numpy as np
import torch
import pdb
from ignite.metrics import Metric

from data.datasets.eval_reid import eval_func
from .re_ranking import re_ranking

from layers.weighted_dist_cp import weighted_dist_comp
from layers.weighted_dist_cp import param_dist_comput


# class FeatureSave(Metric):
#     def __init__(self, num_query, max_rank=50, feat_norm='yes'):
#         super(R1_mAP, self).__init__()
#         self.num_query = num_query
#         self.max_rank = max_rank
#         self.feat_norm = feat_norm

#     def reset(self):
#         self.feats = []
#         self.pids = []
#         self.camids = []
#         self.sceneids = []
#         self.img_path = []

#     def update(self, output):
#         feat, pid, camid, sceneid, img_path = output
#         self.feats.append(feat)
#         self.pids.extend(np.asarray(pid))
#         self.camids.extend(np.asarray(camid))
#         self.sceneids.extend(np.asarray(sceneid))
#         self.img_path.extend(img_path)

#     def compute(self):
#         feats = torch.cat(self.feats, dim=0)
#         if self.feat_norm == 'yes':
#             print("The test feature is normalized")
#             feats = torch.nn.functional.normalize(feats, dim=1, p=2)
#         # query


#         return

class R1_mAP(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.sceneids = []
        self.img_path = []

    def update(self, output):
        feat, pid, camid, sceneid, img_path = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.sceneids.extend(np.asarray(sceneid))
        self.img_path.extend(img_path)

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])

        q_sceneids = np.asarray(self.sceneids[:self.num_query]) # zxp
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])

        g_sceneids = np.asarray(self.sceneids[self.num_query:]) # zxp

        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.cpu().numpy()
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, q_sceneids, g_sceneids)

        return cmc, mAP

class Param_R1_mAP(Metric):
    def __init__(self, num_query, weight_param, max_rank=50, feat_norm='yes'):
        super(Param_R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        # self.weights = []
        self.weight_parm = weight_param

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        # self.weights = []
        self.sceneids = []
        self.img_path = []

    def update(self, output):
        # feat, weight, pid, camid, sceneid, img_path = output
        feat, pid, camid, sceneid, img_path = output
        self.feats.append(feat)
        # self.weights.append(weight)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.sceneids.extend(np.asarray(sceneid))
        self.img_path.extend(img_path)
    
    def compute(self):
        l = len(self.feats[0])
        f_list = [ [] for _ in range(l)]
        for f in self.feats:
            for i in range(l):
                f_list[i].append(f[i])
        f_list = [torch.cat(f, dim=0) for f in f_list]

        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            f_list = [torch.nn.functional.normalize(f) for f in f_list]

        # query
        qf_list = [feats[:self.num_query] for feats in f_list]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        q_sceneids = np.asarray(self.sceneids[:self.num_query]) # zxp

        # gallery
        gf_list = [feats[self.num_query:] for feats in f_list]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        g_sceneids = np.asarray(self.sceneids[self.num_query:]) # zxp

        distmat = param_dist_comput(qf_list, gf_list, self.weight_parm.detach())

        # m, n = qf.shape[0], gf.shape[0]
        # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # distmat.addmm_(1, -2, qf, gf.t())

        distmat = distmat.cpu().numpy()
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, q_sceneids, g_sceneids)
        # cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP


class Weighted_R1_mAP(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(Weighted_R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.weights = []

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.weights = []
        self.sceneids = []
        self.img_path = []

    def update(self, output):
        feat, weight, pid, camid, sceneid, img_path = output
        self.feats.append(feat)
        self.weights.append(weight)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.sceneids.extend(np.asarray(sceneid))
        self.img_path.extend(img_path)

    def compute(self):
        l = len(self.feats[0])
        f_list = [ [] for _ in range(l)]
        w_list = [ [] for _ in range(l)]
        for f, w in zip(self.feats, self.weights):
            for i in range(l):
                f_list[i].append(f[i])
                w_list[i].append(w[i])
        f_list = [torch.cat(f, dim=0) for f in f_list]
        w_list = [torch.cat(w, dim=0) for w in w_list]

        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            f_list = [torch.nn.functional.normalize(f) for f in f_list]

        # query
        qf_list = [feats[:self.num_query] for feats in f_list]
        qw_list = [w[:self.num_query] for w in w_list]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])

        q_sceneids = np.asarray(self.sceneids[:self.num_query]) # zxp
        # gallery
        gf_list = [feats[self.num_query:] for feats in f_list]
        gw_list = [w[self.num_query:] for w in w_list]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        
        g_sceneids = np.asarray(self.sceneids[self.num_query:]) # zxp

        distmat = weighted_dist_comp(qf_list, gf_list, qw_list, gw_list)

        # m, n = qf.shape[0], gf.shape[0]
        # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # distmat.addmm_(1, -2, qf, gf.t())

        distmat = distmat.cpu().numpy()
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, q_sceneids, g_sceneids)
        # cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP

class R1_mAP_reranking(Metric):

    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(R1_mAP_reranking, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.sceneids = []
        self.img_path = []

    def update(self, output):
        feat, pid, camid, sceneid, img_path = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.sceneids.extend(np.asarray(sceneid))
        self.img_path.extend(img_path)

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])

        q_sceneids = np.asarray(self.sceneids[:self.num_query]) # zxp
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])

        g_sceneids = np.asarray(self.sceneids[self.num_query:]) # zxp

        print("Enter reranking")
        distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)

        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, q_sceneids, g_sceneids)

        return cmc, mAP
    

    # def __init__(self, num_query, max_rank=50, feat_norm='yes'):
    #     super(R1_mAP_reranking, self).__init__()
    #     self.num_query = num_query
    #     self.max_rank = max_rank
    #     self.feat_norm = feat_norm

    # def reset(self):
    #     self.feats = []
    #     self.pids = []
    #     self.camids = []

    # def update(self, output):
    #     feat, pid, camid = output
    #     self.feats.append(feat)
    #     self.pids.extend(np.asarray(pid))
    #     self.camids.extend(np.asarray(camid))

    # def compute(self):
    #     feats = torch.cat(self.feats, dim=0)
    #     if self.feat_norm == 'yes':
    #         print("The test feature is normalized")
    #         feats = torch.nn.functional.normalize(feats, dim=1, p=2)

    #     # query
    #     qf = feats[:self.num_query]
    #     q_pids = np.asarray(self.pids[:self.num_query])
    #     q_camids = np.asarray(self.camids[:self.num_query])
    #     # gallery
    #     gf = feats[self.num_query:]
    #     g_pids = np.asarray(self.pids[self.num_query:])
    #     g_camids = np.asarray(self.camids[self.num_query:])
    #     # m, n = qf.shape[0], gf.shape[0]
    #     # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
    #     #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    #     # distmat.addmm_(1, -2, qf, gf.t())
    #     # distmat = distmat.cpu().numpy()
    #     print("Enter reranking")
    #     distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
    #     #pdb.set_trace()
    #     cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)    # modified

    #     return cmc, mAP