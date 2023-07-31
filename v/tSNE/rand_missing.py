import pickle
import torch
import sys
sys.path.append('.')
from utils.reid_metric import R1_mAP

import matplotlib.pyplot as plt
import os
import numpy as np
import random

def get_test_feat():
    # pk_file = r'.\v\tSNE\ours_test.pkl'
    # pk_file = r'.\v\tSNE\ours_rdmiss_ms10.pkl'
    # pk_file = r'.\v\tSNE\ours_rdloss_r3.pkl'
    # pk_file = r'.\v\tSNE\ours_rmloss_m001.pkl'
    # pk_file = r'.\v\tSNE\baseline.pkl'
    pk_file = r'.\v\tSNE\CdC_ALNU_rm.pkl'
    with open(pk_file, 'rb') as f:
        data = pickle.load(f)
        feats = data['bn_f_feat']
        feats = torch.Tensor(feats)
        ids = data['ids']
        camids = data['camids']
        sceneids = data['sceneids']
        pathes = data['paths']
        # import pdb; pdb.set_trace()
    return  feats, ids, camids, sceneids, pathes

def rand_missing(feats, p):
    # feats is a tuple saving features from corresponding modal
    n = feats.shape[0]
    feats = feats.view(n, 2048, 3).permute([0, 2, 1])
    rand = torch.rand(size=(n, 3))
    mask = rand < (1 - p)
    
    # avoid all modal missing
    sum_mask = mask.sum(dim=1)
    clear_mask = sum_mask == 0
    sum_mask[clear_mask] += 1

    # get mean value
    mask = mask.unsqueeze(dim=2).float()
    # import pdb; pdb.set_trace()
    feats = feats * mask
    feats = feats.sum(dim=1)
    feats = feats / sum_mask.unsqueeze(1).float()

    return feats

def rank(feats, ids, camids, sceneids, pathes):
    metrics_r1map = R1_mAP(591, 50, 'yes')
    metrics_r1map.feats = feats.cuda().chunk(2, dim=0)
    metrics_r1map.pids = ids
    metrics_r1map.camids = camids
    metrics_r1map.sceneids = sceneids
    metrics_r1map.img_path = pathes

    cmc, mAP = metrics_r1map.compute()

    return cmc, mAP

def rank_for_n_times(n, p):
    feats, ids, camids, sceneids, pathes = get_test_feat()
    feats = rand_missing(feats, p)
    avg_cmc = [0, 0, 0]
    avg_mAP = 0
    for i in range(n):
        cmc, mAP = rank(feats, ids, camids, sceneids, pathes)
        avg_mAP += mAP
        avg_cmc[0] += cmc[0]
        avg_cmc[1] += cmc[4]
        avg_cmc[2] += cmc[9]
    avg_mAP /= n
    avg_cmc = [v/n for v in avg_cmc]
    return avg_mAP, avg_cmc

def seed_torch(seed=42):
    # https://blog.csdn.net/qq_41645987/article/details/107592810
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # 为当前CPU 设置随机种子
    torch.cuda.manual_seed(seed) # 为当前的GPU 设置随机种子
    torch.cuda.manual_seed_all(seed) # 当使用多块GPU 时，均设置随机种子

if __name__ == '__main__':
    seed_torch()
    map_list = []
    r1_list = []
    r5_list = []
    r10_list = []
    for i in range(11):
        p = i * 0.05
        map, cmc = rank_for_n_times(10, p)
        map_list.append(map)
        r1_list.append(cmc[0])
        r5_list.append(cmc[1])
        r10_list.append(cmc[2])
    for i in range(11):
        p = 0.05 * i
        print('p: {:.2%}'.format(p))
        print("mAP: {:.2%}".format(map_list[i]))
        print("Rank-1:{:.2%} Rank-5:{:.2%} Rank-10:{:.2%}".format(r1_list[i], r5_list[i], r10_list[i]))

    p = [i * 0.05 for i in range(11)]
    
    plt.plot(p, map_list, color='red', linewidth=2, linestyle='--')
    plt.plot(p, r1_list, color='blue', linewidth=2, linestyle='-.')
    plt.show()
    # map, cmc = rank_for_n_times(n=1, p=-1)
    # print("mAP: {:.2%}".format(map))
    # print("Rank-1:{:.2%} Rank-5:{:.2%} Rank-10:{:.2%}".format(cmc[0], cmc[1], cmc[2]))