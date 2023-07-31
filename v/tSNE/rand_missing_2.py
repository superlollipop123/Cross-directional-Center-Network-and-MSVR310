import math
import pickle
import torch
import sys
sys.path.append('.')
from utils.reid_metric import R1_mAP

import matplotlib.pyplot as plt
import os
import numpy as np
import random

def get_test_feat(pk_file):
    # pk_file = r'.\v\tSNE\ours_test.pkl'
    # pk_file = r'.\v\tSNE\ours_rdmiss_ms10.pkl'
    # pk_file = r'.\v\tSNE\ours_rdloss_r3.pkl'
    # pk_file = r'.\v\tSNE\ours_rmloss_m001.pkl'
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
    # p 为 各位置特征随机缺失概率，全部缺失视为无缺失
    # feats is a tuple saving features from corresponding modal
    # import pdb; pdb.set_trace()
    p = 0.5 - math.sqrt(0.25 - p/3)
    n = feats.shape[0]
    feats = feats.view(n, 2048, 3).permute([0, 2, 1])
    rand = torch.rand(size=(n, 3))
    mask = rand < (1 - p) # rand > p
    
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

def rand_missing_2(feats, p):
    # p为存在缺失数据占总数据比
    n = feats.shape[0]

    if feats.shape[1]//3 == 512:
        feats = feats.view(n, 3, 512)
    elif feats.shape[1]//3 == 2048:
        feats = feats.view(n, 3, 2048)

    if p == 0:
        feats = feats.sum(dim=1) / 3
        return feats

    # 缺失样本mask
    n_missing = int(n*p)
    n_missing_1 = n_missing // 2
    n_missing_2 = n_missing - n_missing_1
    n_keep = n - n_missing
    
    select = [3] * n_keep + [2] * n_missing_1 + [1] * n_missing_2
    select = torch.Tensor(select[::-1])
    idx = [i for i in range(n)]
    random.shuffle(idx)
    idx = torch.Tensor(idx).long()
    
    mask_1 = torch.rand((n_missing_1, 3))
    mask_2 = torch.rand((n_missing_2, 3))
    
    mask_1_keep = torch.argmax(mask_1, dim=1)
    mask_1_keep = torch.eye(3)[mask_1_keep]

    mask_2_keep = torch.argmax(mask_2, dim=1)
    mask_2_keep = torch.eye(3)[mask_2_keep]
    mask_2_keep = 1 - mask_2_keep

    mask_3_keep = torch.ones(n_keep*3).view(n_keep, 3)
    
    mask = torch.cat([mask_1_keep, mask_2_keep, mask_3_keep], dim=0)
    mask = mask[idx]
    select = select[idx]
    feats = feats * mask.unsqueeze(2)
    feats = feats.sum(dim=1) / select.unsqueeze(1)
    # import pdb; pdb.set_trace()
    
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

def rank_for_n_times(n, p, file):
    ori_feats, ids, camids, sceneids, pathes = get_test_feat(file)
    # feats = rand_missing(feats, p)
    avg_cmc = [0, 0, 0]
    avg_mAP = 0
    seeds = [i+1 for i in range(n)]
    for i in range(n):
        seed_torch(seeds[i])
        feats = rand_missing_2(ori_feats, p)
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

def get_result(file):
    map_list = []
    r1_list = []
    r5_list = []
    r10_list = []
    RANGE = 11
    STEP = 0.1
    for i in range(0, RANGE):
        p = i * STEP
        print(p)
        map, cmc = rank_for_n_times(10, p, file)
        map_list.append(map)
        r1_list.append(cmc[0])
        r5_list.append(cmc[1])
        r10_list.append(cmc[2])
    for i in range(RANGE):
        p = STEP * i
        print('p: {:.2%}'.format(p))
        print("mAP: {:.2%}".format(map_list[i]))
        print("Rank-1:{:.2%} Rank-5:{:.2%} Rank-10:{:.2%}".format(r1_list[i], r5_list[i], r10_list[i]))
    return map_list, r1_list, r5_list, r10_list

if __name__ == '__main__':
    # pk_file = r'.\v\tSNE\osnet_msvr300_test.pkl'
    # pk_file = r'.\v\tSNE\HAMNET_test.pkl'
    # map_1, r1_1, r5_1, r10_1 = get_result(pk_file)
    # pk_file = r'.\v\tSNE\osnet_msvr300_test.pkl'
    # map_3, r1_3, r5_3, r10_3 = get_result(pk_file)
    # pk_file = r'.\v\tSNE\CdC_lam03alpha06_ALNU_test.pkl'
    # map_2, r1_2, r5_2, r10_2 = get_result(pk_file)
    # # pk_file = r'.\v\tSNE\HAMNET_test.pkl'
    

    # p = [i * 0.1 for i in range(len(map_1))]
    # from v.tSNE.randomColor import PLT_COLOR_MAP1
    # plt.plot(p, map_1, color=PLT_COLOR_MAP1[0], marker = "*", linestyle=':')
    # plt.plot(p, r1_1, color=PLT_COLOR_MAP1[1], marker = "^", linestyle=':')
    # plt.plot(p, r5_1, color=PLT_COLOR_MAP1[2], marker = ".", linestyle=':')
    # plt.plot(p, r10_1, color=PLT_COLOR_MAP1[3], marker = "o", linestyle=':')

    # plt.plot(p, map_2, color=PLT_COLOR_MAP1[0], marker = "*", linestyle='-')
    # plt.plot(p, r1_2, color=PLT_COLOR_MAP1[1], marker = "^", linestyle='-')
    # plt.plot(p, r5_2, color=PLT_COLOR_MAP1[2], marker = ".", linestyle='-')
    # plt.plot(p, r10_2, color=PLT_COLOR_MAP1[3], marker = "o", linestyle='-')

    # plt.plot(p, map_3, color=PLT_COLOR_MAP1[0], marker = "*", linestyle='--')
    # plt.plot(p, r1_3, color=PLT_COLOR_MAP1[1], marker = "^", linestyle='--')
    # plt.plot(p, r5_3, color=PLT_COLOR_MAP1[2], marker = ".", linestyle='--')
    # plt.plot(p, r10_3, color=PLT_COLOR_MAP1[3], marker = "o", linestyle='--')
    # plt.xticks(p)
    # plt.yticks([0.1*i for i in range(1, 9)])
    # plt.show()
    # map, cmc = rank_for_n_times(n=1, p=-1)
    # print("mAP: {:.2%}".format(map))
    # print("Rank-1:{:.2%} Rank-5:{:.2%} Rank-10:{:.2%}".format(cmc[0], cmc[1], cmc[2]))


    CCNet = [
        [34.43, 51.61, 67.68, 73.77],
        [32.99, 50.30, 66.70, 73.57],
        [31.82, 49.10, 66.36, 72.84],
        [30.61, 47.97, 66.01, 72.40],
        [29.37, 46.62, 65.40, 72.01],
        [28.06, 45.52, 64.38, 71.56],
        [26.58, 43.98, 63.03, 70.95],
        [25.45, 42.08, 62.44, 70.44],
        [24.29, 41.30, 61.24, 69.51],
        [22.98, 39.59, 60.30, 68.56],
        [21.80, 37.94, 59.02, 67.83]
    ]

    OSNet = [
        [26.70, 43.15, 63.45, 70.90],
        [24.31, 40.93, 61.44, 69.15],
        [22.19, 39.31, 59.37, 67.53],
        [20.07, 36.57, 57.58, 66.45],
        [18.29, 34.67, 56.11, 65.23],
        [16.64, 32.76, 54.52, 63.57],
        [14.89, 30.76, 52.18, 61.39],
        [13.52, 29.31, 50.07, 59.48],
        [12.26, 26.90, 47.17, 57.41],
        [11.35, 25.53, 46.04, 56.13],
        [11.15, 24.09, 45.21, 55.21]
    ]

    HAMNet = [
        [25.49, 39.09, 56.85, 65.82],
        [24.37, 38.05, 56.13, 65.16],
        [23.27, 36.40, 55.08, 63.86],
        [22.15, 34.72, 54.20, 63.43],
        [21.01, 32.81, 52.99, 62.37],
        [19.95, 31.79, 51.81, 61.24],
        [18.73, 30.24, 50.12, 60.42],
        [17.69, 29.19, 48.90, 59.34],
        [16.57, 27.07, 47.11, 57.97],
        [15.49, 25.62, 45.89, 57.19],
        [14.43, 24.26, 44.23, 55.87]
    ]

    def getIndicators(data):
        L = len(data)
        map = [data[i][0] for i in range(L)]
        r1 = [data[i][1] for i in range(L)]
        r5 = [data[i][2] for i in range(L)]
        r10 = [data[i][3] for i in range(L)]
        return map, r1, r5, r10
    
    map_1, r1_1, r5_1, r10_1 = getIndicators(CCNet)
    map_2, r1_2, r5_2, r10_2 = getIndicators(OSNet)
    map_3, r1_3, r5_3, r10_3 = getIndicators(HAMNet)

    p = [i * 10 for i in range(len(map_1))]
    from v.tSNE.randomColor import PLT_COLOR_MAP1
    # plt.plot(p, map_1, color=PLT_COLOR_MAP1[0], marker = "*", linestyle='--', linewidth=1)
    # plt.plot(p, r1_1, color=PLT_COLOR_MAP1[0], marker = "^", linestyle='-', linewidth=2)
    # plt.plot(p, r5_1, color=PLT_COLOR_MAP1[0], marker = ".", linestyle='-.', linewidth=2)
    # plt.plot(p, r10_1, color=PLT_COLOR_MAP1[0], marker = "o", linestyle='--', linewidth=2)

    # plt.plot(p, map_2, color=PLT_COLOR_MAP1[1], marker = "*", linestyle='--', linewidth=1)
    # plt.plot(p, r1_2, color=PLT_COLOR_MAP1[1], marker = "^", linestyle='-', linewidth=2)
    # plt.plot(p, r5_2, color=PLT_COLOR_MAP1[1], marker = ".", linestyle='-.', linewidth=2)
    # plt.plot(p, r10_2, color=PLT_COLOR_MAP1[1], marker = "o", linestyle='--', linewidth=2)

    # plt.plot(p, map_3, color=PLT_COLOR_MAP1[2], marker = "*", linestyle='--', linewidth=1)
    # plt.plot(p, r1_3, color=PLT_COLOR_MAP1[2], marker = "^", linestyle='-', linewidth=2)
    # plt.plot(p, r5_3, color=PLT_COLOR_MAP1[2], marker = ".", linestyle='-.', linewidth=2)
    # plt.plot(p, r10_3, color=PLT_COLOR_MAP1[2], marker = "o", linestyle='--', linewidth=2)
    plt.plot(p, map_1, color=PLT_COLOR_MAP1[0], marker = "*", linestyle='-', linewidth=2)
    plt.plot(p, r1_1, color=PLT_COLOR_MAP1[1], marker = "^", linestyle='-', linewidth=2)
    plt.plot(p, r5_1, color=PLT_COLOR_MAP1[2], marker = ".", linestyle='-', linewidth=2)
    plt.plot(p, r10_1, color=PLT_COLOR_MAP1[3], marker = "o", linestyle='-', linewidth=2)

    plt.plot(p, map_2, color=PLT_COLOR_MAP1[0], marker = "*", linestyle='--', linewidth=2)
    plt.plot(p, r1_2, color=PLT_COLOR_MAP1[1], marker = "^", linestyle='--', linewidth=2)
    plt.plot(p, r5_2, color=PLT_COLOR_MAP1[2], marker = ".", linestyle='--', linewidth=2)
    plt.plot(p, r10_2, color=PLT_COLOR_MAP1[3], marker = "o", linestyle='--', linewidth=2)

    plt.plot(p, map_3, color=PLT_COLOR_MAP1[0], marker = "*", linestyle=':', linewidth=2)
    plt.plot(p, r1_3, color=PLT_COLOR_MAP1[1], marker = "^", linestyle=':', linewidth=2)
    plt.plot(p, r5_3, color=PLT_COLOR_MAP1[2], marker = ".", linestyle=':', linewidth=2)
    plt.plot(p, r10_3, color=PLT_COLOR_MAP1[3], marker = "o", linestyle=':', linewidth=2)
    plt.xticks(p)
    plt.yticks([10*i for i in range(1, 9)])
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.show()