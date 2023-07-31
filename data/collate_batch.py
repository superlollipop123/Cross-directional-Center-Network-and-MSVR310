# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch


def getTensorData(imgs):
    batchsize = len(imgs)
    modalities = len(imgs[0])
    data = []
    for m in range(modalities):
        tmp = []
        for i in range(batchsize):
            tmp.append(imgs[i][m])
        data.append(torch.stack(tmp, dim=0))
    
    return data

def train_collate_fn(batch):
    imgs, pids, camids, sceneids, img_path = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return getTensorData(imgs), pids, camids, sceneids, img_path


def val_collate_fn(batch):
    imgs, pids, camids, sceneids, img_path = zip(*batch)
    return getTensorData(imgs), pids, camids, sceneids, img_path
