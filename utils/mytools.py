import os
import csv
import torch
from collections import OrderedDict
import numpy as np
import random

def seed_torch(seed=42):
    # https://blog.csdn.net/qq_41645987/article/details/107592810
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # 为当前CPU 设置随机种子
    torch.cuda.manual_seed(seed) # 为当前的GPU 设置随机种子
    torch.cuda.manual_seed_all(seed) # 当使用多块GPU 时，均设置随机种子

def modal_rand_missing(input, prob=0.1):
    # input is a list for multi-modal data
    N_modal = len(input)
    N_sample = input[0].shape[0]
    # prob_missing_1 = N_modal * (prob - 2 * prob**2 + prob**3)
    # prob_missing_2 = N_modal * (prob**2 - prob**3)
    
    rand = torch.rand(size=(N_modal, N_sample))
    mask = rand < prob # mask for missing data

    sum_mask = torch.sum(mask, dim=0, keepdim=True)
    keep_mask = sum_mask.eq(N_modal)
    mask = mask - keep_mask
    
    feat = torch.stack(input)
    # rand = torch.rand_like(feat)
    rand = torch.zeros_like(feat)
    final_feat = feat * (1 - mask).view([N_modal, N_sample, 1, 1, 1]).float() + rand * mask.unsqueeze(2).view([N_modal, N_sample, 1, 1, 1]).float()
    feats = final_feat.chunk(3, dim=0)
    feats = [f.squeeze() for f in feats]
    # import pdb;pdb.set_trace()
    return feats, mask

def load_params(model, trained_path):
    param_dict = torch.load(trained_path)
    net_params_keys = model.state_dict().keys()
    for key in net_params_keys:
        if 'num_batches_tracked' in key:
            continue
        if 'classifier' in key:
            continue
        if key not in param_dict:
            continue
        model.state_dict()[key].copy_(param_dict[key])

class LOSS_ZERO():
    '''
        use this to set some loss to 0 handly, and reduce influence to other code
        add operation is supported
        item function is supported
    '''
    def __add__(self, another_value):
        return another_value

    def __radd__(self, another_value):
        return another_value
    
    def item(self):
        return 0

class ResultSaver():

    def __init__(self) -> None:
        self.saver_dict = OrderedDict()
    
    def add(self, name, result):
        # result is a tuple form as (epoch, map, rank1, rank5, rank10)
        if name in self.saver_dict.keys():
            self.saver_dict[name].append(result)
        else:
            self.saver_dict[name] = []
            self.saver_dict[name].append(result)
        
    def saveResults(self, output_dir):
        path = os.path.join(output_dir, 'results.txt')
        with open(path, 'w', encoding='utf-8') as f:
            for name in sorted(self.saver_dict.keys()):
                value = self.saver_dict[name]
                f.write(name+':\n')
                s = ['Epoch:{:4}  mAP: {:.1%} R-1: {:.1%}  R-5: {:.1%} R-10: {:.1%}\n'.format(r[0], r[1], r[2], r[3], r[4]) for r in value]
                f.writelines(s)
                max_mAP = max([(r[1], r[0]) for r in value[::-1]], key=lambda x: x[0])
                max_r1 = max([(r[2], r[0]) for r in value[::-1]], key=lambda x: x[0])
                max_r5 = max([(r[3], r[0]) for r in value[::-1]], key=lambda x: x[0])
                max_r10 = max([(r[4], r[0]) for r in value[::-1]], key=lambda x: x[0])
                f.write('max_mAP:{:.1%}({:4}) max_r1:{:.1%}({:4}) max_r5:{:.1%}({:4}) max_r10:{:.1%}({:4})\n'.format(max_mAP[0], max_mAP[1], max_r1[0], max_r1[1], \
                    max_r5[0], max_r5[1], max_r10[0], max_r10[1]))
                f.write('\n')
                

    def saveAsCSV(self, output_dir):
        path = os.path.join(output_dir, 'results.csv')
        with open(path, 'w', newline='') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(['name', 'Epoch', 'mAP', 'Rank-1', 'Rank-5', 'Rank-10'])
            # f_csv.writerow([])
            for name, value in self.saver_dict.items():
                # f_csv.writerow(['name', name])
                for line in value:
                    f_csv.writerow([name, *line])
        

if __name__ == "__main__":
    # R = ResultSaver()
    # R.add('test', (10, 0.6663, 0.5553, 0.66663, 0.99999))
    # R.add('test', (20, 0.4663, 0.5553, 0.66663, 0.99999))
    # R.add('test', (30, 0.3663, 0.5553, 0.66663, 0.99999))
    # R.add('test', (40, 0.2663, 0.5553, 0.66663, 0.99999))
    # R.add('test', (50, 0.1663, 0.5553, 0.66663, 0.99999))
    # R.add('test2', (10, 0.6663, 0.5553, 0.66663, 0.99999))
    # R.add('test2', (20, 0.4663, 0.5553, 0.66663, 0.99999))
    # R.add('test2', (30, 0.3663, 0.5553, 0.66663, 0.99999))
    # R.add('test2', (40, 0.2663, 0.5553, 0.66663, 0.99999))
    # R.add('test2', (50, 0.1663, 0.5553, 0.66663, 0.99999))
    # R.saveResults(output_dir='')
    # # R.saveAsCSV(output_dir='')

    inp = [torch.rand((32, 3, 128, 256)), torch.rand((32, 3, 128, 256)), torch.rand((32, 3, 128, 256))]
    modal_rand_missing(inp, 0.5)