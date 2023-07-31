import pickle
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import cv2
import os
import random
import pdb

def getData(filename, TOP_K=5, feat_name='feats'):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        for key, value in data.items():
            data[key] = data[key][591:]
    # ids = [45, 22, 4, 44, 80, 2]
    id_counter = Counter(data['ids'])
    l = [i for i in id_counter.items()]
    l = sorted(l, key=lambda x: x[1], reverse=True)
    # l.pop(0)
    # l.pop(2)
    ids = [vid for vid, num in l[:TOP_K]]
    print(l[:TOP_K])
    print(np.array([0]+[n[1] for n in l[:TOP_K]]).cumsum())
    # import pdb; pdb.set_trace()
    # ids = [24, 20, 26, 94, 59, 75]
    id_map = {vid: i for i, vid in enumerate(ids)}
    id_idx = dict()
    feats = data[feat_name]
    paths = data["paths"]
    feat_list = []
    label_list = []
    path_list = []
    
    # import pdb; pdb.set_trace()
    for i, vid in enumerate(data['ids']):
        if vid in ids:
            feat_list.append(feats[i])
            label_list.append(id_map[vid])
            path_list.append(paths[i])
    return feat_list, label_list, path_list

def _imshow(path):
    # im = plt.imread(path)
    im = cv2.imread(path)[:,:,::-1]
    im = cv2.resize(im, dsize=(512, 256))
    plt.imshow(im)
    # plt.pause(0.00001)

def meanvalue_distribution(path=r'v\tSNE\baseline.pkl'):
    # feats, label, paths = getData(path, feat_name="mid_feat")
    with open(path, 'rb') as f:
        data = pickle.load(f)
        for key, value in data.items():
            print(key, len(data[key]))
            data[key] = data[key][591:]
    
    feats = []
    paths = []
    vids = data["ids"]
    for i in range(1055):
        vids[i] += 200
    # import pdb; pdb.set_trace()
    for i, vid in enumerate(vids):
        feats.append(data["mid_feat"][i])
        paths.append(data["paths"][i])

    modal_feats = ([], [], [])
    modal_means = ([], [], [])
    for f in feats:
        modal_feats[0].append(f[:512])
        modal_means[0].append(f[:512].mean(axis=0))
        modal_feats[1].append(f[512: 1024])
        modal_means[1].append(f[512: 1024].mean(axis=0))
        modal_feats[2].append(f[1024: ])
        modal_means[2].append(f[1024:].mean(axis=0))
    
    N = len(feats)
    print(N)
    
    indexes = [i for i in range(N)]
    random.shuffle(indexes)
    random.shuffle(indexes)

    target_index = []
    fig = plt.figure(figsize=(5, 10))
    for i, idx in enumerate(indexes):
        # print(i, modal_means[0][i], vids[i])
        # pdb.set_trace()
        if vids[idx] == 200+94:
            # plt.scatter(i, modal_means[0][idx], s=18, color=(1, 0, 0, 1), marker='o')
            target_index.append(idx)
        
        plt.subplot(3, 1, 1)
        plt.scatter(i, modal_means[0][idx], s=40, color=(0, 0, 0, 0.1), marker='o')
        plt.subplot(3, 1, 2)
        plt.scatter(i, modal_means[1][idx], s=40, color=(0, 0, 0, 0.1), marker='o')
        plt.subplot(3, 1, 3)
        plt.scatter(i, modal_means[2][idx], s=40, color=(0, 0, 0, 0.1), marker='o')

    step = N // len(target_index)
    for i, idx in zip([i for i in range(0,N,step)], target_index):
        plt.subplot(3, 1, 1)
        plt.scatter(i, modal_means[0][idx], s=40, color=(1, 0, 0, 1), marker='o')
        plt.subplot(3, 1, 2)
        plt.scatter(i, modal_means[1][idx], s=40, color=(1, 0, 0, 1), marker='o')
        plt.subplot(3, 1, 3)
        plt.scatter(i, modal_means[2][idx], s=40, color=(1, 0, 0, 1), marker='o')
        print(paths[idx].split(os.sep)[-1])
        
    plt.subplot(3, 1, 1)
    plt.yticks([0.030, 0.035, 0.040, 0.045])
    plt.xticks([])
    plt.subplot(3, 1, 2)
    plt.yticks([0.030, 0.035, 0.040, 0.045])
    plt.xticks([])
    plt.subplot(3, 1, 3)
    plt.yticks([0.030, 0.035, 0.040, 0.045])
    plt.xticks([])
    # plt.ylabel("mean value")
    plt.show()



if __name__ == "__main__":
    # feats, label, paths = getData(r'v\tSNE\bs_cls_hc_score_test.pkl', feat_name="mid_feat")
    # feats, label, paths = getData(r'v\tSNE\modal_norm_adapt.pkl', feat_name="mid_feat")
    feats, label, paths = getData(r'v\tSNE\baseline.pkl', feat_name="mid_feat")

    # mid_feats = mid_feats.reshape((-1, 3, 512))
    # mid_feats_mean = mid_feats.mean(axis=2) # [N, 3]

    start = 111
    end = 131

    # print(Counter(label))
    # paths= [p[-20:] for p in paths]
    mean_list = []
    for f in feats:
        # import pdb; pdb.set_trace()
        # mean_list.append(f[0:512].mean(axis=0))
        # mean_list.append(f[512:1024].mean(axis=0))
        mean_list.append(f[1024:].mean(axis=0))
    # print(mean_list[start: start+end])

    mean_list = mean_list[start: end]
    paths = paths[start: end]

    # v_min = np.min(mean_list)
    # v_max = np.max(mean_list)
    # mean_list = (mean_list - v_min) / (v_max - v_min)

    index = [i for i in range(len(mean_list))]
    # import pdb; pdb.set_trace()
    index = sorted(index, key=lambda i: mean_list[i])
    print(len(index))
    fig = plt.figure(figsize=(24, 12))
    for i, idx in enumerate(index):
        # print(paths[idx], mean_list[idx])
        ax = plt.subplot(6, 6, i + 1)
        ax.axis("off")
        _imshow(paths[idx])
        # _imshow(paths[idx].replace("vis", "ni"))
        _imshow(paths[idx].replace("vis", "th"))
        plt.title(paths[idx][-20:-4] + " " + str(mean_list[idx]))
    plt.show()

    # meanvalue_distribution(r'v\tSNE\baseline_cls_all.pkl')
        

