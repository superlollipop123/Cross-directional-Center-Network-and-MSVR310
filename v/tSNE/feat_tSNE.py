import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn import manifold
import pickle
from collections import Counter, defaultdict
# from .randomColor import ncolors
from randomColor import ncolors

TOP_K = 6

def getData(filename, TOP_K=8, feat_name='feats'):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        for key, value in data.items():
            data[key] = data[key][591:]
    # ids = [45, 22, 4, 44, 80, 2]
    id_counter = Counter(data['ids'])
    l = [i for i in id_counter.items()]
    l = sorted(l, key=lambda x: x[1], reverse=True)
    l.pop(0)
    l.pop(2)
    ids = [vid for vid, num in l[:TOP_K]]
    print(l[:TOP_K])
    # import pdb; pdb.set_trace()
    # ids = [24, 20, 26, 94, 59, 75]
    id_map = {vid: i for i, vid in enumerate(ids)}
    id_idx = dict()
    feats = data[feat_name]
    feat_list = []
    label_list = []
    for i, vid in enumerate(data['ids']):
        if vid in ids:
            feat_list.append(feats[i])
            label_list.append(id_map[vid])
    return feat_list, label_list, ncolors(TOP_K)

def getData_2(filename, TOP_K=8, feat_name='feats'):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        for key, value in data.items():
            data[key] = data[key][591:]
    # ids = [45, 22, 4, 44, 80, 2]
    id_counter = Counter(data['ids'])
    l = [i for i in id_counter.items()]
    l = sorted(l, key=lambda x: x[1], reverse=True)
    l.pop(0)
    l.pop(2)
    l.pop(5)
    l.pop(5)
    ids = [vid for vid, num in l[:TOP_K]]
    print(l[:TOP_K])
    # import pdb; pdb.set_trace()
    # ids = [24, 20, 26, 94, 59, 75]
    id_map = {vid: i for i, vid in enumerate(ids)}
    feats = data[feat_name]

    # ranges = [(0, 31), (31, 55)]
    
    feat_list = []
    label_list = []
    img_per_id = defaultdict(int)
    for i, vid in enumerate(data['ids']):
        if vid in ids:
            if img_per_id[vid] > 8:
                continue
            else:
                img_per_id[vid] += 1
            feat_list.append(feats[i])
            label_list.append(id_map[vid])
    return feat_list, label_list, ncolors(TOP_K)

def drawPic(X, y, COLORS):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)
    print(X.shape, X_tsne.shape)

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=COLORS[y[i]], fontdict={'weight': 'bold', 'size': 9})
        # plt.scatter(X_norm[i, 0], X_norm[i, 1], s=18, color=COLORS[y[i]])
    plt.xticks([])
    plt.yticks([])

def drawPic2(X, y, COLORS, feat_dim=2048):
    # N, M = X.shape
    n = X.shape[1]//feat_dim # modalities
    m_feats = np.split(X, n, axis=1)
    X = np.concatenate(m_feats, axis=0)
    y = y * n

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)

    # norm
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)

    # draw
    markers = ['^', 'o', '*', '.']
    N = X_norm.shape[0]
    l = N//n # samples
    for i in range(N):
        # plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]), fontdict={'weight': 'bold', 'size': 9})
        plt.scatter(X_norm[i, 0], X_norm[i, 1], s=18, color=COLORS[y[i]], marker=markers[i//l])
    plt.xticks([])
    plt.yticks([])

if __name__ == "__main__":


    feats, labels, colors = getData_2(r'v\tSNE\baseline.pkl', TOP_K, feat_name='bn_f_feat')
    feats = np.stack(feats, axis=0)
    plt.subplot(2, 2, 1)
    drawPic2(feats, labels, colors)

    feats, labels, colors = getData_2(r'v\tSNE\CdC_lambda001_test.pkl', TOP_K, feat_name='bn_f_feat')
    feats = np.stack(feats, axis=0)
    plt.subplot(2, 2, 2)
    drawPic2(feats, labels, colors)

    feats, labels, colors = getData_2(r'v\tSNE\CdC_lambda01_test.pkl', TOP_K, feat_name='bn_f_feat')
    feats = np.stack(feats, axis=0)
    plt.subplot(2, 2, 3)
    drawPic2(feats, labels, colors)

    feats, labels, colors = getData_2(r'v\tSNE\CdC_lambda1_test.pkl', TOP_K, feat_name='bn_f_feat')
    feats = np.stack(feats, axis=0)
    plt.subplot(2, 2, 4)
    drawPic2(feats, labels, colors)

    recs = []
    for i in range(TOP_K):
        recs.append(mpatches.Rectangle((0,0),1,1,fc=colors[i]))
    plt.legend(recs, list(range(TOP_K)), bbox_to_anchor=(1.02, 1), loc=2)

    plt.show() 


