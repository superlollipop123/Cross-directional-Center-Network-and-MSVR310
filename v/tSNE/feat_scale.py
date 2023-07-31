import pickle
import matplotlib.pyplot as plt
import numpy as np

def getData(filename, TOP_K=8, feat_name='feats'):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        for key, value in data.items():
            data[key] = data[key][591:]
    print(data.keys())
    return data

if __name__ == "__main__":
    data = getData(r'v\tSNE\bs_cls_hc_score_test.pkl')
    mid_feats = data['mid_feat']
    labels = data['ids']
    weights = data['weights'] # [N, 3]

    mid_feats = mid_feats.reshape((-1, 3, 512))
    mid_feats_mean = mid_feats.mean(axis=2) # [N, 3]

    N = mid_feats_mean.shape[0]
    # weights = np.random.rand(N, 3) * 0.175 + 0.325
    for i in range(N):
        plt.subplot(2, 3, 1)
        plt.scatter(weights[i, 0], mid_feats_mean[i, 0])
        plt.subplot(2, 3, 2)
        plt.scatter(weights[i, 1], mid_feats_mean[i, 1])
        plt.subplot(2, 3, 3)
        plt.scatter(weights[i, 2], mid_feats_mean[i, 2])
        plt.subplot(2, 3, 4)
        plt.scatter(mid_feats_mean[i, 0], weights[i, 0] * mid_feats_mean[i, 0])
        plt.subplot(2, 3, 5)
        plt.scatter(mid_feats_mean[i, 1], weights[i, 1] * mid_feats_mean[i, 1])
        plt.subplot(2, 3, 6)
        plt.scatter(mid_feats_mean[i, 1], weights[i, 2] * mid_feats_mean[i, 2])
    plt.subplot(2, 3, 1)
    plt.title('rgb')
    plt.xlabel('weights')
    plt.ylabel('feat mean')
    plt.subplot(2, 3, 2)
    plt.title('nir')
    plt.xlabel('weights')
    plt.ylabel('feat mean')
    plt.subplot(2, 3, 3)
    plt.title('tir')
    plt.xlabel('weights')
    plt.ylabel('feat mean')
    plt.subplot(2, 3, 4)
    plt.title('rgb')
    plt.xlabel('feat mean')
    plt.ylabel('feat mean * weight')
    plt.subplot(2, 3, 5)
    plt.title('nir')
    plt.xlabel('feat mean')
    plt.ylabel('feat mean * weight')
    plt.subplot(2, 3, 6)
    plt.title('tir')
    plt.xlabel('feat mean')
    plt.ylabel('feat mean * weight')
    plt.show()

    # plt.subplot(2, 3, 1)
    # plt.plot(feats[0][:2048])
    # plt.subplot(2, 3, 2)
    # plt.plot(feats[0][2048:4096])
    # plt.subplot(2, 3, 3)
    # plt.plot(feats[0][4096:])
    # plt.subplot(2, 3, 4)
    # plt.plot(feats_sc[0][:2048])
    # plt.subplot(2, 3, 5)
    # plt.plot(feats_sc[0][2048:4096])
    # plt.subplot(2, 3, 6)
    # plt.plot(feats_sc[0][4096:])
    # mid_feats = np.split(mid_feats, 3, axis=0)