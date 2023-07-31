import pickle
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import cv2
import os
import random
import pdb

import sys
sys.path.append(".")
from v.tSNE.randomColor import PLT_COLOR_MAP1

def getData(filename, TOP_K=5, f_name=['', '']):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        # for key, value in data.items():
        #     data[key] = data[key][591:]
        # pdb.set_trace()
        count_q, count_g, count_t = 0, 0, 0
        for i, p in enumerate(data["paths"]):
            if "bounding_box_test" in p: count_g += 1
            if "bounding_box_train" in p: count_t += 1
            if "query" in p: count_q += 1
        print(count_q, count_g, count_t)
        
        # query_data = data[:count_q]
        # gallery_data = data[count_q: count_q+count_g]
        # train_data = data[count_q+count_g: ]


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
    
    mean_std_list = []
    label_list = []
    path_list = []
    
    def get_mean_std(index):
        return [data[name][index] for name in f_name]

    # import pdb; pdb.set_trace()
    for i, vid in enumerate(data['ids']):
        # if vid in ids:
            mean_std_list.append(get_mean_std(i))
            # label_list.append(id_map[vid])
            label_list.append(data["ids"][i])
            path_list.append(data["paths"][i])
    return mean_std_list, label_list, path_list

if __name__ == "__main__":
    data, label, path = getData(r"v\save_feat\qualityNorm.pkl", f_name=["f_normed_mean", "f_normed_std"])
    # data2, label2, path2 = getData(r"v\save_feat\baseline_cls.pkl", f_name=["f_ori_mean", "f_ori_std"])
    data2, label2, path2 = getData(r"v\save_feat\qualityNorm.pkl", f_name=["f_ori_mean", "f_ori_std"])
    print(len(path))
    # exit()
    fig = plt.figure(figsize=(12, 8))

    black = (0, 0, 0, 0.2)
    point_size = 30


    # for i in range(len(path)):
    #     plt.subplot(3, 2, 1)
    #     plt.scatter(i, data[i][0][0], s=point_size, color=black, marker='o')
    #     plt.subplot(3, 2, 2)
    #     plt.scatter(i, data2[i][0][0], s=point_size, color=black, marker='o')
    #     # plt.scatter(data[i][2][0], data[i][3][0], s=point_size, color=black, marker='o')
        
    #     plt.subplot(3, 2, 3)
    #     plt.scatter(i, data[i][0][1], s=point_size, color=black, marker='o')
    #     plt.subplot(3, 2, 4)
    #     plt.scatter(i, data2[i][0][1], s=point_size, color=black, marker='o')
    #     # plt.scatter(data[i][2][1], data[i][3][1], s=point_size, color=black, marker='o')
        
    #     plt.subplot(3, 2, 5)
    #     plt.scatter(i, data[i][0][2], s=point_size, color=black, marker='o')
    #     plt.subplot(3, 2, 6)
    #     plt.scatter(i, data2[i][0][2], s=point_size, color=black, marker='o')
    #     # plt.scatter(data[i][2][2], data[i][3][2], s=point_size, color=black, marker='o')

    target_sample = []
    for i in range(len(path)):
        if "0094" in path[i]:
            target_sample.append(i)

        plt.subplot(3, 2, 1)
        plt.scatter(data[i][0][0], data[i][1][0], s=point_size, color=black, marker='o')
        plt.subplot(3, 2, 2)
        plt.scatter(data2[i][0][0], data2[i][1][0], s=point_size, color=black, marker='o')
        

        plt.subplot(3, 2, 3)
        plt.scatter(data[i][0][1], data[i][1][1], s=point_size, color=black, marker='o')
        plt.subplot(3, 2, 4)
        plt.scatter(data2[i][0][1], data2[i][1][1], s=point_size, color=black, marker='o')
        
        plt.subplot(3, 2, 5)
        plt.scatter(data[i][0][2], data[i][1][2], s=point_size, color=black, marker='o')
        plt.subplot(3, 2, 6)
        plt.scatter(data2[i][0][2], data2[i][1][2], s=point_size, color=black, marker='o')
    
    for i in target_sample:
        color = (0.9, 0, 0, 0.6)
        # if '_007' in path[i]:
        #     color = (0, 1, 0, 1)
        # elif '_000' in path[i]:
        #     color = (0, 0, 1, 1)
        # elif '_019' in path[i]:
        #     color = (0, 1, 1, 1)
        # elif '_004' in path[i]:
        #     color = (1, 1, 0, 1)
        # else:
        #     color = (1, 0, 0, 1)
        # for s in ["_007", "_002", "_019", "_004"]:
        #     if s in path[i]:
        #         color = (0, 1, 0, 1)
        
        plt.subplot(3, 2, 1)
        plt.scatter(data[i][0][0], data[i][1][0], s=point_size, color=color, marker='o')
        plt.subplot(3, 2, 2)
        plt.scatter(data2[i][0][0], data2[i][1][0], s=point_size, color=color, marker='o')
        # print(path[i].split(os.sep)[-1], (data[i][0][0], data[i][1][0]), (data2[i][0][0], data2[i][1][0]))

        plt.subplot(3, 2, 3)
        plt.scatter(data[i][0][1], data[i][1][1], s=point_size, color=color, marker='o')
        plt.subplot(3, 2, 4)
        plt.scatter(data2[i][0][1], data2[i][1][1], s=point_size, color=color, marker='o')

        plt.subplot(3, 2, 5)
        plt.scatter(data[i][0][2], data[i][1][2], s=point_size, color=color, marker='o')
        plt.subplot(3, 2, 6)
        plt.scatter(data2[i][0][2], data2[i][1][2], s=point_size, color=color, marker='o')

    # for i in target_sample:
    #     # color = (1, 0, 0, 1)
    #     if '_007' in path[i]:
    #         color = PLT_COLOR_MAP1[1]
    #         # color = (0, 1, 0, 1)
    #     elif '_000' in path[i]:
    #         color = PLT_COLOR_MAP1[2]
    #         # color = (0, 0, 1, 1)
    #     # elif '_019' in path[i]:
    #     #     color = PLT_COLOR_MAP1[4]
    #     elif '_004' in path[i]:
    #         color = PLT_COLOR_MAP1[5]
    #         # color = (0, 1, 1, 1)
    #     elif "_017" in path[i]:
    #         color = PLT_COLOR_MAP1[7]
    #         # color = (1, 0.5, 0.5, 1)
    #     else:
    #         continue
    #     # for s in ["_007", "_002", "_019", "_004"]:
    #     #     if s in path[i]:
    #     #         color = (0, 1, 0, 1)
        
    #     plt.subplot(3, 2, 1)
    #     plt.scatter(data[i][0][0], data[i][1][0], s=point_size, color=color, marker='o')
    #     plt.subplot(3, 2, 2)
    #     plt.scatter(data2[i][0][0], data2[i][1][0], s=point_size, color=color, marker='o')
    #     # print(path[i].split(os.sep)[-1], (data[i][0][0], data[i][1][0]), (data2[i][0][0], data2[i][1][0]))

    #     plt.subplot(3, 2, 3)
    #     plt.scatter(data[i][0][1], data[i][1][1], s=point_size, color=color, marker='o')
    #     plt.subplot(3, 2, 4)
    #     plt.scatter(data2[i][0][1], data2[i][1][1], s=point_size, color=color, marker='o')

    #     plt.subplot(3, 2, 5)
    #     plt.scatter(data[i][0][2], data[i][1][2], s=point_size, color=color, marker='o')
    #     plt.subplot(3, 2, 6)
    #     plt.scatter(data2[i][0][2], data2[i][1][2], s=point_size, color=color, marker='o')
    for i in range(1, 7):
        plt.subplot(3, 2, i)
        plt.yticks(fontproperties = 'Times New Roman', size = 14)
        plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.show()