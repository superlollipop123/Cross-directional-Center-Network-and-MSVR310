import os

import numpy as np
import cv2

import sys
sys.path.append('.')

def get_img_pairs(folder1, folder2):
    for img in os.listdir(os.path.join(r'v\grad_cam', folder1)):
        img1 = os.path.join(r'v\grad_cam', folder1, img)
        img2 = os.path.join(r'v\grad_cam', folder2, img)

        img1 = cv2.imread(img1)
        img2 = cv2.imread(img2)

        merge_img = np.concatenate([img1, img2], axis=1)
        
        cv2.imwrite(os.path.join(r'v\grad_cam\merge_imgs', img), merge_img)

        # break

if __name__ == "__main__":
    get_img_pairs("baseline_cls", "CdC_ALNU")