import torchvision.transforms as T
import os
from PIL import Image, ImageEnhance
# from model import CrossAttentionNet
import numpy as np
import cv2
import torch
import pdb
import matplotlib.pyplot as plt

# from outputs_2.baseline_cls.baseline_zxp import MainNet as Model
from modeling.baseline_zxp import MainNet as Model

MEAN=[0.485, 0.456, 0.406]
STD=[0.229, 0.224, 0.225]

def get_img(path, transfrom):
    if not os.path.exists(path):
        raise IOError('file {} not exist!'.format(path))
    try:
        img = Image.open(path).convert('RGB')
    except:
        print('Can not get the image {}, redo ...'.format(path))

    # img = img * 0.8
    # pdb.set_trace()

    if transfrom:
        img = Transforms(img)
    return img

def Transforms(img):
    return T.Compose([
        T.Resize([128, 256]),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
        ])(img)

def recover_img(feat):  # feat [3, H, W]
    img = feat.permute([1, 2, 0]).numpy()
    # pmin = np.min(img)
    # pmax = np.max(img)
    # img = (img - pmin)/(pmax - pmin + 0.0001)
    for i in range(3):
        img[:, :, i] = img[:, :, i] * STD[i]
        img[:, :, i] = img[:, :, i] + MEAN[i]
    img = (img * 255).astype(np.uint8)
    img = np.clip(img, 0, 255)

    return img[:,:,::-1]

def save_featmap(feat, name, output_dir):
    if not os.path.exists(output_dir):
        p = os.path.abspath(output_dir)
        os.mkdir(p)
        print("dir dose not exist, make it:"+p)
    
    shape = feat.shape
    if len(shape) != 3:
        raise Exception("input feat should be a 3-dim tensor")

    C, H, W = shape
    flag_resize = False
    if H < 32 or W < 32:
        flag_resize = True

    feat = feat.detach().numpy()
    fmin = np.min(feat)
    fmax = np.max(feat)
    print(fmax, fmin)
    for i in range(C):
        map_name = name + '_c{}'.format(i)
        featmap = feat[i, :, :]
        featmap = (featmap - fmin)/(fmax - fmin + 0.0001)
        featmap = (featmap * 255).astype(np.uint8)
        featmap = cv2.applyColorMap(featmap, cv2.COLORMAP_JET)
        if flag_resize:
            featmap = cv2.resize(featmap, (W*5, H*5), interpolation=cv2.INTER_NEAREST)
            map_name += '_upsamp'
        map_name += '.jpg'
        cv2.imwrite(os.path.join(output_dir, map_name), featmap)

def get_imgs(path): # path is the path of some single modal image
    # img_folder = os.path.join(path.split(os.sep)[:-2])
    # img_name = path.split(os.sep)[-1]

    # r_img = os.path.join(img_folder, 'vis', img_name)
    # n_img = os.path.join(img_folder, 'ni', img_name)
    # t_img = os.path.join(img_folder, 'th', img_name)
    
    r_img = path
    n_img = path.replace("vis", "ni")
    t_img = path.replace("vis", "th")

    r_img = get_img(r_img, transfrom=True)
    n_img = get_img(n_img, transfrom=True)
    t_img = get_img(t_img, transfrom=True)

    return r_img, n_img, t_img

def run_once(imgs, weights):
    model = Model(751, 1, None, "bnneck", "after", "resnet50", False, 3).cuda()
    model.load_param(weights)
    print("parameters loaded :" + weights)
    
    model.eval()
    with torch.no_grad():
        imgs = [img.unsqueeze(0).cuda() for img in imgs]
        g_feats, _, bn_f_feat, mid_feat = model(imgs)
        # pdb.set_trace()
        print([mid_feat[i].detach().cpu().mean() for i in range(3)])

def img_processing(path):
    path = path.replace("vis", "vis")
    im = cv2.imread(path)[:,:,::-1]
    # im = cv2.resize(im, dsize=(512, 256))
    
    plt.subplot(1, 2, 1)
    plt.imshow(im)

    # im2 = (im * 0.5).astype(np.uint8).clip(0, 255)
    im2 = cv2.GaussianBlur(im,ksize=(9,9,),sigmaX=3,sigmaY=3)

    plt.subplot(1, 2, 2)
    plt.imshow(im2)

    # plt.imsave(r"C:\Users\zxp\Desktop\imgs\vis\demo4.jpg", im2)

    plt.show()

# ----------------------- dehaze ---------------------------
def zmMinFilterGray(src, r=7):
    '''最小值滤波，r是滤波器半径'''
    return cv2.erode(src, np.ones((2 * r + 1, 2 * r + 1)))


def guidedfilter(I, p, r, eps):
    height, width = I.shape
    m_I = cv2.boxFilter(I, -1, (r, r))
    m_p = cv2.boxFilter(p, -1, (r, r))
    m_Ip = cv2.boxFilter(I * p, -1, (r, r))
    cov_Ip = m_Ip - m_I * m_p

    m_II = cv2.boxFilter(I * I, -1, (r, r))
    var_I = m_II - m_I * m_I

    a = cov_Ip / (var_I + eps)
    b = m_p - a * m_I

    m_a = cv2.boxFilter(a, -1, (r, r))
    m_b = cv2.boxFilter(b, -1, (r, r))
    return m_a * I + m_b


def Defog(m, r, eps, w, maxV1):                 # 输入rgb图像，值范围[0,1]
    '''计算大气遮罩图像V1和光照值A, V1 = 1-t/A'''
    V1 = np.min(m, 2)                           # 得到暗通道图像
    Dark_Channel = zmMinFilterGray(V1, 7)
    # cv2.imshow('20190708_Dark',Dark_Channel)    # 查看暗通道
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    V1 = guidedfilter(V1, Dark_Channel, r, eps)  # 使用引导滤波优化
    bins = 2000
    ht = np.histogram(V1, bins)                  # 计算大气光照A
    d = np.cumsum(ht[0]) / float(V1.size)
    for lmax in range(bins - 1, 0, -1):
        if d[lmax] <= 0.999:
            break
    A = np.mean(m, 2)[V1 >= ht[1][lmax]].max()
    V1 = np.minimum(V1 * w, maxV1)               # 对值范围进行限制
    return V1, A


def deHaze(m, r=81, eps=0.001, w=0.95, maxV1=0.80, bGamma=False):
    Y = np.zeros(m.shape)
    Mask_img, A = Defog(m, r, eps, w, maxV1)             # 得到遮罩图像和大气光照

    for k in range(3):
        Y[:,:,k] = (m[:,:,k] - Mask_img)/(1-Mask_img/A)  # 颜色校正
    Y = np.clip(Y, 0, 1)
    if bGamma:
        Y = Y ** (np.log(0.5) / np.log(Y.mean()))       # gamma校正,默认不进行该操作
    return Y

# ----------------------- dehaze ---------------------------

def img_enhance(path):
    ori_img = cv2.imread(path)
    plt.subplot(1, 2, 1)
    plt.imshow(ori_img[:,:,::-1])

    # img = 255 - ori_img
    # img = ori_img

    # img = img / 255.0
    # img = deHaze(img) * 255
    # img = img.astype(np.int8).clip(0, 255)

    img = ori_img / 255.0
    def norm(im):
        v_min = np.min(im)
        v_max = np.max(im)
        im = (im - v_min) / (v_max - v_min)
        return im
    for i in range(3):
        img[:,:,i] = norm(img[:,:,i])
    
    img = (img * 255).astype(np.uint8).clip(0, 255)[:,:,::-1]
    # img -= ori_img
    # img = 255 - img

    plt.subplot(1, 2, 2)
    plt.imshow(img)

    # plt.imsave(img,)
    # plt.imsave(r"C:\Users\zxp\Desktop\imgs\vis\dark2.jpg", img)

    plt.show()

if __name__ == "__main__":
    img = r"C:\Users\zxp\Desktop\imgs\vis\dark3.jpg"
    img = r"C:\Users\zxp\Desktop\imgs\vis\demo.jpg"
    # img = r"C:\Users\zxp\Desktop\imgs\haze.jpeg"
    # img = r"F:\cropped_data\dataset_test\query\0094\vis\0094_s023_v1_007.jpg"
    weights = r"outputs\modal_adanorm3\resnet50_model_800.pth"
    # weights = r"outputs_2\baseline_cls\resnet50_model_800.pth"

    run_once(get_imgs(img), weights)
    # # img_processing(img)
    # # img_enhance(img)

    # img = cv2.imread(img)
    # img = (deHaze(img / 255.0) * 255).astype(np.uint8).clip(0, 255)
    # plt.imshow(img)
    # plt.show()
    
