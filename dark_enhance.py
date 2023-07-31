from datetime import datetime
import numpy as np
import cv2
from math import ceil
from scipy.sparse import spdiags
# from scipy.misc import imresize
from scipy.optimize import fminbound
from scipy.stats import entropy
from scipy.sparse.linalg import spsolve

from PIL import Image

filepath = r"C:\Users\zxp\Desktop\imgs\th\dark.jpg"

def imresize(img, size, *args, **kargs):
    # img = np.array(Image.fromarray(img).resize((size[1], size[0])))
    # img = cv2.resize(img, size)
    img = np.array(Image.fromarray(img).resize(size))
    return img

def computeTextureWeights(fin, sigma, sharpness):
    # print(fin)
    # fin = fin / 255.0

    dt0_v = np.diff(fin, 1, 0)  # 垂直差分
    dt0_v = np.concatenate((dt0_v, fin[:1, :] - fin[-1:, :]), axis=0)  # 第0行减去最后一行

    dt0_h = np.diff(fin, 1, 1)  # 水平差分
    dt0_h = np.concatenate((dt0_h, fin[:, :1] - fin[:, -1:]), axis=1)  # 第0列减去最后一列

    gauker_h = cv2.filter2D(dt0_h, -1, np.ones((1, sigma)), borderType=cv2.BORDER_CONSTANT)
    gauker_v = cv2.filter2D(dt0_v, -1, np.ones((sigma, 1)), borderType=cv2.BORDER_CONSTANT)
    # cv2这个filter2D（镜像翻转）与MATLAB的filter2（补0）不同

    W_h = 1.0 / (abs(gauker_h) * abs(dt0_h) + sharpness)
    W_v = 1.0 / (abs(gauker_v) * abs(dt0_v) + sharpness)

    return W_h, W_v


def convertCol(tmp):  # 按照列转成列。[[1, 2, 3], [4, 5, 6], [7, 8, 9]] # 转成[147258369].T(竖着)
    return np.reshape(tmp.T, (tmp.shape[0] * tmp.shape[1], 1))


def solveLinearEquation(IN, wx, wy, lambd):
    print('IN', IN.shape)
    r, c, ch = IN.shape[0], IN.shape[1], 1
    k = r * c
    dx = -lambd * convertCol(wx)  # 按列转成一列
    dy = -lambd * convertCol(wy)
    tempx = np.concatenate((wx[:, -1:], wx[:, 0:-1]), 1)  # 最后一列插入到第一列前面
    tempy = np.concatenate((wy[-1:, :], wy[0:-1, :]), 0)  # 最后一行插入到第一行前面
    dxa = -lambd * convertCol(tempx)
    dya = -lambd * convertCol(tempy)
    tempx = np.concatenate((wx[:, -1:], np.zeros((r, c - 1))), 1)  # 取wx最后一列放在第一列，其他为0
    tempy = np.concatenate((wy[-1:, :], np.zeros((r - 1, c))), 0)  # 取wy最后一行放在第一行，其他为0
    dxd1 = -lambd * convertCol(tempx)
    dyd1 = -lambd * convertCol(tempy)
    wx[:, -1:] = 0  # 最后一列置为0
    wy[-1:, :] = 0  # 最后一行置为0
    dxd2 = -lambd * convertCol(wx)
    dyd2 = -lambd * convertCol(wy)

    Ax = spdiags(np.concatenate((dxd1, dxd2), 1).T, np.array([-k + r, -r]), k, k)
    Ay = spdiags(np.concatenate((dyd1, dyd2), 1).T, np.array([-r + 1, -1]), k, k)
    # spdiags，与MATLAB不同，scipy是根据行来构造sp，而matlab是根据列来构造sp

    D = 1 - (dx + dy + dxa + dya)
    A = (Ax + Ay) + (Ax + Ay).T + spdiags(D.T, np.array([0]), k, k)

    A = A / 1000.0  # 需修改

    matCol = convertCol(IN)
    print('spsolve开始', str(datetime.now()))
    OUT = spsolve(A, matCol, permc_spec="MMD_AT_PLUS_A")
    print('spsolve结束', str(datetime.now()))
    OUT = OUT / 1000
    OUT = np.reshape(OUT, (c, r)).T
    return OUT


def tsmooth(I, lambd=0.5, sigma=5, sharpness=0.001):
    # print(I.shape)
    wx, wy = computeTextureWeights(I, sigma, sharpness)
    S = solveLinearEquation(I, wx, wy, lambd)
    return S


def rgb2gm(I):
    print('I', I.shape)
    # I = np.maximum(I, 0.0)
    if I.shape[2] and I.shape[2] == 3:
        I = np.power(np.multiply(np.multiply(I[:, :, 0], I[:, :, 1]), I[:, :, 2]), (1.0 / 3))
    return I


def YisBad(Y, isBad):  # 此处需要修改得更高效
    return Y[isBad >= 1]
    # Z = []
    # [rows, cols] = Y.shape
    # for i in range(rows):
    #     for j in range(cols):
    #         if isBad[i, j] >= 122:
    #             Z.append(Y[i, j])
    # return np.array([Z]).T


def applyK(I, k, a=-0.3293, b=1.1258):
    # print(type(I))
    if not type(I) == 'numpy.ndarray':
        I = np.array(I)
    # print(type(I))
    beta = np.exp((1 - (k ** a)) * b)
    gamma = (k ** a)
    BTF = np.power(I, gamma) * beta
    # try:
    #    BTF = (I ** gamma) * beta
    # except:
    #    print('gamma:', gamma, '---beta:', beta)
    #    BTF = I
    return BTF


def maxEntropyEnhance(I, isBad, mink=1, maxk=10):
    # Y = rgb2gm(np.real(np.maximum(imresize(I, (50, 50), interp='bicubic') / 255.0, 0)))
    # import pdb; pdb.set_trace()
    # Y = imresize(I, (50, 50), interp='bicubic') / 255.0
    Y = np.resize(I, (50, 50))
    Y = rgb2gm(Y)
    # bicubic较为接近
    # Y = rgb2gm(np.real(np.maximum(cv2.resize(I, (50, 50), interpolation=cv2.INTER_LANCZOS4  ), 0)))
    # INTER_AREA 较为接近
    # import matplotlib.pyplot as plt
    # plt.imshow(Y, cmap='gray');plt.show()

    print('isBad', isBad.shape)
    isBad = imresize(isBad.astype(int), (50, 50), interp='nearest')
    print('isBad', isBad.shape)

    # plt.imshow(isBad, cmap='gray');plt.show()

    # 取出isBad为真的Y的值，形成一个列向量Y
    # Y = YisBad(Y, isBad)  # 此处需要修改得更高效
    Y = Y[isBad >= 1]

    # Y = sorted(Y)

    print('-entropy(Y)', -entropy(Y))

    def f(k):
        return -entropy(applyK(Y, k))

    # opt_k = mink
    # k = mink
    # minF = f(k)
    # while k<= maxk:
    #     k+=0.0001
    #     if f(k)<minF:
    #         minF = f(k)
    #         opt_k = k
    opt_k = fminbound(f, mink, maxk)
    print('opt_k:', opt_k)
    # opt_k = 5.363584
    # opt_k = 0.499993757705
    # optk有问题，fminbound正确，f正确，推测Y不一样导致不对
    print('opt_k:', opt_k)
    J = applyK(I, opt_k) - 0.01
    return J


def HDR2dark(I, t_our, W):  # 过亮的地方变暗
    W = 1 - W
    I3 = I * W
    isBad = t_our > 0.8
    J3 = maxEntropyEnhance(I, isBad, 0.1, 0.5)  # 求k和曝光图
    J3 = J3 * (1 - W)  # 曝光图*权重
    fused = I3 + J3  # 增强图
    return I


def oneHDR(I, mu=0.5, a=-0.3293, b=1.1258):
    # mu照度图T的指数，数值越大，增强程度越大
    I = I / 255.0
    t_b = I[:, :, 0]  # t_b表示三通道图转成灰度图（灰度值为RGB中的最大值）,亮度矩阵L
    for i in range(I.shape[2] - 1):  # 防止输入图片非三通道
        t_b = np.maximum(t_b, I[:, :, i + 1])
    # t_b2 = cv2.resize(t_b, (0, 0), fx=0.5, fy=0.5)
    print('t_b', t_b.shape)
    # t_b2 = misc.imresize(t_b, (ceil(t_b.shape[0] / 2), ceil(t_b.shape[1] / 2)),interp='bicubic')
    # print('t_b2', t_b2.shape)
    # t_b2 = t_b / 255.0
    # import pdb; pdb.set_trace()
    t_b2 = imresize(t_b, (256, 256), interp='bicubic', mode='F')  # / 255
    t_our = tsmooth(t_b2)  # 求解照度图T（灰度图）
    print('t_our前', t_our.shape)
    t_our = imresize(t_our, t_b.shape, interp='bicubic', mode='F')  # / 255
    print('t_our后', t_our.shape)

    # W: Weight Matrix 与 I2
    # 照度图L（灰度图） ->  照度图L（RGB图）：灰度值重复3次赋给RGB
    # size为(I, 3) ， 防止与原图尺寸有偏差
    t = np.ndarray(I.shape)
    for ii in range(I.shape[2]):
        t[:, :, ii] = t_our.T
    print('t', t.shape)

    W = t ** mu  # 原图的权重。三维矩阵

    # cv2.imwrite(filepath + 'W.jpg', W * 255)
    # cv2.imwrite(filepath + '1-W.jpg', (1 - W) * 255)
    # cv2.imwrite(filepath + 't.jpg', t * 255)
    # cv2.imwrite(filepath + '1-t.jpg', (1 - t) * 255)

    print('W', W.shape)
    # 变暗
    # isBad = t_our > 0.8  # 是高光照的像素点
    # I = maxEntropyEnhance(I, isBad)  # 求k和曝光图
    # 变暗
    I2 = I * W  # 原图*权重
    # 曝光率->k ->J
    isBad = t_our < 0.5  # 是低光照的像素点
    J = maxEntropyEnhance(I, isBad)  # 求k和曝光图
    J2 = J * (1 - W)  # 曝光图*权重
    fused = I2 + J2  # 增强图
    # 存储中间结果
    # cv2.imwrite(filepath + 'I2.jpg', I2 * 255.0)
    # cv2.imwrite(filepath + 'J2.jpg', J2 * 255.0)
    # 变暗
    # fused = HDR2dark(fused, t_our, W)
    return fused
    # return res

def test():
    inputImg = cv2.imread(filepath)
    outputImg = oneHDR(inputImg)
    # outputImg = outputImg * 255.0
    cv2.imwrite(filepath + '1out.bmp', outputImg * 255)
    print("HDR完成，已保存到本地")
    print('程序结束', str(datetime.now()))

    cv2.imshow('inputImg', inputImg)
    cv2.imshow('outputImg', outputImg)
    # print(inputImg.dtype,outputImg.dtype)
    # outputImg = outputImg.astype(int)
    # print(inputImg.dtype, outputImg.dtype)
    # compare = np.concatenate((inputImg,outputImg),axis=1)
    # cv2.imshow('compare', compare)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test2():
    A = np.array([1, 2, 3, 4])
    B = np.array([1, 0, 0, 1])
    print(A[B>0])


# ---------------------------------------------
# https://blog.csdn.net/u014134327/article/details/102808184?utm_medium=distribute.pc_relevant.none-task-blog-baidujs_title-0&spm=1001.2101.3001.4242

def MinRgb(c):
    return min(c[0], c[1], c[2])

def SumRgb(c):
    return c[0] + c[1] + c[2]

def Invert(img):
    img = 255 - img
    return img

def GetA(R, G, B, k = 100):
    # k默认是原文获取排序后前100个像素点
    rlist = []
    height, width = R.shape[0], R.shape[1]
    for hi in range(height):
        for wi in range(width):
            rlist.append([R[hi][wi], G[hi][wi], B[hi][wi]])
    rlist.sort(key=MinRgb)
    rlist.reverse()
    rlist = rlist[:k]
    rlist.sort(key=SumRgb)
    rlist.reverse()
    return rlist[0][0], rlist[0][1], rlist[0][2]

def CalT(R, G, B, r_A, g_A, b_A, size=1, w=0.76):
    # 计算A值时使用size×size窗口，以图像边缘点为窗口中心时需要进行填充
    # 图像填充时上下左右各填充1行/列255
    ts = (size - 1) // 2
    height, width = R.shape[0], R.shape[1]
    R_f = np.pad(R, ((ts, ts), (ts, ts)), 'constant', constant_values=(255, 255)) / r_A
    G_f = np.pad(G, ((ts, ts), (ts, ts)), 'constant', constant_values=(255, 255)) / g_A
    B_f = np.pad(B, ((ts, ts), (ts, ts)), 'constant', constant_values=(255, 255)) / b_A

    shape = (height, width, size, size)
    strides = R_f.itemsize * np.array([width + ts * 2, 1, width + ts * 2, 1])

    blocks_R = np.lib.stride_tricks.as_strided(R_f, shape=shape, strides=strides)
    blocks_G = np.lib.stride_tricks.as_strided(G_f, shape=shape, strides=strides)
    blocks_B = np.lib.stride_tricks.as_strided(B_f, shape=shape, strides=strides)

    t = np.zeros((height, width))
    for hi in range(height):
        for wi in range(width):
            t[hi, wi] = 1- w * min(np.min(blocks_R[hi, wi]), np.min(blocks_G[hi, wi]), np.min(blocks_B[hi, wi]))
            if t[hi, wi] < 0.5:
                t[hi, wi] = 2 * t[hi, wi] * t[hi, wi]
    return t

def DeHaze(filepath):
    # 根据路径读取照片
    img = Image.open(filepath)
    # 获取图像宽度、高度
    # width, height = img.size
    # 获取图像的RGB数组
    img = np.asarray(img, dtype=np.int32)
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    R, G, B = Invert(R), Invert(G), Invert(B)
    # 计算A值
    r_A, g_A, b_A = GetA(R, G, B)
    t = CalT(R, G, B, r_A, g_A, b_A)
    J_R = (R - r_A) / t + r_A
    J_G = (G - g_A) / t + g_A
    J_B = (B - b_A) / t + b_A
    J_R, J_G, J_B = Invert(J_R), Invert(J_G), Invert(J_B)
    r = Image.fromarray(J_R).convert('L')
    g = Image.fromarray(J_G).convert('L')
    b = Image.fromarray(J_B).convert('L')
    image = Image.merge("RGB", (r, g, b))
    
    # image.save("./dark_result.jpg")
    image.save(r"C:\Users\zxp\Desktop\imgs\th\dark3.jpg")


    image.show()



if __name__ == '__main__':
    # print('程序开始', str(datetime.now()))
    # test()

    DeHaze(filepath=filepath)