import argparse
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models
import cv2
# from model import Resnet18, load_params, CrossAttentionNet

import sys, os
sys.path.append(".")
import pdb


class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers, modal):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)
        if modal == "rgb":
            self.branch = self.model.branch_0
        elif modal == "nir":
            self.branch = self.model.branch_1
        elif modal == "tir":
            self.branch = self.model.branch_2
        else:
            assert 0

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        # pdb.set_trace()

        branch = self.branch
        x = branch.base.conv1(x)
        x = branch.base.bn1(x)
        x = branch.base.maxpool(x)
        x = branch.base.layer1(x)
        x = branch.base.layer2(x)
        x, _ = branch.modalnorm(x)
        x = branch.base.layer3(x)
        target_activations, x = self.feature_extractor(x)
        x = branch.gap(x)
        x = x.view(x.shape[0], -1)
        x = branch.bottleneck(x)
        x = branch.classifier(x)
        
        return target_activations, x

import torchvision.transforms as T
def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    # normalize_transform = T.Normalize(mean=means, std=stds)
    # transform = T.Compose([
    #     T.Resize((128, 256)),
    #     T.ToTensor(),
    #     normalize_transform
    # ])

    # x = transform(img)
    # pdb.set_trace()

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    # pdb.set_trace()
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    # cv2.imwrite("cam.jpg", np.uint8(255 * cam))
    # cv2.imshow("cam", cam)
    # cv2.waitKey(0)

    return cam


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda, modal="rgb"):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names, modal=modal)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]
        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input.permute([0, 1, 3, 2]).shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

from v.grad_cam.model_for_cam import MainNet as TestModel
def get_CAM_for_trainset():
    use_cuda = True
    model = TestModel(num_classes=155)
    # load_params(model, path=r'output/sm_n300/net_119.pth')
    model.load_param(trained_path=r"outputs\ALNU_CdC\resnet50_model_800.pth")
    grad_cam_rgb = GradCam(model=model, feature_module=model.branch_0.base.layer4, \
                       target_layer_names=["1"], use_cuda=use_cuda, modal="rgb")
    grad_cam_nir = GradCam(model=model, feature_module=model.branch_1.base.layer4, \
                       target_layer_names=["1"], use_cuda=use_cuda, modal="nir")
    grad_cam_tir = GradCam(model=model, feature_module=model.branch_2.base.layer4, \
                       target_layer_names=["1"], use_cuda=use_cuda, modal="tir")
    
    img_list = []
    root = r"F:\cropped_data\dataset_test\bounding_box_train"
    for id_folder in os.listdir(root):
        vis_id_path = os.path.join(root, id_folder, 'vis')
        for img in os.listdir(vis_id_path):
            img_list.append(os.path.join(vis_id_path, img))

    def get_img(image_path):
        img = cv2.imread(image_path, 1)
        img = cv2.resize(img, (256, 128))
        img = np.float32(img) / 255
        return img, preprocess_image(img)

    for i, path in enumerate(img_list):
        # if i < 160: continue
        r_path = path
        n_path = path.replace('vis', 'ni')
        t_path = path.replace('vis', 'th')
        
        ori_r_img, r_img = get_img(r_path)
        ori_n_img, n_img = get_img(n_path)
        ori_t_img, t_img = get_img(t_path)

        r_mask = grad_cam_rgb(r_img, None)
        n_mask = grad_cam_nir(n_img, None)
        t_mask = grad_cam_tir(t_img, None)

        final_img_r = show_cam_on_image(ori_r_img, r_mask)
        final_img_n = show_cam_on_image(ori_n_img, n_mask)
        final_img_t = show_cam_on_image(ori_t_img, t_mask)

        final_img = np.concatenate([final_img_r, final_img_n, final_img_t], axis=0)

        cv2.imwrite(os.path.join(r'v\grad_cam\CdC_ALNU', path[-20:]), final_img)
        print(path[-20:])
        # pdb.set_trace()
        # cv2.imshow("fig", final_img)
        # cv2.waitKey(0)
        # break


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args

if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    # args = get_args()
    # # zxp
    # args.use_cuda = True
    # args.image_path = r"F:\cropped_data\dataset_test\bounding_box_train\0015\vis\0015_s017_v5_008.jpg"

    # # Can work with any model, but it assumes that the model has a
    # # feature method, and a classifier method,
    # # as in the VGG models in torchvision.
    # model = TestModel(num_classes=155)
    # # load_params(model, path=r'output/sm_n300/net_119.pth')
    # model.load_param(trained_path=r"outputs\qualityNorm\resnet50_model_800.pth")
    # grad_cam = GradCam(model=model, feature_module=model.branch_0.base.layer4, \
    #                    target_layer_names=["1"], use_cuda=args.use_cuda, modal="rgb")

    # img = cv2.imread(args.image_path, 1)
    # img = cv2.resize(img, (256, 128))
    # img = np.float32(img) / 255
    # input = preprocess_image(img)

    # # If None, returns the map for the highest scoring category.
    # # Otherwise, targets the requested index.
    # target_index = None
    # mask = grad_cam(input, target_index)

    # show_cam_on_image(img, mask)
    get_CAM_for_trainset()
    