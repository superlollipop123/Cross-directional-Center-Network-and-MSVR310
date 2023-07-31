from modeling.backbones.resnet import ResNet
from functools import reduce
import operator
import pdb
import torch
from torch import nn

class AutoEncoder(nn.Module):
    in_planes = 2048

    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.ae1 = BaseAutoEncoder()
        self.ae2 = BaseAutoEncoder()
        self.ae3 = BaseAutoEncoder()

    def forward(self, x):
        f0 = self.ae1(x[0])
        f1 = self.ae2(x[1])
        f2 = self.ae3(x[2])
        return [f0, f1, f2]
        

class BaseAutoEncoder(nn.Module):
    in_planes = 2048

    def __init__(self):
        super(BaseAutoEncoder, self).__init__()
        basenet  = ResNet(last_stride=1, layers=[3, 4, 6, 3])
        self.encoder = nn.Sequential(
            basenet.conv1,
            basenet.bn1,
            basenet.maxpool,
            basenet.layer1,
            basenet.layer2
        )

        self.decoder = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 3, 3, 1, 1, bias=False),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Tanh()
        )

    def forward(self, x):
        f = self.encoder(x)
        return self.decoder(f)