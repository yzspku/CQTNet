import torch
from torch import nn
import torchvision
import torch.nn.functional as F
from collections import OrderedDict
import math
from .basic_module import BasicModule

class CQT_Net(BasicModule):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, 32, kernel_size=(12, 3), dilation=(1, 1), padding=(6, 0), bias=False)),
            ('norm0', nn.BatchNorm2d(32)), ('relu0', nn.ReLU(inplace=True)),
            ('conv1', nn.Conv2d(32, 64, kernel_size=(13, 3), dilation=(1, 2), bias=False)),
            ('norm1', nn.BatchNorm2d(64)), ('relu1', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 1))),

            ('conv2', nn.Conv2d(64, 64, kernel_size=(13, 3), dilation=(1, 1), bias=False)),
            ('norm2', nn.BatchNorm2d(64)), ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(64, 64, kernel_size=(3, 3), dilation=(1, 2), bias=False)),
            ('norm3', nn.BatchNorm2d(64)), ('relu3', nn.ReLU(inplace=True)),
            ('pool3', nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 1))),

            ('conv4', nn.Conv2d(64, 128, kernel_size=(3, 3), dilation=(1, 1), bias=False)),
            ('norm4', nn.BatchNorm2d(128)), ('relu4', nn.ReLU(inplace=True)),
            ('conv5', nn.Conv2d(128, 128, kernel_size=(3, 3), dilation=(1, 2), bias=False)),
            ('norm5', nn.BatchNorm2d(128)), ('relu5', nn.ReLU(inplace=True)),
            ('pool5', nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 1))),

            ('conv6', nn.Conv2d(128, 256, kernel_size=(3, 3), dilation=(1, 1), bias=False)),
            ('norm6', nn.BatchNorm2d(256)), ('relu6', nn.ReLU(inplace=True)),
            ('conv7', nn.Conv2d(256, 256, kernel_size=(3, 3), dilation=(1, 2), bias=False)),
            ('norm7', nn.BatchNorm2d(256)), ('relu7', nn.ReLU(inplace=True)),
            ('pool7', nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 1))),

            ('conv8', nn.Conv2d(256, 512, kernel_size=(3, 3), dilation=(1, 1), bias=False)),
            ('norm8', nn.BatchNorm2d(512)), ('relu8', nn.ReLU(inplace=True)),
            ('conv9', nn.Conv2d(512, 512, kernel_size=(3, 3), dilation=(1, 2), bias=False)),
            ('norm9', nn.BatchNorm2d(512)), ('relu9', nn.ReLU(inplace=True)),
        ]))
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc0 = nn.Linear(512, 300)
        self.fc1 = nn.Linear(300, 10000)

    def forward(self, x):
        # input [N, C, H, W] (W = 396)
        N = x.size()[0]
        x = self.features(x)  # [N, 512, 57, 2~15]
        x = self.pool(x)
        x = x.view(N, -1)
        feature = self.fc0(x)
        x = self.fc1(feature)
        return x, feature
