import torch
import torch.nn as nn

import torchvision.models as models


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


class VGG16_Activations(nn.Module):
    def __init__(self, feature_idx):
        super(VGG16_Activations, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        features = list(vgg16.features)
        self.features = nn.ModuleList(features).eval()
        self.layer_id_list = feature_idx

    def forward(self, x):
        activations = []
        for i, model in enumerate(self.features):
            x = model(x)
            if i in self.layer_id_list:
                activations.append(x)

        return activations


class VGG19_Activations(nn.Module):
    def __init__(self, feature_idx, requires_grad=False):
        super(VGG19_Activations, self).__init__()
        vgg19 = models.vgg19(pretrained=True)
        requires_grad(vgg19, flag=False)
        features = list(vgg19.features)
        self.features = nn.ModuleList(features).eval()
        self.layer_id_list = feature_idx

    def forward(self, x):
        activations = []
        for i, model in enumerate(self.features):
            x = model(x)
            if i in self.layer_id_list:
                activations.append(x)

        return activations


# http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/vgg_face_dag.py
class Vgg_face_dag(nn.Module):
    def __init__(self):
        super(Vgg_face_dag, self).__init__()
        self.meta = {
            "mean": [129.186279296875, 104.76238250732422, 93.59396362304688],
            "std": [1, 1, 1],
            "imageSize": [224, 224, 3],
        }
        self.conv1_1 = nn.Conv2d(
            3, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)
        )
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(
            64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)
        )
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(
            kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False
        )
        self.conv2_1 = nn.Conv2d(
            64, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)
        )
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(
            128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)
        )
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(
            kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False
        )
        self.conv3_1 = nn.Conv2d(
            128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)
        )
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(
            256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)
        )
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(
            256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)
        )
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(
            kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False
        )
        self.conv4_1 = nn.Conv2d(
            256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)
        )
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(
            512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)
        )
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(
            512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)
        )
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(
            kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False
        )
        self.conv5_1 = nn.Conv2d(
            512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)
        )
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(
            512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)
        )
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(
            512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)
        )
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(
            kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False
        )
        self.fc6 = nn.Linear(in_features=25088, out_features=4096, bias=True)
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout6 = nn.Dropout(p=0.5)
        self.fc7 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout7 = nn.Dropout(p=0.5)
        self.fc8 = nn.Linear(in_features=4096, out_features=2622, bias=True)

    def forward(self, x):
        activations = []
        x1 = self.conv1_1(x)
        activations.append(x1)

        x2 = self.relu1_1(x1)
        x3 = self.conv1_2(x2)
        x4 = self.relu1_2(x3)
        x5 = self.pool1(x4)
        x6 = self.conv2_1(x5)
        activations.append(x6)

        x7 = self.relu2_1(x6)
        x8 = self.conv2_2(x7)
        x9 = self.relu2_2(x8)
        x10 = self.pool2(x9)
        x11 = self.conv3_1(x10)
        activations.append(x11)

        x12 = self.relu3_1(x11)
        x13 = self.conv3_2(x12)
        x14 = self.relu3_2(x13)
        x15 = self.conv3_3(x14)
        x16 = self.relu3_3(x15)
        x17 = self.pool3(x16)
        x18 = self.conv4_1(x17)
        activations.append(x18)

        x19 = self.relu4_1(x18)
        x20 = self.conv4_2(x19)
        x21 = self.relu4_2(x20)
        x22 = self.conv4_3(x21)
        x23 = self.relu4_3(x22)
        x24 = self.pool4(x23)
        x25 = self.conv5_1(x24)
        activations.append(x25)

        """
        x26 = self.relu5_1(x25)
        x27 = self.conv5_2(x26)
        x28 = self.relu5_2(x27)
        x29 = self.conv5_3(x28)
        x30 = self.relu5_3(x29)
        x31_preflatten = self.pool5(x30)
        x31 = x31_preflatten.view(x31_preflatten.size(0), -1)
        x32 = self.fc6(x31)
        x33 = self.relu6(x32)
        x34 = self.dropout6(x33)
        x35 = self.fc7(x34)
        x36 = self.relu7(x35)
        x37 = self.dropout7(x36)
        x38 = self.fc8(x37)
        """

        return activations
