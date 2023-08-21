"""
    @Author: Panke
    @Time: 2023-04-06  16:59
    @Email: None
    @File: Heatmap.py
    @Project: MbsCANet
"""
import os
import uuid

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from src.utils.camutils import GradCAM, show_cam_on_image, center_crop_img
from src.models.blocks.MbsCABlock import MultiBranchFrequencyAttentionLayer
import torch.nn as nn
from torch.nn import functional as F


def heatmap(magnification, img_path):
    class ResNetDownBlock_BFca(nn.Module):
        def __init__(self, in_channels, out_channels, stride):
            super(ResNetDownBlock_BFca, self).__init__()
            c2wh = dict([(64, 56), (128, 28), (256, 14), (512, 7)])
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
            self.bn1 = nn.BatchNorm2d(out_channels)

            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
            self.bn2 = nn.BatchNorm2d(out_channels)

            self.extra = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
                nn.BatchNorm2d(out_channels)
            )
            self.fca_att = MultiBranchFrequencyAttentionLayer(out_channels, c2wh[out_channels], c2wh[out_channels])
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            extra_x = self.extra(x)

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.fca_att(out)
            out = self.relu(out + extra_x)

            return out

    class ResNetBasicBlock_BFca(nn.Module):
        def __init__(self, in_channels, out_channels, stride):
            super(ResNetBasicBlock_BFca, self).__init__()

            c2wh = dict([(64, 56), (128, 28), (256, 14), (512, 7)])
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
            self.bn1 = nn.BatchNorm2d(out_channels)

            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.fca_att = MultiBranchFrequencyAttentionLayer(out_channels, c2wh[out_channels], c2wh[out_channels])

        def forward(self, x):
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.fca_att(out)
            out = self.relu(out + x)
            return out

    class ResNet18_BFca(nn.Module):
        def __init__(self):
            super(ResNet18_BFca, self).__init__()
            self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
            self.bn = nn.BatchNorm2d(64)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.layer1 = nn.Sequential(ResNetBasicBlock_BFca(64, 64, 1),
                                        ResNetBasicBlock_BFca(64, 64, 1))

            self.layer2 = nn.Sequential(ResNetDownBlock_BFca(64, 128, [2, 1]),
                                        ResNetBasicBlock_BFca(128, 128, 1))

            self.layer3 = nn.Sequential(ResNetDownBlock_BFca(128, 256, [2, 1]),
                                        ResNetBasicBlock_BFca(256, 256, 1))

            self.layer4 = nn.Sequential(ResNetDownBlock_BFca(256, 512, [2, 1]),
                                        ResNetBasicBlock_BFca(512, 512, 1))

            self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

            self.linear = nn.Linear(512, 2)

        def forward(self, x):
            out = self.conv(x)
            out = self.bn(out)
            out = self.maxpool(out)
            # print(out.shape)
            out = self.layer1(out)
            # print(out.shape)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)

            out = self.avgpool(out)
            out = out.reshape(x.shape[0], -1)
            # print(out.shape)
            out = self.linear(out)
            return out

    net = nn.DataParallel(ResNet18_BFca()).cuda()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    weight_file = 'E:\CLFile\小论文\MbsCANet论文\PythonProject\MbsCANet\weights/' + magnification + '/weight.pth'
    net.load_state_dict(torch.load(weight_file))
    target_layers = [net.module.layer1[-1]]
    # print(target_layers)
    data_transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    # file_name = 'SOB_B_F-14-9133-100-034.png'
    # img_path = '/home/administrator/Caolu/BreastClassification/images/heatmap/img/100/' + file_name
    img_size = 224
    assert os.path.exists(img_path), 'file: {} dose not exist'.format(img_path)
    img = Image.open(img_path).convert('RGB').resize((224, 224))
    img = np.array(img, dtype=np.uint8)

    img = center_crop_img(img, img_size)
    # print(img.shape)
    # img = img.reshape((-1, 224, 224, 3))
    # print(type(img))

    img_tensor = data_transform(img)
    input_tensor = torch.unsqueeze(img_tensor, dim=0)
    cam = GradCAM(model=net, target_layers=target_layers, use_cuda=True)
    grayscale_cam = cam(input_tensor=input_tensor)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255,
                                      grayscale_cam,
                                      use_rgb=True)

    plt.imshow(visualization)
    plt.axis('off')  # 去坐标轴
    plt.xticks([])
    # plt.savefig(
    # '/home/administrator/Caolu/BreastClassification/images/heatmap/bfca/40/'+file_name)
    save_img_name = uuid.uuid1()
    save_img_path = 'E:\\CLFile\\DiasposeProject\\system_service\\src\\main\\resources\\heatmap_folder\\' + str(
        save_img_name) + '.png'
    plt.savefig(save_img_path, bbox_inches='tight', dpi=75, pad_inches=0.0)
    # plt.savefig("C:\q.jpg")
    # plt.show()
    return str(save_img_name) + '.png'

#
# if __name__ == '__main__':
#     heatmap('400', 'E:\CLFile\小论文\MbsCANet论文\PythonProject\MbsCANet\data\\400\SOB_M_DC-14-3909-400-002.png')
