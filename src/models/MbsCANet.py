"""
    @Author: Panke
    @Time: 2022-10-31  18:55
    @Email: None
    @File: MbsCANet.py
    @Project: MbsCANet
"""
import torch.nn as nn
from src.models.blocks.MbsCANetBasicBlock import MbsCANetBasicBlock
from src.models.blocks.MbsCANetDownBlock import MbsCANetDownBlock


class MbsCANet(nn.Module):
    def __init__(self):
        super(MbsCANet, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(MbsCANetBasicBlock(64, 64, 1),
                                    MbsCANetBasicBlock(64, 64, 1))

        self.layer2 = nn.Sequential(MbsCANetDownBlock(64, 128, [2, 1]),
                                    MbsCANetBasicBlock(128, 128, 1))

        self.layer3 = nn.Sequential(MbsCANetDownBlock(128, 256, [2, 1]),
                                    MbsCANetBasicBlock(256, 256, 1))

        self.layer4 = nn.Sequential(MbsCANetDownBlock(256, 512, [2, 1]),
                                    MbsCANetBasicBlock(512, 512, 1))

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
