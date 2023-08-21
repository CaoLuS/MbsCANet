"""
    @Author: Panke
    @Time: 2022-10-31  16:07
    @Email: None
    @File: MbsCANetDownBlock.py
    @Project: MbsCANet
"""
import torch.nn as nn
from src.models.blocks.MbsCABlock import MultiBranchFrequencyAttentionLayer


class MbsCANetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(MbsCANetDownBlock, self).__init__()
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
