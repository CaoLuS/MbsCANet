"""
    @Author: Panke
    @Time: 2022-10-31  16:08
    @Email: None
    @File: MbsCANetBasicBlock.py
    @Project: MbsCANet
"""

import torch.nn as nn
from src.models.blocks.MbsCABlock import MultiBranchFrequencyAttentionLayer


class MbsCANetBasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(MbsCANetBasicBlock, self).__init__()
        c2wh = dict([(64, 56), (128, 28), (256, 14), (512, 7)])
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.fca_att = MultiBranchFrequencyAttentionLayer(out_channel, c2wh[out_channel], c2wh[out_channel])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.fca_att(out)
        out = self.relu(out + x)
        return out
