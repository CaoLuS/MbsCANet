"""
    @Author: Panke
    @Time: 2022-10-31  16:09
    @Email: None
    @File: MbsCABlock.py
    @Project: MbsCANet
"""

import math
import torch
import torch.nn as nn

print("-----Load MbsCA Block-----")


class MultiBranchFrequencyAttentionLayer(nn.Module):
    """
        Multi branch frequency channel attention
    """

    def __init__(self, in_channel, dct_h, dct_w, reduction=16):
        """
            init method
        :param in_channel: Number of input channels
        :param dct_h: High of dct
        :param dct_w: Width of dct
        :param reduction: Reduction in ResNet
        """
        super(MultiBranchFrequencyAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_w = dct_w
        self.dct_h = dct_h
        component_x1, component_y1 = [0], [0]
        component_x2, component_y2 = [6], [3]
        component_x3, component_y3 = [1], [6]
        self.b1 = BranchLayer(channel=in_channel, dct_h=self.dct_h, dct_w=self.dct_w,
                              component_x=component_x1, component_y=component_y1, reduction=self.reduction)
        self.b2 = BranchLayer(channel=in_channel, dct_h=self.dct_h, dct_w=self.dct_w,
                              component_x=component_x2, component_y=component_y2, reduction=self.reduction)
        self.b3 = BranchLayer(channel=in_channel, dct_h=self.dct_h, dct_w=self.dct_w,
                              component_x=component_x3, component_y=component_y3, reduction=self.reduction)

    def forward(self, x):
        x_pooled = x
        b1 = self.b1(x_pooled)
        b2 = self.b2(x_pooled)
        b3 = self.b3(x_pooled)
        return (1 / 3) * (b1 + b2 + b3)


class BranchLayer(nn.Module):
    def __init__(self, channel, dct_w, dct_h, component_x, component_y, reduction):
        super(BranchLayer, self).__init__()
        self.reduction = reduction
        self.dct_w = dct_w
        self.dct_h = dct_h
        component_x = [item * (dct_h // 7) for item in component_x]
        component_y = [item * (dct_w // 7) for item in component_y]
        self.dct_layer = DCTLayer(dct_h, dct_w, component_x, component_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = x
        y = self.dct_layer(x_pooled)
        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


class DCTLayer(nn.Module):
    def __init__(self, height, width, component_x, component_y, channel):
        super(DCTLayer, self).__init__()
        assert len(component_x) == len(component_y)
        assert channel % len(component_x) == 0
        self.num_freq = len(component_x)
        self.register_buffer("weight", self.get_dct_filter(height, width, component_x, component_y, channel))

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got' + str(len(x.shape))
        x = x * self.weight
        result = torch.sum(x, dim=[2, 3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, component_x, component_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(component_x)
        for i, (u_x, v_y) in enumerate(zip(component_x, component_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part:(i + 1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x,
                                                                                          tile_size_x) * self.build_filter(
                        t_y, v_y, tile_size_y)
        return dct_filter
