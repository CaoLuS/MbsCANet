"""
    @Author: Panke
    @Time: 2022-11-05  15:00
    @Email: None
    @File: test.py
    @Project: MbsCANet
"""
import configparser

import torch

x = torch.rand(8, 16, 7, 7)
y = torch.nn.Upsample((16, 16), )
print(y(x).shape)


