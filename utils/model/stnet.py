# -*- coding: utf-8 -*-

"""
@date: 2024/9/17 下午9:50
@file: stnet.py
@author: zj
@description:

# Author:电子科技大学刘俊凯、陈昂
# https://github.com/JKLinUESTC/License-Plate-Recognization-Pytorch

"""

import torch.nn as nn
import torch
import torch.nn.functional as F


class STNet(nn.Module):

    def __init__(self):
        super(STNet, self).__init__()

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=5),
            nn.MaxPool2d(3, stride=3),
            nn.ReLU(True)
        )
        # Regressor for the 3x2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(32 * 14 * 2, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 32 * 14 * 2)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        # /home/zj/anaconda3/envs/yolov5/lib/python3.8/site-packages/torch/nn/functional.py:3828: UserWarning: Default grid_sample and affine_grid behavior has changed to align_corners=False since 1.3.0. Please specify align_corners=True if the old behavior is desired. See the documentation of grid_sample for details.
        #   warnings.warn(
        # align_corners=False is better than align_corners=True
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid)

        return x


if __name__ == '__main__':
    model = STNet()

    a = torch.randn(4, 3, 24, 94)
    output = model(a)
    print(a.shape, output.shape)
