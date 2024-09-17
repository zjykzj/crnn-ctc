# -*- coding: utf-8 -*-

"""
@Time    : 2024/9/8 10:59
@File    : lprnet.py
@Author  : zj
@Description: 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, 0, 0.01)
        nn.init.constant_(module.bias, 0)


class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(small_basic_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )
        self.apply(init_weights)

    def forward(self, x):
        return self.block(x)


class small_basic_block_v2(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(small_basic_block_v2, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )
        self.shortcut = nn.Conv2d(ch_in, ch_out, kernel_size=1)
        self.relu = nn.ReLU()
        self.apply(init_weights)

    def forward(self, x):
        # 主路径
        residual = self.block(x)

        # 快捷路径
        shortcut = self.shortcut(x)

        # 残差连接
        out = residual + shortcut

        # 激活函数
        out = self.relu(out)

        return out


class LPRNet(nn.Module):
    def __init__(self, num_classes, in_channel=3, dropout_rate=0.5, use_origin_block=False, add_stnet=False):
        super(LPRNet, self).__init__()
        self.num_classes = num_classes

        if use_origin_block:
            small_block = small_basic_block
        else:
            small_block = small_basic_block_v2

        self.add_stnet = add_stnet
        if self.add_stnet:
            from utils.model.stnet import STNet
            self.stnet = STNet()

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=3, stride=1),  # 0
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),  # 2
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),
            small_block(ch_in=64, ch_out=128),  # *** 4 ***
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),  # 6
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)),
            small_block(ch_in=64, ch_out=256),  # 8
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 10
            small_block(ch_in=256, ch_out=256),  # *** 11 ***
            nn.BatchNorm2d(num_features=256),  # 12
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)),  # 14
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 4), stride=1),  # 16
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 18
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=(13, 1), stride=1),  # 20
            nn.BatchNorm2d(num_features=num_classes),
            nn.ReLU(),  # *** 22 ***
        )
        self.container = nn.Sequential(
            nn.Conv2d(in_channels=448 + self.num_classes, out_channels=self.num_classes, kernel_size=(1, 1),
                      stride=(1, 1)),
        )

        self.apply(init_weights)

    def forward(self, x):
        if self.add_stnet:
            x = self.stnet(x)

        keep_features = list()
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if i in [2, 6, 13, 22]:  # [2, 4, 8, 11, 22]
                keep_features.append(x)

        global_context = list()
        for i, f in enumerate(keep_features):
            if i in [0, 1]:
                f = nn.AvgPool2d(kernel_size=5, stride=5)(f)
            if i in [2]:
                f = nn.AvgPool2d(kernel_size=(4, 10), stride=(4, 2))(f)
            f_pow = torch.pow(f, 2)
            f_mean = torch.mean(f_pow)
            f = torch.div(f, f_mean)
            global_context.append(f)

        x = torch.cat(global_context, 1)
        # [N, N_Class+448, H, W] -> [N, N_Class, H, W]
        x = self.container(x)
        # [N, N_Class, H, W] -> [N, N_Class, W]
        x = torch.mean(x, dim=2)

        # [N, N_Class, W] -> [N, W, N_Class]
        x = x.permute(0, 2, 1).contiguous()
        logits = F.log_softmax(x, dim=-1)

        return logits


def test_model(data, model, device):
    print(data.shape)
    # Warmup
    for _ in range(3):
        model(data.to(device))
    data = data.to(device)

    t_start = time.time()
    output = model(data)
    t_end = time.time()
    print(f"time: {t_end - t_start}")
    print(data.shape, output.shape)


if __name__ == '__main__':
    import time
    import copy

    data = torch.randn(5, 3, 24, 94)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # LPRNetPlus
    model = LPRNet(num_classes=100, in_channel=3, use_origin_block=False).to(device)
    test_model(copy.deepcopy(data), model, device)

    # LPRNet
    model = LPRNet(num_classes=100, in_channel=3, use_origin_block=True).to(device)
    test_model(copy.deepcopy(data), model, device)

    # LPRNetPlus + STNet
    model = LPRNet(num_classes=100, in_channel=3, use_origin_block=False, add_stnet=True).to(device)
    test_model(copy.deepcopy(data), model, device)
