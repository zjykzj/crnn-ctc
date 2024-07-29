# -*- coding: utf-8 -*-

"""
@date: 2023/10/11 上午11:22
@file: crnn_gru.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CRNN(nn.Module):

    def __init__(self, in_channel, num_classes, cnn_output_height, use_gru=False):
        super().__init__()

        # 特征提取层
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        rnn_input_size = 512 * cnn_output_height
        rnn_hidden_size = rnn_input_size // 2
        if use_gru:
            self.rnn = nn.GRU(input_size=rnn_input_size, hidden_size=rnn_hidden_size, num_layers=2, batch_first=True,
                              bidirectional=True)
        else:
            self.rnn = nn.LSTM(input_size=rnn_input_size, hidden_size=rnn_hidden_size, num_layers=2, batch_first=True,
                               bidirectional=True)

        self.fc = nn.Linear(in_features=rnn_input_size, out_features=num_classes)

    def forward(self, x):
        # CNN 层
        x = self.cnn(x)

        # 调整展平顺序
        # [N, C, H, W] -> [N, W, C, H]
        x = x.permute(0, 3, 1, 2).contiguous()
        # [N, C, H, W] -> [N, W, C*H]
        x = x.view(x.size(0), x.size(1), -1)

        # RNN 层
        x, _ = self.rnn(x)

        # 输出层
        x = self.fc(x)

        out = F.log_softmax(x, dim=-1)
        return out


if __name__ == '__main__':
    import time

    # EMNIST
    model = CRNN(in_channel=1, num_classes=11, cnn_output_height=1)

    data = torch.randn(10, 1, 32, 32 * 5)
    t_start = time.time()
    output = model(data)
    t_end = time.time()
    print(f"time: {t_end - t_start}")
    print(data.shape, output.shape)

    # Plate
    model = CRNN(in_channel=3, num_classes=100, cnn_output_height=2, use_gru=True)
    print(model)

    data = torch.randn(10, 3, 48, 168)

    t_start = time.time()
    output = model(data)
    t_end = time.time()
    print(f"time: {t_end - t_start}")
    print(data.shape, output.shape)
