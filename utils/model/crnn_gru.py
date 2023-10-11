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

    def __init__(self, in_channel, num_classes, cnn_output_height):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=(3, 3)),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(3, 3)),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=2),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(inplace=True),
        )

        self.gru_input_size = cnn_output_height * 64
        gru_hidden_size = 128
        gru_num_layers = 2
        self.rnn = nn.GRU(self.gru_input_size, gru_hidden_size, gru_num_layers, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(gru_hidden_size * 2, num_classes)

    def forward(self, x):
        N = x.shape[0]

        x = self.cnn(x)
        # [N, C, H, W] -> [N, W, H, C]
        x = x.permute(0, 3, 2, 1)
        # [N, W, H, C] -> [N, W, H*C]
        x = x.reshape(N, -1, self.gru_input_size)

        # [N, W, H*C] -> [N, W, gru_hidden_size*2]
        x, _ = self.rnn(x)

        # [N, W, gru_hidden_size*2] -> [N, W, num_classes]
        # out = []
        # for item in x:
        #     tmp = self.fc(item)
        #     log = F.log_softmax(tmp, dim=-1)
        #     out.append(log)
        # out = torch.stack(out)
        out = F.log_softmax(self.fc(x), dim=-1)

        return out


if __name__ == '__main__':
    # EMNIST
    model = CRNN(in_channel=1, num_classes=11, cnn_output_height=4)

    data = torch.randn(10, 1, 28, 28 * 5)
    output = model(data)
    print(data.shape, output.shape)

    # Plate
    model = CRNN(in_channel=3, num_classes=29, cnn_output_height=9)

    data = torch.randn(10, 3, 48, 168)
    output = model(data)
    print(data.shape, output.shape)

    data = torch.randn(10, 3, 48, 300)
    output = model(data)
    print(data.shape, output.shape)
