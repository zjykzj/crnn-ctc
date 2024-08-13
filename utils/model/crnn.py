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

    def __init__(self, in_channel, num_classes, cnn_input_height, is_tiny=True, use_gru=True):
        super().__init__()

        if is_tiny:
            # 特征提取层
            self.cnn = nn.Sequential(
                nn.Conv2d(in_channel, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),

                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, stride=(2, 1), padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),

                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=(2, 1), padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),

                nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=0),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )

            cnn_output_height = cnn_input_height // 2 // 2 // 2 - 1
            rnn_input_size = 64 * cnn_output_height
            rnn_hidden_size = rnn_input_size // 2
        else:
            # 特征提取层
            self.cnn = nn.Sequential(
                nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

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

            cnn_output_height = cnn_input_height // 2 // 2 // 2 // 2 - 1
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

        # PROBLEM: torch/nn/modules/rnn.py:821: UserWarning: RNN module weights are not part of single contiguous chunk of memory. This means they need to be compacted at every call, possibly greatly increasing memory usage. To compact weights again call flatten_parameters(). (Triggered internally at  /pytorch/aten/src/ATen/native/cudnn/RNN.cpp:915.)
        # FIX:
        # 1. https://discuss.pytorch.org/t/rnn-module-weights-are-not-part-of-single-contiguous-chunk-of-memory/6011/20
        # 2. https://pytorch.org/docs/stable/generated/torch.nn.RNNBase.html#torch.nn.RNNBase.flatten_parameters
        self.rnn.flatten_parameters()

        # RNN 层
        x, _ = self.rnn(x)

        # 输出层
        x = self.fc(x)

        out = F.log_softmax(x, dim=-1)
        return out


def test_model(data, model):
    print(data.shape)

    t_start = time.time()
    output = model(data)
    t_end = time.time()
    print(f"time: {t_end - t_start}")
    print(data.shape, output.shape)


if __name__ == '__main__':
    import time
    import copy

    # EMNIST
    data = torch.randn(10, 1, 32, 32 * 5)

    model = CRNN(in_channel=1, num_classes=11, cnn_input_height=32, is_tiny=False, use_gru=True)
    test_model(copy.deepcopy(data), model)

    model = CRNN(in_channel=1, num_classes=11, cnn_input_height=32, is_tiny=True, use_gru=True)
    test_model(copy.deepcopy(data), model)
    # Plate
    data = torch.randn(10, 3, 48, 168)

    model = CRNN(in_channel=3, num_classes=100, cnn_input_height=48, is_tiny=False, use_gru=True)
    test_model(copy.deepcopy(data), model)

    model = CRNN(in_channel=3, num_classes=100, cnn_input_height=48, is_tiny=True, use_gru=True)
    test_model(copy.deepcopy(data), model)
