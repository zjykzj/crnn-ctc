# -*- coding: utf-8 -*-

"""
@date: 2023/10/8 下午4:44
@file: model.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def t_module():
    x = torch.randn(1, 4, 10)
    fc = nn.Linear(10, 5)

    # [N, W, gru_hidden_size*2] -> [N, W, num_classes]
    out = []
    tmp_list = []
    for item in x:
        tmp = fc(item)
        tmp_list.append(tmp)
        log = F.log_softmax(tmp, dim=-1)
        out.append(log)
    out = torch.stack(out)

    out2 = fc(x)
    out2 = F.log_softmax(out2, dim=-1)

    assert torch.all(out == out2)


def t_model():
    num_classes = 11
    gru_hidden_size = 128
    gru_num_layers = 2
    cnn_output_height = 4

    class CRNN(nn.Module):

        def __init__(self):
            super(CRNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
            self.norm1 = nn.InstanceNorm2d(32)
            self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2)
            self.norm2 = nn.InstanceNorm2d(32)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3))
            self.norm3 = nn.InstanceNorm2d(64)
            self.conv4 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=2)
            self.norm4 = nn.InstanceNorm2d(64)
            self.gru_input_size = cnn_output_height * 64
            self.gru = nn.GRU(self.gru_input_size, gru_hidden_size, gru_num_layers, batch_first=True,
                              bidirectional=True)
            self.fc = nn.Linear(gru_hidden_size * 2, num_classes)

        def forward(self, x):
            batch_size = x.shape[0]
            out = self.conv1(x)
            out = self.norm1(out)
            out = F.leaky_relu(out)
            out = self.conv2(out)
            out = self.norm2(out)
            out = F.leaky_relu(out)
            out = self.conv3(out)
            out = self.norm3(out)
            out = F.leaky_relu(out)
            out = self.conv4(out)
            out = self.norm4(out)
            out = F.leaky_relu(out)

            # [N, C, H, W] -> [N, W, H, C]
            out = out.permute(0, 3, 2, 1)
            # [N, W, H, C] -> [N, W, H*C]
            out = out.reshape(batch_size, -1, self.gru_input_size)
            # out: [N, W, H*C]
            out, _ = self.gru(out)
            # out[i]: [W, H*C]
            # 基于列维度计算分类概率
            out = torch.stack([F.log_softmax(self.fc(out[i]), dim=-1) for i in range(out.shape[0])])
            return out

    model = CRNN()
    data = torch.randn([64, 1, 28, 140])

    outputs = model(data)
    print(data.shape, outputs.shape)


if __name__ == '__main__':
    t_model()
    t_module()
