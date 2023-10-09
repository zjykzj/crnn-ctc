# -*- coding: utf-8 -*-

"""
@date: 2023/10/9 下午3:14
@file: loss.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn


class CTCLoss(nn.Module):

    def __init__(self, blank_label=10, cnn_output_width=32):
        super().__init__()
        self.cnn_output_width = cnn_output_width

        self.loss = torch.nn.CTCLoss(blank=blank_label, reduction='mean', zero_infinity=True)

    def forward(self, preds, targets):
        N = preds.shape[0]
        # [N, W, num_classes] -> [W, N, num_classes]
        preds = preds.permute(1, 0, 2)

        input_lengths = torch.IntTensor(N).fill_(self.cnn_output_width)
        target_lengths = torch.IntTensor([len(t) for t in targets])

        loss = self.loss(preds, targets, input_lengths, target_lengths)
        return loss
