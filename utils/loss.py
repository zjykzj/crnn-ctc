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

    def __init__(self, blank_label=0):
        super().__init__()
        self.loss = torch.nn.CTCLoss(blank=blank_label, reduction='mean', zero_infinity=True)

    def forward(self, preds, targets, target_lengths=None):
        N, cnn_output_width = preds.shape[:2]
        # [N, W, num_classes] -> [W, N, num_classes]
        preds = preds.permute(1, 0, 2)

        input_lengths = torch.IntTensor(N).fill_(cnn_output_width).to(preds.device)
        if target_lengths is None:
            # Padded
            target_lengths = torch.IntTensor([len(t) for t in targets]).to(preds.device)

        # RuntimeError: Expected tensor to have CPU Backend, but got tensor with CUDA Backend (while checking arguments for cudnn_ctc_loss)
        # https://github.com/pytorch/pytorch/issues/22234
        with torch.backends.cudnn.flags(enabled=False):
            # print(preds.shape, targets.shape, input_lengths, target_lengths)
            loss = self.loss(preds, targets, input_lengths, target_lengths)
        return loss
