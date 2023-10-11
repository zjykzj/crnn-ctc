# -*- coding: utf-8 -*-

"""
@date: 2023/10/9 下午3:12
@file: evaluator.py
@author: zj
@description: 
"""

import torch

from itertools import groupby


class Evaluator:

    def __init__(self, blank_label=10):
        self.blank_label = blank_label

        self.correct_num = 0.
        self.total_num = 0.

    def reset(self):
        self.correct_num = 0.
        self.total_num = 0.

    def update(self, outputs, targets):
        assert len(outputs) == len(targets)

        correct_num = 0.
        total_num = len(outputs)

        for output, target in zip(outputs, targets):
            _, max_index = torch.max(output, dim=1)

            raw_pred = list(max_index.numpy())
            pred = torch.IntTensor([c for c, _ in groupby(raw_pred) if c != self.blank_label])
            if len(pred) == len(target) and torch.all(pred.eq(target)):
                correct_num += 1

        self.correct_num += correct_num
        self.total_num += total_num

        accuracy = correct_num / total_num
        return accuracy

    def result(self):
        accuracy = self.correct_num / self.total_num

        return accuracy
