# -*- coding: utf-8 -*-

"""
@date: 2023/10/8 下午2:48
@file: dataset.py
@author: zj
@description: 
"""

import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import EMNIST


class EMNISTDataset(Dataset):

    def __init__(self, data_root, is_train=True, num_of_sequences=10000, digits_per_sequence=5):
        self.num_of_sequences = num_of_sequences
        self.digits_per_sequence = digits_per_sequence

        self.emnist = EMNIST(data_root, split="digits", train=is_train, download=True)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(45, fill=0),
            # transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=10, translate=(0.2, 0.15), scale=(0.8, 1.1)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index, return_tf=False):
        assert index < self.num_of_sequences

        indices = np.random.choice(len(self.emnist), size=(self.digits_per_sequence,))
        # [N, 28, 28]
        emnist_images = self.emnist.data[indices]
        # [N]
        emnist_labels = self.emnist.targets[indices]

        transformed_images = []
        for image in emnist_images:
            image = self.transform(image.T)
            transformed_images.append(image)

        # [N, 1, 28, 28] -> [N, 28, 28]
        transformed_images = torch.stack(transformed_images).reshape(self.digits_per_sequence, 28, 28)
        # [N, H, W] -> [H, N*W]
        sequence = np.hstack(transformed_images.numpy())
        # [H, N*W] -> [1, H, N*W]
        sequence = torch.from_numpy(sequence).reshape((1, 28, self.digits_per_sequence * 28))

        if return_tf:
            return sequence, emnist_labels, transformed_images
        else:
            # [1, H, N*W], [N]
            return sequence, emnist_labels

    def __len__(self):
        return self.num_of_sequences
