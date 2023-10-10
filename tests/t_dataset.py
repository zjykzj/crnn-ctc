# -*- coding: utf-8 -*-

"""
@date: 2023/10/8 下午3:17
@file: dataset.py
@author: zj
@description: 
"""

from torchvision.datasets import EMNIST

data_root = "./EMNIST"
is_train = True
emnist = EMNIST(data_root, split="digits", train=is_train, download=True)
print(emnist)

import numpy as np

indices = np.random.choice(len(emnist), size=(5,))
print(indices)

images = emnist.data[indices]
print(images.shape, type(images))

labels = emnist.targets[indices]
print(labels, labels.shape, type(labels))

from dataset import EMNISTDataset

dataset = EMNISTDataset(data_root, is_train=is_train, num_of_sequences=10000, digits_per_sequence=5)
print(dataset)

image, label = dataset.__getitem__(1000)
print(image.shape, label.shape)
