# -*- coding: utf-8 -*-

"""
@date: 2023/10/8 下午2:48
@file: dataset.py
@author: zj
@description: 
"""

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import EMNIST

DIGITS_CHARS = "0123456789#"


def parse_emnist():
    import os

    from torchvision.datasets.mnist import read_image_file, read_label_file

    splits = ('byclass', 'bymerge', 'balanced', 'letters', 'digits', 'mnist')
    gzip_folder = "../datasets/emnist/EMNIST/raw/gzip"
    processed_folder = "../datasets/emnist/EMNIST/processed"

    # process and save as torch files
    for split in splits:
        print('Processing ' + split)
        training_set = (
            read_image_file(os.path.join(gzip_folder, 'emnist-{}-train-images-idx3-ubyte'.format(split))),
            read_label_file(os.path.join(gzip_folder, 'emnist-{}-train-labels-idx1-ubyte'.format(split)))
        )
        test_set = (
            read_image_file(os.path.join(gzip_folder, 'emnist-{}-test-images-idx3-ubyte'.format(split))),
            read_label_file(os.path.join(gzip_folder, 'emnist-{}-test-labels-idx1-ubyte'.format(split)))
        )
        with open(os.path.join(processed_folder, 'training_{}.pt'.format(split)), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(processed_folder, 'test_{}.pt'.format(split)), 'wb') as f:
            torch.save(test_set, f)


class EMNISTDataset(Dataset):

    def __init__(self, data_root, is_train=True, num_of_sequences=100000, digits_per_sequence=5, img_h=32):
        self.num_of_sequences = num_of_sequences
        self.digits_per_sequence = digits_per_sequence
        self.img_h = img_h

        # EMNIST download link is broken #5662
        # https://github.com/pytorch/vision/issues/5662
        # The EMNIST download should be fixed in the next version of torchvision (0.18).
        #
        # So download EMNIST manually from https://www.nist.gov/itl/products-and-services/emnist-dataset
        # parse_emnist()

        self.emnist = EMNIST(data_root, split="digits", train=is_train, download=False)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_h, img_h)),
            transforms.RandomRotation(15, fill=0),  # 减小旋转角度
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # 调整参数
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
            transformed_images.append(np.array(image))
        transformed_images = np.concatenate(transformed_images, axis=-1)

        image = cv2.resize(transformed_images, (self.img_h * self.digits_per_sequence, self.img_h))

        data = torch.from_numpy(image).float() / 255.
        # HW -> CHW
        data = data.unsqueeze(0)

        if return_tf:
            return data, emnist_labels, transformed_images
        else:
            # [1, H, N*W], [N]
            return data, emnist_labels

    def __len__(self):
        return self.num_of_sequences


if __name__ == '__main__':
    dataset = EMNISTDataset("../datasets/emnist/")
    print(dataset)

    data, emnist_labels = dataset.__getitem__(100)
    print(data.shape, emnist_labels)
