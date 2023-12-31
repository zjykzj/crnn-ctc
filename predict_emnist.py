# -*- coding: utf-8 -*-

"""
@date: 2023/10/10 上午9:59
@file: predict.py
@author: zj
@description:

Usage: Predict EMNIST:
    $ python predict_emnist.py runs/emnist_ddp/crnn-emnist-e100.pth ../datasets/EMNIST/ runs/

"""

import argparse

import os
import cv2
import numpy as np
from itertools import groupby
import matplotlib.pyplot as plt

import torch

from utils.model.crnn_gru import CRNN
from utils.dataset.emnist import EMNISTDataset


def parse_opt():
    parser = argparse.ArgumentParser(description='Predict CRNN with EMNIST')
    parser.add_argument('pretrained', metavar='PRETRAINED', type=str, default="runs/emnist/CRNN-e100.pth",
                        help='path to pretrained model')
    parser.add_argument('val_root', metavar='DIR', type=str, help='path to val dataset')
    parser.add_argument('save_dir', metavar='DST', type=str, help='path to save dir')

    args = parser.parse_args()
    print(f"args: {args}")
    return args


@torch.no_grad()
def predict(val_root, pretrained, save_dir):
    model = CRNN(in_channel=1, num_classes=11, cnn_output_height=4)
    print(f"Loading CRNN pretrained: {pretrained}")
    ckpt = torch.load(pretrained, map_location='cpu')
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    digits_per_sequence = 5
    val_dataset = EMNISTDataset(val_root, is_train=False, num_of_sequences=2000,
                                digits_per_sequence=digits_per_sequence)

    plt.figure(figsize=(10, 6))

    blank_label = 10
    for i in range(1, 7):
        random_index = np.random.randint(len(val_dataset))
        sequence, emnist_labels, transformed_images = val_dataset.__getitem__(random_index, return_tf=True)

        # [1, H, N*W] -> [1, 1, H, N*W]
        images = sequence.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(images).cpu()[0]

        _, max_index = torch.max(output, dim=1)
        raw_pred = list(max_index.numpy())
        pred = torch.IntTensor([c for c, _ in groupby(raw_pred) if c != blank_label])
        pred = pred.numpy()
        emnist_labels = emnist_labels.numpy()
        # print(pred, emnist_labels)

        np_images = (transformed_images * 255).numpy()
        np_images = np.hstack(np_images)

        plt.subplot(3, 2, i)
        title = f"Label: {str(emnist_labels)} Pred: {str(pred)}"
        print(title)
        plt.title(title)
        plt.imshow(np_images, cmap='gray')
        plt.axis('off')
    plt.savefig(os.path.join(save_dir, "predict_emnist.jpg"))


def main():
    args = parse_opt()

    predict(args.val_root, args.pretrained, args.save_dir)


if __name__ == '__main__':
    main()
