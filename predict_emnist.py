# -*- coding: utf-8 -*-

"""
@date: 2023/10/10 上午9:59
@file: predict.py
@author: zj
@description:

Usage: Predict EMNIST:
    $ python predict_emnist.py crnn_tiny-emnist.pth ../datasets/emnist/ ./runs/predict/emnist/
    $ python predict_emnist.py crnn-emnist.pth ../datasets/emnist/ ./runs/predict/emnist/ --not-tiny

"""

import argparse

import os
import numpy as np
from itertools import groupby
import matplotlib.pyplot as plt

import torch

from utils.general import load_ocr_model
from utils.dataset.emnist import EMNISTDataset, DIGITS_CHARS


def parse_opt():
    parser = argparse.ArgumentParser(description='Predict CRNN with EMNIST')
    parser.add_argument('pretrained', metavar='PRETRAINED', type=str, help='path to pretrained model')
    parser.add_argument('val_root', metavar='DIR', type=str, help='path to val dataset')
    parser.add_argument('save_dir', metavar='DST', type=str, help='path to save dir')

    parser.add_argument('--use-lstm', action='store_true', help='use nn.LSTM instead of nn.GRU')
    parser.add_argument('--not-tiny', action='store_true', help='Use this flag to specify non-tiny mode')

    args = parser.parse_args()
    print(f"args: {args}")
    return args


@torch.no_grad()
def predict(args, val_root, pretrained, save_dir):
    img_h = 32
    digits_per_sequence = 5

    model, device = load_ocr_model(pretrained=pretrained, shape=(1, 1, img_h, digits_per_sequence * img_h),
                                   num_classes=len(DIGITS_CHARS), not_tiny=args.not_tiny, use_lstm=args.use_lstm)

    val_dataset = EMNISTDataset(val_root, is_train=False, num_of_sequences=50000,
                                digits_per_sequence=digits_per_sequence, img_h=img_h)

    plt.figure(figsize=(10, 6))

    blank_label = len(DIGITS_CHARS) - 1
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

        np_images = transformed_images

        plt.subplot(3, 2, i)
        title = f"Label: {str(emnist_labels)} Pred: {str(pred)}"
        print(title)
        plt.title(title)
        plt.imshow(np_images, cmap='gray')
        plt.axis('off')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, "predict_emnist.jpg"))


def main():
    args = parse_opt()

    predict(args, args.val_root, args.pretrained, args.save_dir)


if __name__ == '__main__':
    main()
