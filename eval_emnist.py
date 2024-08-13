# -*- coding: utf-8 -*-

"""
@date: 2023/10/9 下午4:42
@file: eval.py
@author: zj
@description:

Usage - Single-GPU eval:
    $ python eval_emnist.py crnn_tiny-emnist-b512-e100.pth ../datasets/emnist/
    $ python eval_emnist.py crnn-emnist-b512-e100.pth ../datasets/emnist/ --not-tiny

"""

import argparse

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from utils.general import load_crnn
from utils.dataset.emnist import EMNISTDataset, DIGITS_CHARS
from utils.evaluator import Evaluator


def parse_opt():
    parser = argparse.ArgumentParser(description='Eval CRNN with EMNIST')
    parser.add_argument('pretrained', metavar='PRETRAINED', type=str, help='path to pretrained model')
    parser.add_argument('val_root', metavar='DIR', type=str, help='path to val dataset')

    parser.add_argument('--use-lstm', action='store_true', help='use nn.LSTM instead of nn.GRU')
    parser.add_argument('--not-tiny', action='store_true', help='Use this flag to specify non-tiny mode')

    args = parser.parse_args()
    print(f"args: {args}")
    return args


@torch.no_grad()
def val(args, val_root, pretrained):
    img_h = 32
    digits_per_sequence = 5

    model, device = load_crnn(pretrained=pretrained, shape=(1, 1, img_h, digits_per_sequence * img_h),
                              num_classes=len(DIGITS_CHARS), not_tiny=args.not_tiny, use_lstm=args.use_lstm)

    val_dataset = EMNISTDataset(val_root, is_train=False, num_of_sequences=50000,
                                digits_per_sequence=digits_per_sequence, img_h=img_h)
    batch_size = 1
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False,
                                pin_memory=True)

    blank_label = len(DIGITS_CHARS) - 1
    emnist_evaluator = Evaluator(blank_label=blank_label)

    pbar = tqdm(val_dataloader)
    for idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images).cpu()

        acc = emnist_evaluator.update(outputs, targets)
        info = f"Batch:{idx} ACC:{acc * 100:.3f}"
        pbar.set_description(info)
    acc = emnist_evaluator.result()
    print(f"ACC:{acc * 100:.3f}")


def main():
    args = parse_opt()

    val(args, args.val_root, args.pretrained)


if __name__ == '__main__':
    main()
