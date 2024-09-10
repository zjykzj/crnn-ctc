# -*- coding: utf-8 -*-

"""
@date: 2023/10/9 下午4:42
@file: eval.py
@author: zj
@description:

Usage - Single-GPU eval using CRNN:
    $ python3 eval_plate.py crnn_tiny-plate-b512-e100.pth ../datasets/chinese_license_plate/recog/
    $ python3 eval_plate.py crnn-plate-b512-e100.pth ../datasets/chinese_license_plate/recog/ --not-tiny

Usage - Single-GPU eval using LPRNet:
    $ python3 eval_plate.py lprnetv2-plate-b512-e100.pth ../datasets/chinese_license_plate/recog/ --use-lprnet
    $ python3 eval_plate.py lprnet-plate-b512-e100 ../datasets/chinese_license_plate/recog/ --use-lprnet --use-origin-block

Usage - Specify which dataset to evaluate:
    $ python3 eval_plate.py crnn-plate-b512-e100.pth ../datasets/chinese_license_plate/recog/ --not-tiny --only-ccpd2019
    $ python3 eval_plate.py crnn-plate-b512-e100.pth ../datasets/chinese_license_plate/recog/ --not-tiny --only-ccpd2020
    $ python3 eval_plate.py crnn-plate-b512-e100.pth ../datasets/chinese_license_plate/recog/ --not-tiny --only-others

"""

import argparse

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from utils.general import load_ocr_model
from utils.dataset.plate import PlateDataset, PLATE_CHARS
from utils.evaluator import Evaluator


def parse_opt():
    parser = argparse.ArgumentParser(description='Eval CRNN/LPRNet with CCPD')
    parser.add_argument('pretrained', metavar='PRETRAINED', type=str, help='path to pretrained model')
    parser.add_argument('val_root', metavar='DIR', type=str, help='path to val dataset')

    parser.add_argument('--use-lstm', action='store_true', help='use nn.LSTM instead of nn.GRU')
    parser.add_argument('--not-tiny', action='store_true', help='Use this flag to specify non-tiny mode')

    parser.add_argument("--use-lprnet", action='store_true', help='use LPRNet instead of CRNN')
    parser.add_argument("--use-origin-block", action='store_true', help='use origin small_basic_block impl')

    parser.add_argument('--only-ccpd2019', action='store_true', help='only eval CCPD2019/test dataset')
    parser.add_argument('--only-ccpd2020', action='store_true', help='only eval CCPD2019/test dataset')
    parser.add_argument('--only-others', action='store_true', help='only eval git_plate/val_verify dataset')

    args = parser.parse_args()
    print(f"args: {args}")
    return args


@torch.no_grad()
def val(args, val_root, pretrained):
    # (W, H)
    if args.use_lprnet:
        img_w = 94
        img_h = 24
    else:
        img_w = 168
        img_h = 48
    model, device = load_ocr_model(pretrained=pretrained, shape=(1, 3, img_h, img_w), num_classes=len(PLATE_CHARS),
                                   not_tiny=args.not_tiny, use_lstm=args.use_lstm,
                                   use_lprnet=args.use_lprnet, use_origin_block=args.use_origin_block)

    val_dataset = PlateDataset(val_root, is_train=False, input_shape=(img_w, img_h), only_ccpd2019=args.only_ccpd2019,
                               only_ccpd2020=args.only_ccpd2020, only_others=args.only_others)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, drop_last=False,
                                pin_memory=True)

    blank_label = 0
    emnist_evaluator = Evaluator(blank_label=blank_label)

    pbar = tqdm(val_dataloader)
    for idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        targets = val_dataset.convert(targets)
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
