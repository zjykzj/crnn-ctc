# -*- coding: utf-8 -*-

"""
@date: 2023/10/9 下午4:42
@file: eval.py
@author: zj
@description:

Usage - Single-GPU eval:
    $ python3 eval_plate.py ./runs/crnn_tiny-plate-b512-e100.pth ../datasets/chinese_license_plate/recog/
    $ python3 eval_plate.py ./runs/crnn-plate-b512-e100.pth ../datasets/chinese_license_plate/recog/ --not-tiny

Usage - Specify which dataset to evaluate:
    $ python3 eval_plate.py ./runs/crnn-plate-b512-e100.pth ../datasets/chinese_license_plate/recog/ --not-tiny --only-ccpd2019
    $ python3 eval_plate.py ./runs/crnn-plate-b512-e100.pth ../datasets/chinese_license_plate/recog/ --not-tiny --only-ccpd2020
    $ python3 eval_plate.py ./runs/crnn-plate-b512-e100.pth ../datasets/chinese_license_plate/recog/ --not-tiny --only-others

"""

import argparse

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from utils.model.crnn import CRNN
from utils.dataset.plate import PlateDataset, PLATE_CHARS
from utils.evaluator import Evaluator


def parse_opt():
    parser = argparse.ArgumentParser(description='Eval CRNN with EMNIST')
    parser.add_argument('pretrained', metavar='PRETRAINED', type=str, help='path to pretrained model')
    parser.add_argument('val_root', metavar='DIR', type=str, help='path to val dataset')

    parser.add_argument('--use-lstm', action='store_true', help='use nn.LSTM instead of nn.GRU')
    parser.add_argument('--not-tiny', action='store_true', help='Use this flag to specify non-tiny mode')

    parser.add_argument('--only-ccpd2019', action='store_true', help='only eval CCPD2019/test dataset')
    parser.add_argument('--only-ccpd2020', action='store_true', help='only eval CCPD2019/test dataset')
    parser.add_argument('--only-others', action='store_true', help='only eval git_plate/val_verify dataset')

    args = parser.parse_args()
    print(f"args: {args}")
    return args


@torch.no_grad()
def val(args, val_root, pretrained):
    # (W, H)
    input_shape = (168, 48)

    model = CRNN(in_channel=3, num_classes=len(PLATE_CHARS), cnn_input_height=input_shape[1], is_tiny=not args.not_tiny,
                 use_gru=not args.use_lstm)
    print(f"Loading CRNN pretrained: {pretrained}")
    ckpt = torch.load(pretrained, map_location='cpu')
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    val_dataset = PlateDataset(val_root, is_train=False, input_shape=input_shape, only_ccpd2019=args.only_ccpd2019,
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
