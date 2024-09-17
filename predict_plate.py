# -*- coding: utf-8 -*-

"""
@date: 2023/10/10 上午9:59
@file: predict.py
@author: zj
@description:

Usage: Predict Plate using CRNN:
    $ python predict_plate.py crnn_tiny-plate.pth ./assets/plate/宁A87J92_0.jpg runs/predict/plate/
    $ python predict_plate.py crnn-plate.pth ./assets/plate/宁A87J92_0.jpg runs/predict/plate/ --not-tiny

Usage: Predict Plate using LPRNet:
    $ python predict_plate.py lprnet_plus-plate.pth ./assets/plate/宁A87J92_0.jpg runs/predict/plate/ --use-lprnet
    $ python predict_plate.py lprnet-plate.pth ./assets/plate/宁A87J92_0.jpg runs/predict/plate/ --use-lprnet --use-origin-block

"""

import os
import argparse
import time
from itertools import groupby

import cv2
import matplotlib.pyplot as plt

import torch

# cp assets/fonts/simhei.ttf /usr/share/fonts/truetype/noto/
# rm -rf ~/.cache/matplotlib/*
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题

import importlib

# 根据脚本是否作为主模块运行来决定导入方式
if __name__ == '__main__':
    # 直接运行时，使用绝对导入
    # CRNN = importlib.import_module('utils.model.crnn').CRNN
    # LPRNet = importlib.import_module('utils.model.lprnet').LPRNet
    PLATE_CHARS = importlib.import_module('utils.dataset.plate').PLATE_CHARS
    model_info = importlib.import_module('utils.general').model_info
    load_ocr_model = importlib.import_module('utils.general').load_ocr_model
else:
    # 被导入时，尝试使用相对导入，如果失败则回退到绝对导入
    try:
        # CRNN = importlib.import_module('.utils.model.crnn', package=__package__).CRNN
        # LPRNet = importlib.import_module('.utils.model.lprnet', package=__package__).LPRNet
        PLATE_CHARS = importlib.import_module('.utils.dataset.plate', package=__package__).PLATE_CHARS
        model_info = importlib.import_module('.utils.general', package=__package__).model_info
        load_ocr_model = importlib.import_module('.utils.general', package=__package__).load_ocr_model
    except ValueError:
        # CRNN = importlib.import_module('utils.model.crnn').CRNN
        # LPRNet = importlib.import_module('utils.model.lprnet').LPRNet
        PLATE_CHARS = importlib.import_module('utils.dataset.plate').PLATE_CHARS
        model_info = importlib.import_module('.utils.general').model_info
        load_ocr_model = importlib.import_module('.utils.general').load_ocr_model


def parse_opt():
    parser = argparse.ArgumentParser(description='Predict CRNN/LPRNet with CCPD')
    parser.add_argument('pretrained', metavar='PRETRAINED', type=str, help='path to pretrained model')
    parser.add_argument('image_path', metavar='IMAGE', type=str, help='path to image path')
    parser.add_argument('save_dir', metavar='DST', type=str, help='path to save dir')

    parser.add_argument("--use-lprnet", action='store_true', help='use LPRNet instead of CRNN')
    parser.add_argument("--use-origin-block", action='store_true', help='use origin small_basic_block impl')

    parser.add_argument('--use-lstm', action='store_true', help='use nn.LSTM instead of nn.GRU')
    parser.add_argument('--not-tiny', action='store_true', help='Use this flag to specify non-tiny mode')

    args = parser.parse_args()
    print(f"args: {args}")
    return args


@torch.no_grad()
def predict_plate(image, model=None, device=None, img_h=48, img_w=168):
    start_time = time.time()

    # Data
    resize_image = cv2.resize(image, (img_w, img_h))

    data = torch.from_numpy(resize_image).float() / 255.
    # HWC -> CHW
    data = data.permute(2, 0, 1)

    # Infer
    # [1, H, N*W] -> [1, 1, H, N*W]
    data = data.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(data).cpu()[0]

    _, max_index = torch.max(output, dim=1)
    raw_pred = list(max_index.numpy())
    blank_label = 0
    pred = torch.IntTensor([c for c, _ in groupby(raw_pred) if c != blank_label])
    pred = pred.numpy()

    pred_plate = [PLATE_CHARS[i] for i in pred]
    # pred_plate = ''.join(pred_plate)
    pred_plate = ''.join(pred_plate[:2]) + "·" + ''.join(pred_plate[2:])

    end_time = time.time()
    predict_time = (end_time - start_time) * 1000
    print(f"Pred: {pred_plate} - Predict time: {predict_time :.1f} ms")
    return pred_plate, predict_time


def main():
    args = parse_opt()

    image_path = args.image_path
    assert os.path.isfile(image_path), image_path
    image = cv2.imread(image_path)
    if len(image.shape) == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # Model
    if args.use_lprnet:
        img_w = 94
        img_h = 24
    else:
        img_w = 168
        img_h = 48
    model, device = load_ocr_model(pretrained=args.pretrained, shape=(1, 3, img_h, img_w), num_classes=len(PLATE_CHARS),
                                   not_tiny=args.not_tiny, use_lstm=args.use_lstm,
                                   use_lprnet=args.use_lprnet, use_origin_block=args.use_origin_block)

    # Predict
    pred_plate, _ = predict_plate(image=image, model=model, device=device, img_h=img_h, img_w=img_w)

    # Draw
    plt.figure()
    title = f"Pred: {pred_plate}"

    plt.title(title)
    plt.imshow(image)
    plt.axis('off')

    # Save
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    image_name = os.path.basename(image_path)
    res_path = os.path.join(save_dir, f"plate_{image_name}")
    print(f'Save to {res_path}')
    plt.savefig(res_path)


if __name__ == '__main__':
    main()
