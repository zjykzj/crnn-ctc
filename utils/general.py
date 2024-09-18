# -*- coding: utf-8 -*-

"""
@date: 2023/9/30 上午11:00
@file: general.py
@author: zj
@description: 
"""

import os
import thop
import torch
import random

import platform
import pkg_resources as pkg

import numpy as np
from copy import deepcopy

from .logger import LOGGER
from .model.crnn import CRNN
from .model.lprnet import LPRNet


def emojis(str=''):
    # Return platform-dependent emoji-safe version of string
    return str.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else str


def check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned=False, hard=False, verbose=False):
    # Check version vs. required version
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    s = f'WARNING ⚠️ {name}{minimum} is required by YOLOv5, but {name}{current} is currently installed'  # string
    if hard:
        assert result, emojis(s)  # assert min requirements met
    if verbose and not result:
        LOGGER.warning(s)
    return result


def init_seeds(seed=0, deterministic=False):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe
    # torch.backends.cudnn.benchmark = True  # AutoBatch problem https://github.com/ultralytics/yolov5/issues/9287
    if deterministic and check_version(torch.__version__, '1.12.0'):  # https://github.com/ultralytics/yolov5/pull/8213
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['PYTHONHASHSEED'] = str(seed)


def model_info(model, model_name, verbose=False, img_shape=(1, 3, 48, 168)):
    # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print(f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}")
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:  # FLOPs
        p = next(model.parameters())
        im = torch.empty(img_shape, device=p.device)  # input image in BCHW format
        flops = thop.profile(deepcopy(model), inputs=(im,), verbose=False)[0] / 1E9 * 2  # stride GFLOPs
        fs = f', {flops:.1f} GFLOPs'  # 640x640 GFLOPs
    except Exception:
        fs = ''

    print(f"{model_name} summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}")


def load_ocr_model(pretrained=None, device=None, shape=(1, 3, 48, 168), num_classes=100, not_tiny=False,
                   use_lstm=False, use_lprnet=False, use_origin_block=False, add_stnet=False):
    if use_lprnet:
        model = LPRNet(in_channel=shape[1], num_classes=num_classes, use_origin_block=use_origin_block,
                       add_stnet=add_stnet)
    else:
        model = CRNN(in_channel=shape[1], num_classes=num_classes, cnn_input_height=shape[2], is_tiny=not not_tiny,
                     use_gru=not use_lstm)
    if pretrained is not None:
        if isinstance(pretrained, list):
            pretrained = pretrained[0]
        print(f"Loading CRNN pretrained: {pretrained}")
        ckpt = torch.load(pretrained, map_location='cpu')
        ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
        model.load_state_dict(ckpt, strict=True)
    model.eval()

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Warm
    for _ in range(3):
        data = torch.randn(shape).to(device)
        _ = model(data)

    model_name = os.path.splitext(os.path.basename(pretrained))[0]
    model_info(model, model_name, verbose=False, img_shape=shape)

    return model, device
