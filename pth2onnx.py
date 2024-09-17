# -*- coding: utf-8 -*-

"""
@Time    : 2024/8/11 16:48
@File    : pth2onnx.py
@Author  : zj
@Description:

Usage: Pytorch to ONNX:
    $ python3 pth2onnx.py ./crnn_tiny-emnist.pth ./runs/crnn_tiny-emnist.onnx
    $ python3 pth2onnx.py ./crnn_tiny-plate.pth ./runs/crnn_tiny-plate.onnx

"""

import argparse
import os.path

import numpy as np

import onnx
import onnxruntime

import torch.onnx
import torch.nn as nn

from utils.general import load_ocr_model
from utils.dataset.emnist import DIGITS_CHARS
from utils.dataset.plate import PLATE_CHARS


def parse_opt():
    parser = argparse.ArgumentParser(description="Pytorch to ONNX")
    parser.add_argument("pretrained", metavar="MODEL", type=str, default=None, help="Pytorch Pretrained Model Path")
    parser.add_argument("save", metavar="SAVE", type=str, default=None, help="Saving ONNX Path")

    parser.add_argument('--use-lstm', action='store_true', help='use nn.LSTM instead of nn.GRU')
    parser.add_argument('--not-tiny', action='store_true', help='Use this flag to specify non-tiny mode')

    parser.add_argument("--use-lprnet", action='store_true', help='use LPRNet instead of CRNN')
    parser.add_argument("--use-origin-block", action='store_true', help='use origin small_basic_block impl')

    args = parser.parse_args()
    print(f"args: {args}")

    return args


def check_onnx(onnx_path='pytorch.onnx'):
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)


def check_output(x, torch_out, onnx_path='pytorch.onnx'):
    # See https://blog.csdn.net/zunzunle/article/details/130087922
    print("Supported onnxruntime version: ", onnxruntime.__version__)
    print("Supported Opset versions: ", onnxruntime.get_available_providers())

    # ValueError: This ORT build has ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'] enabled. \
    # Since ORT 1.9, you are required to explicitly set the providers parameter when instantiating InferenceSession.
    # For example, onnxruntime.InferenceSession(..., providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'], ...)
    ort_session = onnxruntime.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    print("Onnx info:")
    print(f"    input: {ort_session.get_inputs()[0]}")
    print(f"    output: {ort_session.get_outputs()[0]}")

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)
    print(x.shape, ort_outs[0].shape)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


def export_to_onnx(torch_model, shape=None, onnx_path="pytorch.onnx", is_dynamic=False):
    assert isinstance(torch_model, nn.Module)

    # Input to the model
    x = torch.randn(shape, requires_grad=True)
    torch_out = torch_model(x)

    # Export the model
    if is_dynamic:
        # variable length axes
        dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    else:
        dynamic_axes = None

    torch.onnx.export(torch_model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      onnx_path,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=12,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes=dynamic_axes
                      )

    check_onnx(onnx_path=onnx_path)
    check_output(x, torch_out, onnx_path=onnx_path)


def main(args):
    # UserWarning: Exporting a model to ONNX with a batch_size other than 1, with a variable length with GRU can cause an error when running the ONNX model with a different batch size.
    # Make sure to save the model with a batch size of 1, or define the initial states (h0/c0) as inputs of the model.
    if 'plate' in os.path.basename(args.pretrained):
        if args.use_lprnet:
            shape = (1, 3, 24, 94)
        else:
            shape = (1, 3, 48, 168)
        num_classes = len(PLATE_CHARS)
    else:
        shape = (1, 1, 32, 160)
        num_classes = len(DIGITS_CHARS)

    model, _ = load_ocr_model(pretrained=args.pretrained, device=torch.device("cpu"),
                              shape=shape, num_classes=num_classes,
                              not_tiny=args.not_tiny, use_lstm=args.use_lstm,
                              use_lprnet=args.use_lprnet, use_origin_block=args.use_origin_block)

    onnx_path = args.save
    export_to_onnx(model, shape=shape, onnx_path=onnx_path, is_dynamic=False)
    print(f"Save to {onnx_path}")


if __name__ == '__main__':
    args = parse_opt()
    main(args)
