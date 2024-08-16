# -*- coding: utf-8 -*-

"""
@Time    : 2024/8/11 15:32
@File    : gradio.py
@Author  : zj
@Description: 
"""
import os.path

import cv2
import time
import torch
import onnxruntime

from datetime import datetime

import numpy as np
import gradio as gr
from itertools import groupby

from utils.dataset.plate import PLATE_CHARS

save_root = "./runs/"
if not os.path.exists(save_root):
    os.makedirs(save_root)


# Model
class ONNXRuntimePredictor:

    def __init__(self, w, device=torch.device('cpu')):
        print(f'Loading {w} for ONNX Runtime inference...')
        providers = ['CPUExecutionProvider']
        session = onnxruntime.InferenceSession(w, providers=providers)
        output_names = [x.name for x in session.get_outputs()]
        meta = session.get_modelmeta().custom_metadata_map  # metadata
        print(f"meta: {meta}")

        self.session = session
        self.output_names = output_names
        self.device = device

    def __call__(self, im):
        im = im.cpu().numpy()  # torch to numpy
        y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})

        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x


device = torch.device("cpu")
model = ONNXRuntimePredictor("./runs/crnn_tiny-plate.onnx", device=device)


# Predict
@torch.no_grad()
def predict_crnn(image, model=None, device=None):
    start_time = time.time()

    # Data
    img_w = 168
    img_h = 48
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


def predict(inp):
    # 获取当前日期和时间
    now = datetime.now()
    # 格式化为字符串，例如 "2024-08-16_21-37-00"
    formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    inp.save(os.path.join(save_root, f"{formatted_time}.jpg"))

    image = np.array(inp)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    pred_plate, predict_time = predict_crnn(image, model=model, device=device)

    return pred_plate


if __name__ == '__main__':
    gr.Interface(fn=predict,
                 inputs=gr.Image(type="pil"),
                 outputs=['text'],
                 examples=["./assets/plate/宁A87J92_0.jpg", "./assets/plate/川A3X7J1_0.jpg"]).launch()
