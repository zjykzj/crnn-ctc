# -*- coding: utf-8 -*-

"""
@date: 2023/10/10 下午4:40
@file: plate.py
@author: zj
@description: 
"""

import os
import cv2
from pathlib import Path

import torch
from torch.utils.data import Dataset

PLATE_CHARS = "#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"


def load_data(data_root, pattern='*.json'):
    assert os.path.isdir(data_root)

    data_list = list()

    p = Path(data_root)
    # for path in tqdm(p.rglob(pattern)):
    for path in p.rglob(pattern):
        data_list.append(str(path).strip())

    return data_list


def is_plate_right(plate_name):
    assert isinstance(plate_name, str), plate_name
    for ch in plate_name:
        if ch not in PLATE_CHARS:
            return False
    return True


def create_plate_label(img_list):
    plate_dict = dict()
    for i in range(len(PLATE_CHARS)):
        plate_dict[PLATE_CHARS[i]] = i

    data_list = list()
    label_dict = dict()
    for img_path in img_list:
        assert os.path.isfile(img_path), img_path

        img_name = os.path.basename(img_path)
        label_name = img_name.split("_")[0]
        if " " in label_name:
            continue
        if not is_plate_right(label_name):
            continue

        label = []
        for i in range(len(label_name)):
            label.append(plate_dict[label_name[i]])
        label_dict[label_name] = label

        data_list.append([img_path, label_name])
    return data_list, label_dict


class PlateDataset(Dataset):

    def __init__(self, data_root, is_train=True, img_h=48, img_w=168):
        self.data_root = data_root
        self.is_train = is_train
        self.img_w = img_w
        self.img_h = img_h

        img_list = load_data(data_root, pattern="*.jpg")
        data_list, label_dict = create_plate_label(img_list)
        print(f"Load {'train' if is_train else 'test'} data: {len(data_list)}")

        self.data_list = data_list
        self.dataset_len = len(data_list)
        self.label_dict = label_dict

    def __getitem__(self, index):
        assert index < self.dataset_len

        img_path, label_name = self.data_list[index]
        image = cv2.imread(img_path)
        if image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        image = cv2.resize(image, (self.img_w, self.img_h))

        data = torch.from_numpy(image).float() / 255.
        # HWC -> CHW
        data = data.permute(2, 0, 1)

        return data, label_name

    def __len__(self):
        return self.dataset_len

    def convert(self, targets):
        labels = []
        for label_name in targets:
            label = self.label_dict[label_name]
            labels.append(torch.IntTensor(label))
        return labels
