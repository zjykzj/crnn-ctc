# -*- coding: utf-8 -*-

"""
@date: 2023/10/10 下午4:40
@file: plate.py
@author: zj
@description: 
"""

import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

RANK = int(os.getenv('RANK', -1))

DELIMITER = '_'

PLATE_CHARS = "#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"

PLATE_DICT = dict()
for i in range(len(PLATE_CHARS)):
    PLATE_DICT[PLATE_CHARS[i]] = i


def load_data(data_root, pattern='*.jpg'):
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
    data_list = list()
    label_dict = dict()
    for img_path in img_list:
        assert os.path.isfile(img_path), img_path

        img_name = os.path.splitext(os.path.basename(img_path))[0]
        label_name = img_name.split(DELIMITER)[0]
        if " " in label_name:
            continue
        if len(label_name) < 3:
            continue
        if not is_plate_right(label_name):
            continue

        if label_name not in label_dict.keys():
            label = []
            for i in range(len(label_name)):
                label.append(PLATE_DICT[label_name[i]])
            label_dict[label_name] = label

        data_list.append([img_path, label_name])
    return data_list, label_dict


class PlateDataset(Dataset):

    def __init__(self, data_root, is_train=True, input_shape=(160, 48),
                 only_ccpd2019=False, only_ccpd2020=False, only_others=False):
        self.data_root = data_root
        self.is_train = is_train
        self.input_shape = input_shape

        if is_train:
            if only_ccpd2019:
                dir_name_list = [
                    'CCPD2019/train',
                    'CCPD2019/val',
                ]
            elif only_ccpd2020:
                dir_name_list = [
                    'CCPD2020/train',
                    'CCPD2020/val',
                ]
            elif only_others:
                dir_name_list = [
                    'git_plate/CCPD_CRPD_OTHER_ALL',
                ]
            else:
                dir_name_list = [
                    'CCPD2019/train',
                    'CCPD2019/val',
                    'CCPD2020/train',
                    'CCPD2020/val',
                    'git_plate/CCPD_CRPD_OTHER_ALL',
                ]
        else:
            if only_ccpd2019:
                dir_name_list = [
                    'CCPD2019/test',
                ]
            elif only_ccpd2020:
                dir_name_list = [
                    'CCPD2020/test',
                ]
            elif only_others:
                dir_name_list = [
                    'git_plate/val_verify',
                ]
            else:
                dir_name_list = [
                    'CCPD2019/test',
                    'CCPD2020/test',
                    'git_plate/val_verify',
                ]

        img_list = []
        for dir_name in dir_name_list:
            data_dir = os.path.join(data_root, dir_name)
            assert os.path.isdir(data_dir), data_dir
            img_list.extend(load_data(data_dir, pattern="*.jpg"))
        assert len(img_list) > 0, data_root
        data_list, label_dict = create_plate_label(img_list)
        if RANK in {-1, 0}:
            print(f"Load {'train' if is_train else 'test'} data: {len(data_list)}")

        self.data_list = data_list
        self.dataset_len = len(data_list)
        self.label_dict = label_dict

        self.transform = transforms.Compose([
            transforms.ToPILImage(),  # 将 numpy array 或 tensor 转换成 PIL Image
            transforms.RandomRotation(15, fill=0),  # 限制旋转角度
            transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # 减小仿射变换的程度
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 适度的颜色变换
            # transforms.ToTensor(),  # 转换为 tensor
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 使用 ImageNet 的均值和标准差进行归一化
        ])

    def __getitem__(self, index):
        assert index < self.dataset_len

        img_path, label_name = self.data_list[index]
        image = cv2.imread(img_path)
        if image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        if self.is_train and random.random() > 0.5:
            image = self.transform(image)
            image = np.array(image, dtype=np.uint8)
        image = cv2.resize(image, self.input_shape)

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
