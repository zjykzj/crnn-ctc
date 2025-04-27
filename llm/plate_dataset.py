# -*- coding: utf-8 -*-

"""
@Time    : 2025/4/26 15:43
@File    : plate_dataset.py
@Author  : zj
@Description: 
"""

import os

from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

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

    def __init__(self, data_root, is_train=True,
                 only_ccpd2019=False, only_ccpd2020=False, only_others=False):
        self.data_root = data_root
        self.is_train = is_train

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

    def __getitem__(self, index):
        assert index < self.dataset_len

        img_path, label_name = self.data_list[index]
        image = Image.open(img_path)

        return image, label_name, img_path

    def __len__(self):
        return self.dataset_len


if __name__ == '__main__':
    data_root = "/home/zjykzj/datasets/chinese_license_plate"
    val_dataset = PlateDataset(data_root, is_train=False)
    print(val_dataset)

    image, label_name = val_dataset.__getitem__(100)
    print(f"image size: {image.size}")
    print(label_name)
