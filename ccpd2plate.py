# -*- coding: utf-8 -*-

"""
@Time    : 2024/7/27 12:00
@File    : ccpd2plate.py
@Author  : zj
@Description:

Download the CCPD2019 and CCPD2020 datasets and store them in the following format:

```text
.
├── CCPD2019
│   ├── ccpd_base
│   ├── ccpd_blur
│   ├── ccpd_challenge
│   ├── ccpd_db
│   ├── ccpd_fn
│   ├── ccpd_np
│   ├── ccpd_rotate
│   ├── ccpd_tilt
│   ├── ccpd_weather
│   ├── LICENSE
│   ├── README.md
│   └── splits
├── CCPD2020
│   └── ccpd_green
```

Crop the license plate area of each image and use the license plate name as the file name.

"""

import sys

import cv2
import glob
import os.path

from tqdm import tqdm

# 获取当前脚本的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 假设你的模块在上级目录中
parent_dir = os.path.dirname(script_dir)

# 添加模块所在的目录到 sys.path
sys.path.insert(0, os.path.join(parent_dir))

from utils.dataset.plate import DELIMITER

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂",
             "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']


def parse_ccpd(file_name):
    # file_name: 025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg

    # ['025', '95_113', '154&383_386&473', '386&473_177&454_154&383_363&402', '0_0_22_27_27_33_16', '37', '15']
    all_infos = file_name.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')
    # print(f"all_infos: {all_infos}")

    # left-top / right-bottom
    # [[x1, y1], [x2, y2]]
    box_xyxy = [[int(eel) for eel in el.split('&')] for el in all_infos[2].split('_')]
    x1, y1 = box_xyxy[0]
    x2, y2 = box_xyxy[1]
    box_xyxy = [x1, y1, x2, y2]

    plate_indexes = all_infos[4].split("_")
    assert len(plate_indexes) >= 7, file_name
    plate_name = []
    plate_name.append(provinces[int(plate_indexes[0])])
    plate_name.append(alphabets[int(plate_indexes[1])])
    for i in range(2, len(plate_indexes)):
        plate_name.append(ads[int(plate_indexes[i])])
    plate_name = ''.join(plate_name)

    return box_xyxy, plate_name


def save_to_dst(file_path, file_name, dst_data_root):
    box_xyxy, plate_name = parse_ccpd(file_name)

    src_img = cv2.imread(file_path)
    x1, y1, x2, y2 = box_xyxy
    plate_img = src_img[y1:y2, x1:x2]

    dst_file_path = os.path.join(dst_data_root, plate_name + f'.jpg')
    if os.path.exists(dst_file_path):
        file_len = len(list(glob.glob(os.path.join(dst_data_root, f"{plate_name}*.jpg"))))
        dst_file_path = os.path.join(dst_data_root, plate_name + f'{DELIMITER}{file_len}.jpg')
    # assert not os.path.exists(dst_file_path), f"{file_name}\n{dst_file_path}"
    cv2.imwrite(dst_file_path, plate_img)


def process_ccpd2019(data_root, dst_root):
    for name in ['splits/train.txt', 'splits/val.txt', 'splits/test.txt']:
        txt_path = os.path.join(data_root, name)
        assert os.path.isfile(txt_path), txt_path
        print('*' * 100)
        print(f"Getting {txt_path} data...")

        cls_name = os.path.basename(name).split('.')[0]
        dst_data_root = os.path.join(dst_root, cls_name)
        if not os.path.exists(dst_data_root):
            os.makedirs(dst_data_root)
        print(f"Save to {dst_data_root}")

        with open(txt_path, 'r') as f:
            for line in tqdm(f.readlines()):
                line = line.strip()
                if line == '':
                    continue

                file_path = os.path.join(data_root, line)
                assert os.path.isfile(file_path), file_path
                assert file_path.endswith('.jpg'), file_path

                save_to_dst(file_path, os.path.basename(file_path), dst_data_root)


def process_ccpd2020(data_root, dst_root):
    for name in ['train', 'val', 'test']:
        data_dir = os.path.join(data_root, name)
        assert os.path.isdir(data_dir), data_dir
        print('*' * 100)
        print(f"Getting {data_dir} data...")

        dst_data_root = os.path.join(dst_root, name)
        if not os.path.exists(dst_data_root):
            os.makedirs(dst_data_root)
        print(f"Save to {dst_data_root}")

        for file_name in tqdm(os.listdir(data_dir)):
            file_path = os.path.join(data_dir, file_name)
            assert os.path.isfile(file_path), file_path
            assert file_path.endswith('.jpg'), file_path

            save_to_dst(file_path, file_name, dst_data_root)


def main():
    dst_root = "../datasets/chinese_license_plate/recog"
    data_root = "../datasets/ccpd"

    ccpd2019_root = os.path.join(data_root, "CCPD2019")
    if os.path.isdir(ccpd2019_root):
        print(f"Process {ccpd2019_root}")
        process_ccpd2019(ccpd2019_root, os.path.join(dst_root, "CCPD2019"))

    ccpd2020_root = os.path.join(data_root, "CCPD2020", "ccpd_green")
    if os.path.isdir(ccpd2020_root):
        print(f"Process {ccpd2020_root}")
        process_ccpd2020(ccpd2020_root, os.path.join(dst_root, "CCPD2020"))


if __name__ == '__main__':
    main()
