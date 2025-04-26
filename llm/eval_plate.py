# -*- coding: utf-8 -*-

"""
@Time    : 2025/4/26 15:47
@File    : eval_plate.py
@Author  : zj
@Description:

Usage - Eval using Ollama_with_MiniCPM-V:
    $ python3 eval_plate.py ../datasets/chinese_license_plate/recog/

Usage - Specify which dataset to evaluate:
    $ python3 eval_plate.py ../datasets/chinese_license_plate/recog/ --only-ccpd2019
    $ python3 eval_plate.py ../datasets/chinese_license_plate/recog/ --only-ccpd2020
    $ python3 eval_plate.py ../datasets/chinese_license_plate/recog/ --only-others

=== 验证结果 ===
总样本数: 5006
正确样本数: 2801
准确率: 55.95%
总耗时: 2563.55 秒
平均每次预测耗时: 0.51 秒

"""

import re
import time
import torch
import argparse
import requests

from io import BytesIO

from llm.plate_dataset import PlateDataset, PLATE_CHARS

# 配置 Ollama API 地址
OLLAMA_API_URL = "http://localhost:11434/api/generate"


def parse_opt():
    parser = argparse.ArgumentParser(description='Eval CRNN/LPRNet with CCPD')
    parser.add_argument('val_root', metavar='DIR', type=str, help='path to val dataset')

    parser.add_argument('--only-ccpd2019', action='store_true', help='only eval CCPD2019/test dataset')
    parser.add_argument('--only-ccpd2020', action='store_true', help='only eval CCPD2019/test dataset')
    parser.add_argument('--only-others', action='store_true', help='only eval git_plate/val_verify dataset')

    args = parser.parse_args()
    print(f"args: {args}")
    return args


def encode_image(image):
    """
    将图像编码为 Base64 格式，以便发送给 Ollama API。
    """
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    import base64
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def query_ollama_with_image(image, prompt, model="minicpm-v:latest"):
    """
    向 Ollama 发送图像和文本提示，获取生成结果。
    """
    # 编码图像为 Base64
    image_base64 = encode_image(image)

    # 构造请求数据
    payload = {
        "model": model,  # 替换为你的多模态模型名称
        "prompt": prompt,
        "images": [image_base64],  # 图像列表
        "stream": False
    }

    # 发送 POST 请求
    response = requests.post(OLLAMA_API_URL, json=payload)
    if response.status_code == 200:
        return response.json().get("response", "")
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")


def clean_license_plate(plate):
    """
    清理车牌号码，移除间隔点或其他非必要字符。

    :param plate: 原始车牌号码（可能包含间隔点）
    :return: 清理后的车牌号码
    """
    # 使用正则表达式移除非字母数字字符
    cleaned_plate = re.sub(r"[^\w]", "", plate)
    return cleaned_plate


@torch.no_grad()
def val(args, val_root):
    val_dataset = PlateDataset(val_root, is_train=False, only_ccpd2019=args.only_ccpd2019,
                               only_ccpd2020=args.only_ccpd2020, only_others=args.only_others)

    correct_num = 0
    total_num = len(val_dataset)

    prompt = """
    任务：识别图像中的车牌号码。
    要求：
    - 只返回车牌号码，不包含任何其他说明、解释或额外信息。
    - 车牌号码必须符合以下格式：
      - 一个汉字（代表省份或直辖市），后跟随字母和数字。
      - 示例格式：沪DH5311 / 皖AD62388 / 皖ADT1060
    - 如果无法识别车牌，请返回 "无法识别"。
    - 不要在输出中包含任何标点符号、空格或其他无关字符。

    请根据上述要求处理图像，并直接返回车牌号码。
    """
    # 记录总时间
    start_time = time.time()
    for idx in range(total_num):
        # 获取图像和真实标签
        image, label_name = val_dataset.__getitem__(idx)

        # 开始单次预测计时
        single_start_time = time.time()

        # 查询 Ollama 模型进行预测
        pred_plate = query_ollama_with_image(image, prompt)
        pred_plate = clean_license_plate(pred_plate)

        # 单次预测结束计时
        single_end_time = time.time()
        single_duration = single_end_time - single_start_time

        # 判断预测是否正确
        is_correct = (pred_plate == label_name)
        if is_correct:
            correct_num += 1

        print(
            f"[{idx + 1}/{total_num}] "
            f"Pred: {pred_plate} | Label: {label_name} | "
            f"Correct: {is_correct} | Time: {single_duration:.2f}s"
        )

    # 总时间计算
    end_time = time.time()
    total_duration = end_time - start_time

    # 计算准确率和平均预测时间
    accuracy = correct_num / total_num
    avg_time_per_pred = total_duration / total_num

    # 打印最终统计信息
    print("\n=== 验证结果 ===")
    print(f"总样本数: {total_num}")
    print(f"正确样本数: {correct_num}")
    print(f"准确率: {accuracy * 100:.2f}%")
    print(f"总耗时: {total_duration:.2f} 秒")
    print(f"平均每次预测耗时: {avg_time_per_pred:.2f} 秒")


def main():
    args = parse_opt()

    val(args, args.val_root)


if __name__ == '__main__':
    main()
