# -*- coding: utf-8 -*-

"""
@Time    : 2025/4/26 15:47
@File    : eval_plate.py
@Author  : zj
@Description:

Usage - Eval using Ollama_with_MiniCPM-V:
    $ python3 llm/eval_plate.py ../datasets/chinese_license_plate/recog/

Usage - Specify which dataset to evaluate:
    $ python3 llm/eval_plate.py ../datasets/chinese_license_plate/recog/ --only-ccpd2019
    $ python3 llm/eval_plate.py ../datasets/chinese_license_plate/recog/ --only-ccpd2020
    $ python3 llm/eval_plate.py ../datasets/chinese_license_plate/recog/ --only-others

=== 验证结果 ===
总样本数: 5006
正确样本数: 3024
准确率: 60.41%
总耗时: 2989.26 秒
平均每次预测耗时: 0.60 秒

"""

import gc
import uuid
import time
import torch
import argparse
import requests

from io import BytesIO

from plate_dataset import PlateDataset

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


def clean_license_plate(text):
    """
    清理 API 返回结果，提取并标准化车牌号码。
    支持清除中间的点（·）、连字符（-）、空格、引号等，并验证车牌号码格式。
    """
    # 去除所有特殊符号和空格
    cleaned_text = text.replace("·", "").replace("-", "").replace(" ", "").replace('"', '').replace("。", "").strip()
    return cleaned_text


def query_ollama_with_image(image, prompt, model="minicpm-v:latest"):
    """
    向 Ollama 发送图像和文本提示，获取生成结果。
    匹配 ChatBox 的配置：temperature=0.7, top_p=1。
    """
    unique_id = str(uuid.uuid4())
    prompt_with_id = f"{prompt} [Request ID: {unique_id}]"
    image_base64 = encode_image(image)

    payload = {
        "model": model,
        "prompt": prompt_with_id,
        "images": [image_base64],
        "stream": False,
        "temperature": 0.7,
        "top_p": 1.0,
        "max_tokens": 50,
    }

    headers = {
        "Cache-Control": "no-cache",
        "Pragma": "no-cache"
    }

    try:
        response = requests.post(OLLAMA_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        raw_response = response.json().get("response", "")
        del image_base64  # 删除已使用的变量
        gc.collect()
        return clean_license_plate(raw_response)
    except requests.exceptions.RequestException as e:
        raise Exception(f"Request failed: {e}")


@torch.no_grad()
def val(args, val_root):
    val_dataset = PlateDataset(val_root, is_train=False, only_ccpd2019=args.only_ccpd2019,
                               only_ccpd2020=args.only_ccpd2020, only_others=args.only_others)

    correct_num = 0
    total_num = len(val_dataset)

    prompt = """
    任务：识别图像中的车牌号码。
    示例：
    - 输入：图像中显示车牌为“沪DH5311”。
    - 输出：沪DH5311
    - 输入：图像中显示车牌为“皖AD16558”。
    - 输出：皖AD16558
    要求：
    - 每次请求是一个独立任务，无需参考之前的对话或上下文。
    - 只返回车牌号码，不要包含任何其他文字、标点符号或空格。
    - 如果图像中的车牌号码模糊不清或被遮挡，也请直接返回“无法识别”。
    """
    # prompt = """
    # 任务：识别图像中的车牌号码。
    #
    # 示例：
    # - 输入：图像中显示车牌为“沪DH5311”。
    # - 输出：沪DH5311
    #
    # - 输入：图像中显示车牌为“皖AD16558”。
    # - 输出：皖AD16558
    #
    # 要求：
    # - 每次请求是一个独立任务，无需参考之前的对话或上下文。
    # - 只返回车牌号码，不要包含任何其他文字、标点符号或空格。
    # - 车牌号码的格式必须是一个汉字 + 字母 + 数字（如“沪DH5311”）。
    # - 如果图像中没有清晰的车牌号码，请直接返回“无法识别”。
    # - 如果图像中的车牌号码模糊不清或被遮挡，也请直接返回“无法识别”。
    #
    # 重要规则：
    # - 不要尝试猜测或生成可能的车牌号码。
    # - 输出结果必须完全符合上述格式要求。
    #
    # 请根据上述要求处理图像，并直接返回车牌号码。
    # """
    # 记录总时间
    start_time = time.time()
    for idx in range(total_num):
        # 获取图像和真实标签
        image, label_name, img_path = val_dataset.__getitem__(idx)

        # 开始单次预测计时
        single_start_time = time.time()

        # 查询 Ollama 模型进行预测
        pred_plate = query_ollama_with_image(image, prompt)

        # 单次预测结束计时
        single_end_time = time.time()
        single_duration = single_end_time - single_start_time

        # 判断预测是否正确
        is_correct = (pred_plate == label_name)
        if is_correct:
            correct_num += 1

        print(
            f"[{idx + 1}/{total_num}] "
            f"Pred: {pred_plate} | Label: {label_name} | img_path: {img_path} | "
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
