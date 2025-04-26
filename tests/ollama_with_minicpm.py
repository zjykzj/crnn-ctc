# -*- coding: utf-8 -*-

"""
@Time    : 2025/4/26 15:29
@File    : ollama_with_minicpm.py
@Author  : zj
@Description: 
"""

import requests
from PIL import Image
from io import BytesIO

# 配置 Ollama API 地址
OLLAMA_API_URL = "http://localhost:11434/api/generate"


def load_image(image_path_or_url):
    """
    加载图像文件或 URL，并返回图像对象。
    """
    if image_path_or_url.startswith("http"):
        response = requests.get(image_path_or_url)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_path_or_url)
    return image


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


if __name__ == "__main__":
    # 用户输入图像路径或 URL
    image_input = input("请输入图像路径或 URL：")
    image = load_image(image_input)

    # 用户输入文本提示
    text_prompt = input("请输入文本提示：")

    # 查询 Ollama
    try:
        result = query_ollama_with_image(image, text_prompt)
        print("生成结果：", result)
    except Exception as e:
        print("发生错误：", str(e))
