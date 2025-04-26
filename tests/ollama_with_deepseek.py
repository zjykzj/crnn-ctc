# -*- coding: utf-8 -*-

"""
@Time    : 2025/4/26 15:33
@File    : ollama_with_deepseek.py
@Author  : zj
@Description: 
"""

import json
import requests

# 配置 Ollama 的本地地址和端口
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"


def chat_with_ollama(prompt, model_name="deepseek-r1:8b"):
    """
    与本地 Ollama 模型进行对话。

    :param prompt: 用户输入的对话内容
    :param model_name: 使用的模型名称，默认为 "deepseek-r1:8b"
    :return: 模型生成的回复
    """
    headers = {
        "Content-Type": "application/json"
    }

    # 构造请求数据
    data = {
        "model": model_name,
        "prompt": prompt,
        "stream": False  # 设置为 False，表示一次性返回完整结果
    }

    try:
        # 发送 POST 请求到 Ollama API
        response = requests.post(OLLAMA_URL, headers=headers, data=json.dumps(data))

        # 检查响应状态码
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "未能获取有效回复")
        else:
            return f"请求失败，状态码：{response.status_code}，错误信息：{response.text}"
    except Exception as e:
        return f"发生错误：{str(e)}"


if __name__ == "__main__":
    print("欢迎使用 Ollama 对话脚本！输入 'exit' 退出。")
    while True:
        user_input = input("\n你: ")
        if user_input.lower() in ["exit", "quit"]:
            print("再见！")
            break

        # 调用 Ollama 进行对话
        response = chat_with_ollama(user_input)
        print(f"Ollama: {response}")
