# -*- coding: utf-8 -*-

"""
@Time    : 2025/4/26 15:47
@File    : eval_plate.py
@Author  : zj
@Description: 使用多模态大模型进行车牌识别评估。
    支持两种后端：
      - Ollama（默认）：本地 Ollama 服务的 /api/chat 接口
      - vLLM / OpenAI-compatible：vLLM 或其他兼容 OpenAI Chat API 的服务

    模型要求：vision-capable 的多模态模型。
    推荐模型（按优先级）：
      - qwen3-vl:2b    (2B, 视觉原生, 中文 OCR 强)
      - glm-ocr         (0.9B, 智谱, OCR 专项 SOTA, 超轻量)
      - deepseek-ocr    (3B, OCR 专项, OmniDocBench 91%)
      - qwen3.5:2b      (2B, 文本模型+视觉能力, 默认)

    Ollama:
      启动 Ollama: ollama serve &
      预热模型:   curl http://localhost:11434/api/chat -d '{"model":"qwen3.5:2b","messages":[{"role":"user","content":"ping"}],"max_tokens":1,"think":false,"stream":false}'

    vLLM:
      启动 vLLM:  vllm serve Qwen/Qwen3.5-2B --port 13000
      启动 vLLM (LM Studio):  python -m vllm.entrypoints.openai.api_server --model Qwen3.5-2B --port 8000

Usage - Ollama (default):
    $ python3 llm/eval_plate.py ../datasets/chinese_license_plate/recog/
    $ python3 llm/eval_plate.py ../datasets/chinese_license_plate/recog/ --model qwen3-vl:2b

Usage - vLLM / OpenAI-compatible:
    $ python3 llm/eval_plate.py ../datasets/chinese_license_plate/recog/ --backend vllm --base-url http://127.0.0.1:13000
    $ python3 llm/eval_plate.py ../datasets/chinese_license_plate/recog/ --backend vllm --base-url http://127.0.0.1:8000 --model Qwen3.5-2B

Usage - Specify which dataset to evaluate:
    $ python3 llm/eval_plate.py ../datasets/chinese_license_plate/recog/ --only-ccpd2019
    $ python3 llm/eval_plate.py ../datasets/chinese_license_plate/recog/ --only-ccpd2020
    $ python3 llm/eval_plate.py ../datasets/chinese_license_plate/recog/ --only-others

Usage - Multi-threaded (e.g., 4 workers):
    $ python3 llm/eval_plate.py ../datasets/chinese_license_plate/recog/ --workers 4

Usage - Custom output path:
    $ python3 llm/eval_plate.py ../datasets/chinese_license_plate/recog/ --output results.json
"""

import os
import sys
import json
import time
import base64
import argparse
import traceback
from io import BytesIO
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests
from PIL import Image
from tqdm import tqdm

# 延迟导入，避免 --help 时依赖 torch
PlateDataset = None

# ---------------------------------------------------------------------------
# 常量配置
# ---------------------------------------------------------------------------

# Ollama Chat API 地址
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

# vLLM / OpenAI-compatible API 默认地址
VLLM_DEFAULT_BASE_URL = "http://127.0.0.1:8000"

# vLLM 模型默认名（不带 tag 后缀）
VLLM_DEFAULT_MODEL = "Qwen3.5-2B"

# 最大重试次数
MAX_RETRIES = 3

# 单次请求超时（秒）
REQUEST_TIMEOUT = 60

# 模型配置：该模型是否支持 think: false 参数
# qwen3-vl:2b 有 bug — think: false 会把输出写入 thinking 字段导致 content 为空
# 匹配规则：按前缀匹配，命中第一个即生效
_MODEL_THINK_CONFIG = {
    "qwen3-vl": False,      # bug: content 为空
    "minicpm-v": False,     # 不支持 think 参数
    # 默认（未命中）: False — 不传 think 参数，安全第一
}

# 模型 fallback: 当 content 为空时，是否从 thinking 字段解析
_MODEL_FALLBACK_THINKING = {
    "qwen3-vl": True,       # bug 导致输出在 thinking 里
}

# 图片预处理：长边最大像素数
MAX_IMAGE_SIZE = 1024

# Prompt 版本标识（随 prompt 修改递增，用于结果溯源）
PROMPT_VERSION = "v4"


# ---------------------------------------------------------------------------
# 结构化数据类
# ---------------------------------------------------------------------------

@dataclass
class PlatePrediction:
    """单次车牌预测的结构化结果"""
    plate: Optional[str] = None         # 识别出的车牌号，None 表示无法识别
    raw_response: str = ""              # API 原始返回文本
    is_valid_json: bool = False         # JSON 解析是否成功
    error: Optional[str] = None         # 异常信息


@dataclass
class EvalConfig:
    """评估配置（保存到结果 JSON 中，便于事后对比）"""
    model: str = ""
    backend: str = "ollama"             # ollama / vllm
    base_url: str = ""                  # vLLM 时使用
    prompt_version: str = ""
    temperature: float = 0.0
    max_image_size: int = MAX_IMAGE_SIZE
    val_root: str = ""
    dataset_filter: str = "all"         # all / ccpd2019 / ccpd2020 / others
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ---------------------------------------------------------------------------
# Prompt 设计
# ---------------------------------------------------------------------------

# --- v3 备份（2025-04 至 2026-07-18, model: Qwen3.5-2B, 准确率 86.04%）---
# SYSTEM_PROMPT = """你是一个精确的车牌识别系统。你的任务是识别图像中的中国机动车号牌，并严格返回 JSON 格式的结果。
#
# 车牌格式规则：
# - 标准车牌（蓝牌/黄牌）：1个汉字(省份简称) + 1个英文字母(发牌机关代号) + 5位数字或字母 = 共7位
# - 新能源车牌（绿牌）：1个汉字(省份简称) + 1个英文字母(发牌机关代号) + 6位数字或字母 = 共8位
# - 省份简称范围：京津沪渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新
# - 发牌机关代号和序号中不会出现字母 I 和 O
#
# 输出要求：
# 1. 如果图像中存在清晰可辨的车牌，返回：{"plate": "完整的车牌号码"}
#   - 车牌号码必须连续书写，禁止在中间插入任何分隔符（如 ·、-、空格、/ 等）
#   - 示例正确输出：{"plate": "沪ADH3906"}，错误输出：{"plate": "沪A·DH3906"}
# 2. 如果图像中没有车牌、车牌模糊不清、或被遮挡无法完整识别，返回：{"plate": null}
# 3. 只返回 JSON，不要包含任何其他文字、解释或标记"""

SYSTEM_PROMPT = """你是一个精确的车牌识别系统。你的任务是识别图像中的中国机动车号牌，并严格返回 JSON 格式的结果。

车牌格式规则：
- 标准车牌（蓝牌/黄牌）：1个汉字(省份简称) + 1个英文字母(发牌机关代号) + 5位数字或字母 = 共7位
- 新能源车牌（绿牌）：1个汉字(省份简称) + 1个英文字母(发牌机关代号) + 6位数字或字母 = 共8位
- 省份简称范围：京津沪渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新
- 发牌机关代号和序号中不会出现字母 I 和 O

⚠️ 关键识别提醒：
- 省份汉字是车牌的【第一个字符】，请仔细观察汉字的笔画和结构，不要凭直觉猜测
- 以下省份字形相似，极易混淆：皖 vs 沪 vs 苏 vs 浙 vs 赣 | 京 vs 琼 | 粤 vs 鄂 | 豫 vs 晋
- 数字和字母易混淆对：0 ↔ O ↔ D ↔ Q | 8 ↔ B | 2 ↔ Z | 5 ↔ S | 1 ↔ I | 7 ↔ T
- 车牌图像可能有倾斜、光照不均或部分遮挡，请以实际看到的为准

输出要求：
1. 如果图像中存在清晰可辨的车牌，返回：{"plate": "完整的车牌号码"}
   - 车牌号码必须连续书写，禁止在中间插入任何分隔符（如 ·、-、空格、/ 等）
   - 示例正确输出：{"plate": "沪ADH3906"}，错误输出：{"plate": "沪A·DH3906"}
2. 如果图像中没有车牌、车牌模糊不清、或被遮挡无法完整识别，返回：{"plate": null}
3. 只返回 JSON，不要包含任何其他文字、解释或标记"""

def build_user_prompt() -> str:
    """构建用户消息中的识别指令。"""
    return "请识别这张图像中的车牌号码。"


# ---------------------------------------------------------------------------
# 图像预处理
# ---------------------------------------------------------------------------

def clean_plate(plate: Optional[str]) -> Optional[str]:
    """
    清洗车牌号：去除分隔符（·、-、空格、/、.、, 等）。
    模型有时会在省份字母和数字之间插入 · 或空格，需要后处理清理。
    返回清洗后的字符串，或 None。
    """
    if plate is None:
        return None
    import re
    # 去掉所有非汉字、非字母、非数字的字符
    cleaned = re.sub(r'[^京津沪渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新'
                     r'A-Za-z0-9]', '', plate)
    return cleaned if cleaned else None


def preprocess_image(image: Image.Image, max_size: int = MAX_IMAGE_SIZE) -> Image.Image:
    """
    将图像长边缩放到 max_size 以内，保持宽高比。
    减小传输体积，加快推理速度。
    """
    w, h = image.size
    longest = max(w, h)
    if longest <= max_size:
        return image
    ratio = max_size / longest
    new_size = (int(w * ratio), int(h * ratio))
    return image.resize(new_size, Image.LANCZOS)


def encode_image(image: Image.Image) -> str:
    """将 PIL Image 编码为 Base64 字符串（JPEG 格式，质量 85）。"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Ollama Chat API 调用
# ---------------------------------------------------------------------------

def _get_model_think_config(model: str) -> bool:
    """根据模型名前缀判断是否传 think: false。"""
    for prefix, enabled in _MODEL_THINK_CONFIG.items():
        if model.startswith(prefix):
            return enabled
    return False  # 默认不传，安全第一


def _need_fallback_thinking(model: str) -> bool:
    """判断该模型是否需要从 thinking 字段 fallback 解析。"""
    for prefix, needed in _MODEL_FALLBACK_THINKING.items():
        if model.startswith(prefix):
            return needed
    return False


def query_ollama_chat(
    image: Image.Image,
    model: str = "qwen3.5:2b",
    temperature: float = 0.0,
    max_tokens: int = 100,
) -> PlatePrediction:
    """
    使用 Ollama Chat API（/api/chat）进行车牌识别。

    - 自动对图像做 resize 预处理
    - 使用 system prompt + user prompt 的 chat 格式
    - 强制 JSON 输出模式（format: "json"）
    - 根据模型名自动决定是否传 think: false
    - 对于 qwen3-vl 等 bug 模型，自动从 thinking 字段 fallback 解析
    - 失败自动重试（指数退避）

    返回 PlatePrediction 结构化结果。
    """
    # 预处理 & 编码
    image = preprocess_image(image)
    image_base64 = encode_image(image)

    use_think = _get_model_think_config(model)
    need_fallback = _need_fallback_thinking(model)

    payload: dict = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": build_user_prompt(),
                "images": [image_base64],
            },
        ],
        "format": "json",
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 1.0,
        },
        "max_tokens": max_tokens,
    }

    if use_think:
        payload["think"] = False

    headers = {
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                OLLAMA_CHAT_URL,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            msg = response.json().get("message", {})
            raw_response = msg.get("content", "").strip()

            # fallback: 部分模型（qwen3-vl）think:false 时 content 为空，
            # 但实际输出在 thinking 字段里
            if not raw_response and need_fallback:
                thinking_text = msg.get("thinking", "").strip()
                if thinking_text:
                    raw_response = thinking_text

            # 尝试解析 JSON
            try:
                data = json.loads(raw_response)
                plate = data.get("plate")
                # plate 必须是 str 或 None
                if plate is not None and not isinstance(plate, str):
                    plate = str(plate)
                return PlatePrediction(
                    plate=plate,
                    raw_response=raw_response,
                    is_valid_json=True,
                )
            except (json.JSONDecodeError, TypeError):
                return PlatePrediction(
                    plate=None,
                    raw_response=raw_response,
                    is_valid_json=False,
                    error=f"JSON parse error: {raw_response[:200]}",
                )

        except requests.exceptions.Timeout:
            last_error = f"Timeout after {REQUEST_TIMEOUT}s"
        except requests.exceptions.ConnectionError:
            last_error = "Connection error"
        except requests.exceptions.RequestException as e:
            last_error = f"Request error: {e}"
        except Exception as e:
            last_error = f"Unexpected error: {traceback.format_exc()[:300]}"

        if attempt < MAX_RETRIES:
            wait = 2 ** attempt  # 指数退避: 2s → 4s → 8s
            tqdm.write(f"  [Retry {attempt}/{MAX_RETRIES}] {last_error}, waiting {wait}s...")
            time.sleep(wait)

    # 所有重试均失败
    return PlatePrediction(
        plate=None,
        raw_response="",
        is_valid_json=False,
        error=f"[Retries exhausted] {last_error}",
    )


def query_vllm_chat(
    image: Image.Image,
    model: str = VLLM_DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 100,
    base_url: str = VLLM_DEFAULT_BASE_URL,
) -> PlatePrediction:
    """
    使用 vLLM / OpenAI-compatible Chat API（/v1/chat/completions）进行车牌识别。

    - 自动对图像做 resize 预处理
    - 使用 system prompt + user prompt 的 chat 格式
    - 强制 JSON 输出模式（response_format: json_object）
    - 失败自动重试（指数退避）

    返回 PlatePrediction 结构化结果。
    """
    # 预处理 & 编码
    image = preprocess_image(image)
    image_base64 = encode_image(image)

    chat_url = f"{base_url.rstrip('/')}/v1/chat/completions"

    # OpenAI-compatible vision format: content is an array of parts
    payload: dict = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_user_prompt()},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                        },
                    },
                ],
            },
        ],
        "response_format": {"type": "json_object"},
        "temperature": temperature,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "stream": False,
    }

    headers = {
        "Content-Type": "application/json",
    }

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                chat_url,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            body = response.json()
            choices = body.get("choices", [])
            if not choices:
                return PlatePrediction(
                    plate=None,
                    raw_response="",
                    is_valid_json=False,
                    error="vLLM returned empty choices",
                )

            raw_response = choices[0].get("message", {}).get("content", "").strip()

            # 尝试解析 JSON
            try:
                data = json.loads(raw_response)
                plate = data.get("plate")
                # plate 必须是 str 或 None
                if plate is not None and not isinstance(plate, str):
                    plate = str(plate)
                return PlatePrediction(
                    plate=plate,
                    raw_response=raw_response,
                    is_valid_json=True,
                )
            except (json.JSONDecodeError, TypeError):
                return PlatePrediction(
                    plate=None,
                    raw_response=raw_response,
                    is_valid_json=False,
                    error=f"JSON parse error: {raw_response[:200]}",
                )

        except requests.exceptions.Timeout:
            last_error = f"Timeout after {REQUEST_TIMEOUT}s"
        except requests.exceptions.ConnectionError:
            last_error = f"Connection error ({chat_url})"
        except requests.exceptions.RequestException as e:
            last_error = f"Request error: {e}"
        except Exception as e:
            last_error = f"Unexpected error: {traceback.format_exc()[:300]}"

        if attempt < MAX_RETRIES:
            wait = 2 ** attempt  # 指数退避: 2s → 4s → 8s
            tqdm.write(f"  [Retry {attempt}/{MAX_RETRIES}] {last_error}, waiting {wait}s...")
            time.sleep(wait)

    # 所有重试均失败
    return PlatePrediction(
        plate=None,
        raw_response="",
        is_valid_json=False,
        error=f"[Retries exhausted] {last_error}",
    )


# ---------------------------------------------------------------------------
# 单样本评估（供并发调用）
# ---------------------------------------------------------------------------

def evaluate_single(task: Tuple[str, str, str, float, str, str]) -> dict:
    """
    评估单张图像。接收一个 tuple 参数以兼容 ThreadPoolExecutor。

    Args:
        task: (img_path, label_name, model, temperature, backend, base_url)

    Returns:
        单条结果 dict，keys: img_path, label, pred, correct, latency_ms
    """
    img_path, label_name, model, temperature, backend, base_url = task

    start = time.time()

    # 加载图像
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        latency_ms = (time.time() - start) * 1000
        return {
            "img_path": img_path,
            "label": label_name,
            "pred": asdict(PlatePrediction(
                plate=None,
                is_valid_json=False,
                error=f"Image load error: {e}",
            )),
            "correct": False,
            "latency_ms": round(latency_ms, 1),
        }

    # 调用模型识别
    if backend == "vllm":
        pred = query_vllm_chat(image, model=model, temperature=temperature, base_url=base_url)
    else:
        pred = query_ollama_chat(image, model=model, temperature=temperature)
    latency_ms = (time.time() - start) * 1000

    # 后处理清洗：去除模型可能插入的分隔符（·、空格、- 等）
    pred.plate = clean_plate(pred.plate)

    is_correct = (pred.plate == label_name)

    return {
        "img_path": img_path,
        "label": label_name,
        "pred": asdict(pred),
        "correct": is_correct,
        "latency_ms": round(latency_ms, 1),
    }


# ---------------------------------------------------------------------------
# 主评估流程
# ---------------------------------------------------------------------------

def _ensure_dataset_import():
    """延迟导入 PlateDataset，避免 --help 时依赖 torch。"""
    global PlateDataset
    if PlateDataset is None:
        from plate_dataset import PlateDataset as _PlateDataset
        PlateDataset = _PlateDataset


def val(args):
    """主评估流程：加载数据 → 逐张/并发推理 → 汇总 → 保存结果。"""
    _ensure_dataset_import()
    val_root = args.val_root

    # ---- 加载数据集 ----
    val_dataset = PlateDataset(
        val_root, is_train=False,
        only_ccpd2019=args.only_ccpd2019,
        only_ccpd2020=args.only_ccpd2020,
        only_others=args.only_others,
    )
    total_num = len(val_dataset)

    # ---- 确定 base_url ----
    base_url = args.base_url
    if args.backend == "vllm" and base_url is None:
        base_url = VLLM_DEFAULT_BASE_URL
    elif base_url is None:
        base_url = ""  # Ollama 不需要 base_url

    # ---- 构建任务列表 ----
    tasks: List[Tuple[str, str, str, float, str, str]] = []
    for idx in range(total_num):
        _, label_name, img_path = val_dataset[idx]
        tasks.append((img_path, label_name, args.model, args.temperature, args.backend, base_url))

    # ---- 确定输出路径 ----
    if args.output:
        output_path = args.output
    else:
        os.makedirs("llm/results", exist_ok=True)
        model_slug = args.model.replace(":", "_").replace("/", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"llm/results/{model_slug}_{timestamp}.json"

    # ---- 评估配置 ----
    dataset_filter = "all"
    if args.only_ccpd2019:
        dataset_filter = "ccpd2019"
    elif args.only_ccpd2020:
        dataset_filter = "ccpd2020"
    elif args.only_others:
        dataset_filter = "others"

    config = EvalConfig(
        model=args.model,
        backend=args.backend,
        base_url=base_url,
        prompt_version=PROMPT_VERSION,
        temperature=args.temperature,
        max_image_size=MAX_IMAGE_SIZE,
        val_root=val_root,
        dataset_filter=dataset_filter,
    )

    # ---- 打印配置 ----
    print(f"\n{'=' * 60}")
    print(f"后端:       {config.backend}")
    if config.backend == "vllm":
        print(f"API 地址:   {base_url}")
    print(f"模型:       {config.model}")
    print(f"温度:       {config.temperature}")
    print(f"Prompt 版本: {config.prompt_version}")
    print(f"数据集:     {val_root}  ({dataset_filter})")
    print(f"样本数:     {total_num}")
    print(f"并发数:     {args.workers}  {'(sequential)' if args.workers <= 1 else ''}")
    print(f"结果输出:   {output_path}")
    print(f"{'=' * 60}\n")

    # ---- 推理 ----
    results: List[dict] = []
    correct_num = 0
    json_failures = 0
    null_predictions = 0
    error_count = 0

    start_time = time.time()

    if args.workers <= 1:
        # 顺序执行
        for task in tqdm(tasks, desc="Evaluating", unit="img"):
            result = evaluate_single(task)
            results.append(result)

            if result["correct"]:
                correct_num += 1
            pred = result["pred"]
            if pred.get("error"):
                error_count += 1
            if not pred.get("is_valid_json"):
                json_failures += 1
            if pred.get("plate") is None and not pred.get("error"):
                null_predictions += 1
    else:
        # 多线程并发执行
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(evaluate_single, task): task for task in tasks}
            with tqdm(total=len(tasks), desc="Evaluating", unit="img") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)

                    if result["correct"]:
                        correct_num += 1
                    pred = result["pred"]
                    if pred.get("error"):
                        error_count += 1
                    if not pred.get("is_valid_json"):
                        json_failures += 1
                    if pred.get("plate") is None and not pred.get("error"):
                        null_predictions += 1

                    pbar.update(1)

        # 按 img_path 排序恢复原始顺序
        results.sort(key=lambda r: r["img_path"])

    total_duration = time.time() - start_time

    # ---- 汇总统计 ----
    accuracy = correct_num / total_num if total_num > 0 else 0.0
    latencies = [r["latency_ms"] for r in results if r["latency_ms"] > 0]
    avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0

    summary = {
        "total": total_num,
        "correct": correct_num,
        "accuracy": round(accuracy, 4),
        "json_parse_failures": json_failures,
        "null_predictions": null_predictions,
        "error_count": error_count,
        "total_duration_s": round(total_duration, 2),
        "avg_latency_ms": round(avg_latency_ms, 1),
    }

    # ---- 保存结果 ----
    output_data = {
        "config": asdict(config),
        "summary": summary,
        "results": results,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 结果已保存到: {output_path}")

    # ---- 打印最终统计 ----
    print("\n" + "=" * 60)
    print("=== 验证结果 ===")
    print(f"总样本数:          {total_num}")
    print(f"正确样本数:        {correct_num}")
    print(f"准确率:            {accuracy * 100:.2f}%")
    print(f"JSON 解析失败:     {json_failures}")
    print(f"无法识别 (null):   {null_predictions}")
    print(f"请求异常:          {error_count}")
    print(f"总耗时:            {total_duration:.2f} 秒")
    print(f"平均每次耗时:      {avg_latency_ms:.1f} ms")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_opt():
    parser = argparse.ArgumentParser(
        description='Eval license plate recognition using Ollama multimodal LLM'
    )
    parser.add_argument('val_root', metavar='DIR', type=str,
                        help='path to val dataset')

    parser.add_argument('--backend', type=str, default='ollama',
                        choices=['ollama', 'vllm'],
                        help='API backend: ollama or vllm (OpenAI-compatible) '
                             '(default: ollama)')
    parser.add_argument('--base-url', type=str, default=None,
                        help='vLLM/OpenAI-compatible API base URL '
                             '(default: http://127.0.0.1:8000). '
                             'Only used when --backend vllm')
    parser.add_argument('--model', type=str, default='qwen3.5:2b',
                        help='Model name. Ollama uses tag suffix (e.g. qwen3.5:2b), '
                             'vLLM uses raw name (e.g. Qwen3.5-2B)')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='LLM temperature for evaluation (default: 0.0)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers (default: 1, sequential)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file path '
                             '(default: llm/results/{model}_{timestamp}.json)')

    parser.add_argument('--only-ccpd2019', action='store_true',
                        help='only eval CCPD2019/test dataset')
    parser.add_argument('--only-ccpd2020', action='store_true',
                        help='only eval CCPD2020/test dataset')
    parser.add_argument('--only-others', action='store_true',
                        help='only eval git_plate/val_verify dataset')

    args = parser.parse_args()
    print(f"args: {args}")
    return args


def main():
    args = parse_opt()
    val(args)


if __name__ == '__main__':
    main()
