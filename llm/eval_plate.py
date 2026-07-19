# -*- coding: utf-8 -*-

"""
@Time    : 2025/4/26 15:47
@File    : eval_plate.py
@Author  : zj
@Description: Evaluate license plate recognition using multimodal LLMs.
    Supports two backends:
      - Ollama (default): local Ollama service via /api/chat endpoint
      - vLLM / OpenAI-compatible: vLLM or other OpenAI Chat API-compatible services

    Model requirements: vision-capable multimodal models.
    Recommended models (by priority):
      - qwen3-vl:2b    (2B, native vision, strong Chinese OCR)
      - glm-ocr         (0.9B, Zhipu, OCR-specialized SOTA, ultra-lightweight)
      - deepseek-ocr    (3B, OCR-specialized, OmniDocBench 91%)
      - qwen3.5:2b      (2B, text model with vision capability, default)

    Ollama:
      Start Ollama:  ollama serve &
      Warm up model:  curl http://localhost:11434/api/chat -d '{"model":"qwen3.5:2b","messages":[{"role":"user","content":"ping"}],"max_tokens":1,"think":false,"stream":false}'

    vLLM:
      Start vLLM:  vllm serve Qwen/Qwen3.5-2B --port 13000
      Start vLLM (LM Studio):  python -m vllm.entrypoints.openai.api_server --model Qwen3.5-2B --port 8000

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

# Deferred import to avoid requiring torch for --help
PlateDataset = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Ollama Chat API endpoint
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

# vLLM / OpenAI-compatible API default base URL
VLLM_DEFAULT_BASE_URL = "http://127.0.0.1:8000"

# vLLM default model name (without tag suffix)
VLLM_DEFAULT_MODEL = "Qwen3.5-2B"

# Maximum retry attempts
MAX_RETRIES = 3

# Single request timeout (seconds)
REQUEST_TIMEOUT = 60

# Model config: whether the model supports the `think: false` parameter.
# qwen3-vl:2b has a bug — `think: false` writes output into the thinking field,
# leaving content empty.
# Matching rule: prefix match; first hit wins.
_MODEL_THINK_CONFIG = {
    "qwen3-vl": False,      # bug: content is empty
    "minicpm-v": False,     # does not support think parameter
    # Default (no match): False — skip think parameter; safety first
}

# Model fallback: when content is empty, whether to parse from thinking field
_MODEL_FALLBACK_THINKING = {
    "qwen3-vl": True,       # bug causes output to land in thinking field
}

# Image preprocessing: maximum pixels on the long side
MAX_IMAGE_SIZE = 1024

# Prompt version identifier (incremented with prompt changes for result traceability)
PROMPT_VERSION = "v4"


# ---------------------------------------------------------------------------
# 结构化数据类
# ---------------------------------------------------------------------------

@dataclass
class PlatePrediction:
    """Structured result for a single license plate prediction"""
    plate: Optional[str] = None         # recognized plate number, None means unrecognizable
    raw_response: str = ""              # raw API response text
    is_valid_json: bool = False         # whether JSON parsing succeeded
    error: Optional[str] = None         # exception info


@dataclass
class EvalConfig:
    """Evaluation config (saved into result JSON for post-hoc comparison)"""
    model: str = ""
    backend: str = "ollama"             # ollama / vllm
    base_url: str = ""                  # used for vLLM
    prompt_version: str = ""
    temperature: float = 0.0
    max_image_size: int = MAX_IMAGE_SIZE
    val_root: str = ""
    dataset_filter: str = "all"         # all / ccpd2019 / ccpd2020 / others
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ---------------------------------------------------------------------------
# Prompt Design
# ---------------------------------------------------------------------------

# --- v3 backup (2025-04 to 2026-07-18, model: Qwen3.5-2B, accuracy 86.04%) ---
# SYSTEM_PROMPT = """You are an accurate license plate recognition system. Your task is to recognize
# Chinese motor vehicle plates in images and return results strictly in JSON format.
#
# Plate format rules:
# - Standard plates (blue/yellow): 1 Chinese character (province abbreviation) + 1 English letter
#   (issuing authority code) + 5 digits/letters = 7 characters total
# - New energy plates (green): 1 Chinese character (province abbreviation) + 1 English letter
#   (issuing authority code) + 6 digits/letters = 8 characters total
# - Province abbreviations: 京津沪渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新
# - Letters I and O never appear in issuing authority codes or serial numbers
#
# Output requirements:
# 1. If a clearly legible plate exists in the image, return: {"plate": "full plate number"}
#   - The plate number must be written consecutively; do NOT insert separators
#     (such as ·, -, space, /, etc.)
#   - Correct: {"plate": "沪ADH3906"}, incorrect: {"plate": "沪A·DH3906"}
# 2. If there is no plate, the plate is blurry, or it is occluded beyond full
#    recognition, return: {"plate": null}
# 3. Return only JSON; do NOT include any other text, explanations, or markup"""

SYSTEM_PROMPT = """You are an accurate license plate recognition system. Your task is to recognize Chinese motor vehicle plates in images and return results strictly in JSON format.

Plate format rules:
- Standard plates (blue/yellow): 1 Chinese character (province abbreviation) + 1 English letter (issuing authority code) + 5 digits/letters = 7 characters total
- New energy plates (green): 1 Chinese character (province abbreviation) + 1 English letter (issuing authority code) + 6 digits/letters = 8 characters total
- Province abbreviations: 京津沪渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新
- Letters I and O never appear in issuing authority codes or serial numbers

⚠️ Critical recognition tips:
- The province character is the **first character** of the plate; carefully observe stroke structure, do not guess by intuition
- These province pairs are visually similar and easily confused: 皖 vs 沪 vs 苏 vs 浙 vs 赣 | 京 vs 琼 | 粤 vs 鄂 | 豫 vs 晋
- Digit/letter confusion pairs: 0 ↔ O ↔ D ↔ Q | 8 ↔ B | 2 ↔ Z | 5 ↔ S | 1 ↔ I | 7 ↔ T
- Plate images may be tilted, unevenly lit, or partially occluded; rely on what you actually see

Output requirements:
1. If a clearly legible plate exists in the image, return: {"plate": "full plate number"}
   - The plate number must be written consecutively; do NOT insert separators (such as ·, -, space, /, etc.)
   - Correct: {"plate": "沪ADH3906"}, incorrect: {"plate": "沪A·DH3906"}
2. If there is no plate, the plate is blurry, or it is occluded beyond full recognition, return: {"plate": null}
3. Return only JSON; do NOT include any other text, explanations, or markup"""

def build_user_prompt() -> str:
    """Build the recognition instruction in the user message."""
    return "请识别这张图像中的车牌号码。"


# ---------------------------------------------------------------------------
# 图像预处理
# ---------------------------------------------------------------------------

def clean_plate(plate: Optional[str]) -> Optional[str]:
    """
    Sanitize a plate string: remove separators (·, -, space, /, ., ,, etc.).
    Post-processing cleanup — models sometimes insert · or spaces between the
    province letter and digits. Returns the cleaned string, or None.
    """
    if plate is None:
        return None
    import re
    # Remove all non-Chinese, non-letter, non-digit characters
    cleaned = re.sub(r'[^京津沪渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新'
                     r'A-Za-z0-9]', '', plate)
    return cleaned if cleaned else None


def preprocess_image(image: Image.Image, max_size: int = MAX_IMAGE_SIZE) -> Image.Image:
    """
    Resize image so the longer side is within max_size, preserving aspect ratio.
    Reduces transfer size and speeds up inference.
    """
    w, h = image.size
    longest = max(w, h)
    if longest <= max_size:
        return image
    ratio = max_size / longest
    new_size = (int(w * ratio), int(h * ratio))
    return image.resize(new_size, Image.LANCZOS)


def encode_image(image: Image.Image) -> str:
    """Encode a PIL Image to a Base64 string (JPEG, quality 85)."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Ollama Chat API 调用
# ---------------------------------------------------------------------------

def _get_model_think_config(model: str) -> bool:
    """Determine whether to pass `think: false` based on model name prefix."""
    for prefix, enabled in _MODEL_THINK_CONFIG.items():
        if model.startswith(prefix):
            return enabled
    return False  # default: don't pass; safety first


def _need_fallback_thinking(model: str) -> bool:
    """Determine whether this model needs fallback parsing from the thinking field."""
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
    Recognize a license plate using the Ollama Chat API (/api/chat).

    - Auto-resizes image as preprocessing
    - Uses system prompt + user prompt chat format
    - Forces JSON output mode (format: "json")
    - Automatically decides whether to pass `think: false` based on model name
    - For buggy models like qwen3-vl, auto-fallbacks to the thinking field
    - Auto-retries on failure (exponential backoff)

    Returns a PlatePrediction structured result.
    """
    # Preprocess & encode
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

            # fallback: some models (qwen3-vl) return empty content
            # when think:false; the actual output is in the thinking field
            if not raw_response and need_fallback:
                thinking_text = msg.get("thinking", "").strip()
                if thinking_text:
                    raw_response = thinking_text

            # Attempt to parse JSON
            try:
                data = json.loads(raw_response)
                plate = data.get("plate")
                # plate must be str or None
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
            wait = 2 ** attempt  # exponential backoff: 2s → 4s → 8s
            tqdm.write(f"  [Retry {attempt}/{MAX_RETRIES}] {last_error}, waiting {wait}s...")
            time.sleep(wait)

    # All retries exhausted
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
    Recognize a license plate using vLLM / OpenAI-compatible Chat API
    (/v1/chat/completions).

    - Auto-resizes image as preprocessing
    - Uses system prompt + user prompt chat format
    - Forces JSON output mode (response_format: json_object)
    - Auto-retries on failure (exponential backoff)

    Returns a PlatePrediction structured result.
    """
    # Preprocess & encode
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

            # Attempt to parse JSON
            try:
                data = json.loads(raw_response)
                plate = data.get("plate")
                # plate must be str or None
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
            wait = 2 ** attempt  # exponential backoff: 2s → 4s → 8s
            tqdm.write(f"  [Retry {attempt}/{MAX_RETRIES}] {last_error}, waiting {wait}s...")
            time.sleep(wait)

    # All retries exhausted
    return PlatePrediction(
        plate=None,
        raw_response="",
        is_valid_json=False,
        error=f"[Retries exhausted] {last_error}",
    )


# ---------------------------------------------------------------------------
# Single-sample evaluation (for concurrent execution)
# ---------------------------------------------------------------------------

def evaluate_single(task: Tuple[str, str, str, float, str, str]) -> dict:
    """
    Evaluate a single image. Accepts a tuple argument for compatibility with
    ThreadPoolExecutor.

    Args:
        task: (img_path, label_name, model, temperature, backend, base_url)

    Returns:
        Single result dict with keys: img_path, label, pred, correct, latency_ms
    """
    img_path, label_name, model, temperature, backend, base_url = task

    start = time.time()

    # Load image
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

    # Call model for recognition
    if backend == "vllm":
        pred = query_vllm_chat(image, model=model, temperature=temperature, base_url=base_url)
    else:
        pred = query_ollama_chat(image, model=model, temperature=temperature)
    latency_ms = (time.time() - start) * 1000

    # Post-process: remove separators model may have inserted (·, space, -, etc.)
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
# Main evaluation flow
# ---------------------------------------------------------------------------

def _ensure_dataset_import():
    """Deferred import of PlateDataset to avoid requiring torch for --help."""
    global PlateDataset
    if PlateDataset is None:
        from plate_dataset import PlateDataset as _PlateDataset
        PlateDataset = _PlateDataset


def val(args):
    """Main evaluation flow: load data → sequential/concurrent inference → summarize → save."""
    _ensure_dataset_import()
    val_root = args.val_root

    # ---- Load dataset ----
    val_dataset = PlateDataset(
        val_root, is_train=False,
        only_ccpd2019=args.only_ccpd2019,
        only_ccpd2020=args.only_ccpd2020,
        only_others=args.only_others,
    )
    total_num = len(val_dataset)

    # ---- Determine base_url ----
    base_url = args.base_url
    if args.backend == "vllm" and base_url is None:
        base_url = VLLM_DEFAULT_BASE_URL
    elif base_url is None:
        base_url = ""  # Ollama does not need base_url

    # ---- Build task list ----
    tasks: List[Tuple[str, str, str, float, str, str]] = []
    for idx in range(total_num):
        _, label_name, img_path = val_dataset[idx]
        tasks.append((img_path, label_name, args.model, args.temperature, args.backend, base_url))

    # ---- Determine output path ----
    if args.output:
        output_path = args.output
    else:
        os.makedirs("llm/results", exist_ok=True)
        model_slug = args.model.replace(":", "_").replace("/", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"llm/results/{model_slug}_{timestamp}.json"

    # ---- Evaluation config ----
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

    # ---- Print config ----
    print(f"\n{'=' * 60}")
    print(f"Backend:         {config.backend}")
    if config.backend == "vllm":
        print(f"API URL:         {base_url}")
    print(f"Model:           {config.model}")
    print(f"Temperature:     {config.temperature}")
    print(f"Prompt version:  {config.prompt_version}")
    print(f"Dataset:         {val_root}  ({dataset_filter})")
    print(f"Samples:         {total_num}")
    print(f"Workers:         {args.workers}  {'(sequential)' if args.workers <= 1 else ''}")
    print(f"Output:          {output_path}")
    print(f"{'=' * 60}\n")

    # ---- Inference ----
    results: List[dict] = []
    correct_num = 0
    json_failures = 0
    null_predictions = 0
    error_count = 0

    start_time = time.time()

    if args.workers <= 1:
        # Sequential execution
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
        # Multi-threaded concurrent execution
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

        # Sort by img_path to restore original order
        results.sort(key=lambda r: r["img_path"])

    total_duration = time.time() - start_time

    # ---- Summary statistics ----
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

    # ---- Save results ----
    output_data = {
        "config": asdict(config),
        "summary": summary,
        "results": results,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Results saved to: {output_path}")

    # ---- Print final statistics ----
    print("\n" + "=" * 60)
    print("=== Evaluation Results ===")
    print(f"Total samples:         {total_num}")
    print(f"Correct:               {correct_num}")
    print(f"Accuracy:              {accuracy * 100:.2f}%")
    print(f"JSON parse failures:   {json_failures}")
    print(f"Unrecognized (null):   {null_predictions}")
    print(f"Request errors:        {error_count}")
    print(f"Total time:            {total_duration:.2f} s")
    print(f"Avg latency:           {avg_latency_ms:.1f} ms")
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
