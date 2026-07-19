# LLM 车牌识别评估

使用多模态大模型（vLLM 部署 + OpenAI-compatible Chat API）对中国车牌进行端到端 OCR 识别，在 CCPD2020/test（5006 张）上对比精度与吞吐。

评估脚本 `eval_plate.py` 支持 Ollama 和 vLLM 两种后端，支持多线程并发，自动重试，结果保存为结构化 JSON。详见 [eval_plate.py](eval_plate.py)。

## 模型精度对比

| 模型 | 参数量 | 准确率 | 无法识别 (null) | 吞吐 (img/s) | 平均延迟 (ms) | 并发数 |
|---|---|---|---|---|---|---|
| Qwen/Qwen3.5-2B | 2B | **86.52%** | 50 | 4.94 | 890.0 | 4 |
| ZhipuAI/GLM-OCR | 0.3B | 85.38% | **0** | **23.98** | **382.8** | 8 |
| Qwen/Qwen3-VL-2B-Instruct | 2B | 83.62% | 109 | 5.08 | 3265.3 | 16 |

> **结论：**
> - **Qwen3.5-2B** 精度最高（86.52%），适合对准确率要求严格的场景。
> - **GLM-OCR** 速度最快（23.98 img/s，延迟 382.8ms），且无 null 输出，适合高吞吐、低延迟场景。参数量仅 0.3B，性价比突出。
> - **Qwen3-VL-2B-Instruct** 精度较低且延迟高（3.3s/张），在有更好替代方案时不推荐。

## 评估环境

- **镜像：** `vllm/vllm-openai:v0.25.1-cu129-ubuntu2404`
- **GPU：** NVIDIA GeForce RTX 4060 (8GB)
- **数据集：** CCPD2020/test，5006 张中国车牌图像
- **Prompt 版本：** v4（含省份相似字形提醒，详见 `eval_plate.py`）
- **通信协议：** OpenAI-compatible `/v1/chat/completions`，`response_format: json_object`

## 服务健康检查

启动 vLLM 服务后，用以下命令验证服务是否就绪：

```bash
curl http://127.0.0.1:13000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-2B",
    "messages": [
      {"role": "user", "content": "你好，请介绍一下你自己"}
    ],
    "temperature": 0.7
  }'
```

将 `model` 替换为实际部署的模型名即可。

---

## Qwen/Qwen3.5-2B

纯文本模型，具备视觉理解能力。精度最高，适合对准确率要求严格的场景。

### 启动 vLLM

```bash
docker run -it --rm \
  --gpus all \
  -p 13000:8000 \
  --ipc=host \
  -v /home/zjykzj/.cache/modelscope/hub/models/Qwen/Qwen3.5-2B:/model \
  -e VLLM_USE_MODELSCOPE=true \
  vllm/vllm-openai:v0.25.1-cu129-ubuntu2404 \
  --model /model \
  --served-model-name Qwen3.5-2B \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.8 \
  --max-num-seqs 32
```

### 运行评估

```bash
python3 llm/eval_plate.py ../../datasets/chinese_license_plate/recog/ \
  --backend vllm \
  --base-url http://127.0.0.1:13000 \
  --model Qwen3.5-2B \
  --only-ccpd2020 \
  --workers 4
```

### 评估结果

```
总样本数:          5006
正确样本数:        4331
准确率:            86.52%
JSON 解析失败:     0
无法识别 (null):   50
请求异常:          0
总耗时:            1013.62 秒
平均每次耗时:      890.0 ms
```

---

## ZhipuAI/GLM-OCR

智谱 OCR 专项模型，参数量仅 0.3B。速度最快、无 null 输出，性价比突出。

### 启动 vLLM

```bash
docker run -it --rm \
  --gpus all \
  -p 13000:8000 \
  --ipc=host \
  -v /home/zjykzj/.cache/modelscope/hub/models/ZhipuAI/GLM-OCR:/model \
  -e VLLM_USE_MODELSCOPE=true \
  -e VLLM_NO_PIN_MEMORY=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=0 \
  vllm/vllm-openai:v0.25.1-cu129-ubuntu2404 \
  --model /model \
  --served-model-name GLM-OCR \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.8 \
  --max-num-seqs 32
```

> **注意：** GLM-OCR 需要额外设置 `VLLM_NO_PIN_MEMORY=1` 和 `VLLM_USE_V2_MODEL_RUNNER=0`，否则可能启动失败。

### 运行评估

```bash
python3 llm/eval_plate.py ../../datasets/chinese_license_plate/recog/ \
  --backend vllm \
  --base-url http://127.0.0.1:13000 \
  --model GLM-OCR \
  --only-ccpd2020 \
  --workers 8
```

### 评估结果

```
总样本数:          5006
正确样本数:        4274
准确率:            85.38%
JSON 解析失败:     0
无法识别 (null):   0
请求异常:          0
总耗时:            208.80 秒
平均每次耗时:      382.8 ms
```

---

## Qwen/Qwen3-VL-2B-Instruct

原生视觉语言模型。精度较低、延迟高，在有更好替代方案时不推荐。

### 启动 vLLM

```bash
docker run -it --rm \
  --gpus all \
  -p 13000:8000 \
  --ipc=host \
  -v /home/zjykzj/.cache/modelscope/hub/models/Qwen/Qwen3-VL-2B-Instruct:/model \
  -e VLLM_USE_MODELSCOPE=true \
  -e VLLM_NO_PIN_MEMORY=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=0 \
  vllm/vllm-openai:v0.25.1-cu129-ubuntu2404 \
  --model /model \
  --served-model-name Qwen3-VL-2B-Instruct \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16 \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.80 \
  --max-num-seqs 4 \
  --enforce-eager \
  --mm-processor-kwargs '{"max_pixels": 360000}'
```

> **注意：** 该模型需要 `--enforce-eager`（关闭 CUDA Graph，节省显存），`--max-model-len 1024`（车牌 OCR 不涉及长文本），以及限制多模态像素 `max_pixels=360000`。

### 运行评估

```bash
python3 llm/eval_plate.py ../../datasets/chinese_license_plate/recog/ \
  --backend vllm \
  --base-url http://127.0.0.1:13000 \
  --model Qwen3-VL-2B-Instruct \
  --only-ccpd2020 \
  --workers 16
```

### 评估结果

```
总样本数:          5006
正确样本数:        4186
准确率:            83.62%
JSON 解析失败:     0
无法识别 (null):   109
请求异常:          0
总耗时:            984.59 秒
平均每次耗时:      3265.3 ms
```

---

## 辅助工具

- **`eval_plate.py`** — 主评估脚本，支持 Ollama / vLLM 后端，多线程并发，结果 JSON 输出。运行 `python3 llm/eval_plate.py --help` 查看完整参数。
- **`reanalyze.py`** — 对已有结果 JSON 进行二次分析：应用 `clean_plate` 清洗分隔符，对比清洗前后准确率，并按错误类型（省份错误、字符错误、长度不匹配等）分类统计。
