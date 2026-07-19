# LLM License Plate Recognition Evaluation

End-to-end Chinese license plate OCR using multimodal LLMs (vLLM deployment + OpenAI-compatible Chat API). Accuracy and throughput benchmarks on CCPD2020/test (5,006 images).

The evaluation script `eval_plate.py` supports Ollama and vLLM backends, multi-threaded concurrency, automatic retries, and structured JSON output. See [eval_plate.py](eval_plate.py).

## Model Accuracy Comparison

| Model | Params | Accuracy | Unrecognized (null) | Throughput (img/s) | Avg Latency (ms) | Workers |
|---|---|---|---|---|---|---|
| Qwen/Qwen3.5-2B | 2B | **86.52%** | 50 | 4.94 | 890.0 | 4 |
| ZhipuAI/GLM-OCR | 0.3B | 85.38% | **0** | **23.98** | **382.8** | 8 |
| Qwen/Qwen3-VL-2B-Instruct | 2B | 83.62% | 109 | 5.08 | 3265.3 | 16 |

> **Summary:**
> - **Qwen3.5-2B** achieves the highest accuracy (86.52%), ideal for accuracy-critical scenarios.
> - **GLM-OCR** is the fastest (23.98 img/s, 382.8ms latency) with zero null outputs — best for high-throughput, low-latency use cases. Only 0.3B parameters, outstanding cost-effectiveness.
> - **Qwen3-VL-2B-Instruct** has lower accuracy and high latency (3.3s/image); not recommended when better alternatives exist.

## Evaluation Environment

- **Image:** `vllm/vllm-openai:v0.25.1-cu129-ubuntu2404`
- **GPU:** NVIDIA GeForce RTX 4060 (8GB)
- **Dataset:** CCPD2020/test, 5,006 Chinese license plate images
- **Prompt version:** v4 (includes province character similarity hints; see `eval_plate.py`)
- **Protocol:** OpenAI-compatible `/v1/chat/completions`, `response_format: json_object`

## Health Check

After starting the vLLM service, verify it is ready:

```bash
curl http://127.0.0.1:13000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-2B",
    "messages": [
      {"role": "user", "content": "Hello, please introduce yourself"}
    ],
    "temperature": 0.7
  }'
```

Replace `model` with the actual deployed model name.

---

## Qwen/Qwen3.5-2B

Text model with vision capability. Highest accuracy, suitable for accuracy-critical scenarios.

### Start vLLM

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

### Run Evaluation

```bash
python3 llm/eval_plate.py ../../datasets/chinese_license_plate/recog/ \
  --backend vllm \
  --base-url http://127.0.0.1:13000 \
  --model Qwen3.5-2B \
  --only-ccpd2020 \
  --workers 4
```

### Results

```
Total samples:      5006
Correct:            4331
Accuracy:           86.52%
JSON parse failures: 0
Unrecognized (null): 50
Request errors:     0
Total time:         1013.62 s
Avg latency:        890.0 ms
```

---

## ZhipuAI/GLM-OCR

Zhipu OCR-specialized model, only 0.3B parameters. Fastest inference, zero null outputs, outstanding cost-effectiveness.

### Start vLLM

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

> **Note:** GLM-OCR requires `VLLM_NO_PIN_MEMORY=1` and `VLLM_USE_V2_MODEL_RUNNER=0`, otherwise it may fail to start.

### Run Evaluation

```bash
python3 llm/eval_plate.py ../../datasets/chinese_license_plate/recog/ \
  --backend vllm \
  --base-url http://127.0.0.1:13000 \
  --model GLM-OCR \
  --only-ccpd2020 \
  --workers 8
```

### Results

```
Total samples:      5006
Correct:            4274
Accuracy:           85.38%
JSON parse failures: 0
Unrecognized (null): 0
Request errors:     0
Total time:         208.80 s
Avg latency:        382.8 ms
```

---

## Qwen/Qwen3-VL-2B-Instruct

Native vision-language model. Lower accuracy and high latency; not recommended when better alternatives exist.

### Start vLLM

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

> **Note:** This model requires `--enforce-eager` (disables CUDA Graph to save VRAM), `--max-model-len 1024` (license plate OCR does not need long context), and `max_pixels=360000` to limit multimodal pixel processing.

### Run Evaluation

```bash
python3 llm/eval_plate.py ../../datasets/chinese_license_plate/recog/ \
  --backend vllm \
  --base-url http://127.0.0.1:13000 \
  --model Qwen3-VL-2B-Instruct \
  --only-ccpd2020 \
  --workers 16
```

### Results

```
Total samples:      5006
Correct:            4186
Accuracy:           83.62%
JSON parse failures: 0
Unrecognized (null): 109
Request errors:     0
Total time:         984.59 s
Avg latency:        3265.3 ms
```

---

## Tools

- **`eval_plate.py`** — Main evaluation script. Supports Ollama / vLLM backends, multi-threaded concurrency, structured JSON output. Run `python3 llm/eval_plate.py --help` for full usage.
- **`reanalyze.py`** — Post-hoc analysis of saved result JSONs: applies `clean_plate` separator sanitization, compares accuracy before/after cleaning, and classifies remaining errors (province mismatch, character errors, length mismatch, etc.).
