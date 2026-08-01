# LoFT CLI

Lightweight finetuning, quantization and local inference for small language models, on CPU.

> **Status: no longer maintained.** I stopped work on LoFT in 2025. Finetuning turned out to be something a team does once and then walks away from, and the users it attracted were mostly hobbyists, so there was no recurring reason to come back to the tool. The code works and the benchmarks below are real. I have left it up as a reference.

Customize small models (1-3B) with LoRA adapters. Train, quantize and run entirely on CPU, including on an 8GB MacBook.

---

## What LoFT does

- Finetune small LLMs (e.g. TinyLlama) using LoRA
- Merge adapters into a standalone Hugging Face model
- Export to GGUF
- Quantize to Q4_0 for CPU inference
- Run the model locally through `llama.cpp`

Everything runs on MacBooks, CPUs and low-RAM laptops. No GPU required.

## Why it existed

Most finetuning tooling assumes a cloud GPU. LoFT was built for the case where you have a laptop and a small dataset, and you want a domain-specific adapter you can run offline.

## Workflow

| Step | Command | Output |
| -------- | --------------- | ------------------------------ |
| Finetune | `loft finetune` | LoRA adapters (`.safetensors`) |
| Merge | `loft merge` | Merged HF model |
| Export | `loft export` | GGUF (F32/FP16) model |
| Quantize | `loft quantize` | Q4_0 GGUF model |
| Chat | `loft chat` | Inference CLI (offline) |

---

## Installation

Requires Python 3.10+.

```bash
# 1. Clone and install LoFT
git clone https://github.com/diptanshu1991/LoFT
cd LoFT

python3 -m venv venv
source venv/bin/activate

pip install -e .
pip install -r requirements.txt
```

```bash
# 2. Build llama.cpp (needed for quantization and inference)
cd ..
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make
cd ../LoFT
```

```bash
# 3. Download the base model (optional but recommended)
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model_id = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
AutoModelForCausalLM.from_pretrained(model_id)
AutoTokenizer.from_pretrained(model_id)
"
```

You now have the `loft` CLI available.

---

## 1. Finetune with LoRA

Uses `peft` with LoRA adapters in float16/float32. Trains only the LoRA layers.

```bash
loft finetune \
  --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset data/sample_finetune_data.json \
  --output_dir adapter/ \
  --num_train_epochs 2 \
  --gradient_checkpointing
```

Takes instruction-tuning format. Works with JSON datasets. Output is a LoRA adapter folder.

## 2. Merge the adapter into the base model

```bash
loft merge \
  --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --adapter_dir adapter/adapter_v1 \
  --output_dir merged_models
```

Produces a single merged HF model with the adapter weights integrated.

## 3. Export and quantize to GGUF

```bash
# Export to GGUF
loft export \
  --output_dir merged_models \
  --format gguf \
  merged_models

# Quantize to 4-bit GGUF (Q4_0)
loft quantize \
  --model_path merged_models/merged_models.gguf \
  --output_path merged_models/merged_models_q4.gguf \
  --quant_type Q4_0
```

Uses llama.cpp's tools. The output works directly with the llama.cpp CLI.

## 4. Run inference

```bash
loft chat \
  --model_path merged_models/merged_models_q4.gguf \
  --prompt "How do I bake a chocolate cake from scratch?" \
  --n_tokens 200
```

Runs under 1GB RAM on CPU.

---

## Benchmarks

MacBook Air, 8GB RAM. Dataset: 20-sample Dolly-style JSON.

| Step | Output | Size | Peak RAM | Time Taken |
| -------- | ------------------------ | ------ | -------- | ---------- |
| Finetune | Adapter (`.safetensors`) | 4.3 MB | 308 MB | 23 min |
| Merge | Merged Model | 4.2 GB | 322 MB | 4.7 min |
| Export | GGUF (F32/FP16) | 2.1 GB | 322 MB | 83 sec |
| Quantize | GGUF (Q4_0) | 607 MB | 322 MB | 21 sec |
| Chat | Response @ 6.9 tok/s | — | 322 MB | 79 sec |

Also tested on 300 samples, where 2 epochs took 1.5 hours. That run is a proof of concept to validate CPU-only finetuning. Production-quality adapters need larger datasets and GPU training.

---

## Project structure

```
LoFT/
├── loft/                          # Core CLI code
│   ├── cli.py                     # CLI parser and dispatcher
│   ├── train.py                   # Finetuning logic
│   ├── merge.py                   # Adapter merge logic
│   ├── export.py                  # GGUF/ONNX export logic
│   └── chat.py                    # CLI chat interface
├── data/
│   └── sample_finetune_data.json  # Sample dataset
├── adapter/
│   └── adapter_v1/                # Example adapter config and tokenizer files
├── merged_models/                 # Exported and quantized models
├── requirements.txt
├── setup.py
└── train_config.yaml
```

## Training data format

```json
[
  {
    "instruction": "Give me a list of basic ingredients for baking cookies",
    "input": "",
    "output": "Flour, sugar, eggs, milk, butter, baking powder, chocolate chips, cinnamon..."
  }
]
```

## Requirements

- Python 3.10+
- `transformers`, `peft`, `datasets`, `accelerate`
- llama.cpp, for quantization and inference
- Optional: `bitsandbytes`, for 4-bit training

## License

MIT.

## Author

Built by [Diptanshu Kumar](https://www.linkedin.com/in/diptanshu-kumar).
