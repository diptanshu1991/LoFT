🪶 LoFT CLI — Lightweight Finetuning + Deployment Toolkit for Custom LLMs

![License](https://img.shields.io/github/license/diptanshu1991/LoFT)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![LoRA](https://img.shields.io/badge/LoRA-compatible-brightgreen)

> 🔧 Customize small language models (1–3B) with LoRA adapters  
> 💻 Train, quantize, and run entirely on CPU — even an 8GB MacBook  
> 🧱 Foundation for an adapter-powered GenAI deployment workflow  

✨ Designed for **developers building local GenAI apps**, not just ML researchers.


---

## 🚀 What is LoFT?

**LoFT CLI** is a lightweight, open-source command-line tool that enables:

- ✅ Finetune lightweight LLMs (like TinyLlama) using LoRA
- ✅ Merge adapters into a standalone Hugging Face model
- ✅ Export to GGUF format
- ✅ Quantize to Q4_0 for CPU inference
- ✅ Run the model locally using `llama.cpp`

Everything works **on MacBooks, CPUs, and low-RAM laptops**.

---

## 🎯 Why LoFT Exists

While others focus on training giant models in the cloud, LoFT empowers developers to:

- 🖥️ Customize open-source models without GPU dependence
- 🔌 Deploy LLMs fully offline — for privacy-first applications
- 🧩 Plug in domain-specific LoRA adapters with one command

Coming soon: **LoFT Recipes** — ready-to-use adapters + fine-tuning guides for real-world use cases like customer support, legal Q&A, and content summarization.

---

## 🧠 TL;DR: Workflow Summary

| Step     | Command         | Output                |
|----------|-----------------|------------------------|
| Finetune | `loft finetune` | LoRA adapters (`.safetensors`) |
| Merge    | `loft merge`    | Merged HF model        |
| Export   | `loft export`   | GGUF (F32/FP16) model  |
| Quantize | `loft quantize` | Q4_0 GGUF model        |
| Chat     | `loft chat`     | Inference CLI (offline) |
---

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/diptanshu1991/LoFT
cd LoFT

# Optional: create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e .

#Download Base Model (Optional but Recommended)
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model_id = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
AutoModelForCausalLM.from_pretrained(model_id)
AutoTokenizer.from_pretrained(model_id)
"
```
#Install Llama.cpp
```
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp

#build the C++ tools
cd llama.cpp
make

```

You now have access to the `loft` CLI.

Install dependencies:

```bash
cd LoFT
pip install -r requirements.txt
```

---

## 🧪 1. Finetune a Model with LoRA

Uses `peft` with LoRA adapters (in float16/float32). Trains only LoRA layers.

```bash
loft finetune \
  --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset data/sample_finetune_data.json \
  --output_dir adapter/ \
  --num_train_epochs 2 \
  --gradient_checkpointing
```

✅ Supports instruction-tuning format

✅ Works with JSON datasets

✅ Output is a LoRA adapter folder

---

## 🔀 2. Merge Adapters into Final Model

```bash
loft merge \
  --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --adapter_dir adapter/adapter_v1 \
  --output_dir merged_models
```

> Produces a single merged HF model with integrated adapter weights.

---

## 🪄 3. Export & Quantize to GGUF

```bash
# Export to GGUF format
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
✅ Uses llama.cpp's Python or compiled tools
✅ Output can be used directly with llama.cpp CLI
> Requires [llama.cpp](https://github.com/ggerganov/llama.cpp) — clone & build using `make`

---

## 💻 4. Inference with CLI Chat

```bash
loft chat \
  --model_path merged_models/merged_models_q4.gguf \
  --prompt "How do I bake a chocolate cake from scratch?" \
  --n_tokens 200
```

> Runs under 1GB RAM. Fast inference on MacBook/CPU. No GPU needed.

📊 Benchmarks (MacBook Air, 8GB RAM)

| Step     | Output                   | Size   | Peak RAM | Time Taken |
| -------- | ------------------------ | ------ | -------- | ---------- |
| Finetune | Adapter (`.safetensors`) | 4.3 MB | 308 MB   | 23 min     |
| Merge    | Merged Model             | 4.2 GB | 322 MB   | 4.7 min    |
| Export   | GGUF (F32/FP16)          | 2.1 GB | 322 MB   | 83 sec     |
| Quantize | GGUF (Q4\_0)             | 607 MB | 322 MB   | 21 sec     |
| Chat     | Response @ 6.9 tok/s     | —      | 322 MB   | 79 sec     |

⚠️ Dataset: 20-sample Dolly-style JSON
🧪 Also tested on 300 samples (2 epochs = 1.5 hours)
⚠️ Note: The 300-sample run is a proof-of-concept to validate CPU-only finetuning.  
For production-quality adapters, larger datasets and GPU training will be recommended.

---


## 📁 Project Structure

```bash
LoFT_v1/
├── loft/                  # Core CLI code
│   ├── cli.py             # CLI parser and dispatcher
│   ├── train.py           # Finetuning logic
│   ├── merge.py           # Adapter merge logic
│   ├── export.py          # GGUF/ONNX export logic
│   └── chat.py            # CLI chat interface (WIP)
├── data/
│   └── sample_finetune_data.json  # Sample dataset
├── adapter/
│   └── adapter_v1/        # Output LoRA adapter files
├── merged_models/
│   ├── merged_models.gguf         # Exported GGUF model
│   ├── merged_models_q4.gguf      # Quantized model (Q4_0)
├── llama.cpp/             # Cloned llama.cpp directory (user must build)
├── README.md
├── requirements.txt
├── setup.py
└── .gitignore
```

---

## 📚 Sample Training Data Format

```json
[
  {
    "instruction": "Who were the children of the legendary Garth Greenhand, the High King of the First Men in the series A Song of Ice and Fire?",
    "input": "",
    "output": "Garth the Gardener, John the Oak, Gilbert of the Vines, Brandon of the Bloody Blade..."
  },
  {
    "instruction": "Give me a list of basic ingredients for baking cookies",
    "input": "",
    "output": "Flour, sugar, eggs, milk, butter, baking powder, chocolate chips, cinnamon..."
  }
]
```

---

## 🛠️ Requirements

* Python 3.10+
* `transformers`, `peft`, `datasets`, `accelerate`
* llama.cpp (for quantization & inference)
* Optional: `bitsandbytes` (for 4-bit training)

---

## 🗺️ Roadmap

* [x] Local LoRA finetuning CLI
* [x] Merge + GGUF Export
* [x] Quantization (Q4/Q8)
* [x] Local CPU Inference
* [ ] Gradio UI for LoFT Chat
* [ ] ONNX Export support
* [ ] SaaS dashboard for inference cost
* [ ] Adapter Marketplace

---

## 🪪 License

MIT License — free to use, modify, and distribute.

---

## 🌍 Author

Built by [@diptanshukumar](https://www.linkedin.com/in/diptanshu-kumar) — strategy consultant turned AI builder. Contributions welcome!
