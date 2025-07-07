#loFT
Open-source CLI toolkit for low-RAM finetuning, quantization, and deployment of LLMs

````markdown
# 🪶 LoFT CLI — Low-RAM Finetuning Toolkit

> 🧠 Fine-tune open-source LLMs with LoRA on **MacBooks, CPUs, or low-RAM devices**  
> 🛠️ Merge, quantize to GGUF, and run locally via `llama.cpp`  
> 💻 No GPU required

---

## 🚀 What is LoFT?

**LoFT CLI** is a lightweight, open-source command-line tool that enables:

✅ Finetuning 1B–3B open-source LLMs with **QLoRA**  
✅ Quantizing models into **GGUF format** for CPU inference  
✅ Running finetuned models locally via **llama.cpp**  
✅ All from the comfort of your **MacBook** or a **low-spec CPU machine**

---

## 🧩 Core Features

| Feature           | Description |
|------------------|-------------|
| 🏋️ Finetune      | Injects LoRA adapters into Hugging Face models and trains on a JSON dataset |
| 🧠 Merge          | Merges base model + adapter into final model weights |
| 🪶 Quantize (GGUF)| Converts merged model into GGUF format via llama.cpp |
| 💬 Chat (WIP)     | Runs a local CLI interface with quantized model |

---

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/your-username/loft-cli.git
cd loft-cli

# Install in development mode
pip install -e .
````

You now have access to the `loft` CLI

---

## 🧪 1. Finetune a Model with LoRA

```bash
loft finetune \
  --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset_path data/finetune_dataset.json \
  --output_dir models/finetuned_model \
  --num_train_epochs 1
```

> ✅ Works on low-RAM MacBooks (4-bit, LoRA adapters only)

---

## 🔀 2. Merge Adapters into Final Model

```bash
loft merge \
  --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --adapter_dir models/finetuned_model/checkpoint-xxx \
  --output_dir models/merged_model
```

> Creates a standalone merged Hugging Face model with adapter weights.

---

## 🪄 3. Quantize to GGUF

```bash
# Convert merged HF model to GGUF (in llama.cpp directory)
python convert_hf_to_gguf.py \
  models/merged_model \
  --outfile TinyLlama.gguf \
  --outdir models/gguf_model

# Quantize to 4-bit Q4_0
./quantize models/gguf_model/TinyLlama.gguf \
  models/gguf_model/TinyLlama-q4.gguf \
  Q4_0
```

> Requires [llama.cpp](https://github.com/ggerganov/llama.cpp) to be cloned and compiled via `make`.

---

## 💻 4. Run Locally via llama.cpp

```bash
./main \
  -m models/gguf_model/TinyLlama-q4.gguf \
  -p "What is LoRA?"
```

> Inference runs <1GB RAM. Blazing fast on CPU. No GPU required.

---

## 📁 Project Structure

```bash
loft-cli/
├── loft/
│   ├── cli.py          # CLI entrypoint
│   ├── finetune.py     # Finetune logic
│   ├── merge.py        # Adapter merge logic
│   ├── export.py       # ONNX export logic (skipped for now)
│   └── chat.py         # (WIP) Local chatbot interface
├── data/
│   └── finetune_dataset.json  # Instruction-style training data
├── models/
│   ├── finetuned_model/
│   ├── merged_model/
│   └── gguf_model/
├── README.md
├── setup.py
└── requirements.txt
```

---

## 📚 Training Data Example

```json
[
  { "text": "Question: What is LoRA? Answer: LoRA is a method to fine-tune LLMs efficiently." },
  { "text": "Question: What is QLoRA? Answer: QLoRA combines LoRA with 4-bit quantization for low-RAM finetuning." }
]
```

---

## 🛠️ Requirements

* Python 3.10+
* `transformers`, `peft`, `datasets`, `accelerate`
* llama.cpp (for GGUF + inference)
* Optional: `bitsandbytes` (for 4-bit training)

Install with:

```bash
pip install -r requirements.txt
```

---

## 📌 Roadmap

* [x] Local LoRA finetuning CLI
* [x] Merge + Quantize (GGUF)
* [x] Run on CPU (llama.cpp)
* [ ] Gradio interface for local chat
* [ ] ONNX export (pending PyTorch operator fix)
* [ ] SaaS dashboard + inference cost estimator
* [ ] Adapter marketplace (future phase)

---

## 🪪 License

MIT License — free to use, modify, and distribute.

---

## 👋 Contribute or Get Involved

This is an early OSS prototype — contributions are welcome!

* ⭐ Star the repo
* 🐛 Report issues

---

## 🌍 Author

Built by [@diptanshukumar](https://www.linkedin.com/in/diptanshukumar) — strategy consultant turned AI builder.

```

---
