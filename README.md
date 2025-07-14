# 🪶 LoFT CLI — Low-RAM Finetuning Toolkit

> 🧠 Fine-tune open-source LLMs with LoRA on **MacBooks, CPUs, or low-RAM devices**
> 🛠️ Merge, quantize to GGUF, and run locally via `llama.cpp`
> 💻 No GPU required

---

## 🚀 What is LoFT?

**LoFT CLI** is a lightweight, open-source command-line tool that enables:

✅ Finetuning 1B–3B open-source LLMs with **LoRA**
✅ Merging adapters and exporting models into **GGUF format** for CPU inference
✅ Running finetuned models locally via `llama.cpp`
✅ All on your **MacBook**, **CPU box**, or **low-spec laptop** — no GPU needed

---

## 🧩 Core Features

| Feature            | Description                                                              |
| ------------------ | ------------------------------------------------------------------------ |
| 🏋️ Finetune       | Inject LoRA adapters into Hugging Face models and train on JSON datasets |
| 🧠 Merge           | Merge base model + adapter into standalone model weights                 |
| 🪶 Quantize (GGUF) | Convert merged model into GGUF format via `llama.cpp` tooling            |
| 💬 Chat (WIP)      | Run a CLI-based chatbot locally using quantized model                    |

---

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/diptanshu1991/LoFT_v1.git
cd LoFT_v1

# Install in development mode
pip install -e .
```

You now have access to the `loft` CLI.

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🧪 1. Finetune a Model with LoRA

Uses `peft` with LoRA adapters (in float16/float32). Trains only LoRA layers.

```bash
loft finetune \
  --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset data/sample_finetune_data.json \
  --output_dir adapter/adapter_v1 \
  --num_train_epochs 3 \
  --gradient_checkpointing
```

> ✅ Works well on low-RAM MacBooks with float-based LoRA adapters

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

---

## 📁 Updated Project Structure

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
