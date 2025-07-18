import os
import time
import psutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset

def run_finetune(model_name, dataset_path, output_dir, num_train_epochs, use_safetensors=True, gradient_checkpointing=False):
    os.makedirs(output_dir, exist_ok=True)

    print("🚀 Starting finetuning benchmark...")

    # Track system stats
    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss / 1024**2
    start_time = time.time()

    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_safetensors=True,
        revision="main"
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # ✅ Unfreeze LayerNorms
    for name, param in model.named_parameters():
        if "layernorm" in name.lower():
            param.requires_grad = True

    # Apply LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1
    )
    model = get_peft_model(model, peft_config)

    # Load and tokenize dataset
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    def tokenize(example):
        prompt = example["instruction"]
        if example.get("input"):
            prompt += "\n" + example["input"]
        prompt += "\n### Response:\n" + example["output"]
        tokens = tokenizer(prompt, truncation=True, padding="max_length", max_length=128)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized_dataset = dataset.map(tokenize)

    # Training config
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=num_train_epochs,
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=10,
        report_to=[],
        gradient_checkpointing=gradient_checkpointing
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset
    )

    trainer.train()

    # Save adapter
    adapter_dir = os.path.join(output_dir, "adapter_v1")
    os.makedirs(adapter_dir, exist_ok=True)

    print("✅ Saving LoRA adapter only (not merged)...")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # Benchmarking summary
    end_time = time.time()
    end_mem = process.memory_info().rss / 1024**2
    adapter_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    adapter_size = os.path.getsize(adapter_path) / 1024**2 if os.path.exists(adapter_path) else 0

    print("\n📊 Finetuning Benchmark Results")
    print(f"Peak RAM Used: {max(start_mem, end_mem):.2f} MB")
    print(f"Training Time: {end_time - start_time:.2f} sec")
    print(f"Adapter Size: {adapter_size:.2f} MB")
    print(f"Adapter saved at: {adapter_path if os.path.exists(adapter_path) else 'Not Found'}")
