import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

def run_merge(base_model, adapter_dir, output_dir):
    print(f"🔗 Merging LoRA adapter from {adapter_dir} into base model {base_model}...")

    os.makedirs(output_dir, exist_ok=True)

    # Load base model
    config = PeftConfig.from_pretrained(adapter_dir)
    base_model = AutoModelForCausalLM.from_pretrained(base_model)

    # Wrap with PEFT
    model = PeftModel.from_pretrained(base_model, adapter_dir)

    # Try merging
    try:
        merged = model.merge_and_unload()
        print("✅ LoRA merged successfully.")
    except Exception as e:
        print("⚠️ Error during merge_and_unload. Likely due to model nesting.")
        print(str(e))
        return

    # Save model and tokenizer
    merged.save_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model.name_or_path)
    tokenizer.save_pretrained(output_dir)

    print(f"✅ Merged model saved to: {output_dir}")
