from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch

adapter_path = "loft-cli_v1/adapter/adapter_v1"

# Load adapter config to get base model path
config = PeftConfig.from_pretrained(adapter_path)
base_model = config.base_model_name_or_path

# Load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained(base_model)
base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float32)

# Load adapter on top of base model
model = PeftModel.from_pretrained(base, adapter_path)
model = model.merge_and_unload()  # ✅ VERY IMPORTANT
model.eval()

# Inference prompt
prompt = "### Instruction:\nHow do I bake a chocolate cake from scratch?\n\n### Response:"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=150)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
