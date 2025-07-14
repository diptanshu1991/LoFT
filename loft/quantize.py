# loft/quantize.py

import os
import subprocess

def run_quantize(model_path, output_path, quant_type="Q4_0"):
    if not os.path.isfile(model_path):
        print(f"❌ Input model not found: {model_path}")
        return

    quantize_bin = os.path.expanduser("~/llama.cpp/build/bin/llama-quantize")

    if not os.path.isfile(quantize_bin):
        print(f"❌ llama-quantize binary not found at: {quantize_bin}")
        print("👉 Please build llama.cpp with: cmake .. && make -j")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    command = [
        quantize_bin,
        model_path,
        output_path,
        quant_type
    ]

    print(f"⚙️  Quantizing model: {model_path}")
    print(f"📦 Output: {output_path} ({quant_type})")

    try:
        subprocess.run(command, check=True)
        print(f"✅ Quantized model saved at: {output_path}")
    except subprocess.CalledProcessError as e:
        print("❌ Quantization failed.")
        print(e)
