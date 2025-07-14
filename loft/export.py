# loft/export.py

import os
import subprocess
import shutil

def run_export(model_dir, format, output_dir, opset=19):
    if format != "gguf":
        raise ValueError("❌ Only 'gguf' export is currently supported in LoFT CLI.")

    os.makedirs(output_dir, exist_ok=True)

    model_name = os.path.basename(model_dir.rstrip("/"))
    output_file = os.path.join(output_dir, f"{model_name}.gguf")

    # 1. Try Python script
    script_path = os.path.expanduser("~/llama.cpp/convert_hf_to_gguf.py")
    # 2. Try compiled C++ binary
    binary_path = os.path.expanduser("~/llama.cpp/build/bin/convert-llama-hf-to-gguf")

    if os.path.isfile(script_path):
        print("📦 Using Python script: convert_hf_to_gguf.py")
        command = ["python3", script_path, model_dir, "--outfile", output_file]
    elif os.path.isfile(binary_path):
        print("📦 Using compiled binary: convert-llama-hf-to-gguf")
        command = [binary_path, model_dir, "--outfile", output_file]
    else:
        print("❌ Could not find GGUF converter (neither script nor binary).")
        print("👉 Please build llama.cpp or set the correct path to convert_hf_to_gguf.")
        return

    print(f"🚀 Converting model: {model_dir} → {output_file}")

    try:
        subprocess.run(command, check=True)
        print(f"✅ GGUF model saved at: {output_file}")
    except subprocess.CalledProcessError as e:
        print("❌ GGUF export failed.")
        print(e)
