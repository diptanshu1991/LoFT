# loft/quantize.py

import os
import subprocess
import time
import psutil

def run_quantize(model_path, output_path, quant_type="Q4_0"):
    if not os.path.isfile(model_path):
        print(f"❌ Input model not found: {model_path}")
        return

    quantize_bin = os.path.expanduser("../llama.cpp/build/bin/llama-quantize") #please change this to original llama.cpp folder

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

    # Benchmark start
    process = psutil.Process(os.getpid())
    start_ram = process.memory_info().rss / 1024 ** 2
    start_time = time.time()

    try:
        subprocess.run(command, check=True)
        print(f"✅ Quantized model saved at: {output_path}")
    except subprocess.CalledProcessError as e:
        print("❌ Quantization failed.")
        print(e)

    # Benchmark end
    end_time = time.time()
    end_ram = process.memory_info().rss / 1024 ** 2
    model_size = os.path.getsize(output_path) / 1024 ** 2 if os.path.exists(output_path) else 0

    # Report
    print("\n📊 Quantization Benchmark")
    print(f"Quantization Time: {end_time - start_time:.2f} sec")
    print(f"Peak RAM Used: {max(start_ram, end_ram):.2f} MB")
    print(f"Quantized GGUF Size: {model_size:.2f} MB")
    print(f"✅ Quantized model saved at: {output_path}")
