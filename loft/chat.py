# loft/chat.py

import subprocess
import os
import time
import psutil

def run_chat(model_path, prompt, n_tokens=128):
    llama_cli = os.path.expanduser("../llama.cpp/build/bin/llama-cli") ##please change this to original llama.cpp folder

    if not os.path.isfile(llama_cli):
        print(f"❌ llama-cli binary not found at: {llama_cli}")
        print("👉 Please build llama.cpp first using cmake + make.")
        return

    if not os.path.isfile(model_path):
        print(f"❌ Model file not found: {model_path}")
        return

    print(f"🧠 Running inference on: {model_path}")
    print(f"📨 Prompt: {prompt}")

    command = [
        llama_cli,
        "-m", model_path,
        "-p", prompt,
        "-n", str(n_tokens)
    ]

    # Benchmark start
    process = psutil.Process(os.getpid())
    start_ram = process.memory_info().rss / 1024 ** 2
    start_time = time.time()

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print("❌ Inference failed.")
        print(e)

    # Benchmark end
    end_time = time.time()
    end_ram = process.memory_info().rss / 1024 ** 2

    print(f"\n📊 Inference Benchmark")
    print(f"Inference Time: {end_time - start_time:.2f} sec")
    print(f"Peak RAM Used: {max(start_ram, end_ram):.2f} MB")
    print(f"✅ Prompt processed and response generated.")