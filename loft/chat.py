# loft/chat.py

import subprocess
import os

def run_chat(model_path, prompt, n_tokens=128):
    llama_cli = os.path.expanduser("../llama.cpp/build/bin/llama-cli") #load downloaded llama.cpp folder

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

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print("❌ Inference failed.")
        print(e)
