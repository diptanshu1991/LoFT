import argparse
from loft.train import run_finetune
from loft.export import run_export
from loft.chat import run_chat
from loft.merge import run_merge
from loft.quantize import run_quantize


def main():
    parser = argparse.ArgumentParser(prog="loft", description="Low-RAM Finetuning Toolkit (LoFT)")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Finetune subcommand
    finetune_parser = subparsers.add_parser("finetune", help="Finetune an open-source LLM with LoRA")
    finetune_parser.add_argument("--model_name", type=str, required=True, help="Hugging Face model ID")
    finetune_parser.add_argument("--dataset_path", type=str, required=True, help="Path to training dataset (JSON format)")
    finetune_parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the trained model")
    finetune_parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    finetune_parser.add_argument("--use_safetensors", action="store_true", help="Use safetensors when loading model")
    finetune_parser.add_argument("--gradient_checkpointing", action="store_true",
                                 help="Enable gradient checkpointing to reduce memory usage")

    # Export subcommand
    export_parser = subparsers.add_parser("export", help="Export a trained model to quantized formats")
    export_parser.add_argument("--model_dir", type=str, required=True, help="Path to trained model directory")
    export_parser.add_argument("--format", type=str, choices=["gguf", "onnx"], required=True, help="Export format")
    export_parser.add_argument("--output_dir", type=str, required=True, help="Directory to save exported model")
    export_parser.add_argument("--opset", type=int, default=19, help="ONNX opset version (default: 19)")

    # Chat subcommand
    chat_parser = subparsers.add_parser("chat", help="Run inference with a quantized GGUF model using llama.cpp")
    chat_parser.add_argument("--model_path", type=str, required=True, help="Path to .gguf quantized model")
    chat_parser.add_argument("--prompt", type=str, required=True, help="Prompt text to run")
    chat_parser.add_argument("--n_tokens", type=int, default=128, help="Number of tokens to generate (default: 128)")

    #Merge subcommand
    merge_parser = subparsers.add_parser("merge", help="Merge base model with LoRA adapter")
    merge_parser.add_argument("--base_model", type=str, required=True, help="Hugging Face model ID or path")
    merge_parser.add_argument("--adapter_dir", type=str, required=True,
                              help="Path to trained LoRA adapter (output of finetune)")
    merge_parser.add_argument("--output_dir", type=str, required=True, help="Where to save the merged model")

    # Quantize subcommand
    quantize_parser = subparsers.add_parser("quantize", help="Quantize a GGUF model")
    quantize_parser.add_argument("--model_path", type=str, required=True, help="Path to input .gguf model")
    quantize_parser.add_argument("--output_path", type=str, required=True,
                                 help="Where to save the quantized .gguf model")
    quantize_parser.add_argument("--quant_type", type=str, default="Q4_0",
                                 help="Quantization type (e.g., Q4_0, Q5_1, Q8_0)")

    args = parser.parse_args()

    if args.command == "finetune":
        run_finetune(
            model_name=args.model_name,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            use_safetensors=args.use_safetensors
        )
    elif args.command == "export":
        run_export(
            model_dir=args.model_dir,
            format=args.format,
            output_dir=args.output_dir,
            opset=args.opset
        )
    elif args.command == "chat":
        run_chat(
            model_path=args.model_path,
            prompt=args.prompt,
            n_tokens=args.n_tokens
        )

    elif args.command == "merge":
        run_merge(
            base_model=args.base_model,
            adapter_dir=args.adapter_dir,
            output_dir=args.output_dir
        )

    elif args.command == "quantize":
        run_quantize(
            model_path=args.model_path,
            output_path=args.output_path,
            quant_type=args.quant_type
        )


    else:
        parser.print_help()
