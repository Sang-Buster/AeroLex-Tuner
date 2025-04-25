#!/usr/bin/env python3
"""
finetune_test.py - Test the fine-tuned ATC model with sample inputs

This script:
1. Loads the fine-tuned model (either adapters or merged)
2. Runs inference on test ATC messages using transformers/unsloth
3. Outputs the improved transcripts and analysis
"""

import argparse
import os
import sys

# Parse arguments
parser = argparse.ArgumentParser(description="Test the fine-tuned ATC model")
parser.add_argument(
    "--model-dir",
    default="./atc_llama_merged",  # Default to merged model dir
    help="Directory containing the fine-tuned model (merged) or adapters (default: ./atc_llama_merged)",
)
parser.add_argument(
    "--base-model-name",
    default="meta-llama/Llama-3.2-3B-Instruct",
    help="Base model name (required if loading adapters from --model-dir, ignored otherwise)",
)
parser.add_argument(
    "--max-seq-length",
    type=int,
    default=2048,
    help="Maximum sequence length used during fine-tuning (default: 2048)",
)
parser.add_argument(
    "--load-in-4bit",
    action="store_true",
    help="Load the model in 4-bit precision (requires bitsandbytes).",
)
parser.add_argument(
    "--load-in-8bit",
    action="store_true",
    help="Load the model in 8-bit precision (requires bitsandbytes, alternative to 4-bit).",
)
parser.add_argument(
    "--offline",
    action="store_true",
    help="Use local model files (model-dir and base-model-name point to local directories)",
)
parser.add_argument(
    "--download",
    metavar="MODEL",
    help="Download a Hugging Face model to use locally (will exit after download)",
)
parser.add_argument(
    "--device",
    default="cuda:0",
    help="Device to run the model on (e.g., 'cuda:0', 'cuda:1', 'cpu'). Default: 'cuda:0'",
)
parser.add_argument(
    "--split-across-gpus",
    action="store_true",
    help="Split model across multiple GPUs if it's too large for a single GPU. Overrides --device.",
)
parser.add_argument(
    "--test-inputs",
    nargs="+",
    default=[
        "southwest five niner two turn left heading three four zero descend and maintain flight level two five zero",
        "united four seven three request clearance to descend due to turbulence",
        "delta six three eight traffic alert cessna two thousand feet twelve oclock expedite climb flight level three five zero",
    ],
    help="Test inputs to process",
)

# Check if help flag is requested before importing anything
if "-h" in sys.argv or "--help" in sys.argv:
    args = parser.parse_args()
    sys.exit(0)

args = parser.parse_args()

# Handle model download if requested
if args.download:
    print(f"Downloading model: {args.download}")
    # Import required libraries only when needed
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Create download directory based on model name
    download_dir = os.path.join("models", args.download.split("/")[-1])
    os.makedirs(download_dir, exist_ok=True)

    print(f"Downloading to {download_dir}...")
    try:
        # Download and save model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.download)
        model = AutoModelForCausalLM.from_pretrained(args.download)

        tokenizer.save_pretrained(download_dir)
        model.save_pretrained(download_dir)

        print(f"Model downloaded successfully to {download_dir}")
        print(
            f"To use this model, run with: --base-model-name {download_dir} --offline"
        )
        sys.exit(0)
    except Exception as e:
        print(f"Error downloading model: {e}")
        sys.exit(1)

# Import remaining libraries after help check and download
import torch  # noqa: E402

# Try importing unsloth, fallback to transformers if unavailable
HAS_UNSLOTH = False
try:
    from unsloth import FastLanguageModel  # noqa: E402

    HAS_UNSLOTH = True
    print("Using Unsloth for model loading.")
except ImportError:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

    print("Unsloth not found, using Hugging Face Transformers.")

# Check for PEFT
HAS_PEFT = False
try:
    from peft import PeftModel  # noqa: E402

    HAS_PEFT = True
except ImportError:
    print(
        "Warning: PEFT not installed. Cannot load LoRA adapters. Install with: uv pip install peft"
    )


def load_model_and_tokenizer():
    """Load the model and tokenizer, handling adapters if present."""
    model_dir = args.model_dir
    base_model_name = args.base_model_name
    max_seq_length = args.max_seq_length
    load_in_4bit = args.load_in_4bit
    load_in_8bit = args.load_in_8bit
    offline_mode = args.offline

    # Set device map based on arguments
    if args.split_across_gpus:
        device_map = "auto"
        print("Using auto device mapping (splitting across available GPUs).")
    elif args.device == "cpu":
        device_map = "cpu"
        print("Using CPU for inference.")
    else:
        # Use the specific device
        device_map = {"": args.device}
        print(f"Using specific device: {args.device}")

    # Check if model_dir contains adapter configuration
    adapter_config_path = os.path.join(model_dir, "adapter_config.json")
    is_adapter = os.path.exists(adapter_config_path)

    if is_adapter and not HAS_PEFT:
        print("Error: Found adapter config but PEFT is not installed. Cannot proceed.")
        sys.exit(1)

    # Determine quantization
    if load_in_4bit and load_in_8bit:
        print(
            "Warning: Both --load-in-4bit and --load-in-8bit specified. Defaulting to 4-bit."
        )
        load_in_8bit = False
    elif not load_in_4bit and not load_in_8bit:
        # Default to 4-bit if unsloth is available and neither is specified
        if HAS_UNSLOTH:
            print("Defaulting to 4-bit quantization with Unsloth.")
            load_in_4bit = True
        else:
            print("No quantization specified, loading in full precision.")

    # --- Loading ---
    try:
        if is_adapter:
            print(
                f"Loading base model '{base_model_name}' to apply adapters from '{model_dir}'"
            )
            if HAS_UNSLOTH:
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=base_model_name,
                    max_seq_length=max_seq_length,
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit,
                    device_map=device_map,  # Use specific device or auto
                    local_files_only=offline_mode,  # Enable offline mode
                )
                print("Loading adapters using PEFT...")
                model = FastLanguageModel.get_peft_model(
                    model, model_dir
                )  # Use Unsloth's PEFT loader
            else:
                # Use transformers and PEFT
                print("Loading base model with transformers...")
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit,
                    torch_dtype=torch.bfloat16,  # Or torch.float16 depending on GPU
                    device_map=device_map,
                    local_files_only=offline_mode,  # Enable offline mode
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    base_model_name,
                    local_files_only=offline_mode,  # Enable offline mode
                )
                print("Loading adapters using PEFT...")
                model = PeftModel.from_pretrained(model, model_dir)
                # Merging is optional for inference, but can sometimes speed it up
                # print("Merging adapters...")
                # model = model.merge_and_unload()
        else:
            print(f"Loading merged model from '{model_dir}'")
            if HAS_UNSLOTH:
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=model_dir,  # Load directly from the directory
                    max_seq_length=max_seq_length,
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit,
                    device_map=device_map,
                    local_files_only=offline_mode,  # Enable offline mode
                )
            else:
                print("Loading merged model with transformers...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit,
                    torch_dtype=torch.bfloat16,  # Or torch.float16
                    device_map=device_map,
                    local_files_only=offline_mode,  # Enable offline mode
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    model_dir,
                    local_files_only=offline_mode,  # Enable offline mode
                )

        print("Model and tokenizer loaded successfully.")
        return model, tokenizer

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        if "bitsandbytes" in str(e).lower():
            print(
                "Hint: You might need to install bitsandbytes: uv pip install bitsandbytes"
            )
        if "peft" in str(e).lower():
            print("Hint: You might need to install peft: uv pip install peft")
        sys.exit(1)


def run_test(model, tokenizer, input_text):
    """Run inference with the loaded model and tokenizer."""
    print("\n" + "=" * 80)
    print(f"TESTING INPUT: {input_text}")
    print("=" * 80)

    # Use the same instruction format as in finetune.py
    instruction = "As an ATC communication expert, improve this transcript and analyze its intentions and data."
    prompt = f"<|begin_of_text|><|header_start|>user<|header_end|>\n\n{instruction}\n\nOriginal: {input_text}<|eot|><|header_start|>assistant<|header_end|>\n\n"  # Ensure prompt ends with assistant start

    # Get the first module's device - with auto splitting, we need to know where to put inputs
    if args.split_across_gpus:
        # Find the device of the first module for input placement
        first_device = None
        for _, module in model.named_modules():
            if hasattr(module, "device") and module.device is not None:
                first_device = module.device
                break
            # Try to get device from parameters
            for param in module.parameters(recurse=False):
                if param.device:
                    first_device = param.device
                    break
            if first_device:
                break

        if not first_device:
            # Fall back to first parameter's device
            first_device = next(model.parameters()).device

        print(
            f"Model is split across GPUs. Placing inputs on first device: {first_device}"
        )
        device = first_device
    else:
        # For single-device models, get the device directly
        device = next(model.parameters()).device
        print(f"Model is on device: {device}")

    # Tokenize and move inputs to the right device
    inputs = tokenizer(prompt, return_tensors="pt")
    # Explicitly move inputs to the appropriate device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Set generation parameters (optional, adjust as needed)
    generation_params = {
        "max_new_tokens": 512,
        "do_sample": False,  # Use greedy decoding for consistent output
        # "temperature": 0.7,
        # "top_p": 0.9,
        "pad_token_id": tokenizer.eos_token_id,  # Set pad_token_id to eos_token_id
    }

    # Generate
    print("Generating response...")
    try:
        with torch.no_grad():  # Ensure no gradients are calculated
            outputs = model.generate(**inputs, **generation_params)

        # Decode
        # We need to decode only the newly generated tokens, excluding the prompt
        # If outputs is on a different device than inputs, move it to CPU first
        if args.split_across_gpus:
            output_text = tokenizer.decode(
                outputs[0, inputs["input_ids"].shape[1] :].cpu(),
                skip_special_tokens=True,
            )
        else:
            output_text = tokenizer.decode(
                outputs[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )

        print("--- MODEL OUTPUT ---")
        print(output_text.strip())
        print("--- END MODEL OUTPUT ---")
        return True

    except Exception as e:
        print(f"Error during generation: {e}")
        print(
            "Try running with --split-across-gpus for large models or --device cuda:0 for smaller models"
        )
        return False


def main():
    """Main function to run tests."""
    print(f"Testing fine-tuned ATC model from: {args.model_dir}")
    if args.split_across_gpus:
        print("Model will be split across available GPUs (auto device mapping)")
    else:
        print(f"Using device: {args.device}")

    # Load model and tokenizer once
    model, tokenizer = load_model_and_tokenizer()

    # Ensure the model is in evaluation mode
    model.eval()

    # Run tests for each input
    success_count = 0
    for i, test_input in enumerate(args.test_inputs, 1):
        print(f"\nRunning test {i}/{len(args.test_inputs)}")
        if run_test(model, tokenizer, test_input):
            success_count += 1

    # Print summary
    print("\n" + "=" * 80)
    print(f"TEST SUMMARY: {success_count}/{len(args.test_inputs)} tests successful")
    print("=" * 80)


if __name__ == "__main__":
    main()
