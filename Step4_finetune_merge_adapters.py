#!/usr/bin/env python3
"""
merge_adapters.py - Merge fine-tuned LoRA adapters with the base model.

This script:
1. Loads the base model.
2. Merges the LoRA adapters from the fine-tuning process.
3. Saves the merged model.
"""

import argparse
import os
import shutil
import subprocess
import sys
import torch
from unsloth import FastLanguageModel, is_bfloat16_supported

# Parse arguments
parser = argparse.ArgumentParser(
    description="Merge fine-tuned LoRA adapters with the base model."
)
parser.add_argument(
    "--adapter-dir",
    default="./atc_llama",
    help="Input directory with the fine-tuned LoRA adapters (default: ./atc_llama)",
)
parser.add_argument(
    "--base-model-name",
    default="meta-llama/Llama-3.2-3B-Instruct",
    help="Base model name used for fine-tuning (default: meta-llama/Llama-3.2-3B-Instruct)",
)
parser.add_argument(
    "--output-dir",
    default="./atc_llama_merged",
    help="Output directory for the merged model (default: ./atc_llama_merged)",
)
args = parser.parse_args()


def load_and_merge_model():
    """Load the base model and merge with LoRA adapters."""
    print("Step 1: Merging LoRA adapters with base model...")
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine dtype
    dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16
    print(f"Using dtype: {dtype}")

    # Load base model
    try:
        print(f"Loading base model: {args.base_model_name}")
        # First try with Unsloth if available
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.base_model_name,
                max_seq_length=2048,
                load_in_4bit=False,
                load_in_8bit=False,
                device_map="auto",
            )
            print("Base model loaded successfully with Unsloth")
        except ImportError:
            # Fall back to regular transformers
            print("Unsloth not available, using regular transformers")
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model = AutoModelForCausalLM.from_pretrained(
                args.base_model_name,
                torch_dtype=dtype,
                device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
            print("Base model loaded successfully with transformers")
    except Exception as e:
        print(f"Error loading base model: {str(e)}")
        return False

    # Now merge with LoRA adapters
    try:
        adapter_config_path = os.path.join(args.adapter_dir, "adapter_config.json")
        adapter_model_path = os.path.join(args.adapter_dir, "adapter_model.safetensors")

        if os.path.exists(adapter_config_path) or os.path.exists(adapter_model_path):
            print(f"Found LoRA adapter in {args.adapter_dir}")

            try:
                from peft import PeftModel

                print(f"Loading adapter from {args.adapter_dir}")
                model = PeftModel.from_pretrained(model, args.adapter_dir)
                print("Merging adapter...")
                model = model.merge_and_unload()
                print("LoRA adapters merged successfully")
            except ImportError:
                print("PEFT not available, please install with: uv pip install peft")
                return False
            except Exception as peft_error:
                print(f"Error loading/merging PEFT adapter: {peft_error}")
                return False
        else:
            print(f"No adapter configuration or weights found in {args.adapter_dir}. Assuming input is already merged or a full model.")
            if os.path.exists(os.path.join(args.adapter_dir, "config.json")):
                 print(f"Copying model files from {args.adapter_dir} to {args.output_dir} as no adapter was found.")
                 shutil.copytree(args.adapter_dir, args.output_dir, dirs_exist_ok=True)
                 print("Model files copied.")
                 if os.path.exists(os.path.join(args.adapter_dir, "tokenizer.json")):
                     tokenizer_files = [f for f in os.listdir(args.adapter_dir) if "token" in f or f == "special_tokens_map.json"]
                     for f_name in tokenizer_files:
                         shutil.copy2(os.path.join(args.adapter_dir, f_name), args.output_dir)
                     print("Tokenizer files copied.")
                 return True
            else:
                 print(f"Error: Neither adapter files nor a full model config found in {args.adapter_dir}.")
                 return False

    except Exception as e:
        print(f"Error during adapter handling: {str(e)}")
        return False

    # Save the merged model
    try:
        print(f"Saving merged model to: {args.output_dir}")
        # Let save_pretrained handle device placement
        model.save_pretrained(args.output_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.output_dir)
        print("Merged model saved successfully")
        return True
    except Exception as e:
        print(f"Error saving merged model: {str(e)}")
        return False

def main():
    """Main function to execute all steps."""
    print("Merging fine-tuned LoRA adapters with the base model")
    print(f"Adapter directory: {args.adapter_dir}")
    print(f"Base model: {args.base_model_name}")
    print(f"Output directory: {args.output_dir}")

    # Perform merge step
    merge_success = load_and_merge_model()

    # Print final message
    if merge_success:
        print("\nProcess completed successfully!")
        print(f"Merged model saved to {args.output_dir}")
    else:
        print("\nProcess completed with errors")
        print("Please check the logs above for details")


if __name__ == "__main__":
    main()
