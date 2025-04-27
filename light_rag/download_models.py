#!/usr/bin/env python3
"""
Download Hugging Face models for offline use.
This script should be run on a machine with internet access
before transferring models to an offline GPU server.
"""

import argparse
import os

from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


def download_model(model_id, output_dir):
    """Download model and tokenizer to local directory."""
    model_dir = os.path.join(output_dir, model_id.replace("/", "_"))

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print(f"Downloading {model_id} to {model_dir}...")

    # For LLMs
    if "Llama" in model_id or model_id == "Sang-Buster/atc-llama":
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)

        tokenizer.save_pretrained(model_dir)
        model.save_pretrained(model_dir)

    # For embedding models
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)

        tokenizer.save_pretrained(model_dir)
        model.save_pretrained(model_dir)

    print(f"✓ Successfully downloaded {model_id}")
    return model_dir


def main():
    parser = argparse.ArgumentParser(
        description="Download Hugging Face models for offline use"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./models",
        help="Directory to save downloaded models",
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # List of models to download
    models = [
        "meta-llama/Llama-3.2-3B-Instruct",
        "Sang-Buster/atc-llama",
        "nomic-ai/nomic-embed-text-v1.5",
    ]

    # Download each model
    downloaded_paths = {}
    for model_id in models:
        model_path = download_model(model_id, args.output)
        downloaded_paths[model_id] = model_path

    # Print summary
    print("\n=== Downloaded Models ===")
    for model_id, path in downloaded_paths.items():
        print(f"{model_id} -> {path}")

    print("\nNow you can copy these models to your offline GPU server.")
    print(
        "Use the paths above with the --local-models flag in lightrag_hf_conccurent_models_demo.py"
    )


if __name__ == "__main__":
    main()
