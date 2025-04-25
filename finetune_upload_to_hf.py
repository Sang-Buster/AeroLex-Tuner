#!/usr/bin/env python3
"""
finetune_upload_to_hf.py - Uploads a local model or adapter directory to Hugging Face Hub.

Example Usage:

# Upload adapters
python finetune_upload_to_hf.py --model-dir ./atc_llama --repo-id Sang-Buster/atc_llama_adapters

# Upload merged model
python finetune_upload_to_hf.py --model-dir ./atc_llama_merged --repo-id Sang-Buster/atc_llama
"""

import argparse
import json
import os
import sys

from huggingface_hub import HfApi, upload_folder


def fix_adapter_config(model_dir, base_model_id="meta-llama/Llama-3.2-3B-Instruct"):
    """
    Fix adapter config by replacing local paths with proper HF model ID.

    Args:
        model_dir: Directory containing adapter files
        base_model_id: HF model ID for the base model
    """
    adapter_config_path = os.path.join(model_dir, "adapter_config.json")

    if os.path.exists(adapter_config_path):
        try:
            with open(adapter_config_path, "r") as f:
                config = json.load(f)
                if "base_model_name_or_path" in config:
                    # Fix local path to proper HF model ID
                    local_path = config["base_model_name_or_path"]
                    if os.path.exists(local_path) or local_path.startswith("models/"):
                        config["base_model_name_or_path"] = base_model_id
                        # Save the updated config
                        with open(adapter_config_path, "w") as f:
                            json.dump(config, f, indent=2)
                            print(
                                f"Updated adapter_config.json with correct base model: {base_model_id}"
                            )
            return True
        except Exception as e:
            print(f"Warning: Error processing adapter config: {e}")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Upload a model/adapter directory to Hugging Face Hub."
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Path to the local directory containing the model/adapter files.",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face Hub repository ID (e.g., your-username/your-repo-name).",
    )
    parser.add_argument(
        "--repo-type",
        default="model",
        choices=["model", "dataset", "space"],
        help="Type of repository on the Hub (default: model).",
    )
    parser.add_argument(
        "--commit-message",
        default="feat: upload model files",
        help="Commit message for the upload.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face API token (optional, uses login cache if not provided).",
    )
    parser.add_argument(
        "--private", action="store_true", help="Create a private repository."
    )
    parser.add_argument(
        "--base-model-id",
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Hugging Face model ID for the base model (for adapters). Default: meta-llama/Llama-3.2-3B-Instruct",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.model_dir):
        print(f"Error: Local directory not found: {args.model_dir}")
        sys.exit(1)

    print(f"Preparing to upload contents of '{args.model_dir}' to '{args.repo_id}'")

    # Check if this is an adapter directory and fix config if needed
    is_adapter = os.path.exists(os.path.join(args.model_dir, "adapter_config.json"))
    if is_adapter:
        print("Detected LoRA adapter directory - fixing configs if needed")
        fix_adapter_config(args.model_dir, args.base_model_id)

    try:
        api = HfApi(token=args.token)

        # Check if repo exists, create if not
        try:
            api.repo_info(repo_id=args.repo_id, repo_type=args.repo_type)
            print(f"Repository '{args.repo_id}' already exists.")
        except Exception:  # More specific exceptions can be caught if needed
            print(f"Repository '{args.repo_id}' not found. Creating...")
            api.create_repo(
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                private=args.private,
                exist_ok=True,  # In case of race condition
            )
            print(f"Repository '{args.repo_id}' created successfully.")

        # Upload the folder
        print("Starting upload...")
        repo_url = upload_folder(
            folder_path=args.model_dir,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            commit_message=args.commit_message,
            token=args.token,  # Pass token explicitly if provided
        )
        print(f"Successfully uploaded to: {repo_url}")

    except Exception as e:
        print(f"An error occurred during upload: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
