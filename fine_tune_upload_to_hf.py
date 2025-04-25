#!/usr/bin/env python3
"""
upload_to_hf.py - Uploads a local model or adapter directory to Hugging Face Hub.

Example Usage:

# Upload adapters
python upload_to_hf.py --model-dir ./atc_llama --repo-id your-username/atc_llama_adapters

# Upload merged model
python upload_to_hf.py --model-dir ./atc_llama_merged --repo-id your-username/atc_llama_merged
"""

import argparse
import os
import sys

from huggingface_hub import HfApi, upload_folder


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

    args = parser.parse_args()

    if not os.path.isdir(args.model_dir):
        print(f"Error: Local directory not found: {args.model_dir}")
        sys.exit(1)

    print(f"Preparing to upload contents of '{args.model_dir}' to '{args.repo_id}'")

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
