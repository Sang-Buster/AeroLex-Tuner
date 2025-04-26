#!/usr/bin/env python3
"""
convert_hf_to_ollama.py - Converts a Hugging Face model to GGUF format for Ollama.

This script uses the convert.py script from the llama.cpp repository to perform
the conversion and quantization. It can also optionally upload the resulting
GGUF file to the Hugging Face Hub.

Prerequisites:
1. Clone the llama.cpp repository:
   git clone https://github.com/ggerganov/llama.cpp.git
2. Install llama.cpp python requirements:
   cd llama.cpp && uv pip install -r requirements.txt && cd ..

Example Usage:

# Convert merged model to Q4_K_M GGUF
python convert_hf_to_ollama.py \
    --model-dir ./atc_llama_merged \
    --llama-cpp-dir ./llama.cpp \
    --output-gguf ./atc_llama_q4_k_m.gguf \
    --quantization-type Q4_K_M

# Convert and upload to HF Hub
python convert_hf_to_ollama.py \
    --model-dir ./atc_llama_merged \
    --llama-cpp-dir ./llama.cpp \
    --output-gguf ./atc_llama_q4_k_m.gguf \
    --quantization-type Q4_K_M \
    --upload-repo your-username/atc-llama-gguf \
    --hf-token YOUR_HF_WRITE_TOKEN
"""

import argparse
import os
import subprocess
import sys

from huggingface_hub import HfApi, upload_file


def main():
    parser = argparse.ArgumentParser(
        description="Convert HF model to GGUF for Ollama using llama.cpp."
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Path to the local directory containing the Hugging Face model files (e.g., the merged model).",
    )
    parser.add_argument(
        "--llama-cpp-dir",
        required=True,
        help="Path to the cloned llama.cpp repository directory.",
    )
    parser.add_argument(
        "--output-gguf",
        required=True,
        help="Path where the output GGUF file will be saved.",
    )
    parser.add_argument(
        "--quantization-type",
        default="Q4_K_M",
        help="Quantization type to apply (e.g., F16, Q4_0, Q4_K_M, Q5_K_M, Q8_0). Default: Q4_K_M.",
    )
    parser.add_argument(
        "--upload-repo",
        default=None,
        help="Optional: Hugging Face Hub repository ID to upload the GGUF file to (e.g., your-username/your-repo-name).",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional: Hugging Face API token with write access for uploading.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="If uploading, create a private repository.",
    )

    args = parser.parse_args()

    # --- Validate paths ---
    if not os.path.isdir(args.model_dir):
        print(f"Error: Input model directory not found: {args.model_dir}")
        sys.exit(1)

    llama_cpp_convert_script = os.path.join(args.llama_cpp_dir, "convert_hf_to_gguf.py")
    if not os.path.isfile(llama_cpp_convert_script):
        print(
            f"Error: llama.cpp convert_hf_to_gguf.py script not found at expected path: {llama_cpp_convert_script}"
        )
        print(
            "Please provide the correct path to the cloned llama.cpp directory using --llama-cpp-dir."
        )
        sys.exit(1)

    output_dir = os.path.dirname(args.output_gguf)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # --- Run llama.cpp conversion ---
    print(f"Starting GGUF conversion for model: {args.model_dir}")
    print(f"Using quantization type: {args.quantization_type}")
    print(f"Output file: {args.output_gguf}")

    # Ensure we use the same Python interpreter that's running this script
    python_executable = sys.executable

    cmd = [
        python_executable,
        llama_cpp_convert_script,
        args.model_dir,
        "--outfile",
        args.output_gguf,
        "--outtype",
        args.quantization_type,
    ]

    try:
        # It's often helpful to see the output of the conversion script
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                print(output.strip())
        rc = process.poll()
        if rc != 0:
            print(f"Error: llama.cpp conversion script failed with exit code {rc}.")
            sys.exit(1)

        print(f"GGUF conversion successful: {args.output_gguf}")

    except FileNotFoundError:
        print(f"Error: Could not execute Python interpreter at '{python_executable}'.")
        print("Please ensure Python is correctly installed and in your PATH.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during conversion: {str(e)}")
        sys.exit(1)

    # --- Optional: Upload to Hugging Face Hub ---
    if args.upload_repo:
        if not os.path.exists(args.output_gguf):
            print(f"Error: GGUF file not found for upload: {args.output_gguf}")
            sys.exit(1)

        print(
            f"Attempting to upload {args.output_gguf} to HF Hub repo: {args.upload_repo}"
        )

        try:
            api = HfApi(token=args.hf_token)

            # Create repo if it doesn't exist
            api.create_repo(
                repo_id=args.upload_repo,
                repo_type="model",
                private=args.private,
                exist_ok=True,  # Don't error if it already exists
            )
            print(f"Ensured repository '{args.upload_repo}' exists.")

            # Upload the single GGUF file
            upload_file(
                path_or_fileobj=args.output_gguf,
                path_in_repo=os.path.basename(
                    args.output_gguf
                ),  # Upload with the same filename
                repo_id=args.upload_repo,
                repo_type="model",
                token=args.hf_token,
                commit_message=f"feat: upload {os.path.basename(args.output_gguf)} ({args.quantization_type})",
            )
            print(
                f"Successfully uploaded GGUF file to: https://huggingface.co/{args.upload_repo}"
            )

        except Exception as e:
            print(f"An error occurred during upload: {str(e)}")
            print(
                "Please ensure you have huggingface_hub installed and are logged in or provided a valid token."
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
