#!/usr/bin/env python3
"""
create_ollama_hf_model.py - Create and upload a GGUF model to Hugging Face with Ollama Modelfile

This script:
1. Creates an optimized Ollama Modelfile for an ATC model
2. Uploads the GGUF file to Hugging Face Hub
3. Creates a README.md with usage instructions for Ollama

Prerequisites:
1. A GGUF file to upload (generate with convert_hf_to_ollama.py)
2. huggingface_hub package installed

Example Usage:

# Upload a GGUF model to Hugging Face with Ollama Modelfile
python create_ollama_hf_model.py \
    --gguf-file ./atc_llama_q4_k_m.gguf \
    --upload-repo Sang-Buster/atc-llama-gguf \
    --hf-token YOUR_HF_WRITE_TOKEN

# Add custom parameters to the Modelfile
python create_ollama_hf_model.py \
    --gguf-file ./atc_llama_q4_k_m.gguf \
    --upload-repo Sang-Buster/atc-llama-gguf \
    --hf-token YOUR_HF_WRITE_TOKEN \
    --parameters "temperature: 0.7, top_p: 0.9, num_ctx: 128000"
"""

import argparse
import os
import sys
import tempfile
from datetime import datetime

from huggingface_hub import HfApi, upload_file

# Hardcoded system prompt specifically for ATC model
ATC_SYSTEM_PROMPT = """You are an Air Traffic Control (ATC) communication expert. Your primary tasks are:

1. Improve and format raw ATC transcripts with proper aviation conventions
2. Analyze communication intentions in ATC transmissions
3. Extract critical data like flight numbers, altitudes, headings, and speeds
4. Identify standard aviation procedures and clearances
5. Explain ATC terminology and protocols when needed

Always follow standard ATC communication protocols, use proper formatting for flight numbers, altitudes, and headings, and maintain aviation safety standards in your responses.
"""

# Example conversations for ATC model
ATC_EXAMPLE_MESSAGES = [
    {
        "role": "user",
        "content": "southwest five niner two turn left heading three four zero descend and maintain flight level two five zero",
    },
    {
        "role": "assistant",
        "content": "Southwest 592, turn left heading 340, descend and maintain FL250.",
    },
    {
        "role": "user",
        "content": "delta six three eight traffic alert cessna two thousand feet twelve oclock expedite climb flight level three five zero",
    },
    {
        "role": "assistant",
        "content": "Delta 638, traffic alert. Cessna at 2,000 feet, 12 o'clock. Expedite climb to FL350.",
    },
    {
        "role": "user",
        "content": "united four seven three request clearance to descend due to turbulence",
    },
    {
        "role": "assistant",
        "content": "United 473, cleared to descend to FL240 due to turbulence. Report when level.",
    },
]

# Recommended template for Llama models with system prompt
LLAMA_TEMPLATE = """{{ if .System }}<|begin_of_text|><|header_start|>system<|header_end|>

{{ .System }}<|eot|>{{ end }}{{ if .Prompt }}<|begin_of_text|><|header_start|>user<|header_end|>

{{ .Prompt }}<|eot|><|header_start|>assistant<|header_end|>

{{ .Response }}<|eot|>{{ end }}"""


def create_modelfile(args):
    """Create an optimized Modelfile with the GGUF for Ollama."""
    model_content = []

    # Add filename as FROM
    model_content.append(f"FROM {os.path.basename(args.gguf_file)}")

    # Add the ATC system prompt
    model_content.append(f'SYSTEM """{ATC_SYSTEM_PROMPT}"""')

    # Add Llama template for modern models
    model_content.append(f'TEMPLATE """{LLAMA_TEMPLATE}"""')

    # Add parameters for optimized performance
    default_params = {
        "temperature": "0.7",
        "top_p": "0.9",
        "top_k": "40",
        "num_ctx": "128000",
        "repeat_penalty": "1.1",
        "stop": "<|header_start|>",
        "stop": "<|header_end|>",  # noqa: F601
        "stop": "<|eot|>",  # noqa: F601
    }

    # Parse user-provided parameters
    if args.parameters:
        params = [p.strip() for p in args.parameters.split(",")]
        for param in params:
            if ":" in param:
                key, value = param.split(":", 1)
                default_params[key.strip()] = value.strip()

    # Add all parameters to the Modelfile
    for key, value in default_params.items():
        model_content.append(f"PARAMETER {key} {value}")

    # Add ATC-specific example conversations
    for i in range(0, len(ATC_EXAMPLE_MESSAGES), 2):
        if i + 1 < len(ATC_EXAMPLE_MESSAGES):
            model_content.append(f"MESSAGE user {ATC_EXAMPLE_MESSAGES[i]['content']}")
            model_content.append(
                f"MESSAGE assistant {ATC_EXAMPLE_MESSAGES[i + 1]['content']}"
            )

    # Add license
    model_content.append(
        'LICENSE """MIT License - Feel free to use this model for any purpose, including commercial applications."""'
    )

    # Create the Modelfile content
    modelfile_content = "\n\n".join(model_content)

    return modelfile_content


def create_readme(args, modelfile_content):
    """Create a README.md file with usage instructions."""

    # Get simplified quantization name for display
    quant_type = "F16"
    if "_" in os.path.basename(args.gguf_file):
        parts = os.path.basename(args.gguf_file).split("_")
        if len(parts) > 1:
            quant_suffix = parts[-1].replace(".gguf", "")
            quant_type = quant_suffix

    # Calculate file size
    file_size_mb = os.path.getsize(args.gguf_file) / (1024 * 1024)

    # Format today's date
    today_date = datetime.now().strftime("%Y-%m-%d")

    # Create front matter for Hugging Face
    front_matter = """---
license: llama3.2
language:
- en
base_model:
- meta-llama/Llama-3.2-3B-Instruct
pipeline_tag: text-generation
tags:
- Speech Recognition
- ATC
- Unsloth
- LoRA-Merged
---"""

    # Create the main README content
    readme = f"""{front_matter}

# ATC Communication Expert - GGUF Model for Ollama

A specialized GGUF model for Air Traffic Control communication analysis and formatting.

## Model Description

This model is optimized for:
- Improving and formatting raw ATC transcripts
- Analyzing communication intentions in ATC transmissions
- Extracting flight numbers, altitudes, headings, and other numeric data
- Identifying standard aviation procedures and clearances
- Explaining ATC terminology and protocols

## Using with Ollama

### Option 1: Direct Run from Hugging Face

```bash
ollama run hf://{args.upload_repo}/{os.path.basename(args.gguf_file)}
```

### Option 2: Create a Local Model

Create a Modelfile with the following content:

```
{modelfile_content}
```

Save this to a file named `Modelfile`, then run:

```bash
# Create the model
ollama create atc-expert -f ./Modelfile 

# Run the model
ollama run atc-expert
```

## Example Conversations

```
User: southwest five niner two turn left heading three four zero descend and maintain flight level two five zero
Assistant: Southwest 592, turn left heading 340, descend and maintain FL250.

User: delta six three eight traffic alert cessna two thousand feet twelve oclock expedite climb flight level three five zero
Assistant: Delta 638, traffic alert. Cessna at 2,000 feet, 12 o'clock. Expedite climb to FL350.
```

## Model Details

- **Format**: GGUF (optimized for Ollama, llama.cpp)
- **Size**: {file_size_mb:.1f} MB
- **Quantization**: {quant_type}
- **Base model**: meta-llama/Llama-3.2-3B-Instruct
- **License**: Llama 3.2 License
- **Created**: {today_date}

## System Prompt

```
{ATC_SYSTEM_PROMPT}
```
"""
    return readme


def upload_to_huggingface(args, modelfile_content):
    """Upload the GGUF file and supporting files to Hugging Face Hub."""
    try:
        api = HfApi(token=args.hf_token)

        # Check file size before upload
        file_size_mb = os.path.getsize(args.gguf_file) / (1024 * 1024)
        print(f"GGUF file size: {file_size_mb:.1f} MB")

        # Create repo if it doesn't exist
        print(f"Ensuring repository '{args.upload_repo}' exists...")
        api.create_repo(
            repo_id=args.upload_repo,
            repo_type="model",
            private=args.private,
            exist_ok=True,
        )

        # Create README.md
        readme_content = create_readme(args, modelfile_content)
        readme_path = os.path.join(tempfile.gettempdir(), "README.md")
        with open(readme_path, "w") as f:
            f.write(readme_content)

        # Create Modelfile
        modelfile_path = os.path.join(tempfile.gettempdir(), "Modelfile")
        with open(modelfile_path, "w") as f:
            f.write(modelfile_content)

        # Upload README.md
        print("Uploading README.md...")
        upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=args.upload_repo,
            repo_type="model",
            token=args.hf_token,
        )

        # Upload Modelfile
        print("Uploading Modelfile...")
        upload_file(
            path_or_fileobj=modelfile_path,
            path_in_repo="Modelfile",
            repo_id=args.upload_repo,
            repo_type="model",
            token=args.hf_token,
        )

        # Upload GGUF file (this may take time)
        print(f"Uploading GGUF file to {args.upload_repo}...")
        print("This may take a while depending on the file size...")

        upload_file(
            path_or_fileobj=args.gguf_file,
            path_in_repo=os.path.basename(args.gguf_file),
            repo_id=args.upload_repo,
            repo_type="model",
            token=args.hf_token,
        )

        print(
            f"Successfully uploaded model files to: https://huggingface.co/{args.upload_repo}"
        )
        print(
            f"To use with Ollama: ollama run hf://{args.upload_repo}/{os.path.basename(args.gguf_file)}"
        )

        # Clean up temp files
        os.remove(readme_path)
        os.remove(modelfile_path)

        return True

    except Exception as e:
        print(f"An error occurred during upload: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Create and upload GGUF model to Hugging Face with Ollama Modelfile."
    )
    parser.add_argument(
        "--gguf-file",
        required=True,
        help="Path to the local GGUF file to upload.",
    )
    parser.add_argument(
        "--upload-repo",
        required=True,
        help="Hugging Face repository ID to upload the GGUF file to (e.g., your-username/atc-llama-gguf).",
    )
    parser.add_argument(
        "--hf-token",
        required=True,
        help="Hugging Face API token with write access for uploading.",
    )
    parser.add_argument(
        "--parameters",
        default=None,
        help="Optional parameters for the model (comma-separated list, e.g. 'temperature: 0.7, top_p: 0.9').",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="If uploading, create a private repository.",
    )

    args = parser.parse_args()

    # Validate GGUF file
    if not os.path.isfile(args.gguf_file):
        print(f"Error: GGUF file not found: {args.gguf_file}")
        sys.exit(1)

    # Create optimized Modelfile
    print(f"Creating optimized Modelfile for {os.path.basename(args.gguf_file)}...")
    print("Using specialized ATC system prompt")
    modelfile_content = create_modelfile(args)

    # Print the Modelfile content
    print("\n--- Modelfile Content ---")
    print(modelfile_content)
    print("------------------------\n")

    # Upload to Hugging Face
    print(f"Uploading to Hugging Face repo: {args.upload_repo}")
    success = upload_to_huggingface(args, modelfile_content)

    if success:
        print("\n--- Upload Successful ---")
        print(f"Model URL: https://huggingface.co/{args.upload_repo}")
        print(
            f"Ollama usage: ollama run hf://{args.upload_repo}/{os.path.basename(args.gguf_file)}"
        )
        print("------------------------\n")
    else:
        print("\n--- Upload Failed ---")
        sys.exit(1)


if __name__ == "__main__":
    main()
