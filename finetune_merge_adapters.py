#!/usr/bin/env python3
"""
finetune_merge_adapters.py - Merge fine-tuned LoRA adapters with the base model.

This script:
1. Loads the base model.
2. Merges the LoRA adapters from the fine-tuning process.
3. Saves the merged model.
"""

import argparse
import os
import shutil
import sys

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
    "--offline",
    action="store_true",
    help="Use local model files (base-model-name points to a local directory)",
)
parser.add_argument(
    "--download",
    metavar="MODEL",
    help="Download a Hugging Face model to use locally (will exit after download)",
)
parser.add_argument(
    "--output-dir",
    default="./atc_llama_merged",
    help="Output directory for the merged model (default: ./atc_llama_merged)",
)

# Check if help flag is requested before importing Unsloth
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

# Import Unsloth and other libraries after help check
import torch  # noqa: E402
from unsloth import FastLanguageModel, is_bfloat16_supported  # noqa: E402


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
                local_files_only=args.offline,  # Enable offline mode
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
                local_files_only=args.offline,  # Enable offline mode
            )
            tokenizer = AutoTokenizer.from_pretrained(
                args.base_model_name,
                local_files_only=args.offline,  # Enable offline mode
            )
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
            print(
                f"No adapter configuration or weights found in {args.adapter_dir}. Assuming input is already merged or a full model."
            )
            if os.path.exists(os.path.join(args.adapter_dir, "config.json")):
                print(
                    f"Copying model files from {args.adapter_dir} to {args.output_dir} as no adapter was found."
                )
                shutil.copytree(args.adapter_dir, args.output_dir, dirs_exist_ok=True)
                print("Model files copied.")
                if os.path.exists(os.path.join(args.adapter_dir, "tokenizer.json")):
                    tokenizer_files = [
                        f
                        for f in os.listdir(args.adapter_dir)
                        if "token" in f or f == "special_tokens_map.json"
                    ]
                    for f_name in tokenizer_files:
                        shutil.copy2(
                            os.path.join(args.adapter_dir, f_name), args.output_dir
                        )
                    print("Tokenizer files copied.")
                return True
            else:
                print(
                    f"Error: Neither adapter files nor a full model config found in {args.adapter_dir}."
                )
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


def create_readme_template(output_dir, base_model_name):
    """Create a detailed README.md file for the merged model with YAML metadata."""
    model_id = base_model_name
    # If base_model_name points to a local directory, use default model ID
    if (
        os.path.exists(base_model_name)
        or base_model_name.startswith("./")
        or base_model_name.startswith("models/")
    ):
        model_id = "meta-llama/Llama-3.2-3B-Instruct"

    # Get model type from base name
    model_type = "llama"
    if "llama" in model_id.lower():
        model_type = "llama"
    elif "mistral" in model_id.lower():
        model_type = "mistral"

    # Get model size if available
    model_size = "3B"
    if "3b" in model_id.lower():
        model_size = "3B"
    elif "7b" in model_id.lower():
        model_size = "7B"
    elif "8b" in model_id.lower():
        model_size = "8B"
    elif "13b" in model_id.lower():
        model_size = "13B"
    elif "34b" in model_id.lower():
        model_size = "34B"
    elif "70b" in model_id.lower():
        model_size = "70B"

    # For merged models - create YAML metadata and detailed README
    readme_content = f"""---
license: llama3.2
language:
- en
base_model:
- {model_id}
pipeline_tag: text-generation
tags:
- Speech Recognition
- ATC
- Unsloth
- LoRA-Merged
---

# ATC Communication Expert Model (Merged)

A fine-tuned model specialized in improving and analyzing Air Traffic Control (ATC) communications, with LoRA adapters merged into the base model.

## Model Details

### Model Description

This model is a fine-tuned version of {model_id} with merged LoRA adapters, optimized for processing Air Traffic Control communications. It can:

- Improve raw ATC transcripts with proper punctuation and formatting
- Identify communication intentions (pilot requests, ATC instructions, etc.)
- Extract key information such as flight numbers, altitudes, headings, and other numerical data
- Analyze speaker roles and communication patterns

The model was created by merging LoRA adapters (fine-tuned on ATC communications) into the {model_type.title()} {model_size} base model, creating a unified model optimized for this specialized domain.

- **Developed by:** ATC NLP Team
- **Model type:** {model_type.title()} {model_size} with merged LoRA adapters
- **Language(s):** English, specialized for ATC terminology
- **License:** Same as the base model
- **Finetuned from model:** {model_id}

## Uses

### Direct Use

This model is intended for:
- Transcribing and formatting raw ATC communications
- Training ATC communication skills
- Analyzing ATC communication patterns
- Extracting structured data from ATC communications
- Educational purposes for those learning ATC communication protocols

### Downstream Use

The model can be integrated into:
- Air traffic management training systems
- Communication analysis tools
- ATC transcript post-processing pipelines
- Aviation safety monitoring systems
- Radio communication enhancement systems

### Out-of-Scope Use

This model is not suitable for:
- Real-time ATC operations or safety-critical decision-making
- Full language translation (it's specialized for ATC terminology only)
- General language processing outside the ATC domain
- Any application where model errors could impact flight safety

## Bias, Risks, and Limitations

- The model is specialized for ATC communications and may not perform well on general text
- It may have limitations with accents or non-standard ATC phraseology
- Performance depends on audio transcription quality for real-world applications
- Not intended for safety-critical applications without human verification
- May have biases based on the training data distribution

### Recommendations

- Always have human verification for safety-critical applications
- Use in conjunction with standard ATC protocols, not as a replacement
- Provide clear domain context for optimal performance
- Test thoroughly with diverse ATC communications before deployment
- Consider fine-tuning further on your specific ATC subdomain if needed

## How to Get Started with the Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "{os.path.basename(output_dir)}",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("{os.path.basename(output_dir)}")

# Process an ATC message
instruction = "As an ATC communication expert, improve this transcript and analyze its intentions and data."
message = "southwest five niner two turn left heading three four zero descend and maintain flight level two five zero"

prompt = f"<|begin_of_text|><|header_start|>user<|header_end|>\\n\\n{{instruction}}\\n\\nOriginal: {{message}}<|eot|><|header_start|>assistant<|header_end|>\\n\\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate improved transcript and analysis
outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
response = tokenizer.decode(outputs[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(response)
```

## Model Creation Process

### Base Model and Adapters

- **Base model:** {model_id}
- **Adapter source:** LoRA adapters fine-tuned on ATC communications data
- **Merge method:** PEFT adapter merging into base model weights

### Merging Procedure

The model creation involved:
1. Loading the base {model_type.title()} {model_size} model
2. Loading LoRA adapters fine-tuned on ATC communications data
3. Merging the adapters into the base model's weights
4. Saving the resulting unified model

## Evaluation

### Testing

The model should be tested on diverse ATC communications, including:
- Clearances and instructions
- Pilot requests and reports
- Emergency communications
- Different accents and speaking patterns

## Technical Specifications

### Model Architecture and Objective

- **Base architecture:** {model_id}
- **Adaptation method:** LoRA adapters merged into base weights
- **Training objective:** Improving and analyzing ATC communications

### Model Card Contact

For issues or questions about this model, please open a discussion in the repository.
"""

    # Write the README.md file
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(readme_content)

    print(f"Created detailed README.md in {output_dir}")
    return readme_path


def main():
    """Main function to execute all steps."""
    print("Merging fine-tuned LoRA adapters with the base model")
    print(f"Adapter directory: {args.adapter_dir}")
    print(f"Base model: {args.base_model_name}")
    print(f"Output directory: {args.output_dir}")

    # Perform merge step
    merge_success = load_and_merge_model()

    # Create README.md for the merged model
    if merge_success:
        create_readme_template(args.output_dir, args.base_model_name)

    # Print final message
    if merge_success:
        print("\nProcess completed successfully!")
        print(f"Merged model saved to {args.output_dir}")
    else:
        print("\nProcess completed with errors")
        print("Please check the logs above for details")


if __name__ == "__main__":
    main()
