# ATC Model Fine-Tuning with Unsloth

This repository contains scripts for fine-tuning, testing, and uploading a Llama 3.2 model specifically for ATC (Air Traffic Control) communications using the Unsloth framework.

## Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended)
- The following Python packages:
  - unsloth
  - pandas
  - datasets
  - transformers
  - torch
  - huggingface-hub
  - setuptools
  - peft (for LoRA adapters)
  - bitsandbytes (for quantization)

Install the required packages:

```bash
uv v -p 3.10
source .venv/bin/activate
uv pip install pandas datasets unsloth transformers torch huggingface-hub setuptools peft bitsandbytes
```

## Files

- `finetune.py`: Script for fine-tuning the model
- `finetune_merge_adapters.py`: Script for merging trained LoRA adapters with base model
- `finetune_test.py`: Script for testing the fine-tuned model
- `finetune_upload_to_hf.py`: Script for uploading models/adapters to Hugging Face Hub

## Downloading Models for Offline Use

Scripts also includes a `--download` option to download models for offline use:

```bash
# Download a model for fine-tuning
python finetune.py --download meta-llama/Llama-3.2-3B-Instruct

# Download a model for merging with adapters
python finetune_merge_adapters.py --download meta-llama/Llama-3.2-3B-Instruct

# Download a model for testing
python finetune_test.py --download meta-llama/Llama-3.2-3B-Instruct
```

After downloading, models will be stored in a `models/` directory, and you can use them with the `--offline` flag.

## Fine-Tuning

The `finetune.py` script fine-tunes a Llama 3.2 model on ATC communications data.

### Basic Usage

```bash
python finetune.py --input-file data/test_llama_70b_punv.json
```

### Offline Mode Usage

```bash
# Use a downloaded model for fine-tuning
python finetune.py --input-file data/atc_data.json --model-name models/Llama-3.2-3B-Instruct --offline
```

### Advanced Options

```bash
python finetune.py \
  --input-file data/test_llama_70b_punv.json \
  --model-name meta-llama/Llama-3.2-3B-Instruct \
  --output-dir ./my_atc_model \
  --epochs 5 \
  --batch-size 8 \
  --learning-rate 1e-4 \
  --rank 32 \
  --alpha 16 \
  --max-seq-length 4096 \
  --gradient-accumulation-steps 4 \
  --warmup-ratio 0.05
```

### Parameter Explanation

- `--input-file`: Path to the JSON file with ATC data (default: `data/test_llama_70b_punv.json`)
- `--epochs`: Number of training epochs (default: 3)
- `--batch-size`: Batch size for training (default: 4)
- `--learning-rate`: Learning rate (default: 2e-4)
- `--rank`: LoRA rank (default: 16)
- `--alpha`: LoRA alpha (default: 16)
- `--max-seq-length`: Maximum sequence length (default: 2048)
- `--model-name`: Model name to fine-tune (default: `meta-llama/Llama-3.2-3B-Instruct`)
- `--offline`: Use local model files (model-name points to a local directory)
- `--download`: Download a Hugging Face model to use locally
- `--output-dir`: Output directory for the fine-tuned model (default: `./atc_llama`)
- `--gradient-accumulation-steps`: Gradient accumulation steps (default: 4)
- `--warmup-ratio`: Warmup ratio (default: 0.03)
- `--max-steps`: Maximum number of training steps (default: None)
- `--full-finetune`: Perform full fine-tuning instead of LoRA (requires more VRAM)
- `--load-in-8bit`: Load the model in 8-bit precision (alternative to 4-bit)

## Merging LoRA Adapters with Base Model

Use the `finetune_merge_adapters.py` script to merge fine-tuned LoRA adapters with the base model:

```bash
python finetune_merge_adapters.py --adapter-dir ./atc_llama --output-dir ./atc_llama_merged
```

### Offline Mode Usage

```bash
python finetune_merge_adapters.py \
  --adapter-dir ./atc_llama \
  --base-model-name models/Llama-3.2-3B-Instruct \
  --output-dir ./atc_llama_merged \
  --offline
```

### Merge Parameters

- `--adapter-dir`: Input directory with fine-tuned LoRA adapters (default: `./atc_llama`)
- `--base-model-name`: Base model name used for fine-tuning (default: `meta-llama/Llama-3.2-3B-Instruct`)
- `--offline`: Use local model files (base-model-name points to a local directory)
- `--download`: Download a Hugging Face model to use locally
- `--output-dir`: Output directory for the merged model (default: `./atc_llama_merged`)

## Testing the Model

Use the `finetune_test.py` script to test your fine-tuned model:

```bash
python finetune_test.py --model-dir ./atc_llama_merged
```

### Offline Mode Usage

```bash
python finetune_test.py --model-dir ./atc_llama_merged --offline
```

### Multi-GPU and Device Control

```bash
# Use a specific GPU
python finetune_test.py --model-dir ./atc_llama_merged --device cuda:1

# Split large models across available GPUs
python finetune_test.py --model-dir ./atc_llama_merged --split-across-gpus
```

### Advanced Testing Options

```bash
python finetune_test.py \
  --model-dir ./my_atc_model \
  --base-model-name meta-llama/Llama-3.2-3B-Instruct \
  --load-in-8bit \
  --test-inputs "delta six three eight traffic alert" "southwest five niner two turn left"
```

### Testing Parameters

- `--model-dir`: Directory containing the fine-tuned model or adapters (default: `./atc_llama_merged`)
- `--base-model-name`: Base model name (required if loading adapters, ignored for merged models)
- `--max-seq-length`: Maximum sequence length (default: 2048)
- `--load-in-4bit`: Load model in 4-bit precision (default behavior with Unsloth)
- `--load-in-8bit`: Load model in 8-bit precision (alternative to 4-bit)
- `--offline`: Use local model files (model-dir and base-model-name point to local directories)
- `--download`: Download a Hugging Face model to use locally
- `--device`: Device to run the model on (e.g., 'cuda:0', 'cuda:1', 'cpu')
- `--split-across-gpus`: Split model across multiple GPUs if it's too large for a single GPU
- `--test-inputs`: List of test inputs to process

## Uploading to Hugging Face Hub

Use the `finetune_upload_to_hf.py` script to upload your models or adapters to the Hugging Face Hub. Both the fine-tuning and merging scripts automatically generate comprehensive model cards (README.md files) that work with Hugging Face Hub.

### Uploading Adapters

```bash
python finetune_upload_to_hf.py --model-dir ./atc_llama --repo-id username/atc-llama-adapters
```

### Uploading Merged Model

```bash
python finetune_upload_to_hf.py --model-dir ./atc_llama_merged --repo-id username/atc-llama
```

### Advanced Upload Options

```bash
python finetune_upload_to_hf.py \
  --model-dir ./atc_llama \
  --repo-id username/atc-llama-adapters \
  --base-model-id meta-llama/Llama-3.2-3B-Instruct \
  --private
```

### Upload Parameters

- `--model-dir`: Path to the local directory containing the model/adapter files
- `--repo-id`: Hugging Face Hub repository ID (e.g., username/model-name)
- `--repo-type`: Type of repository on the Hub (default: model)
- `--commit-message`: Commit message for the upload
- `--token`: Hugging Face API token (optional, uses login cache if not provided)
- `--private`: Create a private repository
- `--base-model-id`: Hugging Face model ID for the base model (for adapters)

## Automatically Generated Model Cards

All models and adapters automatically include comprehensive model cards (README.md files) that include:

### YAML Metadata for Hugging Face Hub
```yaml
---
license: llama3.2
language:
- en
base_model:
- meta-llama/Llama-3.2-3B-Instruct
pipeline_tag: text-generation
tags:
- Speech Recognition
- ATC
- PEFT
- LoRA
---
```

### Detailed Sections
- Model description and capabilities
- Direct and downstream uses
- Limitations and recommendations
- Usage examples with code
- Training details and hyperparameters
- Technical specifications

These model cards are automatically generated when you:

1. Run fine-tuning (`finetune.py`)
2. Merge adapters (`finetune_merge_adapters.py`)

The upload script (`finetune_upload_to_hf.py`) will automatically handle fixing any paths in adapter configs to ensure compatibility with Hugging Face Hub.

## Data Format

The input data should be a JSON file with the following structure:

```json
[
  {
    "original": "united four seven three contact approach on one two five point zero",
    "punctuated": "United 4 7 3 contact approach on 1 2 5 point 0.",
    "pune": "United 473, contact approach on 125.0.",
    "punv": "United 473, contact approach on 125.0.",
    "intentions": {
      "PSC": false,
      "PSR": false,
      "PRP": false,
      "PRQ": false,
      "PRB": false,
      "PAC": false,
      "ASC": false,
      "AGI": true,
      "ACR": false,
      "END": false
    },
    "numbers": {
      "Csgn": "United 473",
      "Freq": "125.0"
    },
    "speaker": "Controller",
    "listener": "United 473",
    "event": "Instructing United 473 to contact approach control.",
    "actions": "United 473 is being instructed to switch to approach control frequency 125.0."
  },
  ...
]
```

## Acknowledgments

- This project uses the [Unsloth](https://github.com/unslothai/unsloth) framework for efficient fine-tuning
- Models are hosted on [Hugging Face](https://huggingface.co/) 