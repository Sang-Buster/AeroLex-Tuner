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
Install the required packages:

```bash
uv v -p 3.10
source .venv/bin/activate
uv pip install pandas datasets unsloth transformers torch huggingface-hub setuptools
```

## Files

- `finetune.py`: Script for fine-tuning the model
- `finetune_test.py`: Script for testing the fine-tuned model
- `finetune_upload.py`: Script for uploading the model to Hugging Face and/or converting to GGUF for Ollama

## Fine-Tuning

The `finetune.py` script fine-tunes a Llama 3.2 model on ATC communications data.

### Basic Usage

```bash
python finetune.py --input-file data/test_llama_70b_punv.json
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
  --warmup-ratio 0.05 \
  --save-to-gguf \
  --quant-method q4_k_m
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
- `--output-dir`: Output directory for the fine-tuned model (default: `./atc_llama`)
- `--gradient-accumulation-steps`: Gradient accumulation steps (default: 4)
- `--warmup-ratio`: Warmup ratio (default: 0.03)
- `--max-steps`: Maximum number of training steps (default: None)
- `--save-to-gguf`: Save the model to GGUF format for Ollama compatibility
- `--quant-method`: Quantization method for GGUF conversion (default: `q4_k_m`)
- `--full-finetune`: Perform full fine-tuning instead of LoRA (requires more VRAM)
- `--load-in-8bit`: Load the model in 8-bit precision (alternative to 4-bit)

## Testing the Model

Use the `finetune_test.py` script to test your fine-tuned model:

```bash
python finetune_test.py --model-path ./atc_llama
```

### Advanced Testing Options

```bash
python finetune_test.py \
  --model-path ./my_atc_model \
  --load-in-8bit \
  --max-new-tokens 1024 \
  --temperature 0.8 \
  --top-p 0.9 \
  --test-case 2
```

### Testing Parameters

- `--model-path`: Path to the fine-tuned model (default: `./atc_llama`)
- `--load-in-4bit`: Load model in 4-bit precision (default: True)
- `--load-in-8bit`: Load model in 8-bit precision (overrides 4-bit if specified)
- `--max-new-tokens`: Maximum number of tokens to generate (default: 512)
- `--temperature`: Temperature for generation (default: 0.7)
- `--top-p`: Top-p sampling parameter (default: 0.95)
- `--test-case`: Test case to use (0: default, 1-3: additional test cases) (default: 0)

## Uploading and Converting

The `finetune_upload.py` script can upload your model to Hugging Face and/or convert it to GGUF format for use with Ollama.

### Uploading to Hugging Face

```bash
python finetune_upload.py --model-path ./atc_llama --repo-id your-username/atc-llama
```

### Converting to GGUF and Creating an Ollama Model

```bash
python finetune_upload.py \
  --model-path ./atc_llama \
  --repo-id local \
  --convert-to-gguf \
  --quant-method q4_k_m \
  --create-modelfile \
  --ollama-model-name atc-llama \
  --create-ollama-model
```

### Upload Parameters

- `--model-path`: Path to the fine-tuned model (default: `./atc_llama`)
- `--repo-id`: Hugging Face repository ID (e.g., `username/model-name`) or `local` to skip HF upload
- `--token`: Hugging Face API token (or set `HF_TOKEN` environment variable)
- `--convert-to-gguf`: Convert the model to GGUF format before uploading
- `--quant-method`: Quantization method for GGUF conversion (default: `q4_k_m`)
- `--create-modelfile`: Create an Ollama-compatible Modelfile
- `--ollama-model-name`: Name for the Ollama model (default: `atc-llama`)
- `--create-ollama-model`: Create the Ollama model after preparing the files (requires Ollama to be installed)

## Using with Ollama

After creating the Ollama model, you can run it with:

```bash
ollama run atc-llama
```

Or if you uploaded to Hugging Face:

```bash
ollama run hf.co/your-username/atc-llama
```

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
- The GGUF conversion uses [llama.cpp](https://github.com/ggerganov/llama.cpp) tools
- Models are hosted on [Hugging Face](https://huggingface.co/) 