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

The scripts include a `--download` option to download models for offline use:

```bash
# Download a model for fine-tuning
uv run finetune.py --download meta-llama/Llama-3.2-3B-Instruct

# Download a model for merging with adapters
uv run finetune_merge_adapters.py --download meta-llama/Llama-3.2-3B-Instruct

# Download a model for testing (can be used as main model or for comparison)
uv run finetune_test.py --download meta-llama/Llama-3.2-3B-Instruct
```

After downloading, models will be stored in a `models/` directory, and you can use them with the `--offline` flag.

```bash
# Use a downloaded model as your main model
uv run finetune_test.py --model-dir models/Llama-3.2-3B-Instruct --offline

# Use a downloaded model as your base model for comparison
uv run finetune_test.py --model-dir ./atc_llama_merged --base-model-dir models/Llama-3.2-3B-Instruct --offline --compare-with-base
```

## Fine-Tuning

The `finetune.py` script fine-tunes a Llama 3.2 model on ATC communications data.

### Basic Usage

```bash
uv run finetune.py --input-file data/test_llama_70b_punv.json
```

### Offline Mode Usage

```bash
# Use a downloaded model for fine-tuning
uv run finetune.py --input-file data/atc_data.json --model-name models/Llama-3.2-3B-Instruct --offline
```

### Advanced Options

```bash
uv run finetune.py \
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
- `--download`: Download a model from Hugging Face to use locally (can be used for both main and base models)
- `--output-dir`: Output directory for the fine-tuned model (default: `./atc_llama`)
- `--gradient-accumulation-steps`: Gradient accumulation steps (default: 4)
- `--warmup-ratio`: Warmup ratio (default: 0.03)
- `--max-steps`: Maximum number of training steps (default: None)
- `--full-finetune`: Perform full fine-tuning instead of LoRA (requires more VRAM)
- `--load-in-8bit`: Load the model in 8-bit precision (alternative to 4-bit)

## Merging LoRA Adapters with Base Model

Use the `finetune_merge_adapters.py` script to merge fine-tuned LoRA adapters with the base model:

```bash
uv run finetune_merge_adapters.py --adapter-dir ./atc_llama --output-dir ./atc_llama_merged
```

### Offline Mode Usage

```bash
uv run finetune_merge_adapters.py \
  --adapter-dir ./atc_llama \
  --base-model-name models/Llama-3.2-3B-Instruct \
  --output-dir ./atc_llama_merged \
  --offline
```

### Merge Parameters

- `--adapter-dir`: Input directory with fine-tuned LoRA adapters (default: `./atc_llama`)
- `--base-model-name`: Base model name used for fine-tuning (default: `meta-llama/Llama-3.2-3B-Instruct`)
- `--offline`: Use local model files (base-model-name points to a local directory)
- `--download`: Download a model from Hugging Face to use locally (can be used for both main and base models)
- `--output-dir`: Output directory for the merged model (default: `./atc_llama_merged`)

## Testing the Model

Use the `finetune_test.py` script to test your fine-tuned model:

```bash
uv run finetune_test.py --model-dir ./atc_llama_merged
```

### Compare with Base Model

```bash
# Compare fine-tuned model with the base model
uv run finetune_test.py --model-dir ./atc_llama_merged --compare-with-base
```

### Test Types and Prompting Styles

The testing script now includes different test categories and prompting styles to better evaluate fine-tuning results:

```bash
# Run with minimal prompting to truly test what was learned during fine-tuning
uv run finetune_test.py --model-dir ./atc_llama_merged --prompt-style minimal

# Test only complex ATC scenarios
uv run finetune_test.py --model-dir ./atc_llama_merged --test-type complex

# Test domain-specific knowledge
uv run finetune_test.py --model-dir ./atc_llama_merged --test-type domain

# Full comparison with base model using minimal prompts on domain knowledge
uv run finetune_test.py --model-dir ./atc_llama_merged --compare-with-base --prompt-style minimal --test-type domain
```

### Offline Mode Usage

```bash
uv run finetune_test.py --model-dir ./atc_llama_merged --offline

# With base model comparison in offline mode
uv run finetune_test.py --model-dir ./atc_llama_merged --base-model-dir models/Llama-3.2-3B-Instruct --offline --compare-with-base
```

# To download a model for offline comparison use:
uv run finetune_test.py --download meta-llama/Llama-3.2-3B-Instruct
# Then use it as the base model:
uv run finetune_test.py --model-dir ./atc_llama_merged --base-model-dir models/Llama-3.2-3B-Instruct --offline --compare-with-base

### Multi-GPU and Device Control

```bash
# Use a specific GPU
uv run finetune_test.py --model-dir ./atc_llama_merged --device cuda:1

# Use different GPUs for fine-tuned and base models
uv run finetune_test.py --model-dir ./atc_llama_merged --compare-with-base --device cuda:0 --base-device cuda:1

# Split large models across available GPUs
uv run finetune_test.py --model-dir ./atc_llama_merged --split-across-gpus
```

### Advanced Testing Options

```bash
uv run finetune_test.py \
  --model-dir ./my_atc_model \
  --base-model-dir models/Llama-3.2-3B-Instruct \
  --compare-with-base \
  --load-in-8bit \
  --prompt-style minimal \
  --test-type all \
  --offline
```

### Testing Parameters

- `--model-dir`: Directory containing the fine-tuned model or adapters (default: `./atc_llama_merged`)
- `--base-model-name`: Base model name (required if loading adapters, ignored for merged models)
- `--compare-with-base`: Also load the base model and compare outputs side by side
- `--base-model-dir`: Local directory containing the base model files (for offline comparison)
- `--prompt-style`: Style of prompts to use (`minimal` or `detailed`)
  - `minimal`: Uses very basic instruction to test what the model learned during fine-tuning
  - `detailed`: Uses comprehensive formatting instructions (default in previous versions)
- `--test-type`: Type of test cases to run (`basic`, `complex`, `domain`, or `all`)
  - `basic`: Simple ATC communications (the original test cases)
  - `complex`: Complex ATC scenarios with multiple instructions and emergencies
  - `domain`: Domain-specific knowledge cases requiring specialized ATC understanding
  - `all`: All test types combined (default)
- `--max-seq-length`: Maximum sequence length (default: 2048)
- `--load-in-4bit`: Load model in 4-bit precision (default behavior with Unsloth)
- `--load-in-8bit`: Load model in 8-bit precision (alternative to 4-bit)
- `--offline`: Use local model files (model-dir and base-model-name point to local directories)
- `--download`: Download a model from Hugging Face to use locally (can be used for both main and base models)
- `--device`: Device to run the model on (e.g., 'cuda:0', 'cuda:1', 'cpu')
- `--base-device`: Device to run the base model on (e.g., 'cuda:1')
- `--split-across-gpus`: Split model across multiple GPUs if it's too large for a single GPU
- `--test-inputs`: Custom test inputs to process (overrides --test-type)

## Uploading to Hugging Face Hub

Use the `finetune_upload_to_hf.py` script to upload your models or adapters to the Hugging Face Hub. Both the fine-tuning and merging scripts automatically generate comprehensive model cards (README.md files) that work with Hugging Face Hub.

### Uploading Adapters

```bash
uv run finetune_upload_to_hf.py --model-dir ./atc_llama --repo-id username/atc-llama-adapters
```

### Uploading Merged Model

```bash
uv run finetune_upload_to_hf.py --model-dir ./atc_llama_merged --repo-id username/atc-llama
```

### Advanced Upload Options

```bash
uv run finetune_upload_to_hf.py \
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

## Converting to GGUF for Ollama

Use the `convert_hf_to_ollama.py` script to convert your fine-tuned model to GGUF format for use with Ollama, llama.cpp, and other compatible applications.

### Prerequisites

1. Clone the llama.cpp repository:
   ```bash
   git clone https://github.com/ggerganov/llama.cpp.git
   cd llama.cpp && uv pip install -r requirements.txt && cd ..
   ```

### Basic Usage

```bash
uv run convert_hf_to_ollama.py \
  --model-dir ./atc_llama_merged \
  --llama-cpp-dir ./llama.cpp \
  --output-gguf ./atc_llama_q4_k_m.gguf
```

### Convert and Upload to Hugging Face

```bash
uv run convert_hf_to_ollama.py \
  --model-dir ./atc_llama_merged \
  --llama-cpp-dir ./llama.cpp \
  --output-gguf ./atc_llama_q4_k_m.gguf \
  --quantization-type Q4_K_M \
  --upload-repo username/atc-llama-gguf \
  --hf-token YOUR_HF_WRITE_TOKEN
```

### Conversion Parameters

- `--model-dir`: Path to the local directory containing the Hugging Face model files
- `--llama-cpp-dir`: Path to the cloned llama.cpp repository directory
- `--output-gguf`: Path where the output GGUF file will be saved
- `--quantization-type`: Quantization type to apply (e.g., F16, Q4_0, Q4_K_M, Q5_K_M, Q8_0)
- `--upload-repo`: Optional: Hugging Face Hub repository ID to upload the GGUF file to
- `--hf-token`: Optional: Hugging Face API token with write access for uploading
- `--private`: If uploading, create a private repository

## Creating Ollama Model from GGUF

Use the `create_ollama_hf_model.py` script to upload a GGUF file to Hugging Face with a specialized Ollama Modelfile, making it easy to use with Ollama.

### Basic Usage

```bash
uv run create_ollama_hf_model.py \
  --gguf-file ./atc_llama_q4_k_m.gguf \
  --upload-repo username/atc-llama-gguf \
  --hf-token YOUR_HF_WRITE_TOKEN
```

### With Custom Parameters

```bash
uv run create_ollama_hf_model.py \
  --gguf-file ./atc_llama_q4_k_m.gguf \
  --upload-repo username/atc-llama-gguf \
  --hf-token YOUR_HF_WRITE_TOKEN \
  --parameters "temperature: 0.7, top_p: 0.9, num_ctx: 128000"
```

### Using the Uploaded Model with Ollama

After uploading, users can run your model directly with:

```bash
ollama run hf://username/atc-llama-gguf/atc_llama_q4_k_m.gguf
```

### Ollama Upload Parameters

- `--gguf-file`: Path to the local GGUF file to upload
- `--upload-repo`: Hugging Face repository ID to upload the GGUF file to
- `--hf-token`: Hugging Face API token with write access for uploading
- `--parameters`: Optional parameters for the model (comma-separated list)
- `--private`: If uploading, create a private repository

## Complete ATC Model Workflow

Here's the complete workflow for creating an optimized ATC model:

1. **Fine-tune the model:**
   ```bash
   uv run finetune.py --input-file data/atc_data.json
   ```

2. **Merge adapters with base model:**
   ```bash
   uv run finetune_merge_adapters.py --adapter-dir ./atc_llama --output-dir ./atc_llama_merged
   ```

3. **Test the fine-tuned model:**
   ```bash
   uv run finetune_test.py --model-dir ./atc_llama_merged --compare-with-base --prompt-style minimal --test-type domain
   ```

4. **Convert to GGUF format:**
   ```bash
   uv run convert_hf_to_ollama.py --model-dir ./atc_llama_merged --llama-cpp-dir ./llama.cpp --output-gguf ./atc_llama_q4_k_m.gguf
   ```

5. **Upload to Hugging Face with Ollama support:**
   ```bash
   uv run create_ollama_hf_model.py --gguf-file ./atc_llama_q4_k_m.gguf --upload-repo username/atc-llama-gguf --hf-token YOUR_HF_WRITE_TOKEN
   ```

6. **Use with Ollama:**
   ```bash
   ollama run hf://username/atc-llama-gguf/atc_llama_q4_k_m.gguf
   ```

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
3. Upload to Hugging Face (`finetune_upload_to_hf.py`, `create_ollama_hf_model.py`)

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
- [Llama.cpp](https://github.com/ggerganov/llama.cpp) for GGUF conversion and quantization
- [Ollama](https://ollama.com) for local model inference
- Models are hosted on [Hugging Face](https://huggingface.co/)