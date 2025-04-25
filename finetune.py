# ruff: noqa: I001
# #!/usr/bin/env python3
"""
finetune.py - Prepares ATC data and fine-tunes Llama3.2-3b model with Unsloth.

This script:
1. Loads processed ATC JSON data
2. Formats it for fine-tuning
3. Fine-tunes llama3.2:3b-instruct with Unsloth (and optimizations)
4. Saves the fine-tuned model or adapters.
"""

import argparse
import json
import os
import sys

import pandas as pd

# Arguments
parser = argparse.ArgumentParser(
    description="Fine-tune models to create an ATC expert model"
)
parser.add_argument(
    "--input-file",
    default="data/test_llama_70b_punv.json",
    help="Input JSON file with processed ATC data (default: data/test_llama_70b_punv.json)",
)
parser.add_argument(
    "--epochs", type=int, default=3, help="Number of training epochs (default: 3)"
)
parser.add_argument(
    "--batch-size", type=int, default=4, help="Batch size for training (default: 4)"
)
parser.add_argument(
    "--learning-rate", type=float, default=2e-4, help="Learning rate (default: 2e-4)"
)
parser.add_argument("--rank", type=int, default=16, help="LoRA rank (default: 16)")
parser.add_argument("--alpha", type=int, default=16, help="LoRA alpha (default: 16)")
parser.add_argument(
    "--max-seq-length",
    type=int,
    default=2048,
    help="Maximum sequence length (default: 2048)",
)
parser.add_argument(
    "--model-name",
    default="meta-llama/Llama-3.2-3B-Instruct",
    help="Model name to fine-tune (default: meta-llama/Llama-3.2-3B-Instruct)",
)
parser.add_argument(
    "--offline",
    action="store_true",
    help="Use local model files (model-name points to a local directory)",
)
parser.add_argument(
    "--download",
    metavar="MODEL",
    help="Download a Hugging Face model to use locally (will exit after download)",
)
parser.add_argument(
    "--output-dir",
    default="./atc_llama",
    help="Output directory for the fine-tuned model/adapters (default: ./atc_llama)",
)
parser.add_argument(
    "--gradient-accumulation-steps",
    type=int,
    default=4,
    help="Gradient accumulation steps (default: 4)",
)
parser.add_argument(
    "--warmup-ratio", type=float, default=0.03, help="Warmup ratio (default: 0.03)"
)
parser.add_argument(
    "--max-steps",
    type=int,
    default=None,
    help="Maximum number of training steps (default: None)",
)
parser.add_argument(
    "--full-finetune",
    action="store_true",
    help="Perform full fine-tuning instead of LoRA (requires more VRAM)",
)
parser.add_argument(
    "--load-in-8bit",
    action="store_true",
    help="Load the model in 8-bit precision (alternative to 4-bit)",
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
        print(f"To use this model, run with: --model-name {download_dir} --offline")
        sys.exit(0)
    except Exception as e:
        print(f"Error downloading model: {e}")
        sys.exit(1)

# Import Unsloth and other libraries after help check
import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from unsloth import (  # noqa: E402
    FastLanguageModel,
    UnslothTrainingArguments,
    is_bfloat16_supported,
)
from transformers import DataCollatorForSeq2Seq, Trainer  # noqa: E402


# uv v -p 3.10
# source .venv/bin/activate
# uv pip install pandas datasets unsloth setuptools torch transformers peft


def prepare_data(input_file):
    """Convert JSON data to a Pandas DataFrame formatted for fine-tuning."""
    print(f"Loading data from {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Create training examples
    training_data = []

    for item in data:
        # Extract message data
        original = item.get("original", "")
        punctuated = item.get("punctuated", "")
        pune = item.get("pune", "")
        intentions = item.get("intentions", {})

        # Additional fields
        punv = item.get("punv", "")
        edit = item.get("edit", "")
        speaker = item.get("speaker", "")
        speaker_confidence = item.get("speaker_confidence", "")
        listener = item.get("listener", "")
        event = item.get("event", "")
        actions = item.get("actions", "")

        # Convert intentions to text description
        intention_text = ""
        for code, is_active in intentions.items():
            if is_active:
                if code == "PSC":
                    intention_text += "- Pilot starts contact to ATC\\n"
                elif code == "PSR":
                    intention_text += (
                        "- Pilot starts contact to ATC with reported info\\n"
                    )
                elif code == "PRP":
                    intention_text += "- Pilot reports information\\n"
                elif code == "PRQ":
                    intention_text += "- Pilot issues requests\\n"
                elif code == "PRB":
                    intention_text += "- Pilot readback\\n"
                elif code == "PAC":
                    intention_text += "- Pilot acknowledge\\n"
                elif code == "ASC":
                    intention_text += "- ATC starts contact to pilot\\n"
                elif code == "AGI":
                    intention_text += "- ATC gives instruction\\n"
                elif code == "ACR":
                    intention_text += "- ATC corrects pilot's readback\\n"
                elif code == "END":
                    intention_text += "- Either party signaling the end of exchange\\n"

        # Extract numbers
        numbers = item.get("numbers", {})
        numbers_text = ""
        for field, value in numbers.items():
            if value:  # Only include non-empty values
                numbers_text += f"- {field}: {value}\n"

        # Create instruction prompt
        instruction = "As an ATC communication expert, improve this transcript and analyze its intentions and data."

        # Create input context
        input_text = f"Original: {original}\nPunctuated: {punctuated}"

        # Create expected output with all fields
        output_text = f"Improved ATC format: {pune}\n\n"

        if punv:
            output_text += f"Punctuated version: {punv}\n\n"

        output_text += f"Intentions:\n{intention_text}\n"
        output_text += f"Extracted data:\n{numbers_text}\n"

        if speaker:
            output_text += f"Speaker: {speaker}\n"
        if speaker_confidence:
            output_text += f"Speaker confidence: {speaker_confidence}\n"
        if listener:
            output_text += f"Listener: {listener}\n"
        if event:
            output_text += f"Event: {event}\n"
        if actions:
            output_text += f"Actions: {actions}\n"
        if edit:
            output_text += f"Edit suggestions: {edit}\n"

        # Add to training data
        training_data.append(
            {"instruction": instruction, "input": input_text, "output": output_text}
        )

    print(f"Prepared {len(training_data)} training examples")

    # Convert to DataFrame
    df = pd.DataFrame(training_data)
    return df


def fine_tune_model(data_df):
    """Fine-tune the model using Unsloth."""

    print("Setting up fine-tuning...")
    model_id = args.model_name  # Use the specified model name

    # Convert pandas DataFrame to Hugging Face Dataset
    dataset = Dataset.from_pandas(data_df)

    print(f"Loading model {model_id}...")

    # Determine whether to use 4-bit or 8-bit quantization
    load_in_4bit = not args.load_in_8bit
    load_in_8bit = args.load_in_8bit

    # Load the base model - don't pass full_finetune here
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=args.max_seq_length,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        local_files_only=args.offline,  # Enable offline mode
    )

    # If we're not doing full fine-tuning, set up LoRA
    if not args.full_finetune:
        # Set up for LoRA fine-tuning directly with FastLanguageModel
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.rank,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=args.alpha,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",  # Optimize memory usage
            random_state=3407,
        )
    else:
        # For full fine-tuning, we don't add LoRA adapters
        # Instead, we ensure gradient checkpointing is still enabled for memory efficiency
        model.enable_input_require_grads()
        model.config.use_cache = False  # Disable KV cache for training
        model.gradient_checkpointing_enable()  # Enable gradient checkpointing
        print("Full fine-tuning enabled - all model parameters will be updated")

    # Prepare the data for training
    def formatting_func(example):
        # Format the data for instruction fine-tuning
        instruction = example["instruction"]
        inp = example["input"]
        output = example["output"]
        formatted_input = f"<|begin_of_text|><|header_start|>user<|header_end|>\n\n{instruction}\n\n{inp}<|eot|><|header_start|>assistant<|header_end|>\n\n{output}<|eot|>"
        return formatted_input

    # Tokenize and prepare the dataset
    def prepare_dataset(example):
        formatted_text = formatting_func(example)
        tokenized = tokenizer(
            formatted_text, truncation=True, max_length=args.max_seq_length
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    # Prepare the dataset
    tokenized_dataset = dataset.map(
        prepare_dataset, remove_columns=dataset.column_names
    )

    # Split into training and validation if there are enough examples
    if len(tokenized_dataset) > 1:
        split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
    else:
        print(
            "Warning: Not enough examples to split into train/test. Using single example for both."
        )
        train_dataset = tokenized_dataset
        eval_dataset = tokenized_dataset

    # Check for bfloat16 support
    bf16_supported = is_bfloat16_supported()

    # Set up training arguments
    training_args_dict = {
        "output_dir": "./results",
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "logging_steps": 10,
        "save_steps": 200,
        "save_total_limit": 3,
        "bf16": bf16_supported,  # Use bfloat16 if supported
        "fp16": not bf16_supported,  # Otherwise use fp16
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": "cosine",
        "optim": "adamw_8bit",  # Use 8-bit optimizer for better memory efficiency
        "group_by_length": True,
        "report_to": "none",
    }

    # Only add max_steps if it's specified and not None
    if args.max_steps is not None:
        training_args_dict["max_steps"] = args.max_steps

    training_args = UnslothTrainingArguments(**training_args_dict)

    # Set up data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train the model
    print("Starting fine-tuning...")
    trainer.train()

    # Save the model - using a custom save function to avoid model card errors
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Custom save function to avoid model card errors
    print(f"Saving model to {output_dir}...")
    try:
        # For LoRA models, save adapter config and weights
        if hasattr(model, "peft_config"):
            # Save adapter config
            model.peft_config.save_pretrained(output_dir)

            # Save adapter weights
            for adapter_name in model.active_adapters:
                adapter_state_dict = {}
                for key, value in model.state_dict().items():
                    if adapter_name in key:
                        adapter_state_dict[key] = value

                # Save adapter model
                torch.save(
                    adapter_state_dict, os.path.join(output_dir, "adapter_model.bin")
                )

            # Save config files needed
            if hasattr(model, "config"):
                model.config.save_pretrained(output_dir)

            print(f"LoRA adapters saved successfully to {output_dir}")
        else:
            # For full fine-tuning, save the entire model
            model.save_pretrained(output_dir, safe_serialization=True)
            print(f"Full model saved successfully to {output_dir}")
    except Exception as e:
        print(f"Error during custom save: {str(e)}")
        print("Falling back to standard save method...")
        try:
            # Try to use the standard save_pretrained with parameters to avoid errors
            if hasattr(model, "save_pretrained"):
                model.save_pretrained(output_dir, safe_serialization=True)
            else:
                trainer.save_model(output_dir)
        except Exception as e2:
            print(f"Standard save also failed: {str(e2)}")
            print("Attempting final fallback method...")
            try:
                # Final fallback for LoRA models
                from peft import get_peft_model_state_dict

                peft_state_dict = get_peft_model_state_dict(model)
                torch.save(
                    peft_state_dict, os.path.join(output_dir, "adapter_model.bin")
                )
                if hasattr(model, "config"):
                    model.config.save_pretrained(output_dir)
                print("Saved model using fallback method")
            except Exception as e3:
                print(f"All save methods failed: {str(e3)}")
                print("Model training was successful but saving failed.")

    # Save the tokenizer separately
    tokenizer.save_pretrained(output_dir)
    print(f"Tokenizer saved to {output_dir}")

    return output_dir


def create_readme_template(output_dir, model_name, is_adapter=True, training_args=None):
    """Create a detailed README.md file for the fine-tuned model/adapter with YAML metadata."""
    model_id = model_name
    # If model_name points to a local directory, use default model ID
    if (
        os.path.exists(model_name)
        or model_name.startswith("./")
        or model_name.startswith("models/")
    ):
        model_id = "meta-llama/Llama-3.2-3B-Instruct"

    # Get model type from base name
    model_type = "llama"
    if "llama" in model_id.lower():
        model_type = "llama"
    elif "mistral" in model_id.lower():
        model_type = "mistral"
    elif "gemma" in model_id.lower():
        model_type = "gemma"
    elif "qwen" in model_id.lower():
        model_type = "qwen"

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

    # Get hyperparameters for training section
    rank = getattr(args, "rank", 16)
    alpha = getattr(args, "alpha", 16)
    learning_rate = getattr(args, "learning_rate", 2e-4)
    batch_size = getattr(args, "batch_size", 4)
    gradient_accumulation_steps = getattr(args, "gradient_accumulation_steps", 4)
    epochs = getattr(args, "epochs", 3)
    warmup_ratio = getattr(args, "warmup_ratio", 0.03)
    max_seq_length = getattr(args, "max_seq_length", 2048)

    if is_adapter:
        # Create the YAML metadata and README content for the adapter
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
- PEFT
- LoRA
- Unsloth
---

# ATC Communication Expert Model (LoRA Adapters)

A specialized set of LoRA adapters fine-tuned for improving and analyzing Air Traffic Control (ATC) communications, extracting relevant information from raw transcripts.

## Model Details

### Model Description

These adapters fine-tune the {model_id} model to specialize in processing Air Traffic Control communications. When applied to the base model, it can:

- Improve raw ATC transcripts with proper punctuation and formatting
- Identify communication intentions (pilot requests, ATC instructions, etc.)
- Extract key information such as flight numbers, altitudes, headings, and other numerical data
- Analyze speaker roles and communication patterns

The adapters were created using LoRA (Low-Rank Adaptation) with PEFT (Parameter-Efficient Fine-Tuning) techniques to efficiently adapt the {model_type.title()} {model_size} model to this specialized domain.

- **Developed by:** ATC NLP Team
- **Model type:** LoRA adapters for {model_id}
- **Language(s):** English, specialized for ATC terminology
- **License:** Same as the base model
- **Finetuned from model:** {model_id}

## Uses

### Direct Use

These adapters are intended for:
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
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    "{model_id}",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("{model_id}")

# Load the adapter
model = PeftModel.from_pretrained(base_model, "{os.path.basename(output_dir)}")

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

## Training Details

### Training Data

The model was trained on a dataset of ATC communications with:
- Original raw transcripts
- Properly punctuated and formatted versions
- Annotated intentions (PSC, PSR, PRP, PRQ, PRB, PAC, ASC, AGI, ACR, END)
- Extracted numerical data (altitudes, headings, flight numbers, etc.)
- Speaker and listener information

### Training Procedure

The model was fine-tuned using LoRA with the following approach:
- Parameter-efficient fine-tuning using PEFT
- LoRA applied to key attention layers (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
- Optimized with Unsloth for efficiency

#### Training Hyperparameters

- **Base model:** {model_id}
- **LoRA rank:** {rank}
- **LoRA alpha:** {alpha}
- **Learning rate:** {learning_rate}
- **Batch size:** {batch_size}
- **Gradient accumulation steps:** {gradient_accumulation_steps}
- **Epochs:** {epochs}
- **Warmup ratio:** {warmup_ratio}
- **Max sequence length:** {max_seq_length}
- **Training regime:** BF16 mixed precision where available, FP16 otherwise
- **Optimizer:** AdamW 8-bit

## Evaluation

### Testing

The adapters should be tested on diverse ATC communications, including:
- Clearances and instructions
- Pilot requests and reports
- Emergency communications
- Different accents and speaking patterns

## Technical Specifications

### Model Architecture and Objective

- **Base architecture:** {model_id}
- **Fine-tuning method:** LoRA with PEFT
- **Optimization library:** Unsloth
- **Training objective:** Improving and analyzing ATC communications

### Compute Infrastructure

- **Framework versions:**
  - PEFT (compatible with the base model)
  - Unsloth (for efficient LoRA fine-tuning)
  - Transformers (compatible with the base model)
  - PyTorch (with BF16 support where available)
"""
    else:
        # For full fine-tuned models
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
---

# ATC Communication Expert Model (Full Fine-tuned)

A fully fine-tuned model specialized in improving and analyzing Air Traffic Control (ATC) communications, extracting relevant information from raw transcripts.

## Model Details

### Model Description

This model is a fully fine-tuned version of {model_id} optimized for processing Air Traffic Control communications. It can:

- Improve raw ATC transcripts with proper punctuation and formatting
- Identify communication intentions (pilot requests, ATC instructions, etc.)
- Extract key information such as flight numbers, altitudes, headings, and other numerical data
- Analyze speaker roles and communication patterns

The model underwent complete fine-tuning to adapt the {model_type.title()} {model_size} model to this specialized domain.

- **Developed by:** ATC NLP Team
- **Model type:** Fully fine-tuned {model_type.title()} {model_size}
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

## Training Details

### Training Data

The model was trained on a dataset of ATC communications with:
- Original raw transcripts
- Properly punctuated and formatted versions
- Annotated intentions (PSC, PSR, PRP, PRQ, PRB, PAC, ASC, AGI, ACR, END)
- Extracted numerical data (altitudes, headings, flight numbers, etc.)
- Speaker and listener information

### Training Procedure

The model was fully fine-tuned with the following approach:
- Complete parameter fine-tuning
- Optimized with Unsloth for efficiency
- Gradient checkpointing for memory efficiency

#### Training Hyperparameters

- **Base model:** {model_id}
- **Learning rate:** {learning_rate}
- **Batch size:** {batch_size}
- **Gradient accumulation steps:** {gradient_accumulation_steps}
- **Epochs:** {epochs}
- **Warmup ratio:** {warmup_ratio}
- **Max sequence length:** {max_seq_length}
- **Training regime:** BF16 mixed precision where available, FP16 otherwise
- **Optimizer:** AdamW 8-bit

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
- **Fine-tuning method:** Full parameter fine-tuning
- **Optimization library:** Unsloth
- **Training objective:** Improving and analyzing ATC communications

### Compute Infrastructure

- **Framework versions:**
  - Unsloth (for efficient fine-tuning)
  - Transformers (compatible with the base model)
  - PyTorch (with BF16 support where available)
"""

    # Write the README.md file
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(readme_content)

    print(f"Created detailed README.md in {output_dir}")
    return readme_path


def main():
    """Main script execution."""

    # Prepare the data
    data_df = prepare_data(args.input_file)

    # Fine-tune the model
    model_dir = fine_tune_model(data_df)

    # Create README.md for the fine-tuned model/adapters
    create_readme_template(model_dir, args.model_name, not args.full_finetune)

    print(f"Fine-tuning completed successfully. Model/adapters saved to {model_dir}")


if __name__ == "__main__":
    main()
