#!/usr/bin/env python3
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

import pandas as pd
import torch
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, Trainer
from unsloth import FastLanguageModel, UnslothTrainingArguments, is_bfloat16_supported

# uv v -p 3.10
# source .venv/bin/activate
# uv pip install pandas datasets unsloth setuptools torch transformers peft

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
args = parser.parse_args()


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


def main():
    """Main script execution."""

    # Prepare the data
    data_df = prepare_data(args.input_file)

    # Fine-tune the model
    model_dir = fine_tune_model(data_df)

    print(f"Fine-tuning completed successfully. Model/adapters saved to {model_dir}")


if __name__ == "__main__":
    main()
