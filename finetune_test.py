#!/usr/bin/env python3
"""
finetune_test.py - Test the fine-tuned ATC model with sample inputs

This script:
1. Loads the fine-tuned model (either adapters or merged)
2. Also loads the original base model for comparison
3. Runs inference on test ATC messages using both models
4. Outputs the transcripts from both models side by side for comparison
"""

import argparse
import os
import sys

# Parse arguments
parser = argparse.ArgumentParser(description="Test the fine-tuned ATC model")
parser.add_argument(
    "--model-dir",
    default="./atc_llama_merged",  # Default to merged model dir
    help="Directory containing the fine-tuned model (merged) or adapters (default: ./atc_llama_merged)",
)
parser.add_argument(
    "--base-model-name",
    default="meta-llama/Llama-3.2-3B-Instruct",
    help="Base model name (required if loading adapters from --model-dir, ignored otherwise)",
)
parser.add_argument(
    "--compare-with-base",
    action="store_true",
    help="Also load the base model and compare outputs side by side",
)
parser.add_argument(
    "--base-model-dir",
    help="Local directory containing the base model files (for offline comparison, overrides base-model-name)",
)
parser.add_argument(
    "--max-seq-length",
    type=int,
    default=2048,
    help="Maximum sequence length used during fine-tuning (default: 2048)",
)
parser.add_argument(
    "--load-in-4bit",
    action="store_true",
    help="Load the model in 4-bit precision (requires bitsandbytes).",
)
parser.add_argument(
    "--load-in-8bit",
    action="store_true",
    help="Load the model in 8-bit precision (requires bitsandbytes, alternative to 4-bit).",
)
parser.add_argument(
    "--offline",
    action="store_true",
    help="Use local model files (model-dir and base-model-name point to local directories)",
)
parser.add_argument(
    "--download",
    metavar="MODEL",
    help="Download a model from Hugging Face to use locally (will exit after download). Use this for both main models and base models.",
)
parser.add_argument(
    "--device",
    default="cuda:0",
    help="Device to run the model on (e.g., 'cuda:0', 'cuda:1', 'cpu'). Default: 'cuda:0'",
)
parser.add_argument(
    "--base-device",
    default=None,
    help="Device to run the base model on (e.g., 'cuda:1'). Defaults to same as --device.",
)
parser.add_argument(
    "--split-across-gpus",
    action="store_true",
    help="Split model across multiple GPUs if it's too large for a single GPU. Overrides --device.",
)
parser.add_argument(
    "--test-type",
    default="all",
    choices=["basic", "complex", "domain", "all"],
    help="Type of test cases to run: basic, complex, domain-specific, or all",
)
parser.add_argument(
    "--prompt-style",
    default="minimal",
    choices=["minimal", "detailed"],
    help="Style of prompts to use: minimal or detailed instructions",
)
parser.add_argument(
    "--test-inputs",
    nargs="+",
    default=None,
    help="Custom test inputs to process (overrides test-type)",
)

# Check if help flag is requested before importing anything
if "-h" in sys.argv or "--help" in sys.argv:
    args = parser.parse_args()
    sys.exit(0)

args = parser.parse_args()

# Define test sets
BASIC_TEST_INPUTS = [
    "southwest five niner two turn left heading three four zero descend and maintain flight level two five zero",
    "united four seven three request clearance to descend due to turbulence",
    "delta six three eight traffic alert cessna two thousand feet twelve oclock expedite climb flight level three five zero",
]

COMPLEX_TEST_INPUTS = [
    "delta one two three four at flight level three five zero request deviation fifteen degrees right for weather and when able direct jackson expect lower in twenty miles",
    "american one zero niner eight heavy loss of cabin pressure emergency descent to one zero thousand feet requesting immediate vectors to nearest suitable airport",
    "air france four five six traffic is a helicopter at your two o'clock position one mile altitude unknown caution wake turbulence previous aircraft was a seven four seven heavy contact departure on one two three point eight five good day",
    "n five four three two x holding short runway two four left wind two three zero at one five gusting two five await takeoff clearance with two souls on board vfr to chicago",
    "speedbird two seven nine squawk seven seven zero zero we have an engine fire commencing emergency descent and turning back to airport please clear airspace below",
]

DOMAIN_KNOWLEDGE_TEST_INPUTS = [
    "lufthansa four niner seven request pan pan medical emergency passenger with suspected heart attack request priority handling and medical assistance on arrival",
    "n one three five seven request frequency change to one two one point five for atis information sierra followed by ils approach runway two seven right",
    "air canada three three zero fuel emergency request direct routing to field and priority handling estimating nine thousand pounds remaining at landing below minimum reserve",
    "southwest six five four request wind check and runway conditions braking action poor reported by previous aircraft",
    "united eight niner seven leaving flight level one niner zero for flight level two three zero requesting ride reports this altitude moderate turbulence",
]


# Handle model download if requested
def download_model(model_name, is_base=False):
    """Download a model from Hugging Face."""
    print(f"Downloading model: {model_name}")
    # Import required libraries only when needed
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Create download directory based on model name
    model_dir_name = model_name.split("/")[-1]
    # Create a unified structure - all models go to the same models directory
    download_dir = os.path.join("models", model_dir_name)

    os.makedirs(download_dir, exist_ok=True)

    print(f"Downloading to {download_dir}...")
    try:
        # Download and save model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        tokenizer.save_pretrained(download_dir)
        model.save_pretrained(download_dir)

        print(f"Model downloaded successfully to {download_dir}")
        return download_dir
    except Exception as e:
        print(f"Error downloading model: {e}")
        if "NewConnectionError" in str(e) or "Max retries exceeded" in str(e):
            print("Network error: Cannot connect to Hugging Face. Are you offline?")
            print(
                "If you're offline, use the --offline flag and provide local model paths."
            )
        return None


if args.download:
    download_dir = download_model(args.download)
    if download_dir:
        print(f"To use this model, run with: --model-dir {download_dir} --offline")
        print(f"Or as a base model with: --base-model-dir {download_dir} --offline")
    sys.exit(0)

# Import remaining libraries after help check and download
import torch  # noqa: E402

# Try importing unsloth, fallback to transformers if unavailable
HAS_UNSLOTH = False
try:
    from unsloth import FastLanguageModel  # noqa: E402

    HAS_UNSLOTH = True
    print("Using Unsloth for model loading.")
except ImportError:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

    print("Unsloth not found, using Hugging Face Transformers.")

# Check for PEFT
HAS_PEFT = False
try:
    from peft import PeftModel  # noqa: E402

    HAS_PEFT = True
except ImportError:
    print(
        "Warning: PEFT not installed. Cannot load LoRA adapters. Install with: uv pip install peft"
    )


def load_models_and_tokenizers():
    """Load both the fine-tuned model and base model for comparison."""
    model_dir = args.model_dir
    base_model_name = args.base_model_name
    base_model_dir = args.base_model_dir
    max_seq_length = args.max_seq_length
    load_in_4bit = args.load_in_4bit
    load_in_8bit = args.load_in_8bit
    offline_mode = args.offline
    compare_with_base = args.compare_with_base
    base_device = args.base_device

    # If base_model_dir is provided, use it instead of base_model_name
    if base_model_dir:
        base_model_name = base_model_dir
        offline_mode = True  # Enforce offline mode when using base_model_dir
        print(f"Using local base model directory: {base_model_dir}")

    # Set device map based on arguments
    if args.split_across_gpus:
        device_map = "auto"
        print("Using auto device mapping (splitting across available GPUs).")
    elif args.device == "cpu":
        device_map = "cpu"
        print("Using CPU for inference.")
    else:
        # Use the specific device
        device_map = {"": args.device}
        print(f"Using specific device: {args.device}")

    # Default the base model device to same as main device if not specified
    if base_device is None:
        base_device = args.device
        # If auto-assigning to a different GPU, inform the user
        if base_device != "cpu" and compare_with_base and torch.cuda.device_count() > 1:
            print(
                "For best results when comparing models, consider using --base-device cuda:1"
            )

    # When running on different devices, inform the user about device assignment
    if base_device != args.device and compare_with_base:
        print(
            f"Base model will run on {base_device}, fine-tuned model on {args.device}"
        )

    # Set up base model device map
    if base_device == "cpu":
        base_device_map = "cpu"
    else:
        base_device_map = {"": base_device}

    # Check if model_dir contains adapter configuration
    adapter_config_path = os.path.join(model_dir, "adapter_config.json")
    is_adapter = os.path.exists(adapter_config_path)

    if is_adapter and not HAS_PEFT:
        print("Error: Found adapter config but PEFT is not installed. Cannot proceed.")
        sys.exit(1)

    # Determine quantization
    if load_in_4bit and load_in_8bit:
        print(
            "Warning: Both --load-in-4bit and --load-in-8bit specified. Defaulting to 4-bit."
        )
        load_in_8bit = False
    elif not load_in_4bit and not load_in_8bit:
        # Default to 4-bit if unsloth is available and neither is specified
        if HAS_UNSLOTH:
            print("Defaulting to 4-bit quantization with Unsloth.")
            load_in_4bit = True
        else:
            print("No quantization specified, loading in full precision.")

    # --- Loading Fine-tuned Model ---
    try:
        if is_adapter:
            print(
                f"Loading base model '{base_model_name}' to apply adapters from '{model_dir}'"
            )
            if HAS_UNSLOTH:
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=base_model_name,
                    max_seq_length=max_seq_length,
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit,
                    device_map=device_map,  # Use specific device or auto
                    local_files_only=offline_mode,  # Enable offline mode
                )
                print("Loading adapters using PEFT...")
                model = FastLanguageModel.get_peft_model(
                    model, model_dir
                )  # Use Unsloth's PEFT loader
            else:
                # Use transformers and PEFT
                print("Loading base model with transformers...")
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit,
                    torch_dtype=torch.bfloat16,  # Or torch.float16 depending on GPU
                    device_map=device_map,
                    local_files_only=offline_mode,  # Enable offline mode
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    base_model_name,
                    local_files_only=offline_mode,  # Enable offline mode
                )
                print("Loading adapters using PEFT...")
                model = PeftModel.from_pretrained(model, model_dir)
                # Merging is optional for inference, but can sometimes speed it up
                # print("Merging adapters...")
                # model = model.merge_and_unload()
        else:
            print(f"Loading merged model from '{model_dir}'")
            if HAS_UNSLOTH:
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=model_dir,  # Load directly from the directory
                    max_seq_length=max_seq_length,
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit,
                    device_map=device_map,
                    local_files_only=offline_mode,  # Enable offline mode
                )
            else:
                print("Loading merged model with transformers...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit,
                    torch_dtype=torch.bfloat16,  # Or torch.float16
                    device_map=device_map,
                    local_files_only=offline_mode,  # Enable offline mode
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    model_dir,
                    local_files_only=offline_mode,  # Enable offline mode
                )

        print("Fine-tuned model and tokenizer loaded successfully.")

        # --- Loading Base Model (for comparison) ---
        base_model = None
        base_tokenizer = None

        if compare_with_base:
            print(f"Loading base model for comparison: {base_model_name}")

            try:
                if HAS_UNSLOTH:
                    # Check if directory exists when in offline mode
                    if offline_mode and not os.path.isdir(base_model_name):
                        print(
                            f"Error: Local base model directory not found: {base_model_name}"
                        )
                        print(
                            "Please specify a valid local directory with --base-model-dir or download with --download"
                        )
                        return model, tokenizer, None, None

                    base_model, base_tokenizer = FastLanguageModel.from_pretrained(
                        model_name=base_model_name,
                        max_seq_length=max_seq_length,
                        load_in_4bit=load_in_4bit,
                        load_in_8bit=load_in_8bit,
                        device_map=base_device_map,
                        local_files_only=offline_mode,
                    )
                else:
                    # Check if directory exists when in offline mode
                    if offline_mode and not os.path.isdir(base_model_name):
                        print(
                            f"Error: Local base model directory not found: {base_model_name}"
                        )
                        print(
                            "Please specify a valid local directory with --base-model-dir or download with --download"
                        )
                        return model, tokenizer, None, None

                    base_model = AutoModelForCausalLM.from_pretrained(
                        base_model_name,
                        load_in_4bit=load_in_4bit,
                        load_in_8bit=load_in_8bit,
                        torch_dtype=torch.bfloat16,
                        device_map=base_device_map,
                        local_files_only=offline_mode,
                    )
                    base_tokenizer = AutoTokenizer.from_pretrained(
                        base_model_name,
                        local_files_only=offline_mode,
                    )
                print("Base model and tokenizer loaded successfully.")
            except Exception as e:
                print(f"Error loading base model: {e}")
                if "NewConnectionError" in str(e) or "Max retries exceeded" in str(e):
                    print(
                        "\nNetwork error: Cannot connect to Hugging Face. Are you offline?"
                    )
                    print("If you're offline, you need to:")
                    print("1. First download the base model using: --download")
                    print(
                        "2. Then run with: --base-model-dir models/<model-name> --offline --compare-with-base"
                    )

                # Continue with just the fine-tuned model
                base_model, base_tokenizer = None, None

        return model, tokenizer, base_model, base_tokenizer

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        if "bitsandbytes" in str(e).lower():
            print(
                "Hint: You might need to install bitsandbytes: uv pip install bitsandbytes"
            )
        if "peft" in str(e).lower():
            print("Hint: You might need to install peft: uv pip install peft")
        if "NewConnectionError" in str(e) or "Max retries exceeded" in str(e):
            print("\nNetwork error: Cannot connect to Hugging Face. Are you offline?")
            print(
                "If you're offline, use the --offline flag and provide local model paths."
            )
        sys.exit(1)


def get_prompt_for_style(input_text, style="minimal"):
    """Generate the prompt based on the selected style."""
    if style == "minimal":
        # Minimal prompt with very basic instruction
        instruction = "Process this ATC message."
        prompt = f"<|begin_of_text|><|header_start|>user<|header_end|>\n\n{instruction}\n\n{input_text}<|eot|><|header_start|>assistant<|header_end|>\n\n"
    elif style == "basic":
        # Simple instruction without formatting details
        instruction = "Improve this ATC transcript and analyze it."
        prompt = f"<|begin_of_text|><|header_start|>user<|header_end|>\n\n{instruction}\n\n{input_text}<|eot|><|header_start|>assistant<|header_end|>\n\n"
    else:  # detailed
        # Detailed instruction with specific formatting guidelines
        instruction = """As an ATC communication expert, improve this transcript and analyze its intentions and data.
        
Follow these specific guidelines for ATC formatting:
1. Convert spelled-out numbers to digits for flight callsigns (e.g., "delta six three eight" → "Delta 638")
2. Format altitude and flight levels properly (e.g., "flight level two five zero" → "FL250" or "flight level 250")
3. Convert headings to proper format (e.g., "heading three four zero" → "heading 340")
4. Handle all numeric values according to standard ATC conventions
5. Provide clear, professional formatting while maintaining the original meaning

Analyze the communication intentions and extract any relevant numerical data like altitudes, headings, and speeds."""
        prompt = f"<|begin_of_text|><|header_start|>user<|header_end|>\n\n{instruction}\n\nOriginal: {input_text}<|eot|><|header_start|>assistant<|header_end|>\n\n"

    return prompt


def run_test(model, tokenizer, base_model, base_tokenizer, input_text, prompt_style):
    """Run inference with both the fine-tuned model and base model for comparison."""
    print("\n" + "=" * 80)
    print(f"TESTING INPUT: {input_text}")
    print("=" * 80)

    # Get prompt based on selected style
    prompt = get_prompt_for_style(input_text, prompt_style)

    # Get the devices for each model
    if model is not None:
        fine_tuned_device = next(model.parameters()).device
        print(f"Fine-tuned model is on device: {fine_tuned_device}")

    if base_model is not None:
        base_device = next(base_model.parameters()).device
        print(f"Base model is on device: {base_device}")
    else:
        print("Base model comparison requested but base model could not be loaded")

    # Set generation parameters (optional, adjust as needed)
    generation_params = {
        "max_new_tokens": 512,
        "do_sample": False,  # Use greedy decoding for consistent output
        # "temperature": 0.7,
        # "top_p": 0.9,
    }

    # Generate with fine-tuned model
    print("Generating response from fine-tuned model...")
    finetuned_output = ""
    try:
        with torch.no_grad():  # Ensure no gradients are calculated
            # Tokenize and move inputs to the right device for fine-tuned model
            finetuned_inputs = tokenizer(prompt, return_tensors="pt")
            finetuned_inputs = {
                k: v.to(fine_tuned_device) for k, v in finetuned_inputs.items()
            }

            # Add pad_token_id for generation
            if tokenizer.pad_token_id is None:
                generation_params["pad_token_id"] = tokenizer.eos_token_id

            finetuned_outputs = model.generate(**finetuned_inputs, **generation_params)

            # Decode only the newly generated tokens, excluding the prompt
            finetuned_output = tokenizer.decode(
                finetuned_outputs[0, finetuned_inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
    except Exception as e:
        print(f"Error during fine-tuned model generation: {e}")
        finetuned_output = f"Error: {str(e)}"

    # Generate with base model if available
    base_output = ""
    if base_model is not None and base_tokenizer is not None:
        print("Generating response from base model...")
        try:
            with torch.no_grad():
                # Important: Create new inputs with the base tokenizer
                base_inputs = base_tokenizer(prompt, return_tensors="pt")
                base_inputs = {k: v.to(base_device) for k, v in base_inputs.items()}

                # Add pad_token_id for generation specific to base model
                base_generation_params = generation_params.copy()
                if base_tokenizer.pad_token_id is None:
                    base_generation_params["pad_token_id"] = base_tokenizer.eos_token_id

                base_outputs = base_model.generate(
                    **base_inputs, **base_generation_params
                )

                # Decode only the newly generated tokens, excluding the prompt
                base_output = base_tokenizer.decode(
                    base_outputs[0, base_inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                )
        except Exception as e:
            print(f"Error during base model generation: {e}")
            base_output = f"Error: {str(e)}"

            if "Expected all tensors to be on the same device" in str(e):
                print("\nHint: The error is due to tensors on different devices.")
                print(
                    "Try using the --base-device argument to explicitly set both models on the same device:"
                )
                print(
                    f"  uv run finetune_test.py --base-model-dir {args.base_model_dir} --offline --compare-with-base --device cuda:0 --base-device cuda:0"
                )

    # Print output comparison
    print("\n" + "=" * 40 + " COMPARISON " + "=" * 40)

    if base_model is not None and base_output and not base_output.startswith("Error:"):
        print("\n" + "-" * 40 + " FINE-TUNED MODEL OUTPUT " + "-" * 40)
        print(finetuned_output.strip())
        print("\n" + "-" * 40 + " BASE MODEL OUTPUT " + "-" * 40)
        print(base_output.strip())
    else:
        # If base model failed to load or generate, only show fine-tuned model
        print("\n" + "-" * 40 + " FINE-TUNED MODEL OUTPUT " + "-" * 40)
        print(finetuned_output.strip())
        if args.compare_with_base:
            print("\n" + "-" * 40 + " BASE MODEL OUTPUT " + "-" * 40)
            print("Base model comparison requested but output not available.")
            print("If you need to run in offline mode, first download the base model:")
            print(f"  uv run finetune_test.py --download {args.base_model_name}")
            print("Then run with:")
            print(
                f"  uv run finetune_test.py --model-dir {args.model_dir} --base-model-dir models/{args.base_model_name.split('/')[-1]} --offline --compare-with-base"
            )

    print("\n" + "=" * 40 + " END COMPARISON " + "=" * 40)

    return True


def main():
    """Main function to run tests."""
    print(f"Testing fine-tuned ATC model from: {args.model_dir}")
    print(f"Comparing with base model: {args.compare_with_base}")
    print(f"Using prompt style: {args.prompt_style}")

    if args.split_across_gpus:
        print("Model will be split across available GPUs (auto device mapping)")
    else:
        print(f"Using device: {args.device}")
        if args.base_device:
            print(f"Using base model device: {args.base_device}")

    # Determine test cases to use
    if args.test_inputs:
        test_inputs = args.test_inputs
        print(f"Using {len(test_inputs)} custom test inputs")
    else:
        if args.test_type == "basic" or args.test_type == "all":
            basic_inputs = BASIC_TEST_INPUTS
        else:
            basic_inputs = []

        if args.test_type == "complex" or args.test_type == "all":
            complex_inputs = COMPLEX_TEST_INPUTS
        else:
            complex_inputs = []

        if args.test_type == "domain" or args.test_type == "all":
            domain_inputs = DOMAIN_KNOWLEDGE_TEST_INPUTS
        else:
            domain_inputs = []

        test_inputs = basic_inputs + complex_inputs + domain_inputs

        test_type_mapping = {
            "basic": "Basic ATC communications",
            "complex": "Complex ATC scenarios",
            "domain": "Domain-specific ATC knowledge",
            "all": "All types of ATC communications",
        }
        print(
            f"Running {len(test_inputs)} tests for {test_type_mapping[args.test_type]}"
        )

    # Load models and tokenizers
    model, tokenizer, base_model, base_tokenizer = load_models_and_tokenizers()

    # Ensure the models are in evaluation mode
    model.eval()
    if base_model is not None:
        base_model.eval()

    # Run tests for each input
    success_count = 0
    for i, test_input in enumerate(test_inputs, 1):
        print(f"\nRunning test {i}/{len(test_inputs)}")
        if run_test(
            model, tokenizer, base_model, base_tokenizer, test_input, args.prompt_style
        ):
            success_count += 1

    # Print summary
    print("\n" + "=" * 80)
    print(f"TEST SUMMARY: {success_count}/{len(test_inputs)} tests successful")
    if base_model is None and args.compare_with_base:
        print(
            "Note: Base model comparison was requested but the base model couldn't be loaded."
        )
        print("If you're offline, first download the base model with --download")
        print(
            f"Then use --base-model-dir models/{args.base_model_name.split('/')[-1]} --offline"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
