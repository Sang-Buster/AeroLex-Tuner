import json
import time
from typing import Dict, List, Optional

import dspy
import mlflow
from pydantic import BaseModel, Field

# Configuration
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1:405b-instruct-q4_K_M"

INPUT_FILE = "data/test_llm.json"
OUTPUT_FILE = "data/test_dspy_llama3.1_405b_punv.json"
OUTPUT_LBS = "data/test_dspy_llama3.1_405b_punv.lbs"

# Set up MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("ATC-Verification")

# Configure DSPy with Ollama - using the correct API format
lm = dspy.LM(model=f"ollama/{OLLAMA_MODEL}", api_base=OLLAMA_HOST)
dspy.configure(lm=lm)  # Remove trace=False parameter to allow MLflow tracing

# Enable MLflow tracing for DSPy
mlflow.dspy.autolog()

# uv pip install dspy mlflow pydantic


# Data models
class ATCNumbers(BaseModel):
    """Model for tracking numeric values in ATC communications."""

    Csgn: Optional[str] = Field(None, description="Aircraft Callsign")
    Rway: Optional[str] = Field(None, description="Runway Number")
    Altd: Optional[str] = Field(None, description="Altitude (feet MSL)")
    FLvl: Optional[str] = Field(None, description="Flight Level (FLxxx)")
    Hdng: Optional[str] = Field(None, description="Heading (degrees)")
    VORr: Optional[str] = Field(None, description="VOR Radial Direction")
    Freq: Optional[str] = Field(None, description="Frequency (e.g., 118.7)")
    ASpd: Optional[str] = Field(None, description="Airspeed (knots)")
    Dist: Optional[str] = Field(None, description="Distance (nautical miles)")
    Squk: Optional[str] = Field(None, description="Transponder Code (Squawk)")
    TZlu: Optional[str] = Field(None, description="Standard Time (Zulu)")
    Amtr: Optional[str] = Field(None, description="Altimeter Setting")
    Wdir: Optional[str] = Field(None, description="Wind Direction")
    Wspd: Optional[str] = Field(None, description="Wind Speed (knots)")
    Tmpr: Optional[str] = Field(None, description="Temperature (°C)")
    DewP: Optional[str] = Field(None, description="Dew Point (°C)")


# Define DSPy modules
class ATCSignature(dspy.Signature):
    """Input/output signature for ATC message analysis."""

    header: str = dspy.InputField(description="Header information for the ATC message")
    original: str = dspy.InputField(
        description="Original transcription of the ATC message"
    )
    processed: str = dspy.InputField(description="Processed version of the message")
    numeric: str = dspy.InputField(description="Numeric representation of the message")
    punctuated: str = dspy.InputField(description="Punctuated version")
    pune: str = dspy.InputField(description="Current PUNE format")
    history: List[Dict] = dspy.InputField(
        description="Previous messages in the conversation"
    )

    punv: str = dspy.OutputField(description="Verified punctuations with improvements")
    edit: str = dspy.OutputField(description="Edit recommendations for the message")
    speaker: str = dspy.OutputField(description="Who is talking (controller/pilot)")
    speaker_confidence: str = dspy.OutputField(
        description="How the speaker was determined (known/inferred/guessed)"
    )
    listener: str = dspy.OutputField(description="Who is being addressed")
    event: str = dspy.OutputField(description="What is happening in this communication")
    actions: str = dspy.OutputField(description="Actions being taken in this message")
    intentions_verified: Dict[str, bool] = dspy.OutputField(
        description="Verified communication intentions"
    )
    numbers_verified: Dict[str, str] = dspy.OutputField(
        description="Verified numerical values"
    )


class ATCAnalyzer(dspy.Module):
    """DSPy module for analyzing ATC messages using ReACT framework."""

    def __init__(self):
        super().__init__()
        self.analyzer = dspy.ChainOfThought(ATCSignature)

    def forward(
        self,
        header,
        original,
        processed,
        numeric,
        punctuated,
        pune,
        history=None,
        intentions=None,
        numbers=None,
    ):
        """Process an ATC message and produce verified analysis."""
        if history is None:
            history = []

        try:
            # Run the analyzer
            result = self.analyzer(
                header=header,
                original=original,
                processed=processed,
                numeric=numeric,
                punctuated=punctuated,
                pune=pune,
                history=history,
            )
            return result
        except Exception as e:
            print(f"Error in analyzer: {str(e)}")
            # Create a fallback result with default values to prevent pipeline failure
            default_result = dspy.Prediction(
                punv=pune,
                edit="No edits needed",
                speaker="Unknown",
                speaker_confidence="Unknown",
                listener="Unknown",
                event="Message analysis failed",
                actions="None",
                intentions_verified={},
                numbers_verified={},
            )
            return default_result


# Optimize the analyzer with a few examples (if available)
def optimize_analyzer():
    """Optimize the ATC analyzer with examples (optional)."""
    try:
        # Load a few examples to bootstrap the optimization
        with open(INPUT_FILE, "r") as f:
            data = json.load(f)

        if len(data) >= 5:
            examples = []
            for i in range(min(5, len(data))):
                # Format example inputs based on your data
                example = dspy.Example(
                    header=data[i]["header"],
                    original=data[i]["original"],
                    processed=data[i]["processed"],
                    numeric=data[i]["numeric"],
                    punctuated=data[i]["punctuated"],
                    pune=data[i]["pune"],
                    history=data[:i] if i > 0 else [],
                ).with_inputs(
                    "header",
                    "original",
                    "processed",
                    "numeric",
                    "punctuated",
                    "pune",
                    "history",
                )
                examples.append(example)

            print(
                "Using unoptimized analyzer due to output format incompatibility with optimizer"
            )
            return ATCAnalyzer()

    except Exception as e:
        print(f"Loading examples failed: {str(e)}")

    # Return unoptimized analyzer if optimization fails
    print("Using default unoptimized analyzer")
    return ATCAnalyzer()


def format_lbs_block(message, analysis):
    """Format a message block for the .lbs file."""
    block = f"{message['header']}\n"
    block += f"ORIG: {message['original']}\n"
    block += f"PROC: {message['processed']}\n"
    block += f"NUMC: {message['numeric']}\n"
    block += f"PUNC: {message['punctuated']}\n"
    block += f"PUNE: {message['pune']}\n"
    block += f"PUNV: {analysis.punv}\n"
    block += f"EDIT: {analysis.edit}\n"
    block += "NOTE:\n"
    block += f"    - Speaker: {analysis.speaker} ({analysis.speaker_confidence})\n\n"
    block += f"    - Listener: {analysis.listener}\n\n"
    block += f"    - Event: {analysis.event}\n\n"
    block += f"    - Actions: {analysis.actions}\n\n"

    # Add intentions checklist
    intention_descriptions = {
        "PSC": "Pilot starts contact to ATC",
        "PSR": "Pilot starts contact to ATC with reported info",
        "PRP": "Pilot reports information",
        "PRQ": "Pilot issues requests",
        "PRB": "Pilot readback",
        "PAC": "Pilot acknowledge",
        "ASC": "ATC starts contact to pilot",
        "AGI": "ATC gives instruction",
        "ACR": "ATC corrects pilot's readback",
        "END": "Either party signaling the end of exchange",
    }

    block += "    - Intention:\n"
    for code, desc in intention_descriptions.items():
        enabled = analysis.intentions_verified.get(code, False)
        mark = "[x]" if enabled else "[]"
        block += f'        - {mark} "{code}": {desc}.\n'

    # Add numbers section
    field_descriptions = {
        "Csgn": "Aircraft Callsign",
        "Rway": "Runway Number",
        "Altd": "Altitude (feet MSL)",
        "FLvl": "Flight Level (FLxxx)",
        "Hdng": "Heading (degrees)",
        "VORr": "VOR Radial Direction",
        "Freq": "Frequency (e.g., 118.7)",
        "ASpd": "Airspeed (knots)",
        "Dist": "Distance (nautical miles)",
        "Squk": "Transponder Code (Squawk)",
        "TZlu": "Standard Time (Zulu)",
        "Amtr": "Altimeter Setting",
        "Wdir": "Wind Direction",
        "Wspd": "Wind Speed (knots)",
        "Tmpr": "Temperature (°C)",
        "DewP": "Dew Point (°C)",
    }

    block += "\n    - Number1:\n"
    for field, desc in field_descriptions.items():
        value = analysis.numbers_verified.get(field, "")
        block += f"        - {field}: {value}  # {desc}\n"

    return block


def process_file():
    """Process the input file and generate verified outputs."""
    with mlflow.start_run(run_name="ATC_Message_Analysis-llama3.1:405b"):
        # Record start time
        start_time = time.time()

        print(f"Processing file: {INPUT_FILE}")

        # Check if Ollama server is running
        try:
            import requests

            response = requests.get(f"{OLLAMA_HOST}/api/tags")
            if response.status_code != 200:
                print(f"Warning: Ollama server returned status {response.status_code}")
                print("Make sure Ollama is running with: 'ollama serve'")
        except Exception as e:
            print(f"Error connecting to Ollama server: {str(e)}")
            print("Please make sure Ollama is running with 'ollama serve'")
            print(
                f"And check that your model '{OLLAMA_MODEL}' is available with 'ollama list'"
            )
            return

        # Get an optimized analyzer
        analyzer = optimize_analyzer()

        # Read input file
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            messages = json.load(f)

        # Process messages
        processed_messages = []
        conversation_history = []
        json_results = []

        total_messages = min(30, len(messages))  # Process max 30 messages

        for i, message in enumerate(messages[:total_messages], 1):
            print(f"\nMessage {i}/{total_messages}")

            # Skip messages with empty PUNE
            if not message.get("pune"):
                print(
                    f"Skipping message with empty PUNE: {message.get('original', '')[:30]}..."
                )
                continue

            # Log this analysis step
            mlflow.log_param(f"message_{i}_original", message.get("original", ""))

            try:
                # Add more detailed error handling and debugging
                print(f"Processing message: {message['original']}")

                # Run analysis with DSPy
                analysis = analyzer(
                    header=message["header"],
                    original=message["original"],
                    processed=message["processed"],
                    numeric=message["numeric"],
                    punctuated=message["punctuated"],
                    pune=message["pune"],
                    history=conversation_history,
                    intentions=message.get("intentions", {}),
                    numbers=message.get("numbers", {}),
                )

                # Validate the analysis result
                if not hasattr(analysis, "punv") or not analysis.punv:
                    print(
                        "Warning: Analysis result is missing required fields, using fallbacks"
                    )
                    analysis.punv = message["pune"]
                if not hasattr(analysis, "edit") or not analysis.edit:
                    analysis.edit = "No edits needed"
                if (
                    not hasattr(analysis, "intentions_verified")
                    or not analysis.intentions_verified
                ):
                    analysis.intentions_verified = {}
                if (
                    not hasattr(analysis, "numbers_verified")
                    or not analysis.numbers_verified
                ):
                    analysis.numbers_verified = {}

                # Create enhanced message with verification
                enhanced_message = {**message}
                enhanced_message.update(
                    {
                        "punv": analysis.punv,
                        "edit": analysis.edit,
                        "speaker": getattr(analysis, "speaker", "Unknown"),
                        "speaker_confidence": getattr(
                            analysis, "speaker_confidence", "Unknown"
                        ),
                        "listener": getattr(analysis, "listener", "Unknown"),
                        "event": getattr(analysis, "event", "Unknown"),
                        "actions": getattr(analysis, "actions", "None"),
                        "intentions_verified": analysis.intentions_verified,
                        "numbers_verified": analysis.numbers_verified,
                    }
                )

                # Format message block for .lbs file
                block = format_lbs_block(message, analysis)
                processed_messages.append(block)

                # Add to JSON results
                json_results.append(enhanced_message)

                # Log metrics
                mlflow.log_metric(f"message_{i}_success", 1)

                # Update conversation history
                conversation_history.append(message)
                if len(conversation_history) > 5:
                    conversation_history.pop(0)

                # Save progress
                with open(OUTPUT_LBS, "w", encoding="utf-8") as f:
                    f.write("\n\n".join(processed_messages))

                with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                    json.dump(json_results, f, indent=2)

                print(f"Saved progress to {OUTPUT_LBS} and {OUTPUT_FILE}")

            except Exception as e:
                print(f"Error processing message {i}: {str(e)}")
                print(f"Error type: {type(e).__name__}")
                # Print more detailed error information
                import traceback

                traceback.print_exc()
                mlflow.log_metric(f"message_{i}_success", 0)
                mlflow.log_param(f"message_{i}_error", str(e))

        # Calculate and display execution time
        end_time = time.time()
        execution_time = end_time - start_time

        # Format time nicely
        hours, remainder = divmod(execution_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours > 0:
            time_str = f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
        elif minutes > 0:
            time_str = f"{int(minutes)}m {seconds:.2f}s"
        else:
            time_str = f"{seconds:.2f}s"

        print(f"\n✓ Total execution time: {time_str}")
        print(f"✓ Completed successfully at {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Log completion information
        mlflow.log_metric("execution_time_seconds", execution_time)
        mlflow.log_metric("messages_processed", len(json_results))

        # View trace in MLflow UI
        print("\nView the analysis trace in MLflow UI at http://localhost:5000")


if __name__ == "__main__":
    process_file()

    # For debugging, you can inspect history
    # dspy.inspect_history(n=5)
