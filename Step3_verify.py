import json
import re
import time
from typing import Any, Dict, List, Optional

import httpx
import ollama
from pydantic import BaseModel, Field

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "llama3.3:70b-instruct-q4_K_M"

# uv pip install httpx ollama pydantic


# Data models
class ATCNumbers(BaseModel):
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


class ATCVerification(BaseModel):
    punv: str = Field(..., description="Verified punctuations with improvements")
    edit: str = Field(..., description="Edit recommendations for the message")
    speaker: str = Field(..., description="Who is talking (controller/pilot)")
    speaker_confidence: str = Field(..., description="How the speaker was determined")
    listener: str = Field(..., description="Who is being addressed")
    event: str = Field(
        ..., description="What happened historically and in this message"
    )
    actions: str = Field(..., description="Actions taken in this message")


# Global state to track aircraft and context
class ConversationState:
    def __init__(self):
        self.aircraft = {}  # Track aircraft positions, altitudes, headings
        self.controllers = {}  # Track controller positions
        self.last_instructions = {}  # Track latest instructions to each aircraft

    def update_from_message(self, message: Dict):
        """Update state based on message content."""
        # Extract callsign
        callsign = message.get("numbers", {}).get("Csgn")
        if not callsign:
            return

        # Track aircraft data
        if callsign not in self.aircraft:
            self.aircraft[callsign] = {}

        # Update altitude if available
        altitude = message.get("numbers", {}).get("Altd")
        if altitude:
            self.aircraft[callsign]["altitude"] = altitude

        # Update heading if available
        heading = message.get("numbers", {}).get("Hdng")
        if heading:
            self.aircraft[callsign]["heading"] = heading

        # Update flight level if available
        flight_level = message.get("numbers", {}).get("FLvl")
        if flight_level:
            self.aircraft[callsign]["flight_level"] = flight_level

        # Track instructions
        if "AGI" in message.get("intentions", {}) and message.get("intentions", {}).get(
            "AGI"
        ):
            self.last_instructions[callsign] = message.get("pune", "")


# Helper functions
def extract_speaker_info_from_header(header: str) -> Dict[str, str]:
    """Extract speaker and listener information from message header."""
    result = {"speaker": "", "listener": "", "speaker_role": ""}

    # Common airline codes
    airline_codes = {
        "AAL": "American",
        "DAL": "Delta",
        "UAL": "United",
        "SWA": "Southwest",
        "USA": "USAir",
        "AWE": "America West",
        "TWA": "TWA",
        "NWA": "Northwest",
        "PAA": "Pan Am",
        "COA": "Continental",
    }

    # Extract parts from header format: {session speaker_turn speaker_identifier start_time end_time}
    # Example: {dca_d1_1 1 dca_d1_1__DR1-1__DAL209__00_01_03 63.040 66.010}
    match = re.search(r"\{(.*?)\s+(\d+)\s+(.*?)\s+(\d+\.\d+)\s+(\d+\.\d+)\}", header)
    if not match:
        return result

    identifier = match.group(3)
    parts = identifier.split("__")

    if len(parts) >= 3:
        # Format: session__speaker__listener__time
        result["speaker"] = parts[1]
        result["listener"] = parts[2]

        # Determine if speaker is ATC or pilot
        if "DR" in result["speaker"]:
            result["speaker_role"] = "ATC"
        else:
            for code in airline_codes:
                if code in result["speaker"]:
                    result["speaker_role"] = "Pilot"
                    break

    return result


def determine_speaker_role(callsign: str) -> str:
    """Determine if the callsign belongs to ATC or a pilot."""
    # Common ATC identifiers
    atc_identifiers = ["DR", "APP", "TWR", "GND", "DEP", "CTR", "ATIS"]

    # Common airline designators
    airline_designators = [
        "AAL",
        "DAL",
        "UAL",
        "SWA",
        "USA",
        "AWE",
        "TWA",
        "NWA",
        "PAA",
        "COA",
        "N",
        "ASA",
        "FFT",
        "BAW",
        "KLM",
        "AFR",
        "DLH",
        "QFA",
        "JBU",
        "VRD",
        "KAL",
        "ANA",
        "EIN",
        "IBE",
        "UAE",
        "CPA",
        "CCA",
        "QTR",
        "SIA",
        "ACA",
    ]

    # Check for ATC identifiers
    for identifier in atc_identifiers:
        if identifier in callsign:
            return "ATC"

    # Check for airline designators or "N" for N-numbered aircraft
    for designator in airline_designators:
        if callsign.startswith(designator) or designator in callsign:
            return "Pilot"

    return "Unknown"


def format_intention_checklist(intentions: Dict[str, bool]) -> str:
    """Format the intentions checklist for the NOTE section."""
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

    checklist = "    - Intention:\n"
    for code, enabled in intentions.items():
        mark = "[x]" if enabled else "[]"
        description = intention_descriptions.get(code, "Unknown intention code")
        checklist += f'        - {mark} "{code}": {description}.\n'

    return checklist


def format_numbers_section(numbers: Dict[str, Any]) -> str:
    """Format the numbers section for the NOTE section."""
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

    section = "    - Number1:\n"
    for field, value in numbers.items():
        description = field_descriptions.get(field, "Unknown field")
        section += f"        - {field}: {value}  # {description}\n"

    return section


def analyze_message_with_react(
    message: Dict,
    conversation_history: List[Dict],
    state: ConversationState,
    max_retries: int = 3,
) -> ATCVerification:
    """Analyze message using the ReACT framework."""
    # Create Ollama client
    client = ollama.Client(host=OLLAMA_HOST)

    # Build context from conversation history
    context_history = []
    for prev_msg in conversation_history:  # Use all available conversation history
        context_entry = {
            "header": prev_msg["header"],
            "original": prev_msg["original"],
            "pune": prev_msg.get("pune", ""),
            "intentions": prev_msg.get("intentions", {}),
            "numbers": prev_msg.get("numbers", {}),
        }
        context_history.append(context_entry)

    # Create system prompt with ReACT framework guidance
    system_prompt = {
        "role": "system",
        "content": """You are an expert in Air Traffic Control (ATC) communications analysis specializing in verification of transcribed communications.

Your task is to analyze ATC messages using the ReACT (Reasoning and Acting) framework:

1. REASON: Think step-by-step about the message context, speaker roles, communication intent, and standard ATC protocols
2. ACT: Generate verified punctuation, suggest edits, and provide event analysis
3. OBSERVE: Analyze the implications and context of the communication
4. THINK AGAIN: Revise your understanding if needed based on all evidence

For each message, provide:

1. PUNV (Verified Punctuation):
   - A verified and potentially improved version of the PUNE punctuation
   - Ensure strict adherence to ATC phraseology standards
   - Correct any subtle errors in phrasing, number grouping, or terminology

2. EDIT (Edit Recommendations):
   - Suggest how the original communication could be improved for clarity and compliance with ATC standards
   - Focus on proper phraseology, standard terminology, and complete information

3. Enhanced NOTE section:
   - Speaker: Identify who is talking (specific controller position or aircraft callsign)
     DO NOT simply extract this from the header information. 
     Analyze the message content and conversation context to determine this.
     
   - Speaker Confidence: Indicate how you determined the speaker as one of:
     "Known from message content" - When the message clearly indicates who is speaking
     "Inferred from context" - When context provides strong clues
     "Guessed from pattern" - When making an educated guess
     "Unknown" - When there's insufficient information
     
   - Listener: Identify who is being addressed, using the same principles as for speaker identification
   
   - Event: Describe what is happening in this communication and its historical context
   
   - Actions: Document specific actions being taken or requested

FOLLOW ATC COMMUNICATION STANDARDS:
- Use proper aviation phraseology
- Format callsigns consistently (e.g., "Delta 209" not "delta two oh nine")
- Format numbers according to ATC standards
- Use appropriate punctuation and separation between phrases

Your analysis must be detailed, precise, and compliant with standard ATC protocols. Focus on providing actionable insights that would improve communication clarity and safety.
""",
    }

    # Prepare aircraft state information
    aircraft_state = {}
    for callsign, data in state.aircraft.items():
        aircraft_state[callsign] = data

    # Create user prompt with ReACT framework structure and context
    user_prompt = {
        "role": "user",
        "content": f"""# CURRENT MESSAGE
Header: {message["header"]}
Original: {message["original"]}
Processed: {message["processed"]}
Numeric: {message["numeric"]}
Punctuated: {message["punctuated"]}
PUNE: {message["pune"]}

# CONTEXT
Header information: 
- Message header: {message["header"]}

Recent conversation history:
{json.dumps(context_history, indent=2)}

Current aircraft states:
{json.dumps(aircraft_state, indent=2)}

# ReACT ANALYSIS PROCESS
Please analyze this message following the ReACT framework. I need your final output in JSON format with the following structure:

```json
{{
  "punv": "Verified punctuation with improvements",
  "edit": "Edit recommendations for better ATC communication",
  "speaker": "Who is talking (specific controller or aircraft)",
  "speaker_confidence": "How speaker was determined (known/inferred/guessed/unknown)",
  "listener": "Who is being addressed",
  "event": "What is happening in this communication",
  "actions": "Specific actions being taken or requested"
}}
```

Make sure to thoroughly reason through the entire message before providing your final output.
""",
    }

    messages = [system_prompt, user_prompt]

    # Process with streaming and retries
    print(f"\nAnalyzing message: {message['original'][:60]}...")

    attempt = 0
    while attempt < max_retries:
        try:
            full_response = ""
            print("Sending request to LLM...")
            print("\n--- LLM RESPONSE START ---")

            for chunk in client.chat(
                model=OLLAMA_MODEL,
                messages=messages,
                stream=True,
            ):
                if chunk.message.content:
                    content = chunk.message.content
                    full_response += content
                    # Print the actual content always
                    print(content, end="", flush=True)

            print("\n\n--- LLM RESPONSE END ---")
            print(" Done!")
            break

        except (httpx.ReadError, ConnectionError) as e:
            attempt += 1
            if attempt < max_retries:
                wait_time = 2**attempt  # Exponential backoff
                print(f"\nConnection error: {str(e)}")
                print(
                    f"Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(wait_time)
            else:
                print(f"\nFailed after {max_retries} attempts: {str(e)}")
                return create_default_verification(message)

    # Parse the JSON response
    try:
        # Extract JSON portion from response (handling potential text before/after)
        json_match = re.search(r"```json\s*(.*?)\s*```", full_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON object directly
            json_match = re.search(r"({[\s\S]*})", full_response)
            if json_match:
                json_str = json_match.group(1)
            else:
                print("Could not extract JSON from response")
                return create_default_verification(message)

        json_data = json.loads(json_str)

        # Extract appropriate fields, handling both direct strings and nested structures
        punv = json_data.get("punv", "")
        if isinstance(punv, dict):
            if "PUNE" in punv:
                punv = punv["PUNE"]
            elif "punctuated" in punv:
                punv = punv["punctuated"]
            else:
                punv = str(punv)

        edit = json_data.get("edit", "")
        if isinstance(edit, dict):
            if "recommended" in edit:
                edit = edit["recommended"]
            else:
                edit = str(edit)

        speaker = json_data.get("speaker", "")
        if isinstance(speaker, dict):
            speaker_type = speaker.get("type", "Unknown")
            speaker_confidence = speaker.get("confidence", "Unknown")
            speaker = f"{speaker_type}"
        else:
            speaker_confidence = json_data.get("speaker_confidence", "Unknown")

        listener = json_data.get("listener", "")
        if isinstance(listener, dict):
            listener_type = listener.get("type", "Unknown")
            listener = f"{listener_type}"

        event = json_data.get("event", "Unknown event")

        actions = json_data.get("actions", "")
        if isinstance(actions, list):
            actions = "; ".join(actions)

        # Create verification object
        verification = ATCVerification(
            punv=punv,
            edit=edit,
            speaker=speaker,
            speaker_confidence=speaker_confidence
            if isinstance(speaker_confidence, str)
            else "Unknown",
            listener=listener,
            event=event,
            actions=actions if isinstance(actions, str) else str(actions),
        )

        return verification

    except Exception as e:
        print(f"Error parsing response: {str(e)}")
        print(f"Raw response: {full_response}")
        return create_default_verification(message)


def create_default_verification(message: Dict) -> ATCVerification:
    """Create a default verification when LLM processing fails."""

    return ATCVerification(
        punv=message.get("pune", ""),
        edit="No edit recommendations available.",
        speaker="Unknown speaker",
        speaker_confidence="Unknown - LLM processing failed",
        listener="Unknown listener",
        event="Unknown event",
        actions="Unknown actions",
    )


def format_lbs_block(message: Dict, verification: ATCVerification) -> str:
    """Format a message block for the .lbs file."""
    block = f"{message['header']}\n"
    block += f"ORIG: {message['original']}\n"
    block += f"PROC: {message['processed']}\n"
    block += f"NUMC: {message['numeric']}\n"
    block += f"PUNC: {message['punctuated']}\n"
    block += f"PUNE: {message['pune']}\n"
    block += f"PUNV: {verification.punv}\n"
    block += f"EDIT: {verification.edit}\n"
    block += "NOTE:\n"
    block += (
        f"    - Speaker: {verification.speaker} ({verification.speaker_confidence})\n\n"
    )
    block += f"    - Listener: {verification.listener}\n\n"
    block += f"    - Event: {verification.event}\n\n"
    block += f"    - Actions: {verification.actions}\n\n"

    # Add intentions checklist
    block += format_intention_checklist(message.get("intentions", {}))

    # Add numbers section
    block += "\n"
    block += format_numbers_section(message.get("numbers", {}))

    return block


def process_file(input_json_path: str, output_json_path: str, output_lbs_path: str):
    """Process an entire file with conversation state tracking."""
    print(f"Processing file: {input_json_path}")

    # Read input file
    with open(input_json_path, "r", encoding="utf-8") as f:
        messages = json.load(f)

    # Initialize conversation state
    state = ConversationState()

    # Process messages
    processed_messages = []
    conversation_history = []
    json_results = []

    total_messages = min(30, len(messages))  # Process max 30 messages

    for i, message in enumerate(messages[:total_messages], 1):
        print(f"\nMessage {i}/{total_messages}")

        # Skip messages with empty PUNE
        if not message.get("pune"):
            print("Skipping message with empty PUNE")
            continue

        # Analyze message with ReACT framework
        verification = analyze_message_with_react(message, conversation_history, state)

        # Create enhanced message with verification
        enhanced_message = {**message}
        enhanced_message.update(
            {
                "punv": verification.punv,
                "edit": verification.edit,
                "speaker": verification.speaker,
                "speaker_confidence": verification.speaker_confidence,
                "listener": verification.listener,
                "event": verification.event,
                "actions": verification.actions,
            }
        )

        # Format message block for .lbs file
        block = format_lbs_block(message, verification)
        processed_messages.append(block)

        # Add to JSON results
        json_results.append(enhanced_message)

        # Update conversation history
        conversation_history.append(message)
        if len(conversation_history) > 5:
            conversation_history.pop(0)

        # Update conversation state
        state.update_from_message(message)

        # Save progress
        with open(output_lbs_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(processed_messages))

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(json_results, f, indent=2)

        print(f"Saved progress to {output_lbs_path} and {output_json_path}")

    print(f"\nCompleted processing {len(processed_messages)} messages!")


if __name__ == "__main__":
    # Record start time
    start_time = time.time()

    # Process all files
    process_file(
        "data/test_llm.json",
        "data/test_llm_verified.json",
        "data/test_llm_verified.lbs",
    )

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
