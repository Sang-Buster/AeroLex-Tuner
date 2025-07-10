import json
import os
import sys
import time
from typing import Dict, List

import httpx
import ollama
from pydantic import BaseModel, Field

# Add utils to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.conversion_helpers import (
    apply_icao_abbreviations,
    apply_proper_casing,
    clean_punctuation,
    convert_numbers_to_words,
    convert_words_to_numbers,
    format_for_output,
    parse_atc_block,
    validate_t1_format,
    validate_t2_format,
)

# Configuration
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "llama3.3:70b-instruct-q4_K_M"

# uv pip install httpx ollama pydantic


class T1FormatResult(BaseModel):
    """T1 format conversion result."""
    t1_text: str = Field(..., description="T1 conversational format with words only")
    confidence: str = Field(..., description="Confidence in conversion quality")
    corrections: str = Field(..., description="Any corrections made during conversion")


class T2FormatResult(BaseModel):
    """T2 format conversion result."""
    t2_text: str = Field(..., description="T2 numerical format with ICAO abbreviations")
    confidence: str = Field(..., description="Confidence in conversion quality")
    corrections: str = Field(..., description="Any corrections made during conversion")


class FormatConversionResult(BaseModel):
    """Combined format conversion result."""
    t1_result: T1FormatResult = Field(..., description="T1 format conversion")
    t2_result: T2FormatResult = Field(..., description="T2 format conversion")
    source_analysis: str = Field(..., description="Analysis of the source PUNC text")
    speaker_context: str = Field(..., description="Speaker context for the conversion")


def parse_atc_blocks(content: str) -> List[Dict]:
    """Parse the entire content into a list of message blocks."""
    lines = content.splitlines()
    blocks = []
    current_block_lines = []
    
    for line in lines:
        if line.strip() == "":
            # Skip empty lines
            continue
        if line.startswith("{") and current_block_lines:
            # Start of a new block; save the previous one
            blocks.append("\n".join(current_block_lines))
            current_block_lines = [line]
        else:
            current_block_lines.append(line)
    
    if current_block_lines:
        blocks.append("\n".join(current_block_lines))
    
    return blocks


def analyze_with_llm(
    message: Dict, 
    conversation_history: List[Dict] = None, 
    max_retries: int = 3
) -> FormatConversionResult:
    """
    Use LLM to analyze ATC message and convert to T1 and T2 formats.
    
    Args:
        message: Parsed ATC message dictionary
        conversation_history: Previous messages for context
        max_retries: Maximum number of retry attempts
        
    Returns:
        FormatConversionResult with T1 and T2 conversions
    """
    if conversation_history is None:
        conversation_history = []

    # Create Ollama client
    client = ollama.Client(host=OLLAMA_HOST)

    # Extract speaker information
    speaker_info = message.get('speaker_info', {})
    speaker = speaker_info.get('speaker', 'Unknown')
    listener = speaker_info.get('listener', 'Unknown')
    speaker_role = speaker_info.get('speaker_role', 'Unknown')
    listener_role = speaker_info.get('listener_role', 'Unknown')

    # Build conversation context
    context_messages = []
    for prev_msg in conversation_history[-3:]:  # Last 3 messages for context
        context_messages.append({
            'original': prev_msg.get('ORIG', ''),
            'punctuated': prev_msg.get('PUNC', ''),
            'speaker': prev_msg.get('speaker_info', {}).get('speaker', ''),
        })

    system_message = {
        "role": "system",
        "content": """You are an expert in Air Traffic Control (ATC) communications with deep expertise in aviation phraseology and format conversion.

Your task is to convert ATC communications from a punctuated format into two specific target formats:

**T1 FORMAT (Conversational)**:
- Use conversational casing and natural punctuation
- Only allowed punctuation: commas (,), periods (.), question marks (?), exclamation marks (!)
- Convert ALL numbers to words (e.g., "209" → "two zero nine", "280" → "two eight zero")
- Use full airline names (e.g., "Delta" not "DAL")
- Use proper capitalization for proper nouns (airports, airlines, positions)
- Example: "Daytona Approach, November one two three, request landing on Runway eighteen left."

**T2 FORMAT (Numerical + ICAO)**:
- Use numerical digits and ICAO abbreviations
- Convert word numbers to digits (e.g., "two zero nine" → "209")
- Apply ICAO abbreviations (e.g., "Runway 18L", "FL240", "N123")
- Use airline codes for commercial aircraft (e.g., "DAL209")
- Maintain proper casing and punctuation
- Example: "Daytona Approach, N123, request landing on Runway 18L."

**AVIATION EXPERTISE REQUIRED**:
- Understand aircraft callsign formats (airline codes vs. general aviation)
- Apply proper runway designations (18L, 09R, etc.)
- Handle altitude formats (flight levels vs. standard altitudes)
- Use correct aviation phraseology and terminology
- Maintain communication intent and meaning exactly

**SPEAKER CONTEXT**:
- Analyze who is speaking (ATC position or aircraft)
- Consider communication flow and intent
- Ensure terminology matches speaker role (pilot vs. controller)

**QUALITY STANDARDS**:
- High confidence means perfect aviation phraseology and format adherence
- Medium confidence means minor uncertainties but correct overall
- Low confidence means significant uncertainties or non-standard phraseology
- Always explain any corrections or assumptions made

Provide detailed analysis and high-quality conversions that maintain aviation safety standards."""
    }

    # Create detailed analysis prompt
    analysis_prompt = {
        "role": "user",
        "content": f"""**MESSAGE TO CONVERT**:
Header: {message.get('header', '')}
Original: {message.get('ORIG', '')}
Processed: {message.get('PROC', '')}
Numeric: {message.get('NUMC', '')}
Punctuated: {message.get('PUNC', '')}

**SPEAKER CONTEXT**:
- Speaker: {speaker} (Role: {speaker_role})
- Listener: {listener} (Role: {listener_role})
- Communication Type: {speaker_role} → {listener_role}

**CONVERSATION HISTORY** (last 3 messages):
{json.dumps(context_messages, indent=2)}

**CONVERSION TASK**:
Please analyze the PUNCTUATED (PUNC) text and convert it to both T1 and T2 formats following the specifications above.

Return your response as a JSON object with this exact structure:

```json
{{
    "t1_result": {{
        "t1_text": "Conversational format with words only",
        "confidence": "High|Medium|Low",
        "corrections": "Description of any corrections made"
    }},
    "t2_result": {{
        "t2_text": "Numerical format with ICAO abbreviations", 
        "confidence": "High|Medium|Low",
        "corrections": "Description of any corrections made"
    }},
    "source_analysis": "Analysis of the source PUNC text quality and content",
    "speaker_context": "How speaker context influenced the conversion"
}}
```

Focus on aviation accuracy, proper phraseology, and format compliance."""
    }

    messages = [system_message, analysis_prompt]

    # Process with streaming and retries
    print(f"\nConverting to T1/T2: {message.get('ORIG', '')[:60]}...")

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
                    print(content, end="", flush=True)

            print("\n\n--- LLM RESPONSE END ---")
            print(" Done!")
            break

        except (httpx.ReadError, ConnectionError) as e:
            attempt += 1
            if attempt < max_retries:
                wait_time = 2**attempt  # Exponential backoff
                print(f"\nConnection error: {str(e)}")
                print(f"Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"\nFailed after {max_retries} attempts: {str(e)}")
                return create_default_conversion(message)

    # Parse the JSON response
    try:
        # Extract JSON portion from response
        import re
        json_match = re.search(r'```json\s*(.*?)\s*```', full_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON object directly
            json_match = re.search(r'({[\s\S]*})', full_response)
            if json_match:
                json_str = json_match.group(1)
            else:
                print("Could not extract JSON from response")
                return create_default_conversion(message)

        json_data = json.loads(json_str)

        # Create result objects
        t1_result = T1FormatResult(**json_data.get('t1_result', {}))
        t2_result = T2FormatResult(**json_data.get('t2_result', {}))

        conversion_result = FormatConversionResult(
            t1_result=t1_result,
            t2_result=t2_result,
            source_analysis=json_data.get('source_analysis', ''),
            speaker_context=json_data.get('speaker_context', '')
        )

        # Validate results
        if not validate_t1_format(t1_result.t1_text):
            print(f"Warning: T1 format validation failed for: {t1_result.t1_text}")
        
        if not validate_t2_format(t2_result.t2_text):
            print(f"Warning: T2 format validation failed for: {t2_result.t2_text}")

        return conversion_result

    except Exception as e:
        print(f"Error parsing LLM response: {str(e)}")
        print(f"Raw response: {full_response}")
        return create_default_conversion(message)


def create_default_conversion(message: Dict) -> FormatConversionResult:
    """Create default conversion when LLM processing fails."""
    punc_text = message.get('PUNC', '')
    
    # Basic rule-based conversion as fallback
    t1_text = apply_proper_casing(convert_numbers_to_words(punc_text))
    t1_text = clean_punctuation(t1_text)
    t1_text = format_for_output(t1_text, 'T1')
    
    t2_text = convert_words_to_numbers(punc_text) 
    t2_text = apply_icao_abbreviations(t2_text)
    t2_text = format_for_output(t2_text, 'T2')

    return FormatConversionResult(
        t1_result=T1FormatResult(
            t1_text=t1_text,
            confidence="Low",
            corrections="Rule-based conversion used due to LLM failure"
        ),
        t2_result=T2FormatResult(
            t2_text=t2_text, 
            confidence="Low",
            corrections="Rule-based conversion used due to LLM failure"
        ),
        source_analysis="LLM analysis unavailable",
        speaker_context="Context analysis unavailable"
    )


def format_combined_lbs_block(message: Dict, conversion: FormatConversionResult) -> str:
    """Format a combined T1/T2 message block for LBS output."""
    block = f"{message['header']}\n"
    block += f"ORIG: {message['ORIG']}\n"
    block += f"PROC: {message['PROC']}\n"
    block += f"NUMC: {message['NUMC']}\n"
    block += f"PUNC: {message['PUNC']}\n"
    block += f"T1: {conversion.t1_result.t1_text}\n"
    block += f"T2: {conversion.t2_result.t2_text}\n"
    block += "NOTE:\n"
    block += f"    - Source Analysis: {conversion.source_analysis}\n"
    block += f"    - T1 Confidence: {conversion.t1_result.confidence}\n"
    block += f"    - T1 Corrections: {conversion.t1_result.corrections}\n"
    block += f"    - T2 Confidence: {conversion.t2_result.confidence}\n"
    block += f"    - T2 Corrections: {conversion.t2_result.corrections}\n"
    block += f"    - Speaker Context: {conversion.speaker_context}\n"
    
    return block


def save_combined_json_results(processed_messages: List[Dict], json_path: str):
    """Save conversion results as combined JSON file."""
    combined_results = []
    
    for msg in processed_messages:
        conversion = msg['conversion']
        
        # Combined T1/T2 result
        combined_entry = {
            'header': msg['header'],
            'original': msg['ORIG'],
            'processed': msg['PROC'],
            'numeric': msg['NUMC'],
            'punctuated': msg['PUNC'],
            't1_text': conversion.t1_result.t1_text,
            't1_confidence': conversion.t1_result.confidence,
            't1_corrections': conversion.t1_result.corrections,
            't2_text': conversion.t2_result.t2_text,
            't2_confidence': conversion.t2_result.confidence,
            't2_corrections': conversion.t2_result.corrections,
            'source_analysis': conversion.source_analysis,
            'speaker_context': conversion.speaker_context,
            'speaker_info': msg.get('speaker_info', {})
        }
        combined_results.append(combined_entry)
    
    # Save combined JSON file
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(combined_results, f, indent=2, ensure_ascii=False)
    
    print(f"Saved combined T1/T2 JSON results to {json_path}")


def process_file(
    input_path: str, 
    combined_lbs_path: str,
    combined_json_path: str
):
    """Process the entire ATC transcription file and generate combined T1/T2 formats."""
    print(f"Processing file: {input_path}")
    
    # Read input file
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Parse ATC blocks
    blocks = parse_atc_blocks(content)
    
    processed_messages = []
    conversation_history = []
    combined_lbs_blocks = []

    total_blocks = min(30, len(blocks))  # Process max 30 messages
    print(f"\nProcessing {total_blocks} messages with LLM conversion...")

    for i, block in enumerate(blocks[:total_blocks], 1):
        if not block.strip():
            continue

        print(f"\nMessage {i}/{total_blocks}")

        # Parse message block
        message = parse_atc_block(block)
        if not message or 'PUNC' not in message:
            print("Skipping malformed message")
            continue

        # Analyze with LLM for T1/T2 conversion
        conversion = analyze_with_llm(message, conversation_history)

        # Store message with conversion
        message['conversion'] = conversion
        processed_messages.append(message)

        # Format combined LBS block
        combined_block = format_combined_lbs_block(message, conversion)
        combined_lbs_blocks.append(combined_block)

        # Update conversation history
        conversation_history.append(message)
        if len(conversation_history) > 5:
            conversation_history.pop(0)

        # Save progress
        with open(combined_lbs_path, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(combined_lbs_blocks))

        # Save JSON progress
        save_combined_json_results(processed_messages, combined_json_path)

        print(f"Saved progress - T1: {conversion.t1_result.confidence}, T2: {conversion.t2_result.confidence}")

    print(f"\nCompleted! Processed {len(processed_messages)} messages")
    print("Results saved to:")
    print(f"  Combined LBS: {combined_lbs_path}")
    print(f"  Combined JSON: {combined_json_path}")


if __name__ == "__main__":
    # Record start time
    start_time = time.time()
    
    # Define paths
    input_file = "data/test.lbs"
    combined_lbs_output = "data/test_t1_t2.lbs"
    combined_json_output = "data/test_t1_t2.json"

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found!")
        print("Please ensure the input file exists.")
        sys.exit(1)

    # Process the file
    process_file(
        input_file,
        combined_lbs_output,
        combined_json_output
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