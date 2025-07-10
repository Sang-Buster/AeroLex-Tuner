import os
import re
import sys

# Add the parent directory to the path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.aviation_mappings import (
    AIRLINE_CODES,
    ATC_POSITIONS,
    AVIATION_NUMBERS,
    ICAO_ABBREVIATIONS,
    NUMBERS_TO_WORDS,
    PROPER_NOUNS,
    PUNCTUATION_CLEANUP,
    T1_ALLOWED_PUNCTUATION,
    WORDS_TO_NUMBERS,
)


def clean_punctuation(text: str, allowed_punctuation: set = None) -> str:
    """
    Clean and standardize punctuation in text.
    
    Args:
        text: Input text to clean
        allowed_punctuation: Set of allowed punctuation marks
        
    Returns:
        Cleaned text with standardized punctuation
    """
    if allowed_punctuation is None:
        allowed_punctuation = T1_ALLOWED_PUNCTUATION
    
    # Apply punctuation cleanup rules
    cleaned = text
    for old, new in PUNCTUATION_CLEANUP.items():
        cleaned = cleaned.replace(old, new)
    
    # Remove disallowed punctuation
    result = ""
    for char in cleaned:
        if char.isalnum() or char.isspace() or char in allowed_punctuation:
            result += char
        elif char not in allowed_punctuation:
            # Replace with space if it's not allowed
            result += " "
    
    # Clean up multiple spaces
    result = re.sub(r'\s+', ' ', result.strip())
    
    return result


def apply_proper_casing(text: str) -> str:
    """
    Apply proper capitalization for aviation terminology.
    
    Args:
        text: Input text to capitalize
        
    Returns:
        Text with proper capitalization
    """
    # Convert to lowercase first
    text = text.lower()
    
    # Capitalize sentence beginnings
    sentences = re.split(r'[.!?]+', text)
    capitalized_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            # Capitalize first letter
            sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
            
            # Apply proper noun capitalization
            for lower_word, proper_word in PROPER_NOUNS.items():
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(lower_word) + r'\b'
                sentence = re.sub(pattern, proper_word, sentence, flags=re.IGNORECASE)
            
            capitalized_sentences.append(sentence)
    
    # Rejoin sentences
    result = '. '.join(capitalized_sentences)
    
    # Handle airline codes specifically
    for code, name in AIRLINE_CODES.items():
        pattern = r'\b' + re.escape(code.lower()) + r'\b'
        result = re.sub(pattern, name, result, flags=re.IGNORECASE)
    
    return result


def convert_numbers_to_words(text: str) -> str:
    """
    Convert numerical digits to words for T1 format.
    
    Args:
        text: Input text containing numbers
        
    Returns:
        Text with numbers converted to words
    """
    # Handle aviation-specific number patterns first
    result = text
    
    # Convert individual digits (aviation style)
    # Handle sequences like "209" -> "two zero nine"
    def convert_digit_sequence(match):
        digits = match.group(0)
        words = []
        for digit in digits:
            if digit in NUMBERS_TO_WORDS:
                words.append(NUMBERS_TO_WORDS[digit])
            else:
                words.append(digit)
        return ' '.join(words)
    
    # Convert multi-digit numbers that should be read as individual digits
    # This includes flight numbers, transponder codes, etc.
    result = re.sub(r'\b\d{3,4}\b', convert_digit_sequence, result)
    
    # Convert remaining individual digits
    for digit, word in NUMBERS_TO_WORDS.items():
        if len(digit) == 1:  # Single digits
            result = re.sub(r'\b' + digit + r'\b', word, result)
    
    # Handle special aviation numbers
    for word, digit in AVIATION_NUMBERS.items():
        if digit in NUMBERS_TO_WORDS:
            result = re.sub(r'\b' + re.escape(digit) + r'\b', NUMBERS_TO_WORDS[digit], result)
    
    return result


def convert_words_to_numbers(text: str) -> str:
    """
    Convert word numbers to digits for T2 format.
    
    Args:
        text: Input text containing word numbers
        
    Returns:
        Text with word numbers converted to digits
    """
    result = text
    
    # Handle aviation-specific conversions first
    for word, digit in AVIATION_NUMBERS.items():
        pattern = r'\b' + re.escape(word) + r'\b'
        result = re.sub(pattern, digit, result, flags=re.IGNORECASE)
    
    # Convert word numbers to digits
    for word, digit in WORDS_TO_NUMBERS.items():
        pattern = r'\b' + re.escape(word) + r'\b'
        result = re.sub(pattern, digit, result, flags=re.IGNORECASE)
    
    # Handle sequences like "two zero nine" -> "209"
    # This is tricky - we need to identify aviation contexts
    def convert_digit_words(match):
        words = match.group(0).split()
        digits = []
        for word in words:
            if word.lower() in AVIATION_NUMBERS:
                digits.append(AVIATION_NUMBERS[word.lower()])
            elif word.lower() in WORDS_TO_NUMBERS:
                digits.append(WORDS_TO_NUMBERS[word.lower()])
        return ''.join(digits)
    
    # Pattern for sequences of digit words (aviation style)
    digit_word_pattern = r'\b(?:' + '|'.join(AVIATION_NUMBERS.keys()) + r')(?:\s+(?:' + '|'.join(AVIATION_NUMBERS.keys()) + r'))*\b'
    result = re.sub(digit_word_pattern, convert_digit_words, result, flags=re.IGNORECASE)
    
    return result


def apply_icao_abbreviations(text: str) -> str:
    """
    Apply ICAO abbreviations for T2 format.
    
    Args:
        text: Input text to apply abbreviations to
        
    Returns:
        Text with ICAO abbreviations applied
    """
    result = text
    
    # Apply ICAO abbreviations
    for phrase, abbreviation in ICAO_ABBREVIATIONS.items():
        pattern = r'\b' + re.escape(phrase) + r'\b'
        result = re.sub(pattern, abbreviation, result, flags=re.IGNORECASE)
    
    # Handle runway designations specially
    # "runway one eight left" -> "Runway 18L"
    runway_pattern = r'runway\s+(\w+)\s+(\w+)\s+(\w+)'
    def format_runway(match):
        number1 = match.group(2) 
        number2 = match.group(3)
        
        # Convert runway numbers
        if number1.lower() in AVIATION_NUMBERS and number2.lower() in AVIATION_NUMBERS:
            runway_num = AVIATION_NUMBERS[number1.lower()] + AVIATION_NUMBERS[number2.lower()]
        else:
            runway_num = number1 + number2
            
        # Convert direction
        direction = ""
        if number2.lower() in ['left', 'right', 'center', 'centre']:
            direction = ICAO_ABBREVIATIONS.get(number2.lower(), number2.upper())
        
        return f"Runway {runway_num}{direction}"
    
    result = re.sub(runway_pattern, format_runway, result, flags=re.IGNORECASE)
    
    # Handle flight levels
    fl_pattern = r'flight\s+level\s+(\w+(?:\s+\w+)*)'
    def format_flight_level(match):
        level_words = match.group(1).split()
        level_digits = []
        for word in level_words:
            if word.lower() in AVIATION_NUMBERS:
                level_digits.append(AVIATION_NUMBERS[word.lower()])
            elif word.lower() in WORDS_TO_NUMBERS:
                level_digits.append(WORDS_TO_NUMBERS[word.lower()])
        return f"FL{''.join(level_digits)}"
    
    result = re.sub(fl_pattern, format_flight_level, result, flags=re.IGNORECASE)
    
    return result


def standardize_callsigns(text: str, format_type: str) -> str:
    """
    Standardize aircraft callsigns for the specified format.
    
    Args:
        text: Input text containing callsigns
        format_type: 'T1' or 'T2'
        
    Returns:
        Text with standardized callsigns
    """
    result = text
    
    if format_type == 'T1':
        # Convert airline codes to full names
        for code, name in AIRLINE_CODES.items():
            # Handle patterns like "DAL209" -> "Delta two zero nine"
            pattern = r'\b' + re.escape(code) + r'(\d+)\b'
            def replace_callsign(match):
                number = match.group(1)
                number_words = convert_numbers_to_words(number)
                return f"{name} {number_words}"
            result = re.sub(pattern, replace_callsign, result, flags=re.IGNORECASE)
            
            # Handle patterns like "Delta 2 0 9" -> "Delta two zero nine"
            pattern = r'\b' + re.escape(name) + r'\s+(\d(?:\s+\d)*)\b'
            def replace_spaced_callsign(match):
                numbers = match.group(1).replace(' ', '')
                number_words = convert_numbers_to_words(numbers)
                return f"{name} {number_words}"
            result = re.sub(pattern, replace_spaced_callsign, result, flags=re.IGNORECASE)
    
    elif format_type == 'T2':
        # Convert full names to codes and digits
        for code, name in AIRLINE_CODES.items():
            # Handle patterns like "Delta two zero nine" -> "DAL209"
            pattern = r'\b' + re.escape(name) + r'\s+(\w+(?:\s+\w+)*)\b'
            def replace_callsign(match):
                number_words = match.group(1)
                number_digits = convert_words_to_numbers(number_words)
                return f"{code}{number_digits}"
            result = re.sub(pattern, replace_callsign, result, flags=re.IGNORECASE)
        
        # Handle general aviation callsigns
        # "November one two three" -> "N123"
        ga_pattern = r'\b(november|lima|mike|papa|quebec|romeo|sierra|tango|uniform|victor|whiskey|x-ray|yankee|zulu)\s+(\w+(?:\s+\w+)*)\b'
        def replace_ga_callsign(match):
            prefix = match.group(1)
            number_words = match.group(2)
            prefix_abbrev = ICAO_ABBREVIATIONS.get(prefix.lower(), prefix.upper())
            number_digits = convert_words_to_numbers(number_words)
            return f"{prefix_abbrev}{number_digits}"
        result = re.sub(ga_pattern, replace_ga_callsign, result, flags=re.IGNORECASE)
    
    return result


def format_aviation_numbers(text: str, format_type: str) -> str:
    """
    Format aviation-specific numbers (altitudes, headings, frequencies).
    
    Args:
        text: Input text containing aviation numbers
        format_type: 'T1' or 'T2'
        
    Returns:
        Text with properly formatted aviation numbers
    """
    result = text
    
    if format_type == 'T1':
        # Convert altitudes to words
        # "17000" -> "one seven thousand"
        altitude_pattern = r'\b(\d+)\s*(?:feet|ft|thousand|hundred)?\b'
        def format_altitude_words(match):
            number = match.group(1)
            return convert_numbers_to_words(number)
        result = re.sub(altitude_pattern, format_altitude_words, result)
        
        # Convert headings to words
        # "280" -> "two eight zero"
        heading_pattern = r'\bheading\s+(\d+)\b'
        def format_heading_words(match):
            number = match.group(1)
            return f"heading {convert_numbers_to_words(number)}"
        result = re.sub(heading_pattern, format_heading_words, result, flags=re.IGNORECASE)
        
    elif format_type == 'T2':
        # Convert altitude words to numbers
        # "one seven thousand" -> "17000"
        # This is handled by convert_words_to_numbers
        
        # Convert heading words to numbers
        # "heading two eight zero" -> "heading 280"
        heading_pattern = r'\bheading\s+(\w+(?:\s+\w+)*)\b'
        def format_heading_digits(match):
            words = match.group(1)
            digits = convert_words_to_numbers(words)
            return f"heading {digits}"
        result = re.sub(heading_pattern, format_heading_digits, result, flags=re.IGNORECASE)
    
    return result


def extract_speaker_info(header: str) -> dict:
    """
    Extract speaker information from message header.
    
    Args:
        header: Header string from the message
        
    Returns:
        Dictionary with speaker information
    """
    # Header format: {session_id speaker_turn speaker_identifier start_time end_time}
    # Example: {dca_d1_1 1 dca_d1_1__DR1-1__DAL209__00_01_03 63.040 66.010}
    
    result = {
        'speaker': '',
        'listener': '',
        'speaker_role': '',
        'listener_role': ''
    }
    
    # Extract the identifier part
    match = re.search(r'\{[^}]+\s+\d+\s+([^}]+)\s+[\d.]+\s+[\d.]+\}', header)
    if not match:
        return result
    
    identifier = match.group(1)
    parts = identifier.split('__')
    
    if len(parts) >= 3:
        # Format: session__speaker__listener__time
        result['speaker'] = parts[1]
        result['listener'] = parts[2]
        
        # Determine speaker role
        if any(atc_id in result['speaker'] for atc_id in ATC_POSITIONS.keys()):
            result['speaker_role'] = 'ATC'
        else:
            result['speaker_role'] = 'Pilot'
            
        # Determine listener role  
        if any(atc_id in result['listener'] for atc_id in ATC_POSITIONS.keys()):
            result['listener_role'] = 'ATC'
        else:
            result['listener_role'] = 'Pilot'
    
    return result


def parse_atc_block(block: str) -> dict:
    """
    Parse an ATC communication block into its components.
    
    Args:
        block: Raw ATC block text
        
    Returns:
        Dictionary with parsed components
    """
    lines = block.strip().split('\n')
    result = {}
    
    # Field prefixes
    field_prefixes = {
        'ORIG': 'ORIG:',
        'PROC': 'PROC:',
        'NUMC': 'NUMC:',
        'PUNC': 'PUNC:',
        'NOTE': 'NOTE:',
    }
    
    current_field = None
    for line in lines:
        line = line.strip()
        if line.startswith('{'):
            # Header line
            result['header'] = line
            result['speaker_info'] = extract_speaker_info(line)
        else:
            # Check for field prefixes
            for field, prefix in field_prefixes.items():
                if line.startswith(prefix):
                    current_field = field
                    content = line[len(prefix):].strip()
                    result[field] = content
                    break
            else:
                # Continuation of current field
                if current_field and current_field in result:
                    result[current_field] += '\n' + line
                elif current_field:
                    result[current_field] = line
    
    return result


def validate_t1_format(text: str) -> bool:
    """
    Validate that text conforms to T1 format requirements.
    
    Args:
        text: Text to validate
        
    Returns:
        True if valid T1 format, False otherwise
    """
    # Check punctuation
    for char in text:
        if not char.isalnum() and not char.isspace() and char not in T1_ALLOWED_PUNCTUATION:
            return False
    
    # Check for digits (shouldn't have any in T1)
    if re.search(r'\d', text):
        return False
    
    return True


def validate_t2_format(text: str) -> bool:
    """
    Validate that text conforms to T2 format requirements.
    
    Args:
        text: Text to validate
        
    Returns:
        True if valid T2 format, False otherwise
    """
    # T2 should have digits and abbreviations
    # This is a basic check - more sophisticated validation could be added
    return True  # For now, assume valid


def format_for_output(text: str, format_type: str) -> str:
    """
    Final formatting for output text.
    
    Args:
        text: Input text
        format_type: 'T1' or 'T2'
        
    Returns:
        Formatted text ready for output
    """
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Ensure proper sentence ending
    if text and not text.endswith(('.', '!', '?')):
        text += '.'
    
    return text 