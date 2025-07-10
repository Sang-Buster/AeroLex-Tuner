#!/usr/bin/env python3
"""
Step 6: Streaming ATC Communication Processor

Real-time processing capabilities for aviation communication with:
- Turn detection and speaker classification
- Information extraction features
- Streaming word-by-word input processing
- Dynamic confidence updates

Features:
- Real-time ASR output processing (word-by-word)
- Turn boundary detection (with/without punctuation)
- Speaker classification with state tracking
- Information extraction (altitude, heading, speed, frequency, squawk)
- Confidence-based feedback filtering
"""

import json
import os
import re
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

# Add utils to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.speaker_state_tracker import SpeakerRole, SpeakerStateTracker


class ProcessingMode(Enum):
    """Processing mode options."""
    REAL_TIME = "real_time"
    BATCH = "batch"
    SIMULATION = "simulation"


class InputFormat(Enum):
    """Input format options."""
    WORDS_ONLY = "words_only"
    WITH_PUNCTUATION = "with_punctuation"
    WITH_SPEAKER_LABELS = "with_speaker_labels"
    FULL_DIARIZATION = "full_diarization"


@dataclass
class StreamingToken:
    """A single token in the streaming input."""
    word: str
    timestamp: float
    confidence: float = 1.0
    speaker_id: Optional[str] = None
    is_punctuation: bool = False


@dataclass
class TurnBoundaryState:
    """State for turn boundary detection."""
    current_confidence: float = 0.0
    boundary_indicators: List[str] = field(default_factory=list)
    word_count: int = 0
    silence_duration: float = 0.0
    last_word_time: Optional[float] = None
    has_end_punctuation: bool = False


@dataclass
class SpeakerState:
    """State for speaker classification."""
    current_speaker: Optional[str] = None
    speaker_confidence: float = 0.0
    role: Optional[SpeakerRole] = None
    position: Optional[str] = None
    flight_phase: Optional[str] = None
    conversation_context: List[str] = field(default_factory=list)


@dataclass
class InformationState:
    """State for information extraction."""
    altitude: Optional[Dict] = None
    heading: Optional[Dict] = None
    speed: Optional[Dict] = None
    frequency: Optional[Dict] = None
    squawk: Optional[str] = None
    extraction_confidence: float = 0.0
    partial_extractions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamingResult:
    """Result from streaming processing."""
    timestamp: float
    partial_transcript: str
    complete_transcript: str
    turn_boundary_detected: bool
    turn_confidence: float
    speaker_classification: Dict[str, Any]
    extracted_information: Dict[str, Any]
    processing_confidence: float
    is_final: bool = False


class StreamingProcessor:
    """
    Real-time streaming processor for ATC communications.
    
    Processes word-by-word input and provides real-time:
    - Turn boundary detection
    - Speaker classification 
    - Information extraction
    """
    
    def __init__(self, 
                 mode: ProcessingMode = ProcessingMode.REAL_TIME,
                 input_format: InputFormat = InputFormat.WORDS_ONLY,
                 buffer_size: int = 50,
                 turn_threshold: float = 0.7,
                 speaker_threshold: float = 0.6,
                 extraction_threshold: float = 0.5,
                 silence_threshold: float = 2.0,
                 ollama_model: str = "llama3.3:70b-instruct-q4_K_M",
                 use_llm: bool = True,
                 verbose: bool = False):
        
        self.mode = mode
        self.input_format = input_format
        self.buffer_size = buffer_size
        self.turn_threshold = turn_threshold
        self.speaker_threshold = speaker_threshold
        self.extraction_threshold = extraction_threshold
        self.silence_threshold = silence_threshold
        self.ollama_model = ollama_model
        self.use_llm = use_llm
        self.verbose = verbose
        
        # Streaming buffers
        self.token_buffer = deque(maxlen=buffer_size)
        self.word_buffer = deque(maxlen=buffer_size)
        self.transcript_buffer = ""
        
        # Processing states
        self.turn_state = TurnBoundaryState()
        self.speaker_state = SpeakerState()
        self.info_state = InformationState()
        
        # Utilities
        self.speaker_tracker = SpeakerStateTracker()
        
        # Callbacks
        self.callbacks: List[Callable[[StreamingResult], None]] = []
        
        # Processing patterns
        self._init_patterns()
        
        # Threading for real-time processing
        self.processing_thread = None
        self.is_processing = False
        self.processing_lock = threading.Lock()

    def _init_patterns(self):
        """Initialize processing patterns."""
        
        # Turn boundary indicators
        self.turn_indicators = {
            'strong': [
                r'good day\b', r'so long\b', r'good afternoon\b', r'good morning\b',
                r'contact\s+\w+', r'thank you\b', r'roger\b', r'wilco\b',
                r'have a good\b', r'see you\b'
            ],
            'medium': [
                r'\.$', r'\?$', r'!$',  # End punctuation
                r'maintain\s+\d+', r'climb\s+and\s+maintain',
                r'turn\s+\w+\s+heading', r'proceed\s+direct'
            ],
            'weak': [
                r'correct\b', r'affirmative\b', r'negative\b',
                r'say again\b', r'standby\b'
            ]
        }
        
        # Information extraction patterns
        self.extraction_patterns = {
            'altitude': {
                'patterns': [
                    r'(?:climb|descend|maintain)\s+(?:and\s+maintain\s+)?(?:flight\s+level\s+)?(\d+)(?:\s+(?:thousand|hundred))?',
                    r'(?:altitude|level)\s+(\d+)',
                    r'FL(\d+)',
                    r'(\d+)(?:\s+thousand)?\s+feet',
                    r'out\s+of\s+(\d+)(?:\s+thousand)?(?:\s+for\s+(\d+)(?:\s+thousand)?)?'
                ],
                'actions': ['climb', 'descend', 'maintain', 'out of', 'for']
            },
            'heading': {
                'patterns': [
                    r'(?:turn\s+)?(?:left|right)\s+heading\s+(\d+)',
                    r'fly\s+heading\s+(\d+)',
                    r'heading\s+(\d+)',
                    r'turn\s+(\d+)'
                ],
                'actions': ['turn left', 'turn right', 'fly']
            },
            'speed': {
                'patterns': [
                    r'(?:maintain|reduce|increase)\s+(?:speed\s+)?(\d+)\s*knots?',
                    r'speed\s+(\d+)',
                    r'(\d+)\s*knots?'
                ],
                'actions': ['maintain', 'reduce', 'increase']
            },
            'frequency': {
                'patterns': [
                    r'contact\s+\w+\s+(\d+\.?\d*)',
                    r'(\d{3}\.\d+)',
                    r'frequency\s+(\d+\.?\d*)',
                    r'(\d+\.\d+)'
                ],
                'actions': ['contact', 'monitor', 'standby']
            },
            'squawk': {
                'patterns': [
                    r'squawk\s+(\d{4})',
                    r'transponder\s+(\d{4})',
                    r'code\s+(\d{4})'
                ],
                'actions': ['squawk', 'reset', 'ident']
            }
        }
        
        # Speaker classification patterns
        self.speaker_patterns = {
            'atc_positions': {
                'ground': [r'ground\b', r'ramp\b', r'clearance\b'],
                'tower': [r'tower\b', r'local\b'],
                'approach': [r'approach\b', r'arrival\b'],
                'departure': [r'departure\b', r'dep\b'],
                'center': [r'center\b', r'control\b']
            },
            'aircraft': {
                'commercial': [r'[A-Z]{2,3}\d{2,4}', r'(american|delta|united|southwest|jetblue)\s+\d+'],
                'general_aviation': [r'N\d+[A-Z]*', r'november\s+\d+'],
                'military': [r'(army|navy|air force|marine)\s+\d+']
            }
        }

    def add_callback(self, callback: Callable[[StreamingResult], None]):
        """Add a callback function for streaming results."""
        self.callbacks.append(callback)

    def remove_callback(self, callback: Callable[[StreamingResult], None]):
        """Remove a callback function."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    def start_processing(self):
        """Start real-time processing in a separate thread."""
        if self.mode == ProcessingMode.REAL_TIME and not self.is_processing:
            self.is_processing = True
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
            if self.verbose:
                print("🚀 Started real-time processing thread")

    def stop_processing(self):
        """Stop real-time processing."""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
            if self.verbose:
                print("⏹️ Stopped real-time processing")

    def process_token(self, token: StreamingToken) -> Optional[StreamingResult]:
        """
        Process a single token and return streaming result if available.
        
        Args:
            token: StreamingToken with word, timestamp, confidence, etc.
            
        Returns:
            StreamingResult if processing yields meaningful output
        """
        with self.processing_lock:
            # Add token to buffer
            self.token_buffer.append(token)
            self.word_buffer.append(token.word)
            
            # Update transcript buffer with proper spacing
            if self.transcript_buffer:
                self.transcript_buffer += f" {token.word}"
            else:
                self.transcript_buffer = token.word
            
            if self.verbose:
                print(f"📝 Token: '{token.word}' (conf: {token.confidence:.2f}) at {token.timestamp:.1f}s")
            
            # Update processing states
            self._update_turn_state(token)
            self._update_speaker_state(token)
            self._update_information_state(token)
            
            # Determine if we should emit a result
            should_emit = self._should_emit_result(token)
            
            if should_emit:
                result = self._create_streaming_result(token)
                self._emit_result(result)
                return result
                
        return None

    def process_text(self, text: str, timestamp: float = None) -> List[StreamingResult]:
        """
        Process a complete text string as streaming tokens.
        
        Args:
            text: Input text to process
            timestamp: Base timestamp (current time if None)
            
        Returns:
            List of StreamingResults
        """
        if timestamp is None:
            timestamp = time.time()
            
        # Tokenize text
        tokens = self._tokenize_text(text, timestamp)
        
        results = []
        for token in tokens:
            result = self.process_token(token)
            if result:
                results.append(result)
                
        return results

    def finalize_turn(self) -> Optional[StreamingResult]:
        """
        Finalize the current turn and return final result.
        
        Returns:
            Final StreamingResult for the completed turn
        """
        with self.processing_lock:
            if not self.transcript_buffer.strip():
                return None
                
            # Force turn boundary detection
            self.turn_state.current_confidence = 1.0
            
            # Create final result
            final_token = StreamingToken(
                word="",
                timestamp=time.time(),
                confidence=1.0
            )
            
            result = self._create_streaming_result(final_token, is_final=True)
            
            # Reset states for next turn
            self._reset_states()
            
            self._emit_result(result)
            return result

    def _tokenize_text(self, text: str, base_timestamp: float) -> List[StreamingToken]:
        """Convert text into streaming tokens."""
        words = text.split()
        tokens = []
        
        for i, word in enumerate(words):
            # Simulate realistic timing (150-200ms per word)
            word_timestamp = base_timestamp + (i * 0.175)
            
            # Detect punctuation
            is_punct = bool(re.search(r'[.!?]$', word))
            
            # Create token
            token = StreamingToken(
                word=word,
                timestamp=word_timestamp,
                confidence=0.95,  # High confidence for simulated input
                is_punctuation=is_punct
            )
            
            tokens.append(token)
            
        return tokens

    def _update_turn_state(self, token: StreamingToken):
        """Update turn boundary detection state."""
        # Update word count and timing
        self.turn_state.word_count += 1
        self.turn_state.last_word_time = token.timestamp
        
        # Check for punctuation
        if token.is_punctuation or re.search(r'[.!?]$', token.word):
            self.turn_state.has_end_punctuation = True
            self.turn_state.current_confidence += 0.4
        
        # Check for turn indicators
        current_text = " ".join(list(self.word_buffer))
        
        for strength, patterns in self.turn_indicators.items():
            for pattern in patterns:
                if re.search(pattern, current_text, re.IGNORECASE):
                    if strength == 'strong':
                        self.turn_state.current_confidence += 0.5
                    elif strength == 'medium':
                        self.turn_state.current_confidence += 0.3
                    else:  # weak
                        self.turn_state.current_confidence += 0.1
                    
                    self.turn_state.boundary_indicators.append(pattern)
        
        # Silence detection (for real-time mode)
        if (self.mode == ProcessingMode.REAL_TIME and 
            self.turn_state.last_word_time and 
            token.timestamp - self.turn_state.last_word_time > self.silence_threshold):
            self.turn_state.silence_duration = token.timestamp - self.turn_state.last_word_time
            self.turn_state.current_confidence += 0.3
        
        # Cap confidence at 1.0
        self.turn_state.current_confidence = min(self.turn_state.current_confidence, 1.0)
        
        if self.verbose and self.turn_state.current_confidence > 0.5:
            print(f"🔄 Turn boundary confidence: {self.turn_state.current_confidence:.2f}")

    def _update_speaker_state(self, token: StreamingToken):
        """Update speaker classification state."""
        current_text = " ".join(list(self.word_buffer))
        
        # Use explicit speaker ID if available
        if token.speaker_id:
            self.speaker_state.current_speaker = token.speaker_id
            self.speaker_state.speaker_confidence = 0.9
            return
        
        # Pattern-based speaker detection
        max_confidence = 0.0
        detected_speaker = None
        detected_role = None
        
        # Check for ATC positions
        for position, patterns in self.speaker_patterns['atc_positions'].items():
            for pattern in patterns:
                if re.search(pattern, current_text, re.IGNORECASE):
                    confidence = 0.8
                    if confidence > max_confidence:
                        max_confidence = confidence
                        detected_speaker = f"{position.title()} Controller"
                        # Map to correct SpeakerRole enum values
                        if position == 'ground':
                            detected_role = SpeakerRole.GROUND
                        elif position == 'tower':
                            detected_role = SpeakerRole.TOWER
                        elif position == 'approach':
                            detected_role = SpeakerRole.APPROACH
                        elif position == 'departure':
                            detected_role = SpeakerRole.DEPARTURE
                        elif position == 'center':
                            detected_role = SpeakerRole.CENTER
                        else:
                            detected_role = SpeakerRole.UNKNOWN
        
        # Check for aircraft callsigns
        for aircraft_type, patterns in self.speaker_patterns['aircraft'].items():
            for pattern in patterns:
                matches = re.findall(pattern, current_text, re.IGNORECASE)
                if matches:
                    confidence = 0.85
                    if confidence > max_confidence:
                        max_confidence = confidence
                        detected_speaker = matches[-1]  # Use the last match
                        detected_role = SpeakerRole.PILOT
        
        # Update state if confidence is high enough
        if max_confidence > self.speaker_threshold:
            self.speaker_state.current_speaker = detected_speaker
            self.speaker_state.speaker_confidence = max_confidence
            self.speaker_state.role = detected_role
        
        if self.verbose and self.speaker_state.current_speaker:
            print(f"👤 Speaker: {self.speaker_state.current_speaker} (conf: {self.speaker_state.speaker_confidence:.2f})")

    def _update_information_state(self, token: StreamingToken):
        """Update information extraction state."""
        current_text = " ".join(list(self.word_buffer))
        
        # Extract information using patterns
        for info_type, config in self.extraction_patterns.items():
            for pattern in config['patterns']:
                matches = re.findall(pattern, current_text, re.IGNORECASE)
                if matches:
                    # Determine action
                    action = None
                    for act in config['actions']:
                        if act.lower() in current_text.lower():
                            action = act
                            break
                    
                    # Store extraction
                    if info_type == 'altitude':
                        if isinstance(matches[0], tuple):
                            # Handle patterns with multiple capture groups
                            value = matches[0][0] if matches[0][0] else matches[0][1]
                        else:
                            value = matches[0]
                        
                        self.info_state.altitude = {
                            'value': int(value) if value.isdigit() else value,
                            'action': action,
                            'confidence': 0.8,
                            'raw_text': current_text
                        }
                    
                    elif info_type == 'heading':
                        self.info_state.heading = {
                            'value': int(matches[0]) if matches[0].isdigit() else matches[0],
                            'action': action,
                            'confidence': 0.9,
                            'raw_text': current_text
                        }
                    
                    elif info_type == 'speed':
                        self.info_state.speed = {
                            'value': int(matches[0]) if matches[0].isdigit() else matches[0],
                            'unit': 'knots',
                            'action': action,
                            'confidence': 0.8,
                            'raw_text': current_text
                        }
                    
                    elif info_type == 'frequency':
                        self.info_state.frequency = {
                            'value': float(matches[0]) if '.' in matches[0] else int(matches[0]),
                            'unit': 'MHz',
                            'confidence': 0.9,
                            'raw_text': current_text
                        }
                    
                    elif info_type == 'squawk':
                        self.info_state.squawk = {
                            'value': matches[0],
                            'confidence': 0.95,
                            'raw_text': current_text
                        }
        
        # Update extraction confidence
        extracted_count = sum(1 for attr in ['altitude', 'heading', 'speed', 'frequency', 'squawk'] 
                            if getattr(self.info_state, attr) is not None)
        self.info_state.extraction_confidence = min(extracted_count * 0.2, 1.0)
        
        if self.verbose and extracted_count > 0:
            print(f"📊 Info extracted: {extracted_count} items (conf: {self.info_state.extraction_confidence:.2f})")

    def _should_emit_result(self, token: StreamingToken) -> bool:
        """Determine if we should emit a streaming result."""
        # Always emit if turn boundary is detected
        if self.turn_state.current_confidence >= self.turn_threshold:
            return True
        
        # Emit periodically for long utterances
        if self.turn_state.word_count > 0 and self.turn_state.word_count % 10 == 0:
            return True
        
        # Emit if significant information is extracted
        if self.info_state.extraction_confidence >= self.extraction_threshold:
            return True
        
        # Emit if speaker is confidently identified
        if self.speaker_state.speaker_confidence >= self.speaker_threshold:
            return True
        
        return False

    def _create_streaming_result(self, token: StreamingToken, is_final: bool = False) -> StreamingResult:
        """Create a streaming result from current state."""
        
        # Determine if this is a complete turn
        is_turn_complete = (self.turn_state.current_confidence >= self.turn_threshold or is_final)
        
        # Format speaker information
        speaker_info = {
            'speaker': self.speaker_state.current_speaker,
            'confidence': self.speaker_state.speaker_confidence,
            'role': self.speaker_state.role.value if self.speaker_state.role else None,
            'position': self.speaker_state.position
        }
        
        # Format extracted information
        extracted_info = {}
        if self.info_state.altitude:
            extracted_info['altitude'] = self.info_state.altitude
        if self.info_state.heading:
            extracted_info['heading'] = self.info_state.heading
        if self.info_state.speed:
            extracted_info['speed'] = self.info_state.speed
        if self.info_state.frequency:
            extracted_info['frequency'] = self.info_state.frequency
        if self.info_state.squawk:
            extracted_info['squawk'] = self.info_state.squawk
        
        # Calculate overall processing confidence
        processing_confidence = (
            self.turn_state.current_confidence * 0.4 +
            self.speaker_state.speaker_confidence * 0.3 +
            self.info_state.extraction_confidence * 0.3
        )
        
        return StreamingResult(
            timestamp=token.timestamp,
            partial_transcript=self.transcript_buffer.strip(),
            complete_transcript=self.transcript_buffer.strip() if is_turn_complete else "",
            turn_boundary_detected=is_turn_complete,
            turn_confidence=self.turn_state.current_confidence,
            speaker_classification=speaker_info,
            extracted_information=extracted_info,
            processing_confidence=processing_confidence,
            is_final=is_final
        )

    def _emit_result(self, result: StreamingResult):
        """Emit result to all registered callbacks."""
        for callback in self.callbacks:
            try:
                callback(result)
            except Exception as e:
                if self.verbose:
                    print(f"❌ Callback error: {e}")

    def _reset_states(self):
        """Reset processing states for next turn."""
        self.transcript_buffer = ""
        self.token_buffer.clear()
        self.word_buffer.clear()
        
        self.turn_state = TurnBoundaryState()
        self.speaker_state = SpeakerState()
        self.info_state = InformationState()

    def _processing_loop(self):
        """Main processing loop for real-time mode."""
        while self.is_processing:
            time.sleep(0.1)  # 100ms processing interval
            
            # Check for silence timeout
            if (self.turn_state.last_word_time and 
                time.time() - self.turn_state.last_word_time > self.silence_threshold):
                
                # Auto-finalize turn due to silence
                if self.transcript_buffer.strip():
                    self.finalize_turn()

    def get_current_state(self) -> Dict[str, Any]:
        """Get current processing state for debugging."""
        return {
            'transcript': self.transcript_buffer,
            'word_count': self.turn_state.word_count,
            'turn_confidence': self.turn_state.current_confidence,
            'speaker': self.speaker_state.current_speaker,
            'speaker_confidence': self.speaker_state.speaker_confidence,
            'extracted_info': {
                'altitude': self.info_state.altitude,
                'heading': self.info_state.heading,
                'speed': self.info_state.speed,
                'frequency': self.info_state.frequency,
                'squawk': self.info_state.squawk
            },
            'extraction_confidence': self.info_state.extraction_confidence
        }


class StreamingDemo:
    """Demo class for testing streaming processor."""
    
    def __init__(self, processor: StreamingProcessor):
        self.processor = processor
        self.results = []
        
        # Register callback
        self.processor.add_callback(self.on_result)
    
    def on_result(self, result: StreamingResult):
        """Handle streaming results."""
        self.results.append(result)
        
        print(f"\n{'='*60}")
        print(f"⏰ {result.timestamp:.1f}s | Turn: {'✅' if result.turn_boundary_detected else '⏳'} ({result.turn_confidence:.2f})")
        print(f"📝 Transcript: {result.partial_transcript}")
        
        if result.speaker_classification['speaker']:
            print(f"👤 Speaker: {result.speaker_classification['speaker']} ({result.speaker_classification['confidence']:.2f})")
        
        if result.extracted_information:
            print(f"📊 Extracted: {result.extracted_information}")
        
        print(f"🎯 Confidence: {result.processing_confidence:.2f}")
        
        if result.is_final:
            print("✅ FINAL RESULT")
        
        print(f"{'='*60}")

    def run_simulation(self, test_messages: List[str]):
        """Run simulation with test messages - live streaming word-by-word."""
        print("🚀 Starting live streaming simulation...\n")
        print("📻 Processing ATC communications word-by-word in real-time")
        print("⏸️  Pauses simulate natural speech patterns\n")
        
        for i, message in enumerate(test_messages):
            print(f"\n📡 Message {i+1}/{len(test_messages)}: '{message}'")
            print("🔊 Live streaming: ", end="", flush=True)
            
            # Process message word-by-word with realistic timing
            words = message.split()
            for j, word in enumerate(words):
                # Print word as it's "spoken"
                print(f"{word} ", end="", flush=True)
                
                # Realistic speech timing with pauses
                if j < len(words) - 1:  # Not the last word
                    if word.lower() in ['and', 'the', 'to', 'for', 'at', 'in', 'on']:
                        time.sleep(0.15)  # Short pause after function words
                    elif word.lower() in ['contact', 'maintain', 'climb', 'turn', 'heading']:
                        time.sleep(0.3)   # Medium pause after action words
                    elif word.endswith(','):
                        time.sleep(0.4)   # Longer pause after commas
                    else:
                        time.sleep(0.2)   # Standard pause
                else:
                    time.sleep(0.6)  # End of message pause

            print(" [COMPLETE]")
            
            # Inter-message pause (simulating radio silence)
            if i < len(test_messages) - 1:
                print("📻 [Radio silence...]", end="", flush=True)
                time.sleep(2.0)  # 2 second pause between messages
                print(" [Next transmission]\n")
        
        print("\n✅ Live streaming simulation completed!")
        print(f"📊 Processed {len(self.results)} streaming results")
        return self.results


def load_orig_messages(file_path="data/test.lbs", max_messages=20):
    """Load ORIG messages from test.lbs file."""
    messages = []
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Extract ORIG lines
        orig_lines = re.findall(r'ORIG: (.+)', content)
        
        # Take first max_messages for demo
        messages = orig_lines[:max_messages]
        
        print(f"📁 Loaded {len(messages)} ORIG messages from {file_path}")
        
    except FileNotFoundError:
        print(f"⚠️  File {file_path} not found, using default messages")
        messages = [
            "Delta two zero nine turn left heading two eight zero",
            "Left to two eight zero Delta two zero nine",
            "American fifteen eighty one contact approach one two four point two",
            "One two four point two American fifteen eighty one",
            "N99G climb and maintain six thousand",
            "Up to six thousand November nine nine golf",
            "Washington departure radar contact climb and maintain one seven thousand",
            "USAir two thirty seven squawk four five six seven",
            "Four five six seven USAir two thirty seven good day"
        ]
    
    return messages


def main():
    """Main function for streaming processor demo."""
    
    print("=== ATC Streaming Processor Demo ===")
    print("Real-time processing with turn detection, speaker classification, and information extraction")
    print("🎯 Streaming ORIG messages from test.lbs word-by-word")
    print()
    
    # Check for verbose mode
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    
    # Load messages from test.lbs
    test_messages = load_orig_messages()
    
    # Create processor
    processor = StreamingProcessor(
        mode=ProcessingMode.SIMULATION,
        input_format=InputFormat.WORDS_ONLY,
        verbose=verbose,
        use_llm=False  # Use rule-based for demo speed
    )
    
    # Create demo with live streaming
    demo = StreamingDemo(processor)
    
    # Run simulation
    results = demo.run_simulation(test_messages)
    
    # Summary statistics
    turn_detections = sum(1 for r in results if r.turn_boundary_detected)
    speaker_identifications = sum(1 for r in results if r.speaker_classification['speaker'])
    info_extractions = sum(1 for r in results if r.extracted_information)
    
    print("\n📊 SUMMARY STATISTICS")
    print(f"Total results: {len(results)}")
    print(f"Turn boundaries detected: {turn_detections}")
    print(f"Speaker identifications: {speaker_identifications}")
    print(f"Information extractions: {info_extractions}")
    print(f"Average confidence: {sum(r.processing_confidence for r in results) / len(results):.2f}")
    
    # Save results
    output_file = "data/test_streaming_live_results.json"
    with open(output_file, 'w') as f:
        json.dump([{
            'timestamp': r.timestamp,
            'transcript': r.partial_transcript,
            'turn_detected': r.turn_boundary_detected,
            'turn_confidence': r.turn_confidence,
            'speaker': r.speaker_classification,
            'extracted_info': r.extracted_information,
            'confidence': r.processing_confidence,
            'is_final': r.is_final
        } for r in results], f, indent=2)
    
    print(f"\n💾 Results saved to {output_file}")
    print("💡 Use --verbose for detailed processing output")
    print("🎯 Live streaming from test.lbs ORIG records complete!")


if __name__ == "__main__":
    main() 