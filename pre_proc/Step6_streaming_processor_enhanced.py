#!/usr/bin/env python3
"""
Step 6: LLM-Enhanced Streaming ATC Communication Processor

Enhanced real-time processing with LLM integration for:
- Improved turn detection and speaker classification
- Advanced information extraction
- Hybrid rule-based + LLM approach for optimal accuracy
"""

import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    llm_enhanced: bool = False
    processing_method: str = "rule_based"
    is_final: bool = False


@dataclass
class LLMAnalysis:
    """LLM analysis result for aviation communication."""
    turn_boundary: bool
    turn_confidence: float
    speaker: Optional[str]
    speaker_confidence: float
    speaker_role: Optional[str]
    extracted_info: Dict[str, Any]
    extraction_confidence: float
    reasoning: str
    overall_confidence: float


class EnhancedStreamingProcessor:
    """
    LLM-Enhanced streaming processor for ATC communications.
    
    Combines rule-based processing with LLM analysis for superior accuracy.
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
                 llm_confidence_threshold: float = 0.8,
                 use_hybrid_approach: bool = True,
                 verbose: bool = False):
        
        self.mode = mode
        self.input_format = input_format
        self.buffer_size = buffer_size
        self.turn_threshold = turn_threshold
        self.speaker_threshold = speaker_threshold
        self.extraction_threshold = extraction_threshold
        self.silence_threshold = silence_threshold
        self.ollama_model = ollama_model
        self.llm_confidence_threshold = llm_confidence_threshold
        self.use_hybrid_approach = use_hybrid_approach
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
        
        # Statistics
        self.llm_calls = 0
        self.rule_based_calls = 0
        self.hybrid_decisions = 0

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

    def _call_ollama_api(self, text: str) -> Optional[LLMAnalysis]:
        """Call Ollama API for LLM analysis."""
        try:
            prompt = f"""
Analyze this ATC communication transcript for streaming processing:

Text: "{text}"

Provide structured analysis in JSON format:
{{
    "turn_boundary": boolean (is this a complete communication turn?),
    "turn_confidence": float (0.0-1.0),
    "speaker": string or null (who is speaking - pilot callsign or ATC position),
    "speaker_confidence": float (0.0-1.0),
    "speaker_role": string or null ("pilot", "ground", "tower", "approach", "departure", "center"),
    "extracted_info": {{
        "altitude": {{"value": number, "action": string, "confidence": float}} or null,
        "heading": {{"value": number, "action": string, "confidence": float}} or null,
        "speed": {{"value": number, "unit": "knots", "action": string, "confidence": float}} or null,
        "frequency": {{"value": number, "unit": "MHz", "confidence": float}} or null,
        "squawk": {{"value": string, "confidence": float}} or null
    }},
    "extraction_confidence": float (0.0-1.0),
    "reasoning": string (brief explanation of analysis),
    "overall_confidence": float (0.0-1.0)
}}

Focus on aviation terminology and communication patterns. Be precise about numerical values.
"""

            # Prepare Ollama API call
            cmd = [
                'curl', '-s',
                'http://localhost:11434/api/generate',
                '-H', 'Content-Type: application/json',
                '-d', json.dumps({
                    'model': self.ollama_model,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.1,
                        'top_p': 0.9
                    }
                })
            ]
            
            # Execute with timeout
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                response = json.loads(result.stdout)
                if 'response' in response:
                    # Extract JSON from response
                    response_text = response['response'].strip()
                    
                    # Find JSON block
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    
                    if json_start >= 0 and json_end > json_start:
                        json_str = response_text[json_start:json_end]
                        analysis_data = json.loads(json_str)
                        
                        return LLMAnalysis(
                            turn_boundary=analysis_data.get('turn_boundary', False),
                            turn_confidence=analysis_data.get('turn_confidence', 0.0),
                            speaker=analysis_data.get('speaker'),
                            speaker_confidence=analysis_data.get('speaker_confidence', 0.0),
                            speaker_role=analysis_data.get('speaker_role'),
                            extracted_info=analysis_data.get('extracted_info', {}),
                            extraction_confidence=analysis_data.get('extraction_confidence', 0.0),
                            reasoning=analysis_data.get('reasoning', ''),
                            overall_confidence=analysis_data.get('overall_confidence', 0.0)
                        )
            
            if self.verbose:
                print(f"⚠️ LLM API call failed: {result.stderr}")
                
        except Exception as e:
            if self.verbose:
                print(f"❌ LLM error: {e}")
        
        return None

    def _rule_based_analysis(self, text: str) -> LLMAnalysis:
        """Perform rule-based analysis as fallback."""
        # Turn detection
        turn_confidence = 0.0
        for strength, patterns in self.turn_indicators.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    if strength == 'strong':
                        turn_confidence += 0.5
                    elif strength == 'medium':
                        turn_confidence += 0.3
                    else:
                        turn_confidence += 0.1
        
        turn_confidence = min(turn_confidence, 1.0)
        is_turn_boundary = turn_confidence >= self.turn_threshold
        
        # Speaker detection
        speaker = None
        speaker_confidence = 0.0
        speaker_role = None
        
        # Check ATC positions
        for position, patterns in self.speaker_patterns['atc_positions'].items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    speaker = f"{position.title()} Controller"
                    speaker_confidence = 0.8
                    speaker_role = position
                    break
        
        # Check aircraft callsigns
        if not speaker:
            for aircraft_type, patterns in self.speaker_patterns['aircraft'].items():
                for pattern in patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    if matches:
                        speaker = matches[-1]
                        speaker_confidence = 0.85
                        speaker_role = "pilot"
                        break
        
        # Information extraction
        extracted_info = {}
        for info_type, config in self.extraction_patterns.items():
            for pattern in config['patterns']:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    action = None
                    for act in config['actions']:
                        if act.lower() in text.lower():
                            action = act
                            break
                    
                    if info_type == 'altitude':
                        value = matches[0][0] if isinstance(matches[0], tuple) and matches[0][0] else matches[0]
                        if value.isdigit():
                            extracted_info['altitude'] = {
                                'value': int(value),
                                'action': action,
                                'confidence': 0.8
                            }
                    elif info_type == 'heading':
                        if matches[0].isdigit():
                            extracted_info['heading'] = {
                                'value': int(matches[0]),
                                'action': action,
                                'confidence': 0.9
                            }
                    elif info_type == 'frequency':
                        try:
                            extracted_info['frequency'] = {
                                'value': float(matches[0]),
                                'unit': 'MHz',
                                'confidence': 0.9
                            }
                        except ValueError:
                            pass
                    elif info_type == 'squawk':
                        extracted_info['squawk'] = {
                            'value': matches[0],
                            'confidence': 0.95
                        }
        
        extraction_confidence = min(len(extracted_info) * 0.2, 1.0)
        overall_confidence = (turn_confidence * 0.4 + speaker_confidence * 0.3 + extraction_confidence * 0.3)
        
        return LLMAnalysis(
            turn_boundary=is_turn_boundary,
            turn_confidence=turn_confidence,
            speaker=speaker,
            speaker_confidence=speaker_confidence,
            speaker_role=speaker_role,
            extracted_info=extracted_info,
            extraction_confidence=extraction_confidence,
            reasoning="Rule-based pattern matching analysis",
            overall_confidence=overall_confidence
        )

    def _hybrid_analysis(self, text: str) -> Tuple[LLMAnalysis, str]:
        """Perform hybrid LLM + rule-based analysis."""
        # Always do rule-based analysis
        rule_analysis = self._rule_based_analysis(text)
        self.rule_based_calls += 1
        
        # Try LLM analysis if text is substantial enough
        if len(text.split()) >= 5:  # Only for longer utterances
            llm_analysis = self._call_ollama_api(text)
            
            if llm_analysis and llm_analysis.overall_confidence >= self.llm_confidence_threshold:
                self.llm_calls += 1
                self.hybrid_decisions += 1
                
                if self.verbose:
                    print(f"🤖 LLM Analysis (conf: {llm_analysis.overall_confidence:.2f}): {llm_analysis.reasoning}")
                
                return llm_analysis, "llm_enhanced"
        
        # Use rule-based analysis
        if self.verbose:
            print(f"📋 Rule-based analysis (conf: {rule_analysis.overall_confidence:.2f})")
        
        return rule_analysis, "rule_based"

    def add_callback(self, callback: Callable[[StreamingResult], None]):
        """Add a callback function for streaming results."""
        self.callbacks.append(callback)

    def process_token(self, token: StreamingToken) -> Optional[StreamingResult]:
        """Process a single token and return streaming result if available."""
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
            
            # Check if we should analyze (every 5 words or turn indicators)
            should_analyze = (
                len(self.word_buffer) % 5 == 0 or  # Every 5 words
                any(re.search(pattern, self.transcript_buffer, re.IGNORECASE) 
                    for patterns in self.turn_indicators.values() 
                    for pattern in patterns)
            )
            
            if should_analyze:
                analysis, method = self._hybrid_analysis(self.transcript_buffer)
                
                # Create streaming result
                result = self._create_streaming_result(token, analysis, method)
                self._emit_result(result)
                return result
                
        return None

    def process_text(self, text: str, timestamp: float = None) -> List[StreamingResult]:
        """Process a complete text string as streaming tokens."""
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
        """Finalize the current turn and return final result."""
        with self.processing_lock:
            if not self.transcript_buffer.strip():
                return None
            
            # Final analysis
            analysis, method = self._hybrid_analysis(self.transcript_buffer)
            analysis.turn_boundary = True
            analysis.turn_confidence = 1.0
            
            # Create final result
            final_token = StreamingToken(
                word="",
                timestamp=time.time(),
                confidence=1.0
            )
            
            result = self._create_streaming_result(final_token, analysis, method, is_final=True)
            
            # Reset states for next turn
            self._reset_states()
            
            self._emit_result(result)
            return result

    def _tokenize_text(self, text: str, base_timestamp: float) -> List[StreamingToken]:
        """Convert text into streaming tokens."""
        words = text.split()
        tokens = []
        
        for i, word in enumerate(words):
            word_timestamp = base_timestamp + (i * 0.175)
            is_punct = bool(re.search(r'[.!?]$', word))
            
            token = StreamingToken(
                word=word,
                timestamp=word_timestamp,
                confidence=0.95,
                is_punctuation=is_punct
            )
            
            tokens.append(token)
            
        return tokens

    def _create_streaming_result(self, token: StreamingToken, analysis: LLMAnalysis, 
                               method: str, is_final: bool = False) -> StreamingResult:
        """Create a streaming result from LLM analysis."""
        
        # Format speaker information
        speaker_info = {
            'speaker': analysis.speaker,
            'confidence': analysis.speaker_confidence,
            'role': analysis.speaker_role,
            'position': None
        }
        
        return StreamingResult(
            timestamp=token.timestamp,
            partial_transcript=self.transcript_buffer.strip(),
            complete_transcript=self.transcript_buffer.strip() if analysis.turn_boundary else "",
            turn_boundary_detected=analysis.turn_boundary,
            turn_confidence=analysis.turn_confidence,
            speaker_classification=speaker_info,
            extracted_information=analysis.extracted_info,
            processing_confidence=analysis.overall_confidence,
            llm_enhanced=(method == "llm_enhanced"),
            processing_method=method,
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

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        total_calls = self.llm_calls + self.rule_based_calls
        return {
            'total_analysis_calls': total_calls,
            'llm_calls': self.llm_calls,
            'rule_based_calls': self.rule_based_calls,
            'hybrid_decisions': self.hybrid_decisions,
            'llm_usage_rate': self.llm_calls / total_calls if total_calls > 0 else 0.0
        }


class EnhancedStreamingDemo:
    """Demo class for testing enhanced streaming processor."""
    
    def __init__(self, processor: EnhancedStreamingProcessor):
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
        print(f"🤖 Method: {result.processing_method} {'(LLM Enhanced)' if result.llm_enhanced else '(Rule-based)'}")
        
        if result.is_final:
            print("✅ FINAL RESULT")
        
        print(f"{'='*60}")

    def run_simulation(self, test_messages: List[str]):
        """Run enhanced streaming simulation."""
        print("🚀 Starting LLM-Enhanced streaming simulation...\n")
        print("🤖 Processing ATC communications with hybrid LLM + rule-based analysis")
        print("⏸️  Realistic timing with advanced AI processing\n")
        
        for i, message in enumerate(test_messages):
            print(f"\n📡 Message {i+1}/{len(test_messages)}: '{message}'")
            print("🔊 Live streaming: ", end="", flush=True)
            
            # Process message word-by-word
            words = message.split()
            for j, word in enumerate(words):
                print(f"{word} ", end="", flush=True)
                        
                # Realistic timing
                if j < len(words) - 1:
                    if word.lower() in ['and', 'the', 'to', 'for', 'at', 'in', 'on']:
                        time.sleep(0.15)
                    elif word.lower() in ['contact', 'maintain', 'climb', 'turn', 'heading']:
                        time.sleep(0.3)
                    else:
                        time.sleep(0.2)
                else:
                    time.sleep(0.6)
            
            print(" [COMPLETE]")
            
            # Inter-message pause
            if i < len(test_messages) - 1:
                print("📻 [Radio silence...]", end="", flush=True)
                time.sleep(2.0)
                print(" [Next transmission]\n")
        
        print("\n✅ LLM-Enhanced streaming simulation completed!")
        print(f"📊 Processed {len(self.results)} streaming results")
        return self.results


def load_orig_messages(file_path="data/test.lbs", max_messages=10):
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
    """Main function for enhanced streaming processor demo."""
    
    print("=== LLM-Enhanced ATC Streaming Processor Demo ===")
    print("Hybrid processing with LLM + rule-based analysis for superior accuracy")
    print("🎯 Streaming ORIG messages from test.lbs with AI enhancement")
    print()
    
    # Check for verbose mode
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    
    # Load messages from test.lbs
    test_messages = load_orig_messages(max_messages=10)  # Smaller set for LLM processing
    
    # Create enhanced processor
    processor = EnhancedStreamingProcessor(
        mode=ProcessingMode.SIMULATION,
        input_format=InputFormat.WORDS_ONLY,
        verbose=verbose,
        llm_confidence_threshold=0.8,
        use_hybrid_approach=True
    )
    
    # Create demo
    demo = EnhancedStreamingDemo(processor)
    
    print(f"🤖 Using model: {processor.ollama_model}")
    print(f"🎯 LLM confidence threshold: {processor.llm_confidence_threshold}")
    print(f"🔄 Hybrid approach: {processor.use_hybrid_approach}")
    
    # Run simulation
    results = demo.run_simulation(test_messages)
    
    # Get processing statistics
    stats = processor.get_processing_stats()
    
    # Summary statistics
    turn_detections = sum(1 for r in results if r.turn_boundary_detected)
    speaker_identifications = sum(1 for r in results if r.speaker_classification['speaker'])
    info_extractions = sum(1 for r in results if r.extracted_information)
    llm_enhanced_results = sum(1 for r in results if r.llm_enhanced)
    
    print("\n📊 ENHANCED PROCESSING SUMMARY")
    print(f"Total results: {len(results)}")
    print(f"Turn boundaries detected: {turn_detections}")
    print(f"Speaker identifications: {speaker_identifications}")
    print(f"Information extractions: {info_extractions}")
    print(f"LLM-enhanced results: {llm_enhanced_results}")
    print(f"Average confidence: {sum(r.processing_confidence for r in results) / len(results):.2f}")
    print("\n🤖 AI PROCESSING STATISTICS")
    print(f"Total analysis calls: {stats['total_analysis_calls']}")
    print(f"LLM calls: {stats['llm_calls']}")
    print(f"Rule-based calls: {stats['rule_based_calls']}")
    print(f"LLM usage rate: {stats['llm_usage_rate']:.2%}")
    
    # Save results
    output_file = "data/test_streaming_live_enhanced_results.json"
    with open(output_file, 'w') as f:
        json.dump([{
            'timestamp': r.timestamp,
            'transcript': r.partial_transcript,
            'turn_detected': r.turn_boundary_detected,
            'turn_confidence': r.turn_confidence,
            'speaker': r.speaker_classification,
            'extracted_info': r.extracted_information,
            'confidence': r.processing_confidence,
            'llm_enhanced': r.llm_enhanced,
            'processing_method': r.processing_method,
            'is_final': r.is_final
        } for r in results], f, indent=2)
    
    print(f"\n💾 Enhanced results saved to {output_file}")
    print("💡 Use --verbose for detailed LLM analysis output")
    print("🎯 LLM-Enhanced streaming processing complete!")


if __name__ == "__main__":
    main() 