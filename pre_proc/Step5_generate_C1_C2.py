#!/usr/bin/env python3
"""
Step 5: Continuous Format Generation (C1/C2)

This script converts T1/T2 individual messages into continuous transcript formats:
- C1: Continuous format derived from T1 (words only)
- C2: Continuous format derived from T2 (numbers/abbreviations)

The continuous formats simulate real-time ASR output for:
- Communication Turn Detection
- Speaker Classification  
- Information Extraction
- Multi-purpose LLM fine-tuning
"""

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

# Add utils to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.speaker_state_tracker import (
    SpeakerStateTracker,
)


@dataclass
class ContinuousMessage:
    """A message in continuous format."""
    timestamp: float
    speaker: str
    speaker_formatted_c1: str
    speaker_formatted_c2: str
    c1_text: str
    c2_text: str
    turn_boundary: bool = False
    turn_confidence: float = 0.0
    extracted_info: Dict[str, Any] = None

    def __post_init__(self):
        if self.extracted_info is None:
            self.extracted_info = {}


@dataclass
class ContinuousSegment:
    """A segment of continuous transcript."""
    start_time: float
    end_time: float
    messages: List[ContinuousMessage]
    c1_segment_text: str
    c2_segment_text: str
    
    
class ContinuousGenerator:
    """
    Generates continuous transcripts (C1/C2) from T1/T2 individual messages.
    
    Features:
    - Speaker diarization and state tracking
    - Turn boundary detection
    - Information extraction
    - Streaming simulation
    """
    
    def __init__(self):
        self.speaker_tracker = SpeakerStateTracker()
        
        # Turn detection patterns (simple rule-based for now)
        self.turn_indicators = [
            r'\.',  # Period
            r'\?',  # Question mark
            r'!',   # Exclamation
            r'good day\b',
            r'roger\b',
            r'wilco\b',
            r'contact\s+\w+',  # Frequency change
            r'so long\b'
        ]
        
        # Information extraction patterns
        self.info_patterns = {
            'altitude': [
                r'(\d+)\s*(?:feet|ft)\b',
                r'flight\s*level\s*(\d+)',
                r'FL(\d+)',
                r'(\d+)\s*thousand',
                r'maintain\s+(\d+)',
                r'climb.*?(\d+)',
                r'descend.*?(\d+)'
            ],
            'heading': [
                r'heading\s+(\d+)',
                r'turn\s+\w+\s+heading\s+(\d+)',
                r'fly\s+heading\s+(\d+)'
            ],
            'speed': [
                r'(\d+)\s*knots?\b',
                r'reduce\s+speed\s+(\d+)',
                r'maintain\s+speed\s+(\d+)',
                r'increase\s+speed\s+(\d+)'
            ],
            'frequency': [
                r'(\d+\.\d+)(?:\s*mhz)?\b',
                r'contact\s+\w+\s+(\d+\.\d+)',
                r'(\d{3}\.\d+)'
            ],
            'squawk': [
                r'squawk\s+(\d{4})',
                r'transponder\s+(\d{4})'
            ]
        }

    def load_t1_t2_data(self, json_path: str) -> List[Dict]:
        """Load T1/T2 data from JSON file."""
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def generate_continuous_formats(self, 
                                  input_json_path: str,
                                  output_c1_c2_lbs: str,
                                  output_c1_c2_json: str) -> List[Dict]:
        """
        Generate combined C1/C2 continuous formats from T1/T2 data.
        Appends C1/C2 information to each individual message record.
        
        Args:
            input_json_path: Path to T1/T2 JSON file
            output_c1_c2_lbs: Output path for combined C1/C2 LBS format
            output_c1_c2_json: Output path for combined C1/C2 JSON format
            
        Returns:
            List of enhanced message records with C1/C2 data
        """
        print(f"Loading T1/T2 data from {input_json_path}")
        
        # Load data
        t1_t2_data = self.load_t1_t2_data(input_json_path)
        
        print(f"Processing {len(t1_t2_data)} messages for continuous generation...")
        
        # Process each message and append C1/C2 data
        enhanced_messages = []
        
        for i, message_data in enumerate(t1_t2_data, 1):
            print(f"Processing message {i}/{len(t1_t2_data)}: {message_data.get('t1_text', '')[:50]}...")
            
            # Extract timing from header
            timestamp = self._extract_timestamp(message_data.get('header', ''))
            speaker = self._extract_speaker(message_data.get('header', ''))
            
            # Update speaker state
            self.speaker_tracker.update_conversation_state(
                timestamp=timestamp,
                speaker=speaker,
                message=message_data.get('original', ''),
                t1_text=message_data.get('t1_text', ''),
                t2_text=message_data.get('t2_text', '')
            )
            
            # Create continuous message with both C1 and C2
            continuous_message = self._create_continuous_message(
                timestamp=timestamp,
                speaker=speaker,
                t1_text=message_data.get('t1_text', ''),
                t2_text=message_data.get('t2_text', '')
            )
            
            # Enhance the original message data with C1/C2 information
            enhanced_message = message_data.copy()
            enhanced_message['c1_text'] = continuous_message.c1_text
            enhanced_message['c2_text'] = continuous_message.c2_text
            enhanced_message['c1_speaker_formatted'] = continuous_message.speaker_formatted_c1
            enhanced_message['c2_speaker_formatted'] = continuous_message.speaker_formatted_c2
            enhanced_message['turn_boundary'] = continuous_message.turn_boundary
            enhanced_message['turn_confidence'] = continuous_message.turn_confidence
            enhanced_message['extracted_info'] = continuous_message.extracted_info
            enhanced_message['timestamp'] = timestamp
            
            enhanced_messages.append(enhanced_message)
        
        # Save results
        self._save_individual_message_formats(enhanced_messages, output_c1_c2_lbs, output_c1_c2_json)
        
        print(f"✓ Processed {len(enhanced_messages)} individual messages with C1/C2 data")
        print(f"✓ Saved combined C1/C2 LBS format to {output_c1_c2_lbs}")
        print(f"✓ Saved combined C1/C2 JSON format to {output_c1_c2_json}")
        
        return enhanced_messages

    def _extract_timestamp(self, header: str) -> float:
        """Extract timestamp from message header."""
        # Extract from pattern like {dca_d1_1 1 dca_d1_1__DR1-1__DAL209__00_01_03 63.040 66.010}
        match = re.search(r'(\d+\.\d+)\s+(\d+\.\d+)\}', header)
        if match:
            return float(match.group(1))  # Start time
        return 0.0

    def _extract_speaker(self, header: str) -> str:
        """Extract speaker from message header."""
        # Extract from pattern like __DR1-1__DAL209__ or __DAL209__DR1-1__
        match = re.search(r'__([A-Z0-9-]+)__([A-Z0-9-]+)__', header)
        if match:
            # First match is usually the speaker
            return match.group(1)
        return "Unknown"

    def _create_continuous_message(self, 
                                 timestamp: float,
                                 speaker: str,
                                 t1_text: str,
                                 t2_text: str) -> ContinuousMessage:
        """Create a continuous message with turn detection and info extraction."""
        
        # Format speaker for both C1 and C2 continuous transcripts
        speaker_formatted_c1 = self.speaker_tracker.format_speaker_for_continuous(
            speaker, "C1"
        )
        speaker_formatted_c2 = self.speaker_tracker.format_speaker_for_continuous(
            speaker, "C2"
        )
        
        # Detect turn boundary (use T1 text for detection)
        turn_boundary, turn_confidence = self._detect_turn_boundary(t1_text)
        
        # Extract information (use T2 text for better numerical extraction)
        extracted_info = self._extract_information(t2_text)
        
        return ContinuousMessage(
            timestamp=timestamp,
            speaker=speaker,
            speaker_formatted_c1=speaker_formatted_c1,
            speaker_formatted_c2=speaker_formatted_c2,
            c1_text=t1_text,
            c2_text=t2_text,
            turn_boundary=turn_boundary,
            turn_confidence=turn_confidence,
            extracted_info=extracted_info
        )

    def _detect_turn_boundary(self, text: str) -> Tuple[bool, float]:
        """Detect if this text represents a turn boundary."""
        text_lower = text.lower()
        
        confidence = 0.0
        is_boundary = False
        
        for pattern in self.turn_indicators:
            if re.search(pattern, text_lower):
                is_boundary = True
                confidence += 0.3
        
        # Additional heuristics
        if text.endswith('.') or text.endswith('!') or text.endswith('?'):
            confidence += 0.4
            is_boundary = True
        
        if any(phrase in text_lower for phrase in ['good day', 'contact', 'so long']):
            confidence += 0.5
            is_boundary = True
        
        return is_boundary, min(confidence, 1.0)

    def _extract_information(self, text: str) -> Dict[str, Any]:
        """Extract aviation information from text."""
        extracted = {}
        
        for info_type, patterns in self.info_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    extracted[info_type] = matches
        
        return extracted

    def _create_segment(self, 
                       start_time: float,
                       end_time: float,
                       messages: List[ContinuousMessage]) -> ContinuousSegment:
        """Create a continuous segment from messages."""
        
        # Build C1 continuous text
        c1_segment_text = ""
        for msg in messages:
            c1_segment_text += f"{msg.speaker_formatted_c1} {msg.c1_text} "
        
        # Build C2 continuous text
        c2_segment_text = ""
        for msg in messages:
            c2_segment_text += f"{msg.speaker_formatted_c2} {msg.c2_text} "
        
        return ContinuousSegment(
            start_time=start_time,
            end_time=end_time,
            messages=messages,
            c1_segment_text=c1_segment_text.strip(),
            c2_segment_text=c2_segment_text.strip()
        )

    def _save_individual_message_formats(self, 
                                        enhanced_messages: List[Dict],
                                        lbs_path: str,
                                        json_path: str):
        """Save individual message records with C1/C2 data appended."""
        
        # Save as LBS file (similar to T1/T2 format but with C1/C2 added)
        with open(lbs_path, 'w', encoding='utf-8') as f:
            for i, message in enumerate(enhanced_messages, 1):
                # Write original header
                f.write(f"{message.get('header', '')}\n")
                f.write(f"ORIG: {message.get('original', '')}\n")
                f.write(f"PROC: {message.get('processed', '')}\n")
                f.write(f"NUMC: {message.get('numeric', '')}\n")
                f.write(f"PUNC: {message.get('punctuated', '')}\n")
                f.write(f"T1: {message.get('t1_text', '')}\n")
                f.write(f"T2: {message.get('t2_text', '')}\n")
                f.write(f"C1: {message.get('c1_speaker_formatted', '')} {message.get('c1_text', '')}\n")
                f.write(f"C2: {message.get('c2_speaker_formatted', '')} {message.get('c2_text', '')}\n")
                f.write("NOTE:\n")
                
                # Copy original NOTE content
                f.write(f"    - Source Analysis: {message.get('source_analysis', '')}\n")
                f.write(f"    - T1 Confidence: {message.get('t1_confidence', '')}\n")
                f.write(f"    - T1 Corrections: {message.get('t1_corrections', '')}\n")
                f.write(f"    - T2 Confidence: {message.get('t2_confidence', '')}\n")
                f.write(f"    - T2 Corrections: {message.get('t2_corrections', '')}\n")
                f.write(f"    - Speaker Context: {message.get('speaker_context', '')}\n")
                
                # Add C1/C2 specific information
                f.write(f"    - C1 Speaker: {message.get('c1_speaker_formatted', '')}\n")
                f.write(f"    - C2 Speaker: {message.get('c2_speaker_formatted', '')}\n")
                f.write(f"    - Turn Boundary: {'Yes' if message.get('turn_boundary', False) else 'No'} ({message.get('turn_confidence', 0.0):.2f})\n")
                f.write(f"    - Info Extractions: {message.get('extracted_info', {})}\n")
                f.write(f"    - Timestamp: {message.get('timestamp', 0.0):.1f}s\n")
                
                f.write("\n\n")
        
        # Save as JSON file (enhanced message structure)
        json_data = []
        for message in enhanced_messages:
            # Create enhanced message record
            enhanced_record = {
                'header': message.get('header', ''),
                'original': message.get('original', ''),
                'processed': message.get('processed', ''),
                'numeric': message.get('numeric', ''),
                'punctuated': message.get('punctuated', ''),
                't1_text': message.get('t1_text', ''),
                't1_confidence': message.get('t1_confidence', ''),
                't1_corrections': message.get('t1_corrections', ''),
                't2_text': message.get('t2_text', ''),
                't2_confidence': message.get('t2_confidence', ''),
                't2_corrections': message.get('t2_corrections', ''),
                'c1_text': message.get('c1_text', ''),
                'c2_text': message.get('c2_text', ''),
                'c1_speaker_formatted': message.get('c1_speaker_formatted', ''),
                'c2_speaker_formatted': message.get('c2_speaker_formatted', ''),
                'turn_boundary': message.get('turn_boundary', False),
                'turn_confidence': message.get('turn_confidence', 0.0),
                'extracted_info': message.get('extracted_info', {}),
                'source_analysis': message.get('source_analysis', ''),
                'speaker_context': message.get('speaker_context', ''),
                'speaker_info': message.get('speaker_info', {}),
                'timestamp': message.get('timestamp', 0.0)
            }
            json_data.append(enhanced_record)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

    def generate_analysis_report(self, 
                               enhanced_messages: List[Dict],
                               output_path: str):
        """Generate detailed analysis report for the continuous formats."""
        
        analysis = {
            'conversation_summary': self.speaker_tracker.get_conversation_summary(),
            'message_statistics': {
                'total_messages': len(enhanced_messages),
                'total_turn_boundaries': sum(1 for msg in enhanced_messages if msg.get('turn_boundary', False)),
                'total_info_extractions': sum(len(msg.get('extracted_info', {})) for msg in enhanced_messages)
            },
            'speaker_statistics': {},
            'turn_analysis': {},
            'information_extraction_summary': {}
        }
        
        # Analyze speakers
        for speaker, info in self.speaker_tracker.conversation_state.active_speakers.items():
            analysis['speaker_statistics'][speaker] = {
                'role': info.role.value,
                'position_name': info.position_name,
                'total_messages': 0,
                'turn_boundaries': 0
            }
        
        # Analyze messages
        total_turns = 0
        total_turn_confidence = 0.0
        info_extractions = {}
        
        for msg in enhanced_messages:
            speaker = self._extract_speaker(msg.get('header', ''))
            if speaker in analysis['speaker_statistics']:
                analysis['speaker_statistics'][speaker]['total_messages'] += 1
                
                if msg.get('turn_boundary', False):
                    analysis['speaker_statistics'][speaker]['turn_boundaries'] += 1
                    total_turns += 1
                    total_turn_confidence += msg.get('turn_confidence', 0.0)
                
                # Collect information extractions
                for info_type, values in msg.get('extracted_info', {}).items():
                    if info_type not in info_extractions:
                        info_extractions[info_type] = []
                    info_extractions[info_type].extend(values)
        
        analysis['turn_analysis'] = {
            'total_turns_detected': total_turns,
            'average_confidence': total_turn_confidence / max(total_turns, 1)
        }
        
        analysis['information_extraction_summary'] = {
            info_type: {
                'total_extractions': len(values),
                'unique_values': len(set(values)),
                'values': list(set(values))
            }
            for info_type, values in info_extractions.items()
        }
        
        # Save analysis
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved continuous analysis to {output_path}")


def main():
    """Main function to generate continuous formats."""
    
    print("=== Phase 2: Continuous Format Generation (C1/C2) ===\n")
    
    # Record start time
    start_time = time.time()
    
    # Define paths
    input_json = "data/test_t1_t2.json"
    output_c1_c2_lbs = "data/test_c1_c2.lbs"
    output_c1_c2_json = "data/test_c1_c2.json"
    analysis_output = "data/test_C1_C2_continuous_analysis.json"
    
    # Check if input file exists
    if not os.path.exists(input_json):
        print(f"Error: Input file {input_json} not found!")
        print("Please run Step4_format_T1_T2.py first to generate T1/T2 data.")
        sys.exit(1)
    
    # Create generator
    generator = ContinuousGenerator()
    
    # Generate continuous formats
    enhanced_messages = generator.generate_continuous_formats(
        input_json_path=input_json,
        output_c1_c2_lbs=output_c1_c2_lbs,
        output_c1_c2_json=output_c1_c2_json
    )
    
    # Generate analysis report
    generator.generate_analysis_report(enhanced_messages, analysis_output)
    
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
    
    print("\n✓ Phase 2 completed successfully!")
    print("✓ Generated combined C1/C2 continuous formats")
    print(f"✓ Total execution time: {time_str}")
    print("✓ Results saved to data/ directory")


if __name__ == "__main__":
    main() 