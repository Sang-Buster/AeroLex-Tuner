#!/usr/bin/env python3
"""
Step 5 Enhanced: LLM-Enhanced Continuous Format Generation (C1/C2)

This script converts T1/T2 individual messages into high-quality continuous transcript formats
using LLM assistance for better accuracy and contextual understanding.

Features:
- Phase 1: LLM-Enhanced Processing
- Phase 2: Hybrid Approach (LLM + Rule-based)
- Phase 3: Quality Assurance & Cross-validation
"""

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Add utils to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.speaker_state_tracker import SpeakerStateTracker


@dataclass
class ContinuousAnalysis:
    """Enhanced analysis results from LLM processing."""
    turn_boundary: bool
    turn_confidence: float
    turn_reasoning: str
    speaker_role: str
    speaker_position: str
    speaker_confidence: float
    speaker_reasoning: str
    c1_speaker_formatted: str
    c2_speaker_formatted: str
    extracted_info: Dict[str, Any]
    extraction_confidence: float
    extraction_reasoning: str
    quality_score: float
    processing_method: str  # "llm", "rule", "hybrid"


@dataclass
class EnhancedContinuousMessage:
    """Enhanced message with comprehensive analysis."""
    timestamp: float
    speaker: str
    c1_text: str
    c2_text: str
    analysis: ContinuousAnalysis
    rule_based_comparison: Optional[Dict] = None


class LLMEnhancedGenerator:
    """
    Enhanced continuous transcript generator using LLM assistance.
    
    Implements three-phase approach:
    - Phase 1: LLM-Enhanced Processing
    - Phase 2: Hybrid Approach
    - Phase 3: Quality Assurance
    """
    
    def __init__(self, ollama_model: str = "llama3.3:70b-instruct-q4_K_M", verbose: bool = True):
        self.ollama_model = ollama_model
        self.speaker_tracker = SpeakerStateTracker()
        self.verbose = verbose
        
        # Quality thresholds
        self.llm_confidence_threshold = 0.8
        self.rule_confidence_threshold = 0.7
        
        # Rule-based patterns for fallback
        self.turn_indicators = [
            r'\.',  # Period
            r'\?',  # Question mark
            r'!',   # Exclamation
            r'good day\b',
            r'roger\b',
            r'wilco\b',
            r'contact\s+\w+',
            r'so long\b'
        ]
        
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

    def generate_enhanced_continuous_formats(self, 
                                           input_json_path: str,
                                           output_c1_c2_lbs: str,
                                           output_c1_c2_json: str,
                                           analysis_output: str) -> List[Dict]:
        """
        Generate enhanced C1/C2 continuous formats with LLM assistance.
        
        Phase 1: LLM-Enhanced Processing
        Phase 2: Hybrid Approach
        Phase 3: Quality Assurance
        """
        print(f"Loading T1/T2 data from {input_json_path}")
        
        # Load data
        t1_t2_data = self.load_t1_t2_data(input_json_path)
        
        print(f"Processing {len(t1_t2_data)} messages with LLM-enhanced analysis...")
        
        # Phase 1: LLM-Enhanced Processing
        enhanced_messages = []
        quality_metrics = {
            'llm_processed': 0,
            'rule_processed': 0,
            'hybrid_processed': 0,
            'high_confidence': 0,
            'medium_confidence': 0,
            'low_confidence': 0
        }
        
        for i, message_data in enumerate(t1_t2_data, 1):
            print(f"Processing message {i}/{len(t1_t2_data)}: {message_data.get('t1_text', '')[:50]}...")
            
            # Extract timing and speaker
            timestamp = self._extract_timestamp(message_data.get('header', ''))
            speaker = self._extract_speaker(message_data.get('header', ''))
            
            # Phase 1: LLM Analysis
            llm_analysis = self._analyze_with_llm(message_data, speaker, timestamp, i)
            
            # Phase 2: Rule-based Analysis for comparison
            rule_analysis = self._analyze_with_rules(message_data, speaker, timestamp)
            
            if self.verbose:
                print("📏 RULE-BASED COMPARISON:")
                print(f"   Turn Boundary: {rule_analysis.get('turn_boundary', False)} (conf: {rule_analysis.get('turn_confidence', 0):.2f})")
                print(f"   Speaker C1: {rule_analysis.get('speaker_formatted_c1', 'N/A')}")
                print(f"   Speaker C2: {rule_analysis.get('speaker_formatted_c2', 'N/A')}")
                print(f"   Extracted Info: {rule_analysis.get('extracted_info', {})}")
                print(f"   Quality Score: {rule_analysis.get('quality_score', 0):.2f}")
                print(f"{'='*80}")
            
            # Phase 2: Hybrid Decision Making
            final_analysis = self._make_hybrid_decision(llm_analysis, rule_analysis)
            
            if self.verbose:
                print("🔀 HYBRID DECISION:")
                print(f"   Final Method: {final_analysis.processing_method}")
                print(f"   Final Quality Score: {final_analysis.quality_score:.2f}")
                print(f"   LLM vs Rule Confidence: {llm_analysis.quality_score:.2f} vs {rule_analysis.get('quality_score', 0):.2f}")
                print(f"{'='*80}\n")
            
            # Update quality metrics
            quality_metrics[f"{final_analysis.processing_method}_processed"] += 1
            if final_analysis.quality_score >= 0.8:
                quality_metrics['high_confidence'] += 1
            elif final_analysis.quality_score >= 0.6:
                quality_metrics['medium_confidence'] += 1
            else:
                quality_metrics['low_confidence'] += 1
            
            # Create enhanced message
            enhanced_message = EnhancedContinuousMessage(
                timestamp=timestamp,
                speaker=speaker,
                c1_text=message_data.get('t1_text', ''),
                c2_text=message_data.get('t2_text', ''),
                analysis=final_analysis,
                rule_based_comparison=rule_analysis
            )
            
            # Update speaker state
            self.speaker_tracker.update_conversation_state(
                timestamp=timestamp,
                speaker=speaker,
                message=message_data.get('original', ''),
                t1_text=message_data.get('t1_text', ''),
                t2_text=message_data.get('t2_text', '')
            )
            
            # Enhance original message data
            enhanced_record = self._create_enhanced_record(message_data, enhanced_message)
            enhanced_messages.append(enhanced_record)
        
        # Phase 3: Quality Assurance
        qa_report = self._generate_quality_assurance_report(enhanced_messages, quality_metrics)
        
        # Save results
        self._save_enhanced_formats(enhanced_messages, output_c1_c2_lbs, output_c1_c2_json)
        self._save_analysis_report(qa_report, analysis_output)
        
        print(f"✓ Enhanced processing completed with {quality_metrics['high_confidence']} high-confidence results")
        print(f"✓ LLM: {quality_metrics['llm_processed']}, Rules: {quality_metrics['rule_processed']}, Hybrid: {quality_metrics['hybrid_processed']}")
        
        return enhanced_messages

    def _analyze_with_llm(self, message_data: Dict, speaker: str, timestamp: float, message_index: int) -> ContinuousAnalysis:
        """Phase 1: LLM-Enhanced Analysis."""
        
        # Get conversation context
        context = self._get_conversation_context(message_index)
        
        # Create LLM prompt
        prompt = self._create_llm_prompt(message_data, speaker, context)
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"🤖 LLM ANALYSIS - Message {message_index}")
            print(f"{'='*80}")
            print("📝 PROMPT:")
            print(f"{prompt}")
            print(f"{'='*80}")
        
        # Call LLM
        try:
            response = self._call_ollama(prompt)
            
            if self.verbose:
                print("🔍 LLM RESPONSE:")
                print(f"{response}")
                print(f"{'='*80}")
            
            analysis = self._parse_llm_response(response)
            analysis.processing_method = "llm"
            
            if self.verbose:
                print("📊 PARSED ANALYSIS:")
                print(f"   Turn Boundary: {analysis.turn_boundary} (conf: {analysis.turn_confidence:.2f})")
                print(f"   Speaker Role: {analysis.speaker_role}")
                print(f"   Speaker Position: {analysis.speaker_position}")
                print(f"   C1 Speaker: {analysis.c1_speaker_formatted}")
                print(f"   C2 Speaker: {analysis.c2_speaker_formatted}")
                print(f"   Extracted Info: {analysis.extracted_info}")
                print(f"   Quality Score: {analysis.quality_score:.2f}")
                print(f"   Processing Method: {analysis.processing_method}")
                print(f"{'='*80}\n")
            
            return analysis
        except Exception as e:
            print(f"❌ LLM analysis failed: {e}, falling back to rules...")
            # Fallback to rule-based analysis
            rule_data = self._analyze_with_rules(message_data, speaker, timestamp)
            return self._dict_to_analysis(rule_data, "rule")

    def _analyze_with_rules(self, message_data: Dict, speaker: str, timestamp: float) -> Dict:
        """Phase 2: Rule-based Analysis for comparison."""
        
        t1_text = message_data.get('t1_text', '')
        t2_text = message_data.get('t2_text', '')
        
        # Turn boundary detection
        turn_boundary, turn_confidence = self._detect_turn_boundary_rules(t1_text)
        
        # Speaker classification
        speaker_formatted_c1, speaker_formatted_c2 = self._classify_speaker_rules(speaker)
        
        # Information extraction
        extracted_info = self._extract_information_rules(t2_text)
        
        return {
            'turn_boundary': turn_boundary,
            'turn_confidence': turn_confidence,
            'speaker_formatted_c1': speaker_formatted_c1,
            'speaker_formatted_c2': speaker_formatted_c2,
            'extracted_info': extracted_info,
            'quality_score': turn_confidence * 0.5 + 0.5  # Simple quality estimate
        }

    def _make_hybrid_decision(self, llm_analysis: ContinuousAnalysis, rule_data: Dict) -> ContinuousAnalysis:
        """Phase 2: Hybrid Decision Making."""
        
        # If LLM confidence is high, use LLM result
        if llm_analysis.quality_score >= self.llm_confidence_threshold:
            return llm_analysis
        
        # Convert rule data to analysis object for comparison
        rule_analysis = self._dict_to_analysis(rule_data, "rule")
        
        # If rule confidence is reasonable and LLM is uncertain, use hybrid
        if rule_analysis.quality_score >= self.rule_confidence_threshold:
            # Create hybrid analysis
            hybrid_analysis = ContinuousAnalysis(
                turn_boundary=llm_analysis.turn_boundary if llm_analysis.turn_confidence > 0.5 else rule_analysis.turn_boundary,
                turn_confidence=max(llm_analysis.turn_confidence, rule_analysis.turn_confidence),
                turn_reasoning=f"Hybrid: LLM conf={llm_analysis.turn_confidence:.2f}, Rule conf={rule_analysis.turn_confidence:.2f}",
                speaker_role=llm_analysis.speaker_role,
                speaker_position=llm_analysis.speaker_position,
                speaker_confidence=llm_analysis.speaker_confidence,
                speaker_reasoning=llm_analysis.speaker_reasoning,
                c1_speaker_formatted=llm_analysis.c1_speaker_formatted or rule_analysis.c1_speaker_formatted,
                c2_speaker_formatted=llm_analysis.c2_speaker_formatted or rule_analysis.c2_speaker_formatted,
                extracted_info=llm_analysis.extracted_info or rule_analysis.extracted_info,
                extraction_confidence=llm_analysis.extraction_confidence,
                extraction_reasoning=llm_analysis.extraction_reasoning,
                quality_score=(llm_analysis.quality_score + rule_analysis.quality_score) / 2,
                processing_method="hybrid"
            )
            return hybrid_analysis
        
        # Default to LLM even if low confidence
        return llm_analysis

    def _create_llm_prompt(self, message_data: Dict, speaker: str, context: str) -> str:
        """Create comprehensive LLM prompt for aviation communication analysis."""
        
        return f"""You are an expert aviation communication analyst. Analyze this ATC communication for continuous transcript generation.

MESSAGE DATA:
- Original: {message_data.get('original', '')}
- T1 (conversational): {message_data.get('t1_text', '')}
- T2 (technical): {message_data.get('t2_text', '')}
- Speaker: {speaker}
- Header: {message_data.get('header', '')}

CONTEXT:
{context}

TASKS:
1. TURN BOUNDARY DETECTION: Determine if this message represents a communication turn boundary
2. SPEAKER CLASSIFICATION: Identify speaker role and position
3. INFORMATION EXTRACTION: Extract aviation-specific information
4. SPEAKER FORMATTING: Format speaker labels for continuous transcripts

OUTPUT FORMAT (JSON):
{{
    "turn_boundary": {{
        "is_boundary": true/false,
        "confidence": 0.0-1.0,
        "reasoning": "explanation"
    }},
    "speaker_classification": {{
        "role": "ATC/Pilot/Ground/Tower/Approach/Departure/Center",
        "position": "specific position name",
        "confidence": 0.0-1.0,
        "reasoning": "explanation"
    }},
    "speaker_formatting": {{
        "c1_format": "(Speaker Label)",
        "c2_format": "(Speaker Label)"
    }},
    "information_extraction": {{
        "altitude": {{"value": "number", "unit": "feet/FL", "action": "maintain/climb/descend"}},
        "heading": {{"value": "number", "action": "turn left/turn right/fly"}},
        "speed": {{"value": "number", "unit": "knots", "action": "maintain/reduce/increase"}},
        "frequency": {{"value": "number", "unit": "MHz"}},
        "squawk": {{"value": "number"}},
        "confidence": 0.0-1.0,
        "reasoning": "explanation"
    }},
    "quality_assessment": {{
        "overall_score": 0.0-1.0,
        "clarity": 0.0-1.0,
        "completeness": 0.0-1.0,
        "accuracy": 0.0-1.0
    }}
}}

Focus on aviation communication patterns, standard phraseology, and contextual understanding."""

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API with the prompt."""
        import json as json_module
        import subprocess
        
        try:
            # Create JSON payload for Ollama API
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 1000
                }
            }
            
            # Call ollama via API
            result = subprocess.run(
                ["curl", "-X", "POST", "http://localhost:11434/api/generate",
                 "-H", "Content-Type: application/json",
                 "-d", json_module.dumps(payload)],
                capture_output=True,
                text=True,
                timeout=120  # Increased timeout for large model
            )
            
            if result.returncode != 0:
                raise Exception(f"Ollama API failed: {result.stderr}")
            
            # Parse response
            response_data = json_module.loads(result.stdout)
            return response_data.get('response', '')
            
        except Exception as e:
            raise Exception(f"Failed to call Ollama: {str(e)}")

    def _parse_llm_response(self, response: str) -> ContinuousAnalysis:
        """Parse LLM response into structured analysis."""
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in response")
            
            data = json.loads(json_match.group())
            
            # Create analysis object
            analysis = ContinuousAnalysis(
                turn_boundary=data.get('turn_boundary', {}).get('is_boundary', False),
                turn_confidence=data.get('turn_boundary', {}).get('confidence', 0.5),
                turn_reasoning=data.get('turn_boundary', {}).get('reasoning', ''),
                speaker_role=data.get('speaker_classification', {}).get('role', 'Unknown'),
                speaker_position=data.get('speaker_classification', {}).get('position', 'Unknown'),
                speaker_confidence=data.get('speaker_classification', {}).get('confidence', 0.5),
                speaker_reasoning=data.get('speaker_classification', {}).get('reasoning', ''),
                c1_speaker_formatted=data.get('speaker_formatting', {}).get('c1_format', '(Unknown)'),
                c2_speaker_formatted=data.get('speaker_formatting', {}).get('c2_format', '(Unknown)'),
                extracted_info=data.get('information_extraction', {}),
                extraction_confidence=data.get('information_extraction', {}).get('confidence', 0.5),
                extraction_reasoning=data.get('information_extraction', {}).get('reasoning', ''),
                quality_score=data.get('quality_assessment', {}).get('overall_score', 0.5),
                processing_method="llm"
            )
            
            return analysis
            
        except Exception as e:
            print(f"Failed to parse LLM response: {e}")
            # Return default analysis
            return ContinuousAnalysis(
                turn_boundary=False,
                turn_confidence=0.3,
                turn_reasoning="Failed to parse LLM response",
                speaker_role="Unknown",
                speaker_position="Unknown",
                speaker_confidence=0.3,
                speaker_reasoning="Failed to parse LLM response",
                c1_speaker_formatted="(Unknown)",
                c2_speaker_formatted="(Unknown)",
                extracted_info={},
                extraction_confidence=0.3,
                extraction_reasoning="Failed to parse LLM response",
                quality_score=0.3,
                processing_method="llm"
            )

    def _get_conversation_context(self, message_index: int) -> str:
        """Get conversation context for LLM analysis."""
        return f"Message {message_index} in conversation sequence."

    def _dict_to_analysis(self, rule_data: Dict, method: str) -> ContinuousAnalysis:
        """Convert rule analysis dictionary to ContinuousAnalysis object."""
        return ContinuousAnalysis(
            turn_boundary=rule_data.get('turn_boundary', False),
            turn_confidence=rule_data.get('turn_confidence', 0.5),
            turn_reasoning=f"Rule-based detection (confidence: {rule_data.get('turn_confidence', 0.5):.2f})",
            speaker_role="Unknown",
            speaker_position="Unknown", 
            speaker_confidence=0.5,
            speaker_reasoning="Rule-based classification",
            c1_speaker_formatted=rule_data.get('speaker_formatted_c1', '(Unknown)'),
            c2_speaker_formatted=rule_data.get('speaker_formatted_c2', '(Unknown)'),
            extracted_info=rule_data.get('extracted_info', {}),
            extraction_confidence=0.7 if rule_data.get('extracted_info') else 0.3,
            extraction_reasoning="Rule-based pattern matching",
            quality_score=rule_data.get('quality_score', 0.5),
            processing_method=method
        )

    def _detect_turn_boundary_rules(self, text: str) -> Tuple[bool, float]:
        """Rule-based turn boundary detection."""
        text_lower = text.lower()
        
        confidence = 0.0
        is_boundary = False
        
        for pattern in self.turn_indicators:
            if re.search(pattern, text_lower):
                is_boundary = True
                confidence += 0.3
        
        if text.endswith('.') or text.endswith('!') or text.endswith('?'):
            confidence += 0.4
            is_boundary = True
        
        return is_boundary, min(confidence, 1.0)

    def _classify_speaker_rules(self, speaker: str) -> Tuple[str, str]:
        """Rule-based speaker classification."""
        # Simple rule-based classification
        if speaker == "Unknown":
            return "(Unknown)", "(Unknown)"
        else:
            return f"({speaker})", f"({speaker})"

    def _extract_information_rules(self, text: str) -> Dict[str, Any]:
        """Rule-based information extraction."""
        extracted = {}
        
        for info_type, patterns in self.info_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    extracted[info_type] = matches
        
        return extracted

    def _extract_timestamp(self, header: str) -> float:
        """Extract timestamp from message header."""
        match = re.search(r'(\d+\.\d+)\s+(\d+\.\d+)\}', header)
        if match:
            return float(match.group(1))
        return 0.0

    def _extract_speaker(self, header: str) -> str:
        """Extract speaker from message header."""
        match = re.search(r'__([A-Z0-9-]+)__([A-Z0-9-]+)__', header)
        if match:
            return match.group(1)
        return "Unknown"

    def _create_enhanced_record(self, message_data: Dict, enhanced_message: EnhancedContinuousMessage) -> Dict:
        """Create enhanced message record with all analysis data."""
        
        analysis = enhanced_message.analysis
        
        enhanced_record = {
            # Original fields
            'header': message_data.get('header', ''),
            'original': message_data.get('original', ''),
            'processed': message_data.get('processed', ''),
            'numeric': message_data.get('numeric', ''),
            'punctuated': message_data.get('punctuated', ''),
            't1_text': message_data.get('t1_text', ''),
            't1_confidence': message_data.get('t1_confidence', ''),
            't1_corrections': message_data.get('t1_corrections', ''),
            't2_text': message_data.get('t2_text', ''),
            't2_confidence': message_data.get('t2_confidence', ''),
            't2_corrections': message_data.get('t2_corrections', ''),
            'source_analysis': message_data.get('source_analysis', ''),
            'speaker_context': message_data.get('speaker_context', ''),
            'speaker_info': message_data.get('speaker_info', {}),
            
            # Enhanced C1/C2 fields
            'c1_text': enhanced_message.c1_text,
            'c2_text': enhanced_message.c2_text,
            'c1_speaker_formatted': analysis.c1_speaker_formatted,
            'c2_speaker_formatted': analysis.c2_speaker_formatted,
            'turn_boundary': analysis.turn_boundary,
            'turn_confidence': analysis.turn_confidence,
            'turn_reasoning': analysis.turn_reasoning,
            'speaker_role': analysis.speaker_role,
            'speaker_position': analysis.speaker_position,
            'speaker_confidence': analysis.speaker_confidence,
            'speaker_reasoning': analysis.speaker_reasoning,
            'extracted_info': analysis.extracted_info,
            'extraction_confidence': analysis.extraction_confidence,
            'extraction_reasoning': analysis.extraction_reasoning,
            'quality_score': analysis.quality_score,
            'processing_method': analysis.processing_method,
            'timestamp': enhanced_message.timestamp,
            
            # Rule-based comparison
            'rule_comparison': enhanced_message.rule_based_comparison
        }
        
        return enhanced_record

    def _save_enhanced_formats(self, enhanced_messages: List[Dict], lbs_path: str, json_path: str):
        """Save enhanced message formats."""
        
        # Save LBS format
        with open(lbs_path, 'w', encoding='utf-8') as f:
            for message in enhanced_messages:
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
                f.write(f"    - Source Analysis: {message.get('source_analysis', '')}\n")
                f.write(f"    - T1 Confidence: {message.get('t1_confidence', '')}\n")
                f.write(f"    - T1 Corrections: {message.get('t1_corrections', '')}\n")
                f.write(f"    - T2 Confidence: {message.get('t2_confidence', '')}\n")
                f.write(f"    - T2 Corrections: {message.get('t2_corrections', '')}\n")
                f.write(f"    - Speaker Context: {message.get('speaker_context', '')}\n")
                f.write("    - Enhanced Analysis:\n")
                f.write(f"        - C1 Speaker: {message.get('c1_speaker_formatted', '')}\n")
                f.write(f"        - C2 Speaker: {message.get('c2_speaker_formatted', '')}\n")
                f.write(f"        - Speaker Role: {message.get('speaker_role', '')}\n")
                f.write(f"        - Speaker Position: {message.get('speaker_position', '')}\n")
                f.write(f"        - Turn Boundary: {'Yes' if message.get('turn_boundary', False) else 'No'} ({message.get('turn_confidence', 0.0):.2f})\n")
                f.write(f"        - Turn Reasoning: {message.get('turn_reasoning', '')}\n")
                f.write(f"        - Info Extractions: {message.get('extracted_info', {})}\n")
                f.write(f"        - Processing Method: {message.get('processing_method', '')}\n")
                f.write(f"        - Quality Score: {message.get('quality_score', 0.0):.2f}\n")
                f.write(f"        - Timestamp: {message.get('timestamp', 0.0):.1f}s\n")
                f.write("\n\n")
        
        # Save JSON format
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_messages, f, indent=2, ensure_ascii=False)

    def _generate_quality_assurance_report(self, enhanced_messages: List[Dict], quality_metrics: Dict) -> Dict:
        """Phase 3: Generate comprehensive quality assurance report."""
        
        total_messages = len(enhanced_messages)
        
        # Calculate quality distribution
        quality_distribution = {
            'excellent': sum(1 for msg in enhanced_messages if msg.get('quality_score', 0) >= 0.9),
            'good': sum(1 for msg in enhanced_messages if 0.7 <= msg.get('quality_score', 0) < 0.9),
            'fair': sum(1 for msg in enhanced_messages if 0.5 <= msg.get('quality_score', 0) < 0.7),
            'poor': sum(1 for msg in enhanced_messages if msg.get('quality_score', 0) < 0.5)
        }
        
        # Method comparison
        method_comparison = {
            'llm_only': quality_metrics['llm_processed'],
            'rule_only': quality_metrics['rule_processed'],
            'hybrid': quality_metrics['hybrid_processed']
        }
        
        # Turn boundary analysis
        turn_boundary_analysis = {
            'total_turns': sum(1 for msg in enhanced_messages if msg.get('turn_boundary', False)),
            'avg_confidence': sum(msg.get('turn_confidence', 0) for msg in enhanced_messages) / total_messages,
            'high_confidence_turns': sum(1 for msg in enhanced_messages if msg.get('turn_confidence', 0) >= 0.8)
        }
        
        # Speaker classification analysis
        speaker_roles = {}
        for msg in enhanced_messages:
            role = msg.get('speaker_role', 'Unknown')
            speaker_roles[role] = speaker_roles.get(role, 0) + 1
        
        # Information extraction analysis
        info_extractions = {}
        for msg in enhanced_messages:
            for info_type in msg.get('extracted_info', {}):
                info_extractions[info_type] = info_extractions.get(info_type, 0) + 1
        
        qa_report = {
            'summary': {
                'total_messages': total_messages,
                'processing_time': time.time(),
                'average_quality_score': sum(msg.get('quality_score', 0) for msg in enhanced_messages) / total_messages
            },
            'quality_distribution': quality_distribution,
            'method_comparison': method_comparison,
            'turn_boundary_analysis': turn_boundary_analysis,
            'speaker_classification': {
                'roles_identified': speaker_roles,
                'avg_confidence': sum(msg.get('speaker_confidence', 0) for msg in enhanced_messages) / total_messages
            },
            'information_extraction': {
                'types_extracted': info_extractions,
                'avg_confidence': sum(msg.get('extraction_confidence', 0) for msg in enhanced_messages) / total_messages
            },
            'recommendations': self._generate_recommendations(enhanced_messages, quality_metrics)
        }
        
        return qa_report

    def _generate_recommendations(self, enhanced_messages: List[Dict], quality_metrics: Dict) -> List[str]:
        """Generate recommendations based on quality analysis."""
        
        recommendations = []
        
        # Quality recommendations
        low_quality_count = sum(1 for msg in enhanced_messages if msg.get('quality_score', 0) < 0.6)
        if low_quality_count > len(enhanced_messages) * 0.1:
            recommendations.append(f"Consider reviewing {low_quality_count} low-quality messages for pattern analysis")
        
        # Method recommendations
        if quality_metrics['rule_processed'] > quality_metrics['llm_processed']:
            recommendations.append("Consider lowering LLM confidence threshold to increase LLM usage")
        
        # Turn boundary recommendations
        avg_turn_confidence = sum(msg.get('turn_confidence', 0) for msg in enhanced_messages) / len(enhanced_messages)
        if avg_turn_confidence < 0.7:
            recommendations.append("Turn boundary detection may need refinement")
        
        # Speaker classification recommendations
        unknown_speakers = sum(1 for msg in enhanced_messages if msg.get('speaker_role', '') == 'Unknown')
        if unknown_speakers > len(enhanced_messages) * 0.2:
            recommendations.append(f"High number of unknown speakers ({unknown_speakers}), consider improving speaker classification")
        
        return recommendations

    def _save_analysis_report(self, qa_report: Dict, output_path: str):
        """Save quality assurance report."""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(qa_report, f, indent=2, ensure_ascii=False)


def main():
    """Main function for enhanced C1/C2 generation."""
    
    print("=== Enhanced C1/C2 Continuous Format Generation ===")
    print("Phase 1: LLM-Enhanced Processing")
    print("Phase 2: Hybrid Approach")
    print("Phase 3: Quality Assurance")
    print()
    
    # Check for verbose mode
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    if verbose:
        print("🔊 VERBOSE MODE ENABLED - Showing LLM interactions")
        print()
    
    # Record start time
    start_time = time.time()
    
    # Define paths
    input_json = "data/test_t1_t2.json"
    output_c1_c2_lbs = "data/test_c1_c2_enhanced.lbs"
    output_c1_c2_json = "data/test_c1_c2_enhanced.json"
    analysis_output = "data/test_C1_C2_enhanced_analysis.json"
    
    # Check if input file exists
    if not os.path.exists(input_json):
        print(f"Error: Input file {input_json} not found!")
        print("Please run Step4_format_T1_T2.py first to generate T1/T2 data.")
        sys.exit(1)
    
    # Create enhanced generator
    generator = LLMEnhancedGenerator(verbose=verbose)
    
    # Generate enhanced continuous formats
    enhanced_messages = generator.generate_enhanced_continuous_formats(
        input_json_path=input_json,
        output_c1_c2_lbs=output_c1_c2_lbs,
        output_c1_c2_json=output_c1_c2_json,
        analysis_output=analysis_output
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
    
    print("\n✓ Enhanced C1/C2 generation completed successfully!")
    print(f"✓ Processed {len(enhanced_messages)} messages with LLM assistance")
    print(f"✓ Total execution time: {time_str}")
    print("✓ Results saved to data/ directory")
    print(f"✓ Quality assurance report: {analysis_output}")
    print("\n💡 Tip: Use --verbose or -v to see detailed LLM analysis output")


if __name__ == "__main__":
    main() 