"""
Speaker State Tracker for ATC Continuous Format Generation

This module handles speaker identification, role tracking, and state management
for generating continuous transcripts (C1/C2) from individual messages.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class SpeakerRole(Enum):
    """ATC speaker roles in communication flow."""
    GROUND = "Ground"
    TOWER = "Tower" 
    APPROACH = "Approach"
    DEPARTURE = "Departure"
    CENTER = "Center"
    PILOT = "Pilot"
    UNKNOWN = "Unknown"


class FlightPhase(Enum):
    """Aircraft flight phases for determining expected speaker sequence."""
    GROUND_OPS = "Ground Operations"
    TAKEOFF_SEQUENCE = "Takeoff Sequence"
    DEPARTURE = "Departure"
    EN_ROUTE = "En Route"
    APPROACH = "Approach"
    LANDING_SEQUENCE = "Landing Sequence"
    UNKNOWN = "Unknown"


@dataclass
class SpeakerInfo:
    """Information about a speaker in the conversation."""
    callsign: str
    role: SpeakerRole
    position_name: Optional[str] = None
    is_active: bool = True
    last_seen_time: float = 0.0
    
    
@dataclass
class ConversationState:
    """Current state of the ATC conversation."""
    active_speakers: Dict[str, SpeakerInfo]
    current_phase: FlightPhase
    primary_aircraft: Optional[str] = None
    last_speaker: Optional[str] = None
    conversation_timeline: List[Tuple[float, str, str]] = None  # (time, speaker, message)
    
    def __post_init__(self):
        if self.conversation_timeline is None:
            self.conversation_timeline = []


class SpeakerStateTracker:
    """
    Tracks speaker states and roles for continuous transcript generation.
    
    Handles:
    - Speaker identification and role assignment
    - ATC position sequence tracking (Ground → Tower → Departure → Center)
    - Aircraft callsign standardization
    - Communication flow analysis
    """
    
    def __init__(self):
        self.conversation_state = ConversationState(
            active_speakers={},
            current_phase=FlightPhase.UNKNOWN
        )
        
        # ATC position patterns
        self.atc_patterns = {
            SpeakerRole.GROUND: [
                r'ground\b', r'ramp\b', r'clearance\b', r'delivery\b'
            ],
            SpeakerRole.TOWER: [
                r'tower\b', r'local\b', r'control\b'
            ],
            SpeakerRole.APPROACH: [
                r'approach\b', r'arrival\b'
            ],
            SpeakerRole.DEPARTURE: [
                r'departure\b', r'dep\b'
            ],
            SpeakerRole.CENTER: [
                r'center\b', r'centre\b', r'artcc\b'
            ]
        }
        
        # Aircraft callsign patterns
        self.aircraft_patterns = [
            r'^N\d+[A-Z]*$',  # General aviation (N123AB)
            r'^[A-Z]{3}\d+$',  # Commercial airline (DAL209, AAL1581)
            r'^\d+[A-Z]*$',   # Numeric callsigns
        ]
        
        # Airline code mapping for callsign standardization
        self.airline_codes = {
            'delta': 'DAL',
            'american': 'AAL', 
            'united': 'UAL',
            'southwest': 'SWA',
            'jetblue': 'JBU',
            'alaska': 'ASA',
            'spirit': 'NKS',
            'frontier': 'FFT',
        }

    def analyze_speaker(self, callsign: str, message_text: str = "") -> SpeakerInfo:
        """
        Analyze a callsign and message to determine speaker role and information.
        
        Args:
            callsign: The speaker's callsign or identifier
            message_text: The message content for context analysis
            
        Returns:
            SpeakerInfo with determined role and details
        """
        callsign = callsign.strip().upper()
        message_lower = message_text.lower()
        
        # Check if it's an ATC position
        atc_role = self._identify_atc_role(callsign, message_lower)
        if atc_role != SpeakerRole.UNKNOWN:
            return SpeakerInfo(
                callsign=callsign,
                role=atc_role,
                position_name=self._extract_position_name(callsign, message_lower)
            )
        
        # Check if it's an aircraft
        if self._is_aircraft_callsign(callsign):
            return SpeakerInfo(
                callsign=self._standardize_aircraft_callsign(callsign),
                role=SpeakerRole.PILOT
            )
        
        # Unknown speaker type
        return SpeakerInfo(
            callsign=callsign,
            role=SpeakerRole.UNKNOWN
        )

    def _identify_atc_role(self, callsign: str, message: str) -> SpeakerRole:
        """Identify ATC role from callsign and message content."""
        combined_text = f"{callsign} {message}".lower()
        
        for role, patterns in self.atc_patterns.items():
            for pattern in patterns:
                if re.search(pattern, combined_text):
                    return role
        
        return SpeakerRole.UNKNOWN

    def _extract_position_name(self, callsign: str, message: str) -> Optional[str]:
        """Extract the full position name from callsign or message."""
        # Common position name patterns
        position_patterns = [
            r'(\w+\s+ground)',
            r'(\w+\s+tower)',
            r'(\w+\s+approach)',
            r'(\w+\s+departure)', 
            r'(\w+\s+center)',
            r'(\w+\s+control)',
        ]
        
        combined_text = f"{callsign} {message}".lower()
        
        for pattern in position_patterns:
            match = re.search(pattern, combined_text)
            if match:
                return match.group(1).title()
        
        return None

    def _is_aircraft_callsign(self, callsign: str) -> bool:
        """Check if callsign matches aircraft patterns."""
        for pattern in self.aircraft_patterns:
            if re.match(pattern, callsign):
                return True
        return False

    def _standardize_aircraft_callsign(self, callsign: str) -> str:
        """Standardize aircraft callsign format."""
        # Already in standard format
        if re.match(r'^[A-Z]{3}\d+$', callsign) or re.match(r'^N\d+[A-Z]*$', callsign):
            return callsign
            
        # Handle various formats
        # This could be extended based on actual data patterns
        return callsign

    def update_conversation_state(self, 
                                timestamp: float,
                                speaker: str, 
                                message: str,
                                t1_text: str = "",
                                t2_text: str = "") -> ConversationState:
        """
        Update the conversation state with a new message.
        
        Args:
            timestamp: Message timestamp
            speaker: Speaker identifier
            message: Original message text
            t1_text: T1 format text
            t2_text: T2 format text
            
        Returns:
            Updated conversation state
        """
        # Analyze speaker
        speaker_info = self.analyze_speaker(speaker, message)
        
        # Update active speakers
        self.conversation_state.active_speakers[speaker] = speaker_info
        speaker_info.last_seen_time = timestamp
        
        # Update conversation timeline
        self.conversation_state.conversation_timeline.append(
            (timestamp, speaker, message)
        )
        
        # Determine primary aircraft if not set
        if (speaker_info.role == SpeakerRole.PILOT and 
            self.conversation_state.primary_aircraft is None):
            self.conversation_state.primary_aircraft = speaker
        
        # Update flight phase based on communication patterns
        self._update_flight_phase(speaker_info, message)
        
        # Update last speaker
        self.conversation_state.last_speaker = speaker
        
        return self.conversation_state

    def _update_flight_phase(self, speaker_info: SpeakerInfo, message: str):
        """Update flight phase based on speaker and message content."""
        message_lower = message.lower()
        
        # Analyze message content for phase indicators
        if any(word in message_lower for word in ['taxi', 'gate', 'pushback', 'startup']):
            self.conversation_state.current_phase = FlightPhase.GROUND_OPS
        elif any(word in message_lower for word in ['takeoff', 'departure', 'runway']):
            self.conversation_state.current_phase = FlightPhase.TAKEOFF_SEQUENCE
        elif any(word in message_lower for word in ['climb', 'maintain', 'altitude']):
            self.conversation_state.current_phase = FlightPhase.DEPARTURE
        elif any(word in message_lower for word in ['cruise', 'level', 'direct']):
            self.conversation_state.current_phase = FlightPhase.EN_ROUTE
        elif any(word in message_lower for word in ['descend', 'approach', 'vector']):
            self.conversation_state.current_phase = FlightPhase.APPROACH
        elif any(word in message_lower for word in ['land', 'final', 'cleared']):
            self.conversation_state.current_phase = FlightPhase.LANDING_SEQUENCE

    def get_speaker_sequence(self) -> List[str]:
        """Get the expected speaker sequence for current flight phase."""
        phase = self.conversation_state.current_phase
        
        if phase == FlightPhase.TAKEOFF_SEQUENCE:
            return ['Ground', 'Tower', 'Departure']
        elif phase == FlightPhase.DEPARTURE:
            return ['Departure', 'Center']
        elif phase == FlightPhase.APPROACH:
            return ['Center', 'Approach']
        elif phase == FlightPhase.LANDING_SEQUENCE:
            return ['Approach', 'Tower', 'Ground']
        else:
            return ['Unknown']

    def format_speaker_for_continuous(self, 
                                    speaker: str, 
                                    format_type: str = "C1") -> str:
        """
        Format speaker identifier for continuous transcript.
        
        Args:
            speaker: Speaker identifier
            format_type: "C1" (with words) or "C2" (with numbers/abbreviations)
            
        Returns:
            Formatted speaker identifier
        """
        if speaker not in self.conversation_state.active_speakers:
            return f"({speaker})"
        
        speaker_info = self.conversation_state.active_speakers[speaker]
        
        if speaker_info.role == SpeakerRole.PILOT:
            if format_type == "C1":
                # Use conversational format for aircraft
                return f"({self._callsign_to_words(speaker)})"
            else:  # C2
                return f"({speaker})"
        else:
            # ATC position
            position_name = speaker_info.position_name or speaker_info.role.value
            return f"({position_name})"

    def _callsign_to_words(self, callsign: str) -> str:
        """Convert aircraft callsign to word format for C1."""
        # This is a simplified conversion - could be enhanced
        if re.match(r'^N\d+[A-Z]*$', callsign):
            # General aviation: N123AB → November one two three Alpha Bravo
            return f"November {callsign[1:]}"  # Simplified
        elif re.match(r'^[A-Z]{3}\d+$', callsign):
            # Airline: DAL209 → Delta two zero nine
            airline_code = callsign[:3]
            flight_num = callsign[3:]
            
            # Find airline name
            airline_name = "Unknown"
            for name, code in self.airline_codes.items():
                if code == airline_code:
                    airline_name = name.title()
                    break
            
            return f"{airline_name} {flight_num}"
        
        return callsign

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conversation state."""
        return {
            'active_speakers': {
                speaker: {
                    'role': info.role.value,
                    'position_name': info.position_name,
                    'last_seen': info.last_seen_time
                }
                for speaker, info in self.conversation_state.active_speakers.items()
            },
            'current_phase': self.conversation_state.current_phase.value,
            'primary_aircraft': self.conversation_state.primary_aircraft,
            'total_messages': len(self.conversation_state.conversation_timeline),
            'expected_sequence': self.get_speaker_sequence()
        }

    def reset_state(self):
        """Reset the conversation state for a new conversation."""
        self.conversation_state = ConversationState(
            active_speakers={},
            current_phase=FlightPhase.UNKNOWN
        ) 