"""Pydantic models for every pipeline stage's output. All fields are
optional or have safe defaults, matching the original pipeline's
philosophy: a model returning slightly malformed JSON should degrade
gracefully, not crash the run.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class CallsignMention(BaseModel):
    """Every time a callsign is spoken, by whom, in which turn — the raw
    material RuleBasedChecker needs to detect drift without an LLM call."""

    turn_index: int
    speaker: str
    callsign_text: str


class CriticalInstruction(BaseModel):
    turn_index: int
    instruction_type: str
    raw_instruction: str
    key_values: Dict[str, str] = {}
    requires_readback: bool = True
    readback_must_include: List[str] = []


class Readback(BaseModel):
    turn_index: int
    readback_for_turn: int
    readback_for_type: str
    pilot_text: str
    values_extracted: Dict[str, str] = {}


class ConversationContext(BaseModel):
    """Stage 1 output — the LLM's ONE job is faithful extraction, not
    judgment. Every error-detection decision downstream is either a rule
    over this structured data or a narrowly-scoped LLM question."""

    primary_callsign: str
    callsign_variations: List[str] = []
    callsign_mentions: List[CallsignMention] = []
    abbreviated_callsign_valid_from_turn: Optional[int] = None
    aircraft_type: Optional[str] = None
    conversation_phase: str = "other"
    facility_type: str = "unknown"
    critical_instructions: List[CriticalInstruction] = []
    actual_readbacks: List[Readback] = []
    total_turns: int
    controller_turns: int
    pilot_turns: int
    unknown_turns: int = 0
    emergency_declared: bool = False
    emergency_type: Optional[str] = None
    key_events: List[str] = []
    traffic_mentioned: List[str] = []
    runway_in_use: Optional[str] = None
    void_time: Optional[str] = None


class DetectedError(BaseModel):
    turn_index: int
    speaker: str
    error_type: str
    severity: Literal["critical", "high", "medium", "low"]
    expected_behavior: str = ""
    actual_behavior: str = ""
    explanation: str = ""
    related_turns: List[int] = []
    safety_impact: str = ""
    values_involved: Dict[str, str] = {}
    source: Literal["rule", "llm"] = "llm"
    confidence: Literal["confirmed", "uncertain"] = "confirmed"


class TurnAnalysis(BaseModel):
    turn_index: int
    speaker: str
    text: str
    errors_in_turn: List[str] = []


class ConversationReport(BaseModel):
    acn: Optional[int] = None
    event: Optional[str] = None
    analysis_timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    pipeline_version: str = "faa_cxer_ng-v1"
    conversation_summary: str = ""
    context: ConversationContext
    errors_detected: List[DetectedError] = []
    total_errors: int = 0
    errors_by_source: Dict[str, int] = {}
    critical_errors: List[str] = []
    safety_risk_level: str = "unknown"
    operational_status: str = "unknown"
    turn_analysis: List[TurnAnalysis] = []
    executive_summary: str = ""
    recommendations: List[str] = []
    error_statistics: Dict[str, int] = {}


def default_context(total_turns: int, controller_turns: int, pilot_turns: int, unknown_turns: int) -> dict:
    return ConversationContext(
        primary_callsign="UNKNOWN",
        total_turns=total_turns,
        controller_turns=controller_turns,
        pilot_turns=pilot_turns,
        unknown_turns=unknown_turns,
    ).model_dump()
