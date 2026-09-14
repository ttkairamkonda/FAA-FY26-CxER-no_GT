"""Curated ATC communication error taxonomy.

Revises the original 43-type list from `atc_pipeline.py`:

  - Four types were merged into a near-duplicate that already covers the
    same underlying phenomenon, which was fragmenting the LLM's attention
    across labels that mean almost the same thing:
      callsign_confusion       -> callsign_drift / wrong_aircraft_responding
      readback_order_wrong     -> non_standard_phraseology
      incorrect_acknowledgement-> roger_only_readback / protocol_violation
      controller_no_correction -> hearback_failure

  - Every remaining type is tagged with a CheckMode saying HOW it gets
    detected:
      RULE   - a deterministic check over stage-1's structured extraction
               (values compared via ATCValueNormalizer, presence/absence
               checks, turn-position logic). Zero LLM calls, zero
               hallucination risk, unit-testable.
      HYBRID - a rule check handles the common case; genuinely ambiguous
               instances get escalated into the semantic LLM prompt.
      LLM    - inherently requires contextual/semantic judgment (was the
               emergency response operationally adequate? is this phrasing
               ambiguous? did the controller notice and correct an error?)
               that no rule can decide.

This split is what lets the pipeline reserve its one semantic-detection
LLM call for questions that actually need judgment, instead of asking a
single prompt to check all 39 categories across an entire conversation at
once (the "only catches the obvious ones" problem in the original
`atc_pipeline.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class CheckMode(str, Enum):
    RULE = "rule"
    HYBRID = "hybrid"
    LLM = "llm"


class Category(str, Enum):
    CALLSIGN = "callsign"
    READBACK = "readback"
    HEARBACK = "hearback"
    VALUE_MISMATCH = "value_mismatch"
    CLEARANCE = "clearance"
    EMERGENCY = "emergency"
    PHRASEOLOGY = "phraseology"
    SEMANTIC = "semantic"
    SEQUENCING = "sequencing"


@dataclass(frozen=True)
class ErrorTypeSpec:
    name: str
    category: Category
    check_mode: CheckMode
    default_severity: str  # "critical" | "high" | "medium" | "low"
    description: str


ERROR_TYPES: Dict[str, ErrorTypeSpec] = {
    spec.name: spec
    for spec in [
        # ---- Callsign ------------------------------------------------------
        ErrorTypeSpec("callsign_format_error", Category.CALLSIGN, CheckMode.RULE, "high",
                      "A single turn's callsign digits/letters don't match the established primary callsign."),
        ErrorTypeSpec("callsign_drift", Category.CALLSIGN, CheckMode.RULE, "high",
                      "A wrong callsign variant is used consistently across 2+ turns instead of a one-off slip."),
        ErrorTypeSpec("callsign_omitted", Category.CALLSIGN, CheckMode.RULE, "low",
                      "A readback turn has no recognizable callsign at all."),
        ErrorTypeSpec("invalid_abbreviated_callsign", Category.CALLSIGN, CheckMode.RULE, "medium",
                      "A shortened callsign form is used before ICAO rules allow it (controller hasn't abbreviated first)."),
        ErrorTypeSpec("wrong_aircraft_responding", Category.CALLSIGN, CheckMode.LLM, "critical",
                      "An instruction addressed to one aircraft appears to be acted on by a different one."),

        # ---- Readback --------------------------------------------------------
        ErrorTypeSpec("missing_readback", Category.READBACK, CheckMode.RULE, "high",
                      "A critical instruction requiring readback has no corresponding pilot readback at all."),
        ErrorTypeSpec("roger_only_readback", Category.READBACK, CheckMode.RULE, "high",
                      "Pilot acknowledges with only 'roger'/'wilco' — not a valid readback under ICAO Annex 11."),
        ErrorTypeSpec("partial_readback", Category.READBACK, CheckMode.RULE, "medium",
                      "The readback omits one or more required elements (per the instruction's readback_must_include list)."),
        ErrorTypeSpec("incorrect_readback_value", Category.READBACK, CheckMode.HYBRID, "critical",
                      "A readback value differs from the instruction's value after normalizing ICAO phonetics/number words."),
        ErrorTypeSpec("wrong_clearance_type", Category.READBACK, CheckMode.LLM, "high",
                      "Pilot's readback implies a different clearance type than what was actually issued."),

        # ---- Hearback (controller-side) --------------------------------------
        ErrorTypeSpec("hearback_failure", Category.HEARBACK, CheckMode.LLM, "critical",
                      "Controller fails to notice/correct a wrong or incomplete pilot readback."),

        # ---- Value mismatches (instruction vs readback) -----------------------
        ErrorTypeSpec("altitude_mismatch", Category.VALUE_MISMATCH, CheckMode.HYBRID, "critical", "Altitude value differs between instruction and readback."),
        ErrorTypeSpec("flight_level_mismatch", Category.VALUE_MISMATCH, CheckMode.HYBRID, "critical", "Flight level value differs between instruction and readback."),
        ErrorTypeSpec("heading_mismatch", Category.VALUE_MISMATCH, CheckMode.HYBRID, "high", "Heading value differs between instruction and readback."),
        ErrorTypeSpec("runway_mismatch", Category.VALUE_MISMATCH, CheckMode.HYBRID, "critical", "Runway designator differs between instruction and readback."),
        ErrorTypeSpec("speed_mismatch", Category.VALUE_MISMATCH, CheckMode.HYBRID, "medium", "Speed value differs between instruction and readback."),
        ErrorTypeSpec("frequency_mismatch", Category.VALUE_MISMATCH, CheckMode.HYBRID, "medium", "Frequency value differs between instruction and readback."),
        ErrorTypeSpec("squawk_mismatch", Category.VALUE_MISMATCH, CheckMode.HYBRID, "high", "Transponder code differs between instruction and readback."),
        ErrorTypeSpec("qnh_altimeter_mismatch", Category.VALUE_MISMATCH, CheckMode.HYBRID, "medium", "QNH/altimeter setting differs between instruction and readback."),
        ErrorTypeSpec("crossing_restriction_mismatch", Category.VALUE_MISMATCH, CheckMode.HYBRID, "high", "A crossing restriction value differs between instruction and readback."),

        # ---- Clearance -----------------------------------------------------
        ErrorTypeSpec("incomplete_clearance", Category.CLEARANCE, CheckMode.HYBRID, "high",
                      "Controller's clearance omits an element required for its instruction type (checked against a per-type checklist)."),
        ErrorTypeSpec("missing_wind_in_clearance", Category.CLEARANCE, CheckMode.RULE, "low",
                      "Takeoff/landing clearance omits wind information (FAA AIM 4-3-9)."),
        ErrorTypeSpec("clearance_void_time_missed", Category.CLEARANCE, CheckMode.RULE, "high",
                      "A clearance that requires a void time doesn't state one."),
        ErrorTypeSpec("conditional_clearance_error", Category.CLEARANCE, CheckMode.LLM, "critical",
                      "A conditional ('behind the X') clearance's condition isn't correctly read back first."),
        ErrorTypeSpec("hold_short_violation", Category.CLEARANCE, CheckMode.LLM, "critical",
                      "Aircraft appears to proceed past a hold-short point without clearance."),
        ErrorTypeSpec("runway_incursion_risk", Category.CLEARANCE, CheckMode.LLM, "critical",
                      "Conversation content suggests a runway incursion risk beyond a simple value mismatch."),

        # ---- Emergency -------------------------------------------------------
        ErrorTypeSpec("emergency_not_declared", Category.EMERGENCY, CheckMode.LLM, "critical",
                      "Described events (engine failure, bird strike, etc.) warrant a mayday/pan-pan that was never declared."),
        ErrorTypeSpec("wrong_emergency_squawk", Category.EMERGENCY, CheckMode.RULE, "high",
                      "Emergency declared but squawk 7700 is never mentioned."),
        ErrorTypeSpec("emergency_nits_incomplete", Category.EMERGENCY, CheckMode.RULE, "medium",
                      "Nature/Intentions/Time-or-fuel/Souls-on-board elements are not all present during a declared emergency."),
        ErrorTypeSpec("lost_comm_procedure_error", Category.EMERGENCY, CheckMode.LLM, "critical",
                      "Lost-communication procedures aren't followed correctly."),

        # ---- Phraseology -----------------------------------------------------
        ErrorTypeSpec("non_standard_phraseology", Category.PHRASEOLOGY, CheckMode.HYBRID, "low",
                      "Non-ICAO phrasing (filler words, wrong clearance word order) — checked against known bad patterns, novel cases go to the LLM."),
        ErrorTypeSpec("protocol_violation", Category.PHRASEOLOGY, CheckMode.LLM, "medium",
                      "Required communication protocol elements are missing in a way not covered by a more specific type."),
        ErrorTypeSpec("ambiguous_transmission", Category.PHRASEOLOGY, CheckMode.LLM, "medium",
                      "A transmission is interpretable in more than one way."),

        # ---- Semantic --------------------------------------------------------
        ErrorTypeSpec("semantic_misunderstanding", Category.SEMANTIC, CheckMode.LLM, "high",
                      "Pilot's response indicates a different understanding of the instruction than what was said."),
        ErrorTypeSpec("instruction_not_followed", Category.SEMANTIC, CheckMode.LLM, "critical",
                      "Described subsequent action contradicts an issued instruction."),
        ErrorTypeSpec("wrong_traffic_callsign", Category.SEMANTIC, CheckMode.LLM, "high",
                      "A traffic advisory's callsign is confused with the addressed aircraft's own callsign."),

        # ---- Sequencing --------------------------------------------------------
        ErrorTypeSpec("out_of_sequence_response", Category.SEQUENCING, CheckMode.LLM, "medium",
                      "A response addresses something out of logical/temporal order."),
        ErrorTypeSpec("missed_position_report", Category.SEQUENCING, CheckMode.RULE, "medium",
                      "Controller requested a position report ('report downwind', 'report established') that never came."),
        ErrorTypeSpec("premature_frequency_change", Category.SEQUENCING, CheckMode.LLM, "medium",
                      "Pilot appears to change frequency before being released to do so."),
    ]
}

RULE_CHECKABLE = {name for name, spec in ERROR_TYPES.items() if spec.check_mode in (CheckMode.RULE, CheckMode.HYBRID)}
LLM_JUDGMENT = {name for name, spec in ERROR_TYPES.items() if spec.check_mode in (CheckMode.LLM, CheckMode.HYBRID)}
ALL_ERROR_TYPES = sorted(ERROR_TYPES)

# Value-mismatch categories -> the ATCValueNormalizer field_type used to
# compare their instruction/readback values.
VALUE_MISMATCH_FIELD_TYPES = {
    "altitude_mismatch": "altitude",
    "flight_level_mismatch": "flight_level",
    "heading_mismatch": "heading",
    "runway_mismatch": "runway",
    "speed_mismatch": "speed",
    "frequency_mismatch": "frequency",
    "squawk_mismatch": "squawk",
    "qnh_altimeter_mismatch": "generic",
    "crossing_restriction_mismatch": "generic",
}

# Required key_values elements per instruction_type, for incomplete_clearance.
# Not exhaustive — an instruction_type not listed here is skipped by the
# rule check (nothing to compare against) rather than guessed at.
REQUIRED_ELEMENTS_BY_INSTRUCTION_TYPE = {
    "takeoff_clearance": {"runway"},
    "landing_clearance": {"runway"},
    "taxi": {"runway"},
    "hold_short": {"runway"},
    "altitude": {"altitude"},
    "heading": {"heading"},
    "frequency_change": {"frequency"},
    "squawk": {"squawk"},
}


def spec_for(error_type: str) -> Optional[ErrorTypeSpec]:
    return ERROR_TYPES.get(error_type)
