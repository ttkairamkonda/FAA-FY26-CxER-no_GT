"""Stage 2 prompt: semantic error detection — ONLY for the error types that
genuinely need judgment (`error_taxonomy.LLM_JUDGMENT`), plus any specific
instruction/readback value pairs `RuleBasedChecker` couldn't confidently
resolve on its own.

This is the key difference from the original `atc_pipeline.py`, which
asked one prompt to check all 43 types across the whole conversation at
once. Here the mechanical categories are already handled for free by
`RuleBasedChecker` — this prompt only has to reason about the ~15
categories that actually require understanding meaning, intent, or
cross-turn context, so it isn't competing with mechanical checks for the
model's attention.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from ..conversation import Turn
from ..error_taxonomy import ERROR_TYPES, LLM_JUDGMENT


class SemanticErrorPromptBuilder:
    def build(self, turns: List[Turn], context: Dict[str, Any], uncertain_value_pairs: List[dict]) -> str:
        conv_text = "\n".join(f"Turn {t.turn_index} [{t.speaker}]: {t.text}" for t in turns)
        primary = context.get("primary_callsign", "UNKNOWN")
        aircraft = context.get("aircraft_type") or "unknown type"
        instructions = json.dumps(context.get("critical_instructions", []), indent=2)
        readbacks = json.dumps(context.get("actual_readbacks", []), indent=2)
        emergency = context.get("emergency_declared", False)
        events = context.get("key_events", [])
        runway = context.get("runway_in_use") or "unknown"

        type_lines = "\n".join(
            f"  {name}: {ERROR_TYPES[name].description}" for name in sorted(LLM_JUDGMENT)
        )

        uncertain_section = ""
        if uncertain_value_pairs:
            uncertain_section = (
                "\nVALUE PAIRS A RULE-BASED CHECK COULD NOT CONFIDENTLY RESOLVE — decide "
                "for each whether the readback value actually matches the instruction value "
                "in MEANING (even if worded completely differently), and only flag it as an "
                "error (incorrect_readback_value) if it genuinely differs:\n"
                + json.dumps(uncertain_value_pairs, indent=2) + "\n"
            )

        return (
            "You are an expert ATC safety analyst. This conversation's mechanical/structural "
            "errors (missing readbacks, roger-only acknowledgements, digit/value mismatches, "
            "callsign format, emergency checklist presence) have ALREADY been checked "
            "separately. Your job is ONLY to find errors that require understanding MEANING, "
            "INTENT, or the relationship between turns — do not re-report a simple value or "
            "format mismatch; only report it here if it's in the uncertain list below.\n\n"
            f'PRIMARY CALLSIGN: "{primary}" ({aircraft})\n'
            f"Runway: {runway} | Emergency declared: {emergency} | Key events: {events}\n\n"
            f"INSTRUCTIONS ISSUED:\n{instructions}\n\n"
            f"READBACKS GIVEN:\n{readbacks}\n\n"
            f"CONVERSATION:\n{conv_text}\n"
            f"{uncertain_section}\n"
            f"CHECK ONLY THESE ERROR TYPES (each needs judgment, not just a lookup):\n{type_lines}\n\n"
            "OUTPUT RULES:\n"
            "- Output a JSON array. Output [] if you find nothing.\n"
            "- Start with [ and end with ]. No text before or after. No markdown fences.\n\n"
            "EXAMPLE OUTPUT (structure only):\n"
            "[\n"
            "  {\n"
            '    "turn_index": 9,\n'
            '    "speaker": "controller",\n'
            '    "error_type": "hearback_failure",\n'
            '    "severity": "critical",\n'
            '    "expected_behavior": "Controller should notice the pilot is using the wrong callsign and correct it",\n'
            '    "actual_behavior": "Controller continues addressing the aircraft as Delta 3456 without correction",\n'
            '    "explanation": "The controller accepted and perpetuated an incorrect callsign across multiple turns.",\n'
            '    "related_turns": [6, 7, 9],\n'
            '    "safety_impact": "Instructions could be misapplied if a different aircraft with a similar callsign exists.",\n'
            '    "values_involved": {}\n'
            "  }\n"
            "]\n\n"
            "Now output the JSON array for the conversation above:"
        )
