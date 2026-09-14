"""Stage 3 prompt: turn the merged rule-based + LLM-detected errors into a
human-readable report (summary, risk level, per-turn notes,
recommendations). No new error detection happens here — this stage only
narrates what stages 1+2 already found.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..conversation import Turn


class SynthesisPromptBuilder:
    def build(self, turns: List[Turn], context: Dict[str, Any], errors: List[dict], acn: Optional[int]) -> str:
        conv_text = "\n".join(f"Turn {t.turn_index} [{t.speaker}]: {t.text}" for t in turns)
        by_sev: Dict[str, int] = {}
        for e in errors:
            by_sev[e.get("severity", "unknown")] = by_sev.get(e.get("severity", "unknown"), 0) + 1

        return (
            "You are an expert ATC safety analyst. Write the final human-readable report "
            "for this ALREADY-ANALYZED conversation. Do not invent new errors — only "
            "narrate and assess the ones listed below.\n\n"
            f"ACN: {acn}\n"
            f"TOTAL ERRORS: {len(errors)} {by_sev}\n"
            f"CONTEXT: {json.dumps(context, indent=2)}\n"
            f"ERRORS FOUND: {json.dumps(errors, indent=2)}\n\n"
            f"CONVERSATION:\n{conv_text}\n\n"
            "PRODUCE A JSON OBJECT with these exact fields:\n\n"
            "conversation_summary: 2-4 sentences — what aircraft, what happened, the outcome.\n\n"
            "safety_risk_level: critical/high/medium/low/none\n"
            "  critical = an error that could directly cause an accident\n"
            "  high = one critical-severity error, or multiple high-severity errors\n"
            "  medium = multiple medium-severity errors\n"
            "  low = only minor deviations\n"
            "  none = zero errors\n\n"
            "operational_status: unsafe_operation/significant_concerns/minor_issues/safe_operation\n\n"
            f"turn_analysis: one entry for EVERY turn 0 to {len(turns) - 1}: "
            "turn_index, speaker, text, errors_in_turn (list of error_type strings that "
            "occurred in that turn, [] if none)\n\n"
            "executive_summary: 4-6 sentences for management/investigators.\n\n"
            "recommendations: 3-6 specific, actionable items based on the ACTUAL errors found.\n\n"
            "OUTPUT RULES:\n"
            "- Output a single JSON object, starting with { and ending with }\n"
            "- No text before or after the JSON, no markdown code fences\n\n"
            "EXAMPLE OUTPUT (structure only):\n"
            "{\n"
            '  "conversation_summary": "Delta 345 conducted a runway 28R approach after declaring an engine failure emergency and landed safely with fire trucks standing by.",\n'
            '  "safety_risk_level": "high",\n'
            '  "operational_status": "significant_concerns",\n'
            '  "turn_analysis": [\n'
            '    {"turn_index": 0, "speaker": "controller", "text": "Delta 345 taxi...", "errors_in_turn": []},\n'
            '    {"turn_index": 1, "speaker": "pilot", "text": "Taxi to runway...", "errors_in_turn": []}\n'
            "  ],\n"
            '  "executive_summary": "Delta 345 experienced an engine failure requiring an emergency return...",\n'
            '  "recommendations": ["Controllers should verify callsign accuracy before issuing further instructions after any discrepancy is noticed."]\n'
            "}\n\n"
            "Now output the JSON object for the conversation above:"
        )
