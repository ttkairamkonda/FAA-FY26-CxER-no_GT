"""Stage 1 prompt: pure extraction, no judgment. The LLM's only job is to
faithfully pull structured facts out of the conversation — every
error-detection decision downstream is either a rule over this data or a
narrowly-scoped question in the stage-2 semantic prompt.

Extends the original context prompt with `callsign_mentions` (every time a
callsign is spoken, by whom, in which turn) — the raw material
`RuleBasedChecker` needs to detect callsign drift without an LLM call.
"""

from __future__ import annotations

from typing import List

from ..conversation import Turn


class ContextExtractionPromptBuilder:
    def build(self, turns: List[Turn]) -> str:
        n = len(turns)
        nc = sum(1 for t in turns if t.speaker == "controller")
        npil = sum(1 for t in turns if t.speaker == "pilot")
        nu = sum(1 for t in turns if t.speaker == "unknown")
        conv_text = "\n".join(f"Turn {t.turn_index} [{t.speaker}]: {t.text}" for t in turns)

        # NOTE: this string is built with plain concatenation, not a single
        # f-string/`.format()` call, so JSON braces below are written as
        # single `{`/`}` — NOT the `{{`/`}}` escaping that would be needed
        # inside an f-string or a str.format() template. Getting this wrong
        # once meant the example JSON literally contained double braces,
        # and the model dutifully copied that invalid syntax into every
        # response (verified against a real Llama-3.3-70B run).
        return (
            "You are an expert ATC safety analyst. Extract structured facts from this "
            "conversation. Extract ONLY what is stated — do not judge whether anything is "
            "correct or an error; that happens in a later stage.\n\n"
            f"CONVERSATION ({n} turns, controller={nc}, pilot={npil}, unknown={nu}):\n"
            f"{conv_text}\n\n"
            "EXTRACT ALL OF THE FOLLOWING:\n"
            "1. primary_callsign: the callsign used by the correct/established aircraft "
            "(the most frequently and earliest used correct-looking form)\n"
            "2. callsign_variations: every distinct callsign wording seen, correct or not\n"
            "3. callsign_mentions: EVERY time a callsign is spoken, in ANY turn, by ANY "
            "speaker, exactly as spoken (do not normalize numbers to digits) — "
            "{turn_index, speaker, callsign_text}\n"
            "4. abbreviated_callsign_valid_from_turn: the turn where the controller first "
            "uses/accepts a shortened callsign form (null if never)\n"
            "5. aircraft_type: aircraft type if mentioned, else null\n"
            "6. conversation_phase: ground/taxi/takeoff/departure/enroute/approach/landing/"
            "emergency/other\n"
            "7. facility_type: ground/tower/approach/departure/center/unknown\n"
            "8. critical_instructions: EVERY controller instruction that normally requires a "
            "pilot readback (altitude/heading/speed assignments, runway/taxi/hold-short "
            "instructions, frequency changes, squawk codes, clearances, crossing "
            "restrictions, position-report requests). For each: turn_index, "
            "instruction_type (short snake_case label, e.g. 'takeoff_clearance', "
            "'altitude', 'hold_short'), raw_instruction (verbatim), key_values (a dict of "
            "the SPECIFIC values in the instruction, e.g. {\"runway\": \"28R\", "
            "\"altitude\": \"6000\"} using the exact words spoken), requires_readback, "
            "readback_must_include (list of value KEYS the readback should contain, e.g. "
            "[\"runway\", \"taxiway\"])\n"
            "9. actual_readbacks: for each pilot turn that reads back an instruction: "
            "turn_index, readback_for_turn (which instruction's turn_index this responds "
            "to), readback_for_type, pilot_text (verbatim), values_extracted (a dict with "
            "the SAME KEYS as that instruction's key_values, using the exact words the "
            "pilot said for each one the pilot actually included — omit a key entirely if "
            "the pilot didn't mention it)\n"
            "10. total_turns, controller_turns, pilot_turns, unknown_turns\n"
            "11. emergency_declared: true ONLY if 'mayday' or 'pan-pan' is actually said\n"
            "12. emergency_type, key_events (notable events: bird strike, engine failure, "
            "medical, etc.), traffic_mentioned (other aircraft referenced)\n"
            "13. runway_in_use, void_time (if a clearance void time is stated)\n\n"
            "OUTPUT RULES:\n"
            "- Output a single JSON object, starting with { and ending with }\n"
            "- No text before or after the JSON, no markdown code fences\n\n"
            "EXAMPLE OUTPUT (structure only — replace every value with the real "
            "conversation's content):\n"
            "{\n"
            '  "primary_callsign": "Delta 345",\n'
            '  "callsign_variations": ["Delta 345", "Delta 3456"],\n'
            '  "callsign_mentions": [\n'
            '    {"turn_index": 0, "speaker": "controller", "callsign_text": "Delta 345"},\n'
            '    {"turn_index": 6, "speaker": "controller", "callsign_text": "Delta 3456"}\n'
            "  ],\n"
            '  "abbreviated_callsign_valid_from_turn": null,\n'
            '  "aircraft_type": null,\n'
            '  "conversation_phase": "landing",\n'
            '  "facility_type": "tower",\n'
            '  "critical_instructions": [\n'
            "    {\n"
            '      "turn_index": 0,\n'
            '      "instruction_type": "hold_short",\n'
            '      "raw_instruction": "taxi to runway two eight right via Alpha, hold short of runway two eight right",\n'
            '      "key_values": {"runway": "28R", "taxiway": "Alpha"},\n'
            '      "requires_readback": true,\n'
            '      "readback_must_include": ["runway", "taxiway"]\n'
            "    }\n"
            "  ],\n"
            '  "actual_readbacks": [\n'
            "    {\n"
            '      "turn_index": 1,\n'
            '      "readback_for_turn": 0,\n'
            '      "readback_for_type": "hold_short",\n'
            '      "pilot_text": "Taxi to runway two eight right via Alpha hold short Delta 345",\n'
            '      "values_extracted": {"runway": "two eight right", "taxiway": "Alpha"}\n'
            "    }\n"
            "  ],\n"
            f'  "total_turns": {n},\n'
            f'  "controller_turns": {nc},\n'
            f'  "pilot_turns": {npil},\n'
            f'  "unknown_turns": {nu},\n'
            '  "emergency_declared": false,\n'
            '  "emergency_type": null,\n'
            '  "key_events": [],\n'
            '  "traffic_mentioned": [],\n'
            '  "runway_in_use": "28R",\n'
            '  "void_time": null\n'
            "}\n\n"
            "Now output the JSON object for the conversation above:"
        )
