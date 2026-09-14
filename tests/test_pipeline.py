"""End-to-end wiring test with a fake LLM backend — no GPU/vLLM needed.
Verifies stage1 -> rule-check -> stage2 -> stage3 actually connects
correctly and produces a valid report, and that exactly 3 batched calls
happen regardless of how many conversations are in the batch."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faa_cxer_ng.pipeline import ConversationErrorPipeline

STAGE1_RESPONSE = json.dumps({
    "primary_callsign": "Delta 345",
    "callsign_variations": ["Delta 345", "Delta 3456"],
    "callsign_mentions": [
        {"turn_index": 0, "speaker": "controller", "callsign_text": "Delta 345"},
        {"turn_index": 2, "speaker": "controller", "callsign_text": "Delta 3456"},
    ],
    "abbreviated_callsign_valid_from_turn": None,
    "aircraft_type": None,
    "conversation_phase": "landing",
    "facility_type": "tower",
    "critical_instructions": [
        {"turn_index": 0, "instruction_type": "hold_short", "raw_instruction": "taxi to runway 28R via Alpha, hold short of 28R",
         "key_values": {"runway": "28R", "taxiway": "Alpha"}, "requires_readback": True, "readback_must_include": ["runway", "taxiway"]},
    ],
    "actual_readbacks": [
        {"turn_index": 1, "readback_for_turn": 0, "readback_for_type": "hold_short",
         "pilot_text": "taxi to runway two eight right via alpha hold short delta 345",
         "values_extracted": {"runway": "two eight right", "taxiway": "alpha"}},
    ],
    "total_turns": 4, "controller_turns": 2, "pilot_turns": 2, "unknown_turns": 0,
    "emergency_declared": False, "emergency_type": None, "key_events": [], "traffic_mentioned": [],
    "runway_in_use": "28R", "void_time": None,
})

STAGE2_RESPONSE = json.dumps([
    {
        "turn_index": 2, "speaker": "controller", "error_type": "hearback_failure", "severity": "critical",
        "expected_behavior": "notice the wrong callsign", "actual_behavior": "used Delta 3456 without correction",
        "explanation": "test", "related_turns": [2], "safety_impact": "could confuse aircraft", "values_involved": {},
    }
])

STAGE3_RESPONSE = json.dumps({
    "conversation_summary": "Test conversation summary.",
    "safety_risk_level": "high",
    "operational_status": "significant_concerns",
    "turn_analysis": [{"turn_index": i, "speaker": "controller" if i % 2 == 0 else "pilot", "text": "x", "errors_in_turn": []} for i in range(4)],
    "executive_summary": "Test executive summary.",
    "recommendations": ["Fix the callsign handling."],
})


class FakeBackend:
    """Returns canned responses per stage, and counts how many times
    generate_batch was called (should be exactly 3 for one run_batch call,
    no matter how many conversations are in it)."""

    def __init__(self):
        self.call_count = 0
        self._responses_by_call = [STAGE1_RESPONSE, STAGE2_RESPONSE, STAGE3_RESPONSE]

    def generate_batch(self, prompts, max_tokens):
        response = self._responses_by_call[self.call_count]
        self.call_count += 1
        return [response for _ in prompts]


CONVERSATIONS = [
    {"ACN": 1001, "event": "test", "conversation": [
        "controller : Delta 345, taxi to runway two eight right via Alpha, hold short of runway two eight right.",
        "pilot : Taxi to runway two eight right via Alpha hold short Delta 345.",
        "controller : Delta 3456, cleared for takeoff runway two eight right.",
        "pilot : Cleared for takeoff runway two eight right, Delta 3456.",
    ]},
    {"ACN": 1002, "event": "test2", "conversation": [
        "controller : Delta 345, taxi to runway two eight right via Alpha, hold short of runway two eight right.",
        "pilot : Taxi to runway two eight right via Alpha hold short Delta 345.",
        "controller : Delta 3456, cleared for takeoff runway two eight right.",
        "pilot : Cleared for takeoff runway two eight right, Delta 3456.",
    ]},
]


def test_pipeline_produces_valid_reports_for_every_conversation():
    backend = FakeBackend()
    pipeline = ConversationErrorPipeline(backend)
    reports = pipeline.run_batch(CONVERSATIONS, verbose=False)

    assert len(reports) == 2
    for report in reports:
        assert report["acn"] in (1001, 1002)
        assert report["context"]["primary_callsign"] == "Delta 345"
        assert report["safety_risk_level"] == "high"
        assert len(report["turn_analysis"]) == 4


def test_exactly_three_batched_calls_regardless_of_conversation_count():
    backend = FakeBackend()
    pipeline = ConversationErrorPipeline(backend)
    pipeline.run_batch(CONVERSATIONS, verbose=False)
    assert backend.call_count == 3


def test_errors_include_both_rule_and_llm_sources():
    backend = FakeBackend()
    pipeline = ConversationErrorPipeline(backend)
    reports = pipeline.run_batch(CONVERSATIONS, verbose=False)

    report = reports[0]
    sources = {e["source"] for e in report["errors_detected"]}
    assert "rule" in sources  # the callsign mismatch at turn 2 is rule-detected
    assert "llm" in sources   # the hearback_failure came from the fake stage-2 response
    assert report["errors_by_source"]["rule"] >= 1
    assert report["errors_by_source"]["llm"] == 1


def test_malformed_llm_output_degrades_gracefully():
    class BrokenBackend:
        def generate_batch(self, prompts, max_tokens):
            return ["not json at all" for _ in prompts]

    pipeline = ConversationErrorPipeline(BrokenBackend())
    reports = pipeline.run_batch(CONVERSATIONS[:1], verbose=False)
    assert len(reports) == 1
    assert reports[0]["context"]["primary_callsign"] == "UNKNOWN"
    assert reports[0]["safety_risk_level"] == "unknown"
