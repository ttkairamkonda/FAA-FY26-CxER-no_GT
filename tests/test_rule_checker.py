import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faa_cxer_ng.conversation import Turn
from faa_cxer_ng.rule_checker import RuleBasedChecker
from faa_cxer_ng.schemas import CallsignMention, ConversationContext, CriticalInstruction, Readback

checker = RuleBasedChecker()


def _turns(n):
    return [Turn(turn_index=i, speaker="controller" if i % 2 == 0 else "pilot", text=f"turn {i}") for i in range(n)]


def _base_context(**overrides):
    defaults = dict(primary_callsign="Delta 345", total_turns=4, controller_turns=2, pilot_turns=2)
    defaults.update(overrides)
    return ConversationContext(**defaults)


# ---- callsign: verbalization difference must NOT be flagged ---------------

def test_verbalization_difference_is_not_flagged_as_error():
    ctx = _base_context(callsign_mentions=[
        CallsignMention(turn_index=0, speaker="controller", callsign_text="Delta 345"),
        CallsignMention(turn_index=1, speaker="pilot", callsign_text="Delta three four five"),
    ])
    errors, uncertain = checker.check(_turns(2), ctx)
    callsign_errors = [e for e in errors if e["error_type"].startswith("callsign")]
    assert callsign_errors == []


def test_isolated_callsign_slip_is_format_error_not_drift():
    ctx = _base_context(callsign_mentions=[
        CallsignMention(turn_index=0, speaker="controller", callsign_text="Delta 345"),
        CallsignMention(turn_index=6, speaker="controller", callsign_text="Delta 3456"),  # one-off slip
        CallsignMention(turn_index=7, speaker="pilot", callsign_text="Delta 345"),  # back to correct
    ])
    errors, _ = checker.check(_turns(8), ctx)
    types = [e["error_type"] for e in errors if e["error_type"].startswith("callsign")]
    assert "callsign_format_error" in types
    assert "callsign_drift" not in types


def test_sustained_wrong_callsign_is_drift_not_repeated_format_errors():
    ctx = _base_context(callsign_mentions=[
        CallsignMention(turn_index=0, speaker="controller", callsign_text="Delta 345"),
        CallsignMention(turn_index=6, speaker="controller", callsign_text="Delta 3456"),
        CallsignMention(turn_index=7, speaker="pilot", callsign_text="Delta 3456"),
        CallsignMention(turn_index=8, speaker="controller", callsign_text="Delta 3456"),
    ])
    errors, _ = checker.check(_turns(9), ctx)
    types = [e["error_type"] for e in errors if e["error_type"].startswith("callsign")]
    assert types.count("callsign_drift") == 1
    assert "callsign_format_error" not in types


def test_valid_abbreviation_after_established_turn_is_not_an_error():
    ctx = _base_context(
        abbreviated_callsign_valid_from_turn=5,
        callsign_mentions=[
            CallsignMention(turn_index=0, speaker="controller", callsign_text="Delta 345"),
            CallsignMention(turn_index=6, speaker="pilot", callsign_text="Delta 45"),  # valid abbreviation
        ],
    )
    errors, _ = checker.check(_turns(7), ctx)
    assert [e for e in errors if e["error_type"] == "invalid_abbreviated_callsign"] == []


def test_abbreviation_before_established_turn_is_flagged():
    ctx = _base_context(
        abbreviated_callsign_valid_from_turn=5,
        callsign_mentions=[
            CallsignMention(turn_index=0, speaker="controller", callsign_text="Delta 345"),
            CallsignMention(turn_index=2, speaker="pilot", callsign_text="Delta 45"),  # too early
        ],
    )
    errors, _ = checker.check(_turns(3), ctx)
    assert any(e["error_type"] == "invalid_abbreviated_callsign" for e in errors)


# ---- readback: missing / roger-only / partial / value mismatch -----------

def test_missing_readback_detected():
    ctx = _base_context(critical_instructions=[
        CriticalInstruction(turn_index=0, instruction_type="altitude", raw_instruction="climb to 6000",
                            key_values={"altitude": "6000"}, requires_readback=True),
    ])
    errors, _ = checker.check(_turns(1), ctx)
    assert any(e["error_type"] == "missing_readback" for e in errors)


def test_roger_only_readback_detected():
    ctx = _base_context(
        critical_instructions=[CriticalInstruction(turn_index=0, instruction_type="altitude",
                                                    raw_instruction="climb to 6000", key_values={"altitude": "6000"})],
        actual_readbacks=[Readback(turn_index=1, readback_for_turn=0, readback_for_type="altitude", pilot_text="Roger")],
    )
    errors, _ = checker.check(_turns(2), ctx)
    assert any(e["error_type"] == "roger_only_readback" for e in errors)


def test_value_mismatch_after_normalization_is_a_real_mismatch():
    ctx = _base_context(
        critical_instructions=[CriticalInstruction(turn_index=0, instruction_type="runway",
                                                    raw_instruction="cleared to land runway 27L",
                                                    key_values={"runway": "27L"})],
        actual_readbacks=[Readback(turn_index=1, readback_for_turn=0, readback_for_type="runway",
                                   pilot_text="cleared to land runway two seven right",
                                   values_extracted={"runway": "two seven right"})],
    )
    errors, _ = checker.check(_turns(2), ctx)
    assert any(e["error_type"] == "runway_mismatch" for e in errors)


def test_value_match_after_normalization_is_not_flagged():
    ctx = _base_context(
        critical_instructions=[CriticalInstruction(turn_index=0, instruction_type="runway",
                                                    raw_instruction="cleared to land runway 27L",
                                                    key_values={"runway": "27L"})],
        actual_readbacks=[Readback(turn_index=1, readback_for_turn=0, readback_for_type="runway",
                                   pilot_text="cleared to land runway two seven left",
                                   values_extracted={"runway": "two seven left"})],
    )
    errors, _ = checker.check(_turns(2), ctx)
    assert [e for e in errors if e["error_type"] == "runway_mismatch"] == []


def test_uncertain_value_pair_is_escalated_not_guessed():
    ctx = _base_context(
        critical_instructions=[CriticalInstruction(turn_index=0, instruction_type="taxi",
                                                    raw_instruction="taxi via alpha", key_values={"taxiway": "the assigned route"})],
        actual_readbacks=[Readback(turn_index=1, readback_for_turn=0, readback_for_type="taxi",
                                   pilot_text="taxiing now", values_extracted={"taxiway": "unclear mumbling"})],
    )
    errors, uncertain = checker.check(_turns(2), ctx)
    assert any(u["field"] == "taxiway" for u in uncertain)


def test_partial_readback_lists_missing_elements():
    ctx = _base_context(
        critical_instructions=[CriticalInstruction(turn_index=0, instruction_type="hold_short",
                                                    raw_instruction="taxi to runway 28R via Alpha, hold short of 28R",
                                                    key_values={"runway": "28R", "taxiway": "Alpha"})],
        actual_readbacks=[Readback(turn_index=1, readback_for_turn=0, readback_for_type="hold_short",
                                   pilot_text="taxi via alpha", values_extracted={"taxiway": "Alpha"})],
    )
    errors, _ = checker.check(_turns(2), ctx)
    partials = [e for e in errors if e["error_type"] == "partial_readback"]
    assert len(partials) == 1
    assert "runway" in partials[0]["values_involved"]["missing"]


# ---- emergency --------------------------------------------------------------

def test_emergency_without_squawk_7700_is_flagged():
    ctx = _base_context(emergency_declared=True, total_turns=1, controller_turns=0, pilot_turns=1)
    turns = [Turn(0, "pilot", "mayday mayday, engine failure, souls on board two, requesting return")]
    errors, _ = checker.check(turns, ctx)
    assert any(e["error_type"] == "wrong_emergency_squawk" for e in errors)


def test_emergency_with_squawk_7700_not_flagged():
    ctx = _base_context(emergency_declared=True, total_turns=1, controller_turns=0, pilot_turns=1)
    turns = [Turn(0, "pilot", "mayday, engine failure, squawking 7700, two souls on board, fuel remaining two hours, intend to return")]
    errors, _ = checker.check(turns, ctx)
    assert [e for e in errors if e["error_type"] == "wrong_emergency_squawk"] == []


def test_no_emergency_checks_when_not_declared():
    ctx = _base_context(emergency_declared=False, total_turns=1, controller_turns=1, pilot_turns=0)
    errors, _ = checker.check(_turns(1), ctx)
    assert [e for e in errors if e["error_type"] in ("wrong_emergency_squawk", "emergency_nits_incomplete")] == []


# ---- phraseology --------------------------------------------------------------

def test_filler_phrase_flagged():
    turns = [Turn(0, "pilot", "Taxi to apron via Bravo, thank you, Delta 345")]
    errors = checker._check_phraseology(turns)
    assert any(e["error_type"] == "non_standard_phraseology" for e in errors)


def test_word_order_violation_flagged():
    turns = [Turn(0, "controller", "Delta 345, runway two eight right, cleared to land")]
    errors = checker._check_phraseology(turns)
    assert any(e["error_type"] == "non_standard_phraseology" for e in errors)
