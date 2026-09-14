import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faa_cxer_ng.conversation import resolve_turns_field


def test_default_conversation_field():
    record = {"ACN": 1, "conversation": ["a : hi", "b : hello"]}
    assert resolve_turns_field(record) == ["a : hi", "b : hello"]


def test_auto_detects_dialogue_field():
    # matches data/ROWAN_samples.json's schema
    record = {"ACN": 1, "dialogue": ["approach : hi", "pilot : hello"]}
    assert resolve_turns_field(record) == ["approach : hi", "pilot : hello"]


def test_explicit_field_overrides_autodetect():
    record = {"ACN": 1, "conversation": ["x"], "dialogue": ["y"]}
    assert resolve_turns_field(record, field="dialogue") == ["y"]


def test_explicit_field_missing_raises_with_available_keys_listed():
    record = {"ACN": 1, "narrative_1": "..."}
    with pytest.raises(KeyError, match="narrative_1"):
        resolve_turns_field(record, field="conversation")


def test_no_recognized_field_raises_instead_of_silently_empty():
    record = {"ACN": 1, "narrative_1": "a free-text report with no turn structure"}
    with pytest.raises(KeyError, match="--conversation-field"):
        resolve_turns_field(record)
