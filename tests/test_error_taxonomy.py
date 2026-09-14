import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faa_cxer_ng.error_taxonomy import (
    ALL_ERROR_TYPES,
    ERROR_TYPES,
    LLM_JUDGMENT,
    RULE_CHECKABLE,
    CheckMode,
    spec_for,
)


def test_every_type_has_a_valid_severity():
    for name, spec in ERROR_TYPES.items():
        assert spec.default_severity in ("critical", "high", "medium", "low"), name


def test_rule_and_llm_sets_partition_by_check_mode():
    for name, spec in ERROR_TYPES.items():
        if spec.check_mode == CheckMode.RULE:
            assert name in RULE_CHECKABLE and name not in LLM_JUDGMENT
        elif spec.check_mode == CheckMode.LLM:
            assert name in LLM_JUDGMENT and name not in RULE_CHECKABLE
        else:
            assert name in RULE_CHECKABLE and name in LLM_JUDGMENT


def test_spec_for_unknown_type_returns_none():
    assert spec_for("not_a_real_error_type") is None


def test_all_error_types_sorted_and_matches_registry():
    assert ALL_ERROR_TYPES == sorted(ERROR_TYPES)


def test_merged_legacy_types_are_gone():
    # These were folded into other types when the taxonomy was curated —
    # they should not reappear as separate entries.
    for legacy in ("callsign_confusion", "readback_order_wrong", "incorrect_acknowledgement", "controller_no_correction"):
        assert legacy not in ERROR_TYPES
