"""Deterministic error checks over stage-1's structured extraction — zero
LLM calls. Handles every RULE and HYBRID error type from `error_taxonomy`.

Every value comparison goes through `ATCValueNormalizer.values_equivalent`,
which returns True / False / None. `None` (uncertain) is never treated as
a match or a mismatch — those items are returned separately so the caller
can put them in front of the LLM instead of guessing.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from .conversation import Turn
from .error_taxonomy import REQUIRED_ELEMENTS_BY_INSTRUCTION_TYPE, spec_for
from .normalization import ATCValueNormalizer
from .schemas import ConversationContext, DetectedError

# instruction key_values field name -> (error_type if mismatched, normalizer field_type)
_FIELD_CHECK = {
    "runway": ("runway_mismatch", "runway"),
    "altitude": ("altitude_mismatch", "altitude"),
    "flight_level": ("flight_level_mismatch", "flight_level"),
    "heading": ("heading_mismatch", "heading"),
    "speed": ("speed_mismatch", "speed"),
    "frequency": ("frequency_mismatch", "frequency"),
    "squawk": ("squawk_mismatch", "squawk"),
    "qnh": ("qnh_altimeter_mismatch", "qnh"),
    "altimeter": ("qnh_altimeter_mismatch", "qnh"),
    "crossing_restriction": ("crossing_restriction_mismatch", "generic"),
}

_ROGER_ONLY_RE = re.compile(r"^(roger|wilco|copy|copy that|affirmative|okay|ok)\.?$")

_BAD_PHRASES = ["thank you", "thanks", "no problem"]
_WORD_ORDER_VIOLATIONS = [
    (re.compile(r"\brunway\b.{0,40}?\bcleared to land\b", re.I),
     "ICAO order is 'cleared to land runway X', not 'runway X ... cleared to land'"),
]

_NITS_KEYWORDS = {
    "nature_of_emergency": ["engine", "fire", "strike", "failure", "medical", "fuel", "smoke",
                            "depressurization", "hydraulic", "bird", "gear", "electrical"],
    "intentions": ["intend", "request", "return", "divert", "continue", "land", "proceeding"],
    "souls_on_board": ["souls", "passengers", "persons on board", "pob", "people on board"],
    "time_or_fuel_remaining": ["fuel remaining", "hours of fuel", "minutes of fuel", "time remaining", "fuel on board"],
}

_POSITION_KEYWORDS = ["downwind", "base", "final", "established", "outer marker", "field in sight"]


def _err(**kwargs) -> dict:
    kwargs.setdefault("source", "rule")
    kwargs.setdefault("confidence", "confirmed")
    return DetectedError(**kwargs).model_dump()


class RuleBasedChecker:
    """`check()` returns `(confirmed_errors, uncertain_items)`. Confirmed
    errors are ready to merge into the final report as-is. Uncertain items
    are dicts describing a specific value comparison the normalizer
    couldn't resolve — the semantic LLM prompt gets asked about exactly
    those, instead of re-deriving everything from scratch."""

    def __init__(self, normalizer: ATCValueNormalizer = None):
        self.normalizer = normalizer or ATCValueNormalizer()

    def check(self, turns: List[Turn], context: ConversationContext) -> Tuple[List[dict], List[dict]]:
        errors: List[dict] = []
        uncertain: List[dict] = []

        errors += self._check_callsign_mismatches(context)
        errors += self._check_callsign_omitted(context)

        rb_errors, rb_uncertain = self._check_instruction_readback_pairs(context)
        errors += rb_errors
        uncertain += rb_uncertain

        errors += self._check_clearance_completeness(context)
        errors += self._check_emergency(turns, context)
        errors += self._check_phraseology(turns)
        errors += self._check_position_reports(turns, context)

        return errors, uncertain

    # ---- callsign ---------------------------------------------------------

    def _primary_digits(self, context: ConversationContext):
        parsed = self.normalizer.callsign_from_words(context.primary_callsign)
        return parsed.split("|", 1)[1] if parsed else None

    def _check_callsign_mismatches(self, context: ConversationContext) -> List[dict]:
        errors: List[dict] = []
        primary_digits = self._primary_digits(context)
        mentions = sorted(context.callsign_mentions, key=lambda m: m.turn_index)

        run_key, run_turns = None, []

        def flush():
            if not run_turns:
                return
            if len(run_turns) >= 2:
                errors.append(_err(
                    turn_index=run_turns[0], speaker="unknown", error_type="callsign_drift",
                    severity=spec_for("callsign_drift").default_severity,
                    expected_behavior=f"Use established callsign '{context.primary_callsign}'",
                    actual_behavior=f"A mismatched callsign variant was used consistently across turns {run_turns}",
                    related_turns=list(run_turns),
                    values_involved={"expected": context.primary_callsign},
                ))
            else:
                errors.append(_err(
                    turn_index=run_turns[0], speaker="unknown", error_type="callsign_format_error",
                    severity=spec_for("callsign_format_error").default_severity,
                    expected_behavior=f"Use established callsign '{context.primary_callsign}'",
                    actual_behavior="Callsign digits/format did not match the established callsign",
                    related_turns=list(run_turns),
                    values_involved={"expected": context.primary_callsign},
                ))

        for m in mentions:
            eq = self.normalizer.values_equivalent(context.primary_callsign, m.callsign_text, "callsign")
            if eq is True:
                flush()
                run_key, run_turns = None, []
                continue
            if eq is None:
                continue  # nothing parseable to compare — not this check's job

            parsed = self.normalizer.callsign_from_words(m.callsign_text)
            mention_digits = parsed.split("|", 1)[1] if parsed else None

            if primary_digits and mention_digits and len(mention_digits) < len(primary_digits) \
                    and primary_digits.endswith(mention_digits):
                valid_from = context.abbreviated_callsign_valid_from_turn
                if valid_from is None or m.turn_index < valid_from:
                    flush()
                    run_key, run_turns = None, []
                    errors.append(_err(
                        turn_index=m.turn_index, speaker=m.speaker, error_type="invalid_abbreviated_callsign",
                        severity=spec_for("invalid_abbreviated_callsign").default_severity,
                        expected_behavior="Full callsign until the controller establishes an abbreviated form",
                        actual_behavior=f"Used abbreviated callsign '{m.callsign_text}'",
                        related_turns=[m.turn_index],
                    ))
                # else: a validly abbreviated callsign — not an error
                continue

            key = mention_digits or m.callsign_text
            if key == run_key:
                run_turns.append(m.turn_index)
            else:
                flush()
                run_key, run_turns = key, [m.turn_index]

        flush()
        return errors

    def _check_callsign_omitted(self, context: ConversationContext) -> List[dict]:
        errors = []
        primary_digits = self._primary_digits(context)
        if not primary_digits:
            return errors
        for rb in context.actual_readbacks:
            parsed = self.normalizer.callsign_from_words(rb.pilot_text)
            text_digits = parsed.split("|", 1)[1] if parsed else None
            if not text_digits:
                errors.append(_err(
                    turn_index=rb.turn_index, speaker="pilot", error_type="callsign_omitted",
                    severity=spec_for("callsign_omitted").default_severity,
                    expected_behavior=f"Readback should include callsign '{context.primary_callsign}'",
                    actual_behavior="No callsign found in this readback turn.",
                    related_turns=[rb.turn_index],
                ))
        return errors

    # ---- instruction/readback pairs ---------------------------------------

    def _check_instruction_readback_pairs(self, context: ConversationContext) -> Tuple[List[dict], List[dict]]:
        errors: List[dict] = []
        uncertain: List[dict] = []

        by_turn: Dict[int, list] = {}
        for rb in context.actual_readbacks:
            by_turn.setdefault(rb.readback_for_turn, []).append(rb)

        for instr in context.critical_instructions:
            if not instr.requires_readback:
                continue
            matches = by_turn.get(instr.turn_index, [])
            if not matches:
                errors.append(_err(
                    turn_index=instr.turn_index, speaker="controller", error_type="missing_readback",
                    severity=spec_for("missing_readback").default_severity,
                    expected_behavior=f"Pilot readback of {instr.instruction_type}",
                    actual_behavior="No matching readback found for this instruction.",
                    related_turns=[instr.turn_index],
                ))
                continue

            rb = matches[0]
            canon = self.normalizer.canonical_text(rb.pilot_text).strip()
            if _ROGER_ONLY_RE.match(canon):
                errors.append(_err(
                    turn_index=rb.turn_index, speaker="pilot", error_type="roger_only_readback",
                    severity=spec_for("roger_only_readback").default_severity,
                    expected_behavior="A substantive readback of the instruction's content",
                    actual_behavior=f"Pilot replied only '{rb.pilot_text.strip()}'",
                    related_turns=[instr.turn_index, rb.turn_index],
                ))
                continue

            missing_elements = []
            for key, expected_val in instr.key_values.items():
                actual_val = rb.values_extracted.get(key)
                error_type, field_type = _FIELD_CHECK.get(key, ("incorrect_readback_value", "generic"))

                if actual_val is None:
                    missing_elements.append(key)
                    continue

                eq = self.normalizer.values_equivalent(expected_val, actual_val, field_type)
                if eq is True:
                    continue
                if eq is False:
                    errors.append(_err(
                        turn_index=rb.turn_index, speaker="pilot", error_type=error_type,
                        severity=spec_for(error_type).default_severity,
                        expected_behavior=f"{key} = {expected_val}", actual_behavior=f"{key} = {actual_val}",
                        related_turns=[instr.turn_index, rb.turn_index],
                        values_involved={"field": key, "expected": expected_val, "actual": actual_val},
                    ))
                else:
                    uncertain.append({
                        "instruction_turn": instr.turn_index, "readback_turn": rb.turn_index,
                        "field": key, "expected": expected_val, "actual": actual_val,
                    })

            if missing_elements:
                errors.append(_err(
                    turn_index=rb.turn_index, speaker="pilot", error_type="partial_readback",
                    severity=spec_for("partial_readback").default_severity,
                    expected_behavior=f"Readback should include: {sorted(missing_elements)}",
                    actual_behavior="These elements were not found in the readback's extracted values.",
                    related_turns=[instr.turn_index, rb.turn_index],
                    values_involved={"missing": ",".join(sorted(missing_elements))},
                ))

        return errors, uncertain

    # ---- clearance completeness --------------------------------------------

    def _check_clearance_completeness(self, context: ConversationContext) -> List[dict]:
        errors = []
        for instr in context.critical_instructions:
            required = REQUIRED_ELEMENTS_BY_INSTRUCTION_TYPE.get(instr.instruction_type)
            if required:
                missing = required - set(instr.key_values.keys())
                if missing:
                    errors.append(_err(
                        turn_index=instr.turn_index, speaker="controller", error_type="incomplete_clearance",
                        severity=spec_for("incomplete_clearance").default_severity,
                        expected_behavior=f"Instruction should specify: {sorted(missing)}",
                        actual_behavior="Not found in the instruction as extracted.",
                        related_turns=[instr.turn_index],
                        values_involved={"missing": ",".join(sorted(missing))},
                    ))

            if instr.instruction_type in ("takeoff_clearance", "landing_clearance"):
                text = self.normalizer.canonical_text(instr.raw_instruction)
                if "wind" not in text:
                    errors.append(_err(
                        turn_index=instr.turn_index, speaker="controller", error_type="missing_wind_in_clearance",
                        severity=spec_for("missing_wind_in_clearance").default_severity,
                        expected_behavior="Wind information included per FAA AIM 4-3-9",
                        actual_behavior="No wind information found in this clearance.",
                        related_turns=[instr.turn_index],
                    ))

            if "void" in instr.raw_instruction.lower() and not instr.key_values.get("void_time"):
                errors.append(_err(
                    turn_index=instr.turn_index, speaker="controller", error_type="clearance_void_time_missed",
                    severity=spec_for("clearance_void_time_missed").default_severity,
                    expected_behavior="An explicit void time",
                    actual_behavior="Instruction references a void condition but states no time.",
                    related_turns=[instr.turn_index],
                ))
        return errors

    # ---- emergency ----------------------------------------------------------

    def _check_emergency(self, turns: List[Turn], context: ConversationContext) -> List[dict]:
        if not context.emergency_declared:
            return []
        errors = []
        full_text = " ".join(t.text for t in turns).lower()

        if not any(kw in full_text for kw in ("squawk 7700", "seven seven zero zero", "7700")):
            errors.append(_err(
                turn_index=turns[-1].turn_index if turns else 0, speaker="unknown", error_type="wrong_emergency_squawk",
                severity=spec_for("wrong_emergency_squawk").default_severity,
                expected_behavior="Squawk 7700 mentioned during a declared emergency",
                actual_behavior="No mention of squawk 7700 found anywhere in the conversation.",
            ))

        missing_nits = [label for label, kws in _NITS_KEYWORDS.items() if not any(kw in full_text for kw in kws)]
        if missing_nits:
            errors.append(_err(
                turn_index=turns[-1].turn_index if turns else 0, speaker="pilot", error_type="emergency_nits_incomplete",
                severity=spec_for("emergency_nits_incomplete").default_severity,
                expected_behavior="Nature / Intentions / Souls on board / Time-or-fuel remaining, all communicated",
                actual_behavior=f"No mention found for: {missing_nits}",
                values_involved={"missing_nits_elements": ",".join(missing_nits)},
            ))
        return errors

    # ---- phraseology ----------------------------------------------------------

    def _check_phraseology(self, turns: List[Turn]) -> List[dict]:
        errors = []
        for t in turns:
            lower = t.text.lower()
            for phrase in _BAD_PHRASES:
                if phrase in lower:
                    errors.append(_err(
                        turn_index=t.turn_index, speaker=t.speaker, error_type="non_standard_phraseology",
                        severity=spec_for("non_standard_phraseology").default_severity,
                        expected_behavior="Standard ICAO phraseology (no filler acknowledgements)",
                        actual_behavior=f"Used non-standard phrase: '{phrase}'",
                        related_turns=[t.turn_index], values_involved={"phrase": phrase},
                    ))
                    break
            for pattern, note in _WORD_ORDER_VIOLATIONS:
                if pattern.search(t.text):
                    errors.append(_err(
                        turn_index=t.turn_index, speaker=t.speaker, error_type="non_standard_phraseology",
                        severity=spec_for("non_standard_phraseology").default_severity,
                        expected_behavior=note, actual_behavior=t.text, related_turns=[t.turn_index],
                    ))
        return errors

    # ---- position reports -----------------------------------------------------

    def _check_position_reports(self, turns: List[Turn], context: ConversationContext) -> List[dict]:
        errors = []
        pilot_text_all = " ".join(t.text.lower() for t in turns if t.speaker == "pilot")
        for instr in context.critical_instructions:
            if "report" not in instr.raw_instruction.lower():
                continue
            for kw in _POSITION_KEYWORDS:
                if kw in instr.raw_instruction.lower() and kw not in pilot_text_all:
                    errors.append(_err(
                        turn_index=instr.turn_index, speaker="pilot", error_type="missed_position_report",
                        severity=spec_for("missed_position_report").default_severity,
                        expected_behavior=f"Pilot reports '{kw}' as requested",
                        actual_behavior="No matching position report found in any pilot turn.",
                        related_turns=[instr.turn_index],
                    ))
                    break
        return errors
