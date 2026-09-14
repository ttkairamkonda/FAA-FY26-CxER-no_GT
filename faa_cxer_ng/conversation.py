"""Parses raw `"speaker : text"` conversation lines into structured turns.

Ported from the original `atc_pipeline.py`'s `parse_turns`/`_norm_speaker`,
unchanged in behavior — this part already worked fine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

# Field names tried, in order, when a conversation record doesn't use the
# default "conversation" key — e.g. data/ROWAN_samples.json uses "dialogue".
CONVERSATION_FIELD_CANDIDATES = ["conversation", "dialogue", "turns", "transcript"]


def resolve_turns_field(record: dict, field: Optional[str] = None) -> List[str]:
    """Finds the list of "speaker : text" lines in a conversation record.

    Deliberately does NOT fall back to an empty list on a miss — a record
    silently treated as "0 turns" produces a report that looks like a
    normal empty analysis instead of an error, which is worse than a loud
    failure. Pass `field` to force a specific key (e.g. "dialogue" for
    ROWAN_samples.json); leave it unset to auto-detect from
    `CONVERSATION_FIELD_CANDIDATES`.
    """
    if field is not None:
        if field not in record:
            raise KeyError(
                f"'{field}' not found in conversation record (ACN={record.get('ACN')}). "
                f"Available keys: {sorted(record.keys())}"
            )
        return record[field]

    for candidate in CONVERSATION_FIELD_CANDIDATES:
        value = record.get(candidate)
        if value:
            return value

    raise KeyError(
        f"No conversation field found (tried {CONVERSATION_FIELD_CANDIDATES}) in record "
        f"(ACN={record.get('ACN')}). Available keys: {sorted(record.keys())}. "
        f"Pass --conversation-field explicitly if this dataset uses a different name."
    )


_CONTROLLER_ALIASES = {
    "controller", "atc", "atco", "tower", "ground", "approach",
    "departure", "center", "radar", "tracon", "clearance", "delivery",
    "c", "ctrl", "ctl",
}
_PILOT_ALIASES = {
    "pilot", "aircraft", "ac", "flight", "crew", "p", "pil", "flightcrew", "fp",
}


@dataclass
class Turn:
    turn_index: int
    speaker: str  # "controller" | "pilot" | "unknown"
    text: str

    def as_dict(self) -> dict:
        return {"turn_index": self.turn_index, "speaker": self.speaker, "text": self.text}


def _normalize_speaker_label(raw: str) -> str:
    key = raw.strip().lower().rstrip(".")
    if key in _CONTROLLER_ALIASES:
        return "controller"
    if key in _PILOT_ALIASES:
        return "pilot"
    return "unknown"


class ConversationParser:
    """Turns a list of raw `"speaker : text"` lines into `Turn` objects."""

    def parse(self, conversation: List[str]) -> List[Turn]:
        turns = []
        for idx, line in enumerate(conversation):
            if ":" in line:
                prefix, _, text = line.partition(":")
                speaker = _normalize_speaker_label(prefix.strip())
                text = text.strip()
            else:
                speaker, text = "unknown", line.strip()

            if speaker == "unknown":
                speaker = self._guess_speaker(text)

            turns.append(Turn(turn_index=idx, speaker=speaker, text=text))
        return turns

    @staticmethod
    def _guess_speaker(text: str) -> str:
        """Linguistic fallback when there's no explicit "speaker:" prefix.
        Controllers tend to open with the callsign (capitalized word);
        pilots tend to close a transmission with their callsign."""
        words = text.split()
        if not words:
            return "unknown"
        if re.match(r"^[A-Z][a-z]", text):
            return "controller"
        if re.match(r"^[A-Z0-9]+$", words[-1]):
            return "pilot"
        return "unknown"

    @staticmethod
    def format_turns(turns: List[Turn]) -> str:
        return "\n".join(f"Turn {t.turn_index} [{t.speaker}]: {t.text}" for t in turns)
