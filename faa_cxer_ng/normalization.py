"""ATC-domain text normalization: resolves phonetic/verbalization
differences ("United 123" vs "United one two three", "tree fife zero" vs
"350") so the rule-based checker doesn't flag a wording difference as a
real error — the exact problem flagged when this package was scoped:
"if it's contextually the same but character-wise different [...] handle
them appropriately."

Every `normalize_*`/`*_equivalent` method returns `None` (not a guess)
when it can't confidently parse its input. `None` means "uncertain" —
callers (the rule checker) treat that as a signal to escalate to the LLM
rather than silently assuming a match OR a mismatch.
"""

from __future__ import annotations

import re
from typing import List, Optional

# ICAO phonetic alphabet -> letter. Used for callsign suffixes and runway
# letters ("Cessna Alpha Bravo" -> "AB", "runway niner right" plus a
# phonetic taxiway "Alpha" -> "A").
PHONETIC_ALPHABET = {
    "alpha": "A", "alfa": "A", "bravo": "B", "charlie": "C", "delta": "D",
    "echo": "E", "foxtrot": "F", "golf": "G", "hotel": "H", "india": "I",
    "juliet": "J", "juliett": "J", "kilo": "K", "lima": "L", "mike": "M",
    "november": "N", "oscar": "O", "papa": "P", "quebec": "Q", "romeo": "R",
    "sierra": "S", "tango": "T", "uniform": "U", "victor": "V",
    "whiskey": "W", "xray": "X", "x-ray": "X", "yankee": "Y", "zulu": "Z",
}

# Digit-by-digit words — how ATC actually reads callsigns, headings,
# frequencies, squawk codes, flight levels, and runway numbers (never as a
# compound cardinal number: "flight level three five zero", not "flight
# level three hundred fifty").
DIGIT_WORDS = {
    "zero": "0", "oh": "0", "one": "1", "wun": "1", "two": "2", "too": "2",
    "three": "3", "tree": "3", "four": "4", "fower": "4", "five": "5",
    "fife": "5", "six": "6", "seven": "7", "eight": "8", "ait": "8",
    "nine": "9", "niner": "9",
}

DECIMAL_WORDS = {"decimal", "point"}
RUNWAY_SIDE_WORDS = {"left": "L", "right": "R", "center": "C", "centre": "C"}

# Cardinal-number grammar, for altitudes/speeds spoken as compound numbers
# ("six thousand", "one thousand two hundred") rather than digit by digit.
_ONES = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
_TEENS = {
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_SCALES = {"hundred": 100, "thousand": 1000}
_UNIT_WORDS = {"feet", "ft", "knots", "kts", "foot"}


def _words(text: str) -> List[str]:
    cleaned = re.sub(r"[^\w\s./-]", "", text or "").strip().lower()
    return cleaned.split()


class ATCValueNormalizer:
    """Resolves ATC verbalization differences to a canonical form. See
    module docstring for the None-means-uncertain contract."""

    # ---- free-text canonicalization (for substring/containment checks) --

    def canonical_text(self, text: str) -> str:
        """Lowercase, de-punctuate, and inline-expand digit/phonetic words
        — "united one two three" -> "united 123". For containment checks
        ("does this readback mention the runway"), not exact equality."""
        tokens = []
        for w in _words(text):
            if w in DIGIT_WORDS:
                tokens.append(DIGIT_WORDS[w])
            elif w in PHONETIC_ALPHABET:
                tokens.append(PHONETIC_ALPHABET[w].lower())
            elif w in RUNWAY_SIDE_WORDS:
                tokens.append(RUNWAY_SIDE_WORDS[w].lower())
            elif w in DECIMAL_WORDS:
                tokens.append(".")
            else:
                tokens.append(w)
        return " ".join(self._collapse_digit_runs(tokens))

    @staticmethod
    def _collapse_digit_runs(tokens: List[str]) -> List[str]:
        """Glues adjacent single-digit tokens (and a "." decimal marker)
        into one run: ["3", "5", "0"] -> ["350"], ["1","2","1",".","5"] -> ["121.5"]."""
        merged: List[str] = []
        for tok in tokens:
            prev = merged[-1] if merged else None
            if tok == "." and prev is not None and re.fullmatch(r"\d+", prev):
                merged[-1] = prev + "."
            elif prev is not None and prev.endswith(".") and tok.isdigit():
                merged[-1] = prev + tok
            elif tok.isdigit() and prev is not None and prev.isdigit():
                merged[-1] = prev + tok
            else:
                merged.append(tok)
        return merged

    # ---- digit-by-digit fields -------------------------------------------

    def digits_from_words(self, text: str) -> Optional[str]:
        """"tree fife zero" -> "350", "one two one decimal five" -> "121.5".
        None if no recognizable digit words are present at all."""
        tokens, saw_digit = [], False
        for w in _words(text):
            if w in DIGIT_WORDS:
                tokens.append(DIGIT_WORDS[w])
                saw_digit = True
            elif w in DECIMAL_WORDS:
                tokens.append(".")
            elif re.fullmatch(r"\d+(\.\d+)?", w):
                tokens.append(w)
                saw_digit = True
            # unrecognized words (units, "flight", "level") carry no digit
            # information and are dropped rather than invalidating the parse
        if not saw_digit:
            return None
        return "".join(self._collapse_digit_runs(tokens))

    def runway_from_words(self, text: str) -> Optional[str]:
        """"two seven left" -> "27L", "runway 9R" -> "9R"."""
        digits, side, found = [], "", False
        for w in _words(text):
            if w == "runway":
                continue
            if w in DIGIT_WORDS:
                digits.append(DIGIT_WORDS[w])
                found = True
            elif w in RUNWAY_SIDE_WORDS:
                side = RUNWAY_SIDE_WORDS[w]
                found = True
            else:
                m = re.fullmatch(r"(\d{1,2})([LRC]?)", w.upper())
                if m:
                    digits.append(m.group(1))
                    side = m.group(2) or side
                    found = True
        if not found:
            return None
        return "".join(digits) + side

    def callsign_from_words(self, text: str) -> Optional[str]:
        """Splits an operator name from its digit sequence so verbalization
        doesn't matter: "united one two three" and "united 123" both ->
        "united|123". None if no digits are found (nothing to key on).

        Phonetic-alphabet words (used for GA tail-number suffixes, e.g.
        "...one two three Alpha") only count as letters AFTER a digit has
        already appeared — several real airline names ARE ICAO phonetic
        words ("Delta" = D), so a leading phonetic word is the airline
        name, not a letter, until a digit shows this is a tail-number tail.
        """
        name_parts, digit_parts = [], []
        seen_digit = False
        for w in _words(text):
            if w in DIGIT_WORDS:
                digit_parts.append(DIGIT_WORDS[w])
                seen_digit = True
            elif re.fullmatch(r"\d+", w):
                digit_parts.extend(list(w))
                seen_digit = True
            elif w in PHONETIC_ALPHABET and seen_digit:
                digit_parts.append(PHONETIC_ALPHABET[w])
            else:
                name_parts.append(w)
        if not digit_parts:
            return None
        return "{}|{}".format(" ".join(name_parts), "".join(digit_parts))

    # ---- cardinal numbers (altitudes/speeds spoken as compound numbers) -

    def cardinal_number_from_words(self, text: str) -> Optional[int]:
        """"six thousand" -> 6000, "one thousand two hundred" -> 1200.
        Enforces real English number grammar (a ones-word can't directly
        follow another ones-word without an intervening scale word) so
        digit-by-digit speech like "two five zero" is correctly rejected
        here (-> None, forcing the digits_from_words fallback) instead of
        being misparsed as 2+5+0=7."""
        words = [w for w in _words(text) if w not in _UNIT_WORDS]
        if not words:
            return None

        total, current, last_type = 0, 0, None
        for w in words:
            if w in _ONES or w.isdigit():
                if last_type not in (None, "scale", "tens"):
                    return None
                current += _ONES[w] if w in _ONES else int(w)
                last_type = "ones"
            elif w in _TEENS:
                if last_type not in (None, "scale"):
                    return None
                current += _TEENS[w]
                last_type = "teens"
            elif w in _TENS:
                if last_type not in (None, "scale"):
                    return None
                current += _TENS[w]
                last_type = "tens"
            elif w in _SCALES:
                if last_type not in ("ones", "teens", "tens", None):
                    return None
                scale = _SCALES[w]
                current = (current or 1) * scale
                if scale == 1000:
                    total += current
                    current = 0
                last_type = "scale"
            else:
                return None  # unrecognized word -> not a clean cardinal number

        return total + current

    # ---- top-level equivalence check --------------------------------------

    def values_equivalent(self, expected: str, actual: str, field_type: str = "generic") -> Optional[bool]:
        """True/False if `field_type`'s parser confidently reads both
        sides; None ("uncertain" — escalate to the LLM) if either side
        doesn't parse cleanly. `field_type`: callsign, heading, frequency,
        squawk, flight_level, qnh, runway, altitude, speed, or generic.

        "generic" (arbitrary free text — e.g. a taxiway description) can
        only ever confidently return True (identical after canonicalizing).
        A wording difference in free text might still mean the same thing,
        so it returns None rather than guessing a mismatch — only fields
        with a well-defined ICAO verbalization (numbers, callsigns,
        runways) can confidently report a real difference.
        """
        if expected is None or actual is None:
            return None
        expected, actual = str(expected).strip(), str(actual).strip()
        if not expected or not actual:
            return None

        if field_type == "callsign":
            e, a = self.callsign_from_words(expected), self.callsign_from_words(actual)
        elif field_type == "runway":
            e, a = self.runway_from_words(expected), self.runway_from_words(actual)
        elif field_type in ("altitude", "speed"):
            # ATC speaks these as compound cardinals OR digit-by-digit
            # (flight levels) — try cardinal first, fall back to digits.
            e, a = self.cardinal_number_from_words(expected), self.cardinal_number_from_words(actual)
            if e is None or a is None:
                e2, a2 = self.digits_from_words(expected), self.digits_from_words(actual)
                if e2 is not None and a2 is not None:
                    e, a = e2, a2
        elif field_type in ("heading", "frequency", "squawk", "flight_level", "qnh"):
            e, a = self.digits_from_words(expected), self.digits_from_words(actual)
        else:
            ce, ca = self.canonical_text(expected), self.canonical_text(actual)
            return True if ce == ca else None

        if e is None or a is None:
            return None
        return e == a
