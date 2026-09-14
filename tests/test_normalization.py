import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faa_cxer_ng.normalization import ATCValueNormalizer

norm = ATCValueNormalizer()


# ---- callsigns: verbalization differences must be equivalent -------------

def test_callsign_digit_words_match_digits():
    assert norm.values_equivalent("United 123", "United one two three", "callsign") is True


def test_callsign_real_drift_is_a_mismatch():
    assert norm.values_equivalent("Delta 345", "Delta 3456", "callsign") is False


def test_callsign_case_insensitive():
    assert norm.values_equivalent("delta 345", "DELTA 345", "callsign") is True


# ---- the exact bug this normalizer exists to avoid -----------------------

def test_digit_by_digit_is_not_misread_as_cardinal_sum():
    # "two five zero" must become "250", NOT 2+5+0=7.
    assert norm.cardinal_number_from_words("two five zero") is None
    assert norm.digits_from_words("two five zero") == "250"


def test_flight_level_digit_by_digit():
    assert norm.values_equivalent("three five zero", "350", "flight_level") is True
    assert norm.values_equivalent("three five zero", "351", "flight_level") is False


# ---- altitudes: compound cardinal numbers ---------------------------------

def test_altitude_cardinal_word_forms():
    assert norm.cardinal_number_from_words("six thousand feet") == 6000
    assert norm.cardinal_number_from_words("one thousand two hundred") == 1200
    assert norm.cardinal_number_from_words("eight thousand five hundred") == 8500
    assert norm.cardinal_number_from_words("twenty five") == 25


def test_altitude_equivalent_across_wordings():
    assert norm.values_equivalent("six thousand feet", "6000 feet", "altitude") is True
    assert norm.values_equivalent("eight thousand feet", "6000 feet", "altitude") is False


# ---- runway ---------------------------------------------------------------

def test_runway_words_vs_digits():
    assert norm.values_equivalent("two seven left", "27L", "runway") is True
    assert norm.values_equivalent("two seven left", "27R", "runway") is False
    assert norm.values_equivalent("niner right", "9R", "runway") is True


# ---- frequency (decimal) ---------------------------------------------------

def test_frequency_decimal():
    assert norm.values_equivalent("one two one decimal five", "121.5", "frequency") is True
    assert norm.values_equivalent("one two one decimal five", "122.5", "frequency") is False


# ---- heading / squawk ------------------------------------------------------

def test_heading_words_vs_digits():
    assert norm.values_equivalent("two seven zero", "270", "heading") is True
    assert norm.values_equivalent("two seven zero", "two seven", "heading") is False


def test_squawk_words_vs_digits():
    assert norm.values_equivalent("seven seven zero zero", "7700", "squawk") is True


# ---- uncertain cases must return None, not a guess ------------------------

def test_unparseable_returns_none_not_a_guess():
    assert norm.values_equivalent("descend to the assigned altitude", "6000 feet", "altitude") is None
    assert norm.values_equivalent("", "6000", "altitude") is None
    assert norm.values_equivalent(None, "6000", "altitude") is None


def test_generic_text_containment_style():
    assert norm.canonical_text("United one two three") == "united 123"
    assert norm.canonical_text("runway two seven left") == "runway 27 l"
