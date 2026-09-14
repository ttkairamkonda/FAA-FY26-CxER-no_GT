import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faa_cxer_ng.json_extractor import JSONExtractor

extractor = JSONExtractor()


def test_clean_object():
    assert extractor.extract('{"a": 1}') == {"a": 1}


def test_clean_array():
    assert extractor.extract('[{"a": 1}]') == [{"a": 1}]


def test_markdown_fence():
    raw = 'Here you go:\n```json\n{"a": 1}\n```'
    assert extractor.extract(raw) == {"a": 1}


def test_leading_trailing_text():
    raw = 'Sure, here is the analysis: {"a": 1} — hope that helps!'
    assert extractor.extract(raw) == {"a": 1}


def test_trailing_comma_and_single_quoted_keys():
    raw = "{'a': 1, 'b': 2,}"
    assert extractor.extract(raw) == {"a": 1, "b": 2}


def test_unparseable_raises():
    import pytest

    with pytest.raises(ValueError):
        extractor.extract("I refuse to output JSON today.")


def test_as_dict_unwraps_wrapper_key():
    assert extractor.as_dict({"context": {"a": 1}}) == {"a": 1}


def test_as_dict_unwraps_single_item_list():
    assert extractor.as_dict([{"a": 1}]) == {"a": 1}


def test_as_list_unwraps_wrapper_key():
    assert extractor.as_list({"errors": [1, 2, 3]}) == [1, 2, 3]


def test_as_list_passthrough():
    assert extractor.as_list([1, 2]) == [1, 2]


def test_as_list_returns_empty_for_unrecognized_shape():
    assert extractor.as_list({"unrelated": "value"}) == []
