"""Robust extraction of a JSON object/array from raw LLM completion text.
Same multi-strategy approach proven in both `atc_pipeline.py` and
`FAA_CxER`'s judge pipeline, kept standalone here rather than imported
cross-package."""

from __future__ import annotations

import json
import re
from typing import Any, List, Optional


class JSONExtractor:
    """5-strategy JSON recovery: direct parse, markdown fence, balanced
    brace/bracket scan, common-mistake cleanup (trailing commas, JS
    comments, single-quoted keys), then a final cleanup + scan pass."""

    _FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```")
    _TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
    _LINE_COMMENT_RE = re.compile(r"//[^\n]*\n")
    _BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
    _SINGLE_QUOTED_KEY_RE = re.compile(r"'([^'\n]*?)':")

    def extract(self, text: str) -> Any:
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        m = self._FENCE_RE.search(text)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                pass

        found = self._scan_balanced(text)
        if found is not None:
            return found

        cleaned = self._TRAILING_COMMA_RE.sub(r"\1", text)
        cleaned = self._LINE_COMMENT_RE.sub("\n", cleaned)
        cleaned = self._BLOCK_COMMENT_RE.sub("", cleaned)
        cleaned = self._SINGLE_QUOTED_KEY_RE.sub(r'"\1":', cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        found = self._scan_balanced(cleaned)
        if found is not None:
            return found

        raise ValueError("All JSON extraction strategies failed. Output: {}".format(text[:300]))

    @staticmethod
    def _scan_balanced(text: str) -> Optional[Any]:
        for open_ch, close_ch in (("{", "}"), ("[", "]")):
            start = text.find(open_ch)
            if start == -1:
                continue
            depth = 0
            for i in range(start, len(text)):
                if text[i] == open_ch:
                    depth += 1
                elif text[i] == close_ch:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start : i + 1])
                        except json.JSONDecodeError:
                            break
        return None

    @staticmethod
    def as_dict(raw: Any) -> Optional[dict]:
        """Unwrap common model wrapping patterns ({"context": {...}}, a
        single-item list) to get a plain dict."""
        if isinstance(raw, list) and raw and isinstance(raw[0], dict):
            return raw[0]
        if not isinstance(raw, dict):
            return None
        for key in ("context", "report", "analysis", "result", "data", "output"):
            if key in raw and isinstance(raw[key], dict):
                return raw[key]
        return raw

    @staticmethod
    def as_list(raw: Any) -> List:
        """Unwrap common model wrapping patterns to get a plain list."""
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            for key in ("errors", "error_list", "errors_detected", "items", "results", "data"):
                if key in raw and isinstance(raw[key], list):
                    return raw[key]
            for v in raw.values():
                if isinstance(v, list):
                    return v
        return []
