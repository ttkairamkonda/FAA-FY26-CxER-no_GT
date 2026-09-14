"""Corpus-level summary of a batch of analysis reports: error-type
frequency, severity distribution, and — importantly — the rule-vs-LLM
detection split, which is the number that validates whether the hybrid
design is actually doing what it's supposed to (most of the mechanical
categories caught for free, the LLM call reserved for judgment calls)."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

import pandas as pd

from .error_taxonomy import spec_for


class ReportAggregator:
    def __init__(self, reports: List[dict]):
        self.reports = reports

    def error_type_counts(self) -> pd.DataFrame:
        counts: Counter = Counter()
        for r in self.reports:
            for e in r.get("errors_detected", []):
                counts[e.get("error_type", "unknown")] += 1
        rows = [
            {"error_type": t, "count": c, "category": spec_for(t).category.value if spec_for(t) else "unknown",
             "check_mode": spec_for(t).check_mode.value if spec_for(t) else "unknown"}
            for t, c in counts.most_common()
        ]
        return pd.DataFrame(rows)

    def source_split(self) -> Dict[str, int]:
        """How many detected errors came from the rule checker vs. the LLM
        — validates that the hybrid design is actually offloading most of
        the mechanical categories away from the LLM call."""
        counts = {"rule": 0, "llm": 0}
        for r in self.reports:
            for e in r.get("errors_detected", []):
                counts[e.get("source", "llm")] = counts.get(e.get("source", "llm"), 0) + 1
        return counts

    def severity_counts(self) -> Dict[str, int]:
        counts: Counter = Counter()
        for r in self.reports:
            for e in r.get("errors_detected", []):
                counts[e.get("severity", "unknown")] += 1
        return dict(counts)

    def safety_risk_distribution(self) -> Dict[str, int]:
        counts: Counter = Counter()
        for r in self.reports:
            counts[r.get("safety_risk_level", "unknown")] += 1
        return dict(counts)

    def summary_table(self) -> pd.DataFrame:
        rows = []
        for r in self.reports:
            rows.append({
                "acn": r.get("acn"),
                "event": r.get("event"),
                "total_errors": r.get("total_errors", 0),
                "rule_errors": r.get("errors_by_source", {}).get("rule", 0),
                "llm_errors": r.get("errors_by_source", {}).get("llm", 0),
                "safety_risk_level": r.get("safety_risk_level"),
                "operational_status": r.get("operational_status"),
            })
        return pd.DataFrame(rows)

    def print_summary(self) -> None:
        split = self.source_split()
        total = sum(split.values())
        rule_pct = 100 * split["rule"] / total if total else 0

        print(f"Conversations analyzed: {len(self.reports)}")
        print(f"Total errors detected: {total}")
        print(f"  Rule-based (0 LLM calls): {split['rule']} ({rule_pct:.0f}%)")
        print(f"  LLM semantic judgment:    {split['llm']} ({100 - rule_pct:.0f}%)")
        print(f"\nSeverity distribution: {self.severity_counts()}")
        print(f"Safety risk level distribution: {self.safety_risk_distribution()}")
        print("\nTop error types:")
        print(self.error_type_counts().head(10).to_string(index=False))
