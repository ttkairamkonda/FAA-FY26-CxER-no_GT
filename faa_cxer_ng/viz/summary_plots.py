"""Two corpus-level plots: error-type frequency (colored by rule/LLM/hybrid
check mode) and the overall rule-vs-LLM detection split."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from ..aggregate import ReportAggregator

_MODE_COLORS = {"rule": "#2ca02c", "hybrid": "#DD8452", "llm": "#4C72B0"}


class SummaryPlotter:
    def __init__(self, aggregator: ReportAggregator):
        self.aggregator = aggregator

    def plot_error_type_frequency(self, output_path: str, dpi: int = 150, show: bool = False):
        df = self.aggregator.error_type_counts()
        if df.empty:
            print("No errors to plot.")
            return

        fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(df))))
        colors = [_MODE_COLORS.get(m, "#888") for m in df["check_mode"]]
        ax.barh(df["error_type"][::-1], df["count"][::-1], color=colors[::-1])
        ax.set_xlabel("Count")
        ax.set_title("Detected Error Types (color = how it was checked)")

        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in _MODE_COLORS.values()]
        ax.legend(handles, _MODE_COLORS.keys(), loc="lower right")

        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {output_path}")
        if show:
            plt.show()
        plt.close(fig)

    def plot_source_split(self, output_path: str, dpi: int = 150, show: bool = False):
        split = self.aggregator.source_split()
        total = sum(split.values()) or 1

        fig, ax = plt.subplots(figsize=(5, 5))
        labels = [f"Rule-based\n({split['rule']}, {100 * split['rule'] / total:.0f}%)",
                  f"LLM judgment\n({split['llm']}, {100 * split['llm'] / total:.0f}%)"]
        ax.pie([split["rule"], split["llm"]], labels=labels, colors=["#2ca02c", "#4C72B0"],
               autopct=None, startangle=90)
        ax.set_title("Where Did Each Detected Error Come From?")

        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {output_path}")
        if show:
            plt.show()
        plt.close(fig)
