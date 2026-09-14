#!/usr/bin/env python3
"""Corpus-level summary of a completed analysis run: error-type frequency,
severity distribution, and the rule-vs-LLM detection split (the number
that shows whether the hybrid design is actually working as intended).

No GPU/vLLM required — this only reads the JSON `01_run_pipeline.py` wrote.

Example:
    python scripts/02_summarize_results.py
"""

import argparse

import _bootstrap  # noqa: F401
from faa_cxer_ng.aggregate import ReportAggregator
from faa_cxer_ng.config import default_config
from faa_cxer_ng.io_utils import load_json
from faa_cxer_ng.viz import SummaryPlotter


def main():
    cfg = default_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=str(cfg.results_path))
    parser.add_argument("--output-dir", default=str(cfg.outputs_dir))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    reports = load_json(args.input)
    print(f"Loaded {len(reports)} analyzed conversations from {args.input}\n")

    aggregator = ReportAggregator(reports)
    aggregator.print_summary()

    print("\nPer-conversation table:")
    print(aggregator.summary_table().to_string(index=False))

    if args.no_plots:
        return

    plotter = SummaryPlotter(aggregator)
    plotter.plot_error_type_frequency(f"{args.output_dir}/error_type_frequency.png")
    plotter.plot_source_split(f"{args.output_dir}/rule_vs_llm_source_split.png")


if __name__ == "__main__":
    main()
