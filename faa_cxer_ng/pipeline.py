"""Orchestrates the full 3-stage pipeline across a batch of conversations.

`ConversationErrorPipeline.run_batch` processes an entire list of
conversations per stage in ONE `generate_batch()` call each — stage 1 for
every conversation, then stage 2 for every conversation, then stage 3 —
instead of looping "conversation -> stage1 -> stage2 -> stage3" and
issuing single-prompt LLM calls along the way. Three batched calls total,
regardless of how many conversations are in the run.

`ConversationAnalysisRunner` adds resumoption + incremental, crash-safe
saving on top, chunking very large datasets so a run can be interrupted
and continued without redoing already-completed conversations.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from .conversation import ConversationParser, Turn, resolve_turns_field
from .error_taxonomy import spec_for
from .io_utils import PathLike, load_json, save_json_atomic
from .json_extractor import JSONExtractor
from .llm_backend import VLLMBackend
from .prompts import ContextExtractionPromptBuilder, SemanticErrorPromptBuilder, SynthesisPromptBuilder
from .rule_checker import RuleBasedChecker
from .schemas import ConversationContext, ConversationReport, DetectedError, default_context


class ConversationErrorPipeline:
    def __init__(
        self,
        backend: VLLMBackend,
        context_max_tokens: int = 2000,
        semantic_max_tokens: int = 2500,
        synthesis_max_tokens: int = 2500,
    ):
        self.backend = backend
        self.context_max_tokens = context_max_tokens
        self.semantic_max_tokens = semantic_max_tokens
        self.synthesis_max_tokens = synthesis_max_tokens

        self.conversation_parser = ConversationParser()
        self.json_extractor = JSONExtractor()
        self.rule_checker = RuleBasedChecker()
        self.context_prompts = ContextExtractionPromptBuilder()
        self.semantic_prompts = SemanticErrorPromptBuilder()
        self.synthesis_prompts = SynthesisPromptBuilder()

    # ---- stage 1: context extraction ---------------------------------------

    def _parse_context(self, raw: str, turns: List[Turn]) -> dict:
        nc = sum(1 for t in turns if t.speaker == "controller")
        npil = sum(1 for t in turns if t.speaker == "pilot")
        nu = sum(1 for t in turns if t.speaker == "unknown")

        parsed = None
        if raw:
            try:
                parsed = self.json_extractor.extract(raw)
            except ValueError:
                parsed = None
        ctx = self.json_extractor.as_dict(parsed) if parsed is not None else None
        if ctx is None:
            return default_context(len(turns), nc, npil, nu)

        ctx.setdefault("total_turns", len(turns))
        ctx.setdefault("controller_turns", nc)
        ctx.setdefault("pilot_turns", npil)
        ctx.setdefault("unknown_turns", nu)
        ctx.setdefault("primary_callsign", "UNKNOWN")
        ctx.setdefault("conversation_phase", "other")
        ctx.setdefault("facility_type", "unknown")
        ctx.setdefault("emergency_declared", False)
        for key in ("callsign_variations", "callsign_mentions", "critical_instructions",
                    "actual_readbacks", "key_events", "traffic_mentioned"):
            ctx.setdefault(key, [])

        try:
            return ConversationContext(**ctx).model_dump()
        except Exception as e:
            print(f"  WARNING stage 1 Pydantic validation failed ({e}) — using safe default")
            print(f"  Extracted (but discarded) dict was: {str(ctx)[:500]}")
            return default_context(len(turns), nc, npil, nu)

    # ---- stage 2b: semantic LLM errors --------------------------------------

    def _parse_errors(self, raw: str) -> List[dict]:
        if not raw:
            return []
        try:
            parsed = self.json_extractor.extract(raw)
        except ValueError:
            return []
        items = self.json_extractor.as_list(parsed)

        validated = []
        for e in items:
            if not isinstance(e, dict):
                continue
            e.setdefault("speaker", "unknown")
            e.setdefault("severity", "low")
            e.setdefault("source", "llm")
            e.setdefault("confidence", "confirmed")
            spec = spec_for(e.get("error_type", ""))
            if spec is None:
                continue  # not a recognized error type — drop rather than guess a taxonomy entry
            try:
                validated.append(DetectedError(**e).model_dump())
            except Exception:
                continue
        return validated

    # ---- stage 3: synthesis --------------------------------------------------

    def _build_report(self, conv: dict, turns: List[Turn], ctx: dict, errors: List[dict], raw: str) -> dict:
        synth = None
        if raw:
            try:
                parsed = self.json_extractor.extract(raw)
                synth = self.json_extractor.as_dict(parsed)
            except ValueError:
                synth = None
        if not synth:
            synth = {
                "conversation_summary": "Synthesis failed — see errors_detected directly.",
                "safety_risk_level": "unknown",
                "operational_status": "unknown",
                "turn_analysis": [
                    {"turn_index": t.turn_index, "speaker": t.speaker, "text": t.text, "errors_in_turn": []}
                    for t in turns
                ],
                "executive_summary": "Synthesis stage failed to produce valid output.",
                "recommendations": [],
            }

        errors_by_source = {
            "rule": sum(1 for e in errors if e.get("source") == "rule"),
            "llm": sum(1 for e in errors if e.get("source") == "llm"),
        }
        error_stats: Dict[str, int] = {}
        for e in errors:
            k = e.get("error_type", "unknown")
            error_stats[k] = error_stats.get(k, 0) + 1
        critical = sorted({e["error_type"] for e in errors if e.get("severity") == "critical"})

        report = {
            "acn": conv.get("ACN"),
            "event": conv.get("event"),
            "context": ctx,
            "errors_detected": errors,
            "errors_by_source": errors_by_source,
            "critical_errors": critical,
            "error_statistics": error_stats,
        }
        report.update(synth)
        report["total_errors"] = len(errors)  # authoritative count, not whatever synthesis echoed back

        try:
            return ConversationReport(**report).model_dump()
        except Exception as e:
            print(f"  WARNING stage 3 Pydantic validation failed ({e}) — using raw dict")
            return report

    # ---- main entry point: one batched call per stage -----------------------

    def run_batch(self, conversations: List[dict], conversation_field: Optional[str] = None, verbose: bool = True) -> List[dict]:
        n = len(conversations)
        t0 = time.time()
        parsed_turns = [
            self.conversation_parser.parse(resolve_turns_field(c, conversation_field))
            for c in conversations
        ]

        if verbose:
            print(f"[1/3] Context extraction — batching {n} conversations into one generate() call...")
        stage1_prompts = [self.context_prompts.build(turns) for turns in parsed_turns]
        stage1_raw = self.backend.generate_batch(stage1_prompts, self.context_max_tokens)
        contexts = [self._parse_context(raw, turns) for raw, turns in zip(stage1_raw, parsed_turns)]

        if verbose:
            print("[2a/3] Rule-based checks (0 LLM calls)...")
        rule_results = [
            self.rule_checker.check(turns, ConversationContext(**ctx))
            for turns, ctx in zip(parsed_turns, contexts)
        ]
        rule_errors_list = [r[0] for r in rule_results]
        uncertain_list = [r[1] for r in rule_results]
        if verbose:
            n_rule_errors = sum(len(e) for e in rule_errors_list)
            n_uncertain = sum(len(u) for u in uncertain_list)
            print(f"       {n_rule_errors} rule-confirmed errors, {n_uncertain} value pairs escalated to stage 2b")

        if verbose:
            print(f"[2b/3] Semantic error detection — batching {n} conversations into one generate() call...")
        stage2_prompts = [
            self.semantic_prompts.build(turns, ctx, uncertain)
            for turns, ctx, uncertain in zip(parsed_turns, contexts, uncertain_list)
        ]
        stage2_raw = self.backend.generate_batch(stage2_prompts, self.semantic_max_tokens)
        llm_errors_list = [self._parse_errors(raw) for raw in stage2_raw]

        merged_errors_list = [
            rule_errs + llm_errs for rule_errs, llm_errs in zip(rule_errors_list, llm_errors_list)
        ]

        if verbose:
            print(f"[3/3] Synthesis — batching {n} conversations into one generate() call...")
        stage3_prompts = [
            self.synthesis_prompts.build(turns, ctx, errors, conv.get("ACN"))
            for turns, ctx, errors, conv in zip(parsed_turns, contexts, merged_errors_list, conversations)
        ]
        stage3_raw = self.backend.generate_batch(stage3_prompts, self.synthesis_max_tokens)

        reports = [
            self._build_report(conv, turns, ctx, errors, raw)
            for conv, turns, ctx, errors, raw in zip(conversations, parsed_turns, contexts, merged_errors_list, stage3_raw)
        ]

        if verbose:
            total_errors = sum(r["total_errors"] for r in reports)
            elapsed = time.time() - t0
            print(f"Done: {n} conversations, {total_errors} total errors, {elapsed:.1f}s "
                  f"({elapsed / n:.1f}s/conversation, 3 LLM calls total regardless of n)")

        return reports


class ConversationAnalysisRunner:
    """Resumable wrapper around `ConversationErrorPipeline`: chunks a large
    dataset, skips conversations already present in the output file, and
    saves incrementally (atomic temp-file + rename) after every chunk."""

    def __init__(self, pipeline: ConversationErrorPipeline, chunk_size: int = 50):
        self.pipeline = pipeline
        self.chunk_size = chunk_size

    def run(
        self,
        conversations: List[dict],
        output_path: PathLike,
        skip_existing: bool = True,
        conversation_field: Optional[str] = None,
    ) -> List[dict]:
        output_path = str(output_path)
        existing: Dict[Any, dict] = {}
        try:
            prior = load_json(output_path)
            existing = {r.get("acn", i): r for i, r in enumerate(prior)}
            print(f"Resuming: {len(existing)} conversations already analyzed in {output_path}")
        except FileNotFoundError:
            pass

        pending = []
        for i, conv in enumerate(conversations):
            key = conv.get("ACN", i)
            if skip_existing and key in existing:
                continue
            pending.append((key, conv))

        if not pending:
            print("Nothing pending — all conversations already analyzed.")
            return list(existing.values())

        print(f"{len(pending)}/{len(conversations)} conversations pending")

        for start in range(0, len(pending), self.chunk_size):
            chunk = pending[start : start + self.chunk_size]
            keys = [k for k, _ in chunk]
            convs = [c for _, c in chunk]

            reports = self.pipeline.run_batch(convs, conversation_field=conversation_field)
            for key, report in zip(keys, reports):
                existing[key] = report

            save_json_atomic(output_path, list(existing.values()))
            print(f"  Saved progress: {len(existing)}/{len(conversations)} -> {output_path}")

        return list(existing.values())
