#!/usr/bin/env python3
"""Run the no-ground-truth ATC error detection pipeline over a dataset of
synthetic conversations.

Exactly 3 LLM calls total for the whole dataset (context extraction,
semantic error detection, synthesis), regardless of how many conversations
are in it — every stage batches all conversations into one `generate()`
call instead of looping conversation-by-conversation. Safe to Ctrl-C and
re-run: already-analyzed conversations (by ACN) are skipped.

Requires a CUDA GPU host with vLLM. CUDA_VISIBLE_DEVICES must be set
BEFORE this script imports torch/vllm.

Example:
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    python scripts/01_run_pipeline.py
    python scripts/01_run_pipeline.py --limit 2   # sanity check first

    # A different dataset, e.g. data/ROWAN_samples.json (which stores its
    # turn list under "dialogue" instead of "conversation"):
    python scripts/01_run_pipeline.py --input data/ROWAN_samples.json \\
        --output data/rowan_results.json --conversation-field dialogue
"""

import argparse
import os

import _bootstrap  # noqa: F401
from faa_cxer_ng.config import DEFAULT_JUDGE_MODEL, default_config
from faa_cxer_ng.io_utils import load_json


def load_model(model_name: str, tensor_parallel_size: int = 4, gpu_memory_utilization: float = 0.8, max_model_len: int = 10000):
    import torch
    from vllm import LLM

    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype="float16",
        max_model_len=max_model_len,
        # A bare int is the version-stable way to say "no torch.compile"
        # across vLLM releases (older releases had a `level` field on
        # CompilationConfig, newer ones renamed it to a `mode` enum — see
        # FAA_CxER's faa_cxer/judge/pipeline.py for the same fix).
        compilation_config=0,
    )


def main():
    cfg = default_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=str(cfg.conversations_path))
    parser.add_argument("--output", default=str(cfg.results_path))
    parser.add_argument("--conversation-field", default=None,
                         help="Key holding each record's turn list, if not 'conversation'/'dialogue'/'turns'/"
                              "'transcript' (auto-detected). E.g. --conversation-field dialogue for ROWAN_samples.json")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=50, help="Conversations per resumable save point")
    parser.add_argument("--limit", type=int, default=None, help="Only analyze the first N conversations (sanity check)")
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()

    from faa_cxer_ng.llm_backend import VLLMBackend
    from faa_cxer_ng.pipeline import ConversationAnalysisRunner, ConversationErrorPipeline

    conversations = load_json(args.input)
    if args.limit:
        conversations = conversations[: args.limit]
    print(f"Loaded {len(conversations)} conversations from {args.input}")

    llm = load_model(args.judge_model, tensor_parallel_size=args.tensor_parallel_size)
    backend = VLLMBackend(llm)
    pipeline = ConversationErrorPipeline(backend)
    runner = ConversationAnalysisRunner(pipeline, chunk_size=args.chunk_size)

    runner.run(conversations, args.output, skip_existing=not args.no_skip_existing, conversation_field=args.conversation_field)


if __name__ == "__main__":
    main()
