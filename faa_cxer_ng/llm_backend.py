"""Batched vLLM generation.

The original `atc_pipeline.py`/`CER_Summary.py` called
`self.llm.generate([one_prompt], sp)` — one prompt per Python-level call,
issued from inside a `ThreadPoolExecutor`, once per conversation per stage.
That throws away vLLM's actual strength: handing it many prompts in ONE
`.generate(prompts, sp)` call lets its continuous batching scheduler pack
them across the GPUs itself. `VLLMBackend.generate_batch` does that: every
conversation's stage-N prompt goes into one call, for all three stages.

This is the same OOM-halving-and-retry pattern proven in FAA_CxER's judge
pipeline, kept standalone here.
"""

from __future__ import annotations

from typing import List


class VLLMBackend:
    def __init__(self, llm, temperature: float = 0.05, top_p: float = 0.95, min_batch_size: int = 4):
        self.llm = llm
        self.temperature = temperature
        self.top_p = top_p
        self.min_batch_size = min_batch_size

    def generate_batch(self, prompts: List[str], max_tokens: int) -> List[str]:
        """Generate one completion per prompt, in order. Empty list in,
        empty list out. On GPU OOM, halves the batch and retries each half
        recursively down to `min_batch_size`, matching prompts that fail
        even at the floor to empty strings (caller treats "" as a failure
        needing the default/fallback path)."""
        if not prompts:
            return []
        from vllm import SamplingParams

        sp = SamplingParams(max_tokens=max_tokens, temperature=self.temperature, top_p=self.top_p)
        return self._generate_with_backoff(prompts, sp)

    def _generate_with_backoff(self, prompts: List[str], sp) -> List[str]:
        try:
            outputs = self.llm.generate(prompts, sp)
            return [o.outputs[0].text if o.outputs else "" for o in outputs]
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            import torch

            torch.cuda.empty_cache()

            if len(prompts) <= self.min_batch_size:
                print(f"  ⚠ OOM at floor batch size ({len(prompts)}) — these will fall back to the stage's default")
                return ["" for _ in prompts]

            mid = max(len(prompts) // 2, self.min_batch_size)
            print(f"  ⚠ OOM — retrying {len(prompts)} prompts as {mid} + {len(prompts) - mid}")
            return self._generate_with_backoff(prompts[:mid], sp) + self._generate_with_backoff(prompts[mid:], sp)
