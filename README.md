# FAA-CxER-no-groundtruth

**Finding operationally significant ATC communication errors in a conversation
with nothing to compare it against.**

This is a sibling project to [`FAA_CxER`](../FAA_CxER), but a genuinely
different problem. FAA_CxER judges an ASR *transcription* against a known
*reference* — it always has a ground truth to compare to. Here there is no
reference at all: the input is a multi-turn pilot/controller conversation
reconstructed by an LLM from real [ASRS](https://asrs.arc.nasa.gov/)
incident-report narratives — no real audio, no real transcript. The task
is to read the conversation on its own and decide whether the pilot and
controller actually communicated correctly: right callsign, complete and
correct readbacks, controller catching pilot mistakes, proper emergency
procedure, standard phraseology.

If you're new to this project, read this file top to bottom. Section 5 is
a complete walkthrough of every file — read it before changing anything.

---

## 1. What existed before, and what was wrong with it

`../Context extraction/` contains three earlier attempts at this problem:

1. **`CER_Summary.py`/`.ipynb`** — a LangChain-flavored first prototype:
   extract conversation context, detect errors, synthesize a report. Three
   LLM calls per conversation.
2. **A turn-by-turn approach** (`_synth_ATC_eval_results.json`,
   `synth_CE_pred.json`/`synth_CE_ref.json`) that tried to fake a "ground
   truth" by generating a clean and a corrupted version of each
   conversation, then running the *same pairwise CxER judge* FAA_CxER uses
   on ASR output — one LLM call **per utterance**. This is where "too many
   calls" bites hardest: **140 seconds per single-utterance call** in the
   saved results, meaning a 20-turn conversation could take the better
   part of an hour. It also doesn't fit this problem: FAA_CxER's judge
   needs a reference to compare against, and here there fundamentally
   isn't one — the "reference" conversations were themselves synthetic and
   invented for the exercise.
3. **`atc_pipeline.py`** — the refined successor to (1): same 3-call
   structure (context → error detection → synthesis), no LangChain, a
   5-strategy JSON extractor, and graceful fallbacks instead of crashes.
   This is the best of the three, but stage 2 asks ONE prompt to check all
   43 error types across an entire multi-turn conversation at once — a lot
   to ask of a single generation, and in practice it reliably catches only
   the obvious errors (a blatant callsign digit swap) and misses subtler
   ones (a controller quietly accepting a wrong callsign for five turns).

Both remaining problems — **too many LLM calls** and **only catches the
obvious ones** — turned out to share a fix.

## 2. The fix

**Efficiency**: `atc_pipeline.py` and `CER_Summary.py` both call
`llm.generate([one_prompt], sp)` — one prompt per Python-level call, issued
from inside a `ThreadPoolExecutor`, once per conversation per stage. That
throws away vLLM's actual strength: handing it *many* prompts in one
`generate(prompts, sp)` call lets its continuous batching scheduler pack
them across the GPUs itself. This pipeline batches every stage across the
**entire dataset**: stage 1 for all N conversations is one `generate()`
call, stage 2 is another, stage 3 is another. **Exactly 3 LLM calls total,
whether the dataset has 2 conversations or 2,000.**

**Recall**: stage 1 already extracts structured data — instruction
`key_values` vs. readback `values_extracted`, callsign variants, which
turn each one appeared in. Most of the original 43 error types are just
*comparisons over that structured data* (did the readback's altitude match
the instruction's altitude?), not something that needs an LLM's judgment
at all. A `RuleBasedChecker` handles those deterministically — for free,
instantly, and without the risk of an LLM hallucinating a category. The
one remaining semantic-detection LLM call is reserved for the ~15
categories that genuinely require understanding meaning or cross-turn
context (did the controller notice and correct a mistake? is this
emergency response adequate?) — it isn't competing with 43 categories at
once, so it has a much better chance of catching the subtle ones.

The one wrinkle: ATC callsigns/altitudes/etc. are spoken in many
equivalent ways ("United 123" = "United one two three" = "United tree
too tree" phonetically). A naive string comparison would flag every one of
those as a false "error." `ATCValueNormalizer` resolves ICAO phonetics and
number words to a canonical form before comparing — and when it can't
confidently parse a value, it returns "uncertain" rather than guessing, so
that specific comparison gets escalated into the LLM prompt instead of
being silently wrong in either direction.

```
 CONVERSATION (list of "speaker : text" turns, no reference available)
            │
            │  Stage 1 — ContextExtractionPromptBuilder (LLM, 1 batched call
            │  for the whole dataset)
            ▼
 STRUCTURED CONTEXT: callsign + every mention of it, instructions with
 key_values, readbacks with values_extracted, emergency flags, key events
            │
            ├──────────────────────────────┐
            │                               │
            ▼                               ▼
 Stage 2a — RuleBasedChecker         Stage 2b — SemanticErrorPromptBuilder
 (Python, 0 LLM calls)               (LLM, 1 batched call for the whole
 handles: callsign format/drift/     dataset) handles: hearback failure,
 omission, missing/roger-only/       wrong-aircraft-responding, ambiguous
 partial readback, all *_mismatch    phrasing, semantic misunderstanding,
 value comparisons, clearance        emergency-adequacy judgment, and any
 completeness, emergency squawk/     value pair the rule checker marked
 NITS keyword presence, known bad    "uncertain" instead of guessing
 phraseology patterns
            │                               │
            └──────────────┬────────────────┘
                            ▼
                  MERGED ERROR LIST (each tagged source="rule"|"llm")
                            │
                            │  Stage 3 — SynthesisPromptBuilder (LLM, 1
                            │  batched call for the whole dataset)
                            ▼
                  FINAL REPORT (summary, risk level, per-turn notes,
                  recommendations) — narrates what stages 1+2 already found,
                  detects nothing new
```

## 3. The error taxonomy

`faa_cxer_ng/error_taxonomy.py` curates the original 43 error types down to
39 (four were near-duplicates fragmenting the same phenomenon across
labels — see the module docstring for exactly which and why) and tags each
one with how it's detected:

| Check mode | Meaning | Count |
|---|---|---|
| `RULE` | A deterministic comparison over stage-1's structured data. Zero LLM calls. | 12 |
| `HYBRID` | Rule-checked in the common case; genuinely ambiguous instances escalate to the LLM. | 12 |
| `LLM` | Inherently requires contextual/semantic judgment no rule can decide. | 15 |

(`RULE` + `HYBRID` = 24 of 39 types resolved with zero or conditional LLM
involvement — the semantic-detection call only has to reason about the
remaining 15 plus whatever `HYBRID` comparisons come back "uncertain.")

This taxonomy reflects my own read of which ATC communication failures are
mechanically decidable vs. which need judgment — you know this domain
better than any default I could pick, so treat `error_taxonomy.py` as a
starting point to edit, not a fixed list.

**This design was verified against the real model, not just tested with
mocks.** A live run against `meta-llama/Llama-3.3-70B-Instruct` on 4 GPUs
correctly extracted real callsigns/instructions/readbacks and produced a
mix of rule-caught errors (`wrong_emergency_squawk`, `missed_position_report`,
a filler-word `non_standard_phraseology`) and genuinely LLM-judged ones
(`hearback_failure` — "controller responds with 'roger' without addressing
the pilot's non-compliance"; `instruction_not_followed` — "pilot states
they are flying runway heading instead of the instructed heading"), an
80%-rule/20%-LLM split on that sample.

## 4. Using your own dataset

Every conversation record needs an `ACN` (used as its identifier for
resuming/skipping) and a list of `"speaker : text"` lines. The list can
live under any key — `--input`/`--conversation-field` tell the pipeline
where to find it:

```bash
# Default dataset, default field name ("conversation")
python scripts/01_run_pipeline.py

# A different file, auto-detecting the field name (tries "conversation",
# "dialogue", "turns", "transcript" in that order — see
# faa_cxer_ng/conversation.py:resolve_turns_field)
python scripts/01_run_pipeline.py --input data/ROWAN_samples.json --output data/rowan_results.json

# Same, but naming the field explicitly (recommended once you know it —
# auto-detect is a convenience, not a guarantee, if a dataset happens to
# have more than one candidate key present)
python scripts/01_run_pipeline.py \
    --input data/ROWAN_samples.json \
    --output data/rowan_results.json \
    --conversation-field dialogue
```

`data/ROWAN_samples.json` (16 real ASRS-derived conversations across
altitude/heading/speed deviations, runway incursions, a bird strike, and
an NMAC) is a good example of *why* the field name matters: each record
stores its turn list under `"dialogue"`, not `"conversation"`, alongside
fields this pipeline doesn't use yet (`narrative_1`/`narrative_2` — the
original ASRS reporter narratives the dialogue was generated from,
`callback_1`/`callback_2`, `synopsis`, `aircraft_make`, `operating_under`,
`flight_plan`, `flight_phase`) and one worth knowing about:
**`dialogue_analysis`** — a pre-existing, independently produced per-turn
annotation (controller/pilot utterance type, callsign error, readback
error, missing/incorrect entities) plus a `score`. Nothing in this
pipeline reads `dialogue_analysis` today, but it's a ready-made point of
comparison if you want to sanity-check this pipeline's output against a
different labeling approach on the same conversations.

<!-- **Why this needed a code change, not just a flag on your end:** before
this, `ConversationErrorPipeline.run_batch` did `conv.get("conversation", [])`
— a record with no `"conversation"` key silently became a **0-turn**
conversation instead of an error, which then runs through all 3 LLM
stages and produces a report that *looks* normal (an empty one) instead of
telling you the input wasn't even being read. `resolve_turns_field` in
`faa_cxer_ng/conversation.py` fixes this: it raises `KeyError` (listing
the keys that *are* present) whenever it can't find a turn list, rather
than ever returning an empty list by default. -->

## 5. Code walkthrough — every file explained

This section exists so you can make changes without reverse-engineering
the code first. It's ordered the way a request actually flows through the
system, not alphabetically.

### `faa_cxer_ng/config.py`
One `Config` dataclass holding every default file path
(`conversations_path`, `results_path`, `data_dir`, `outputs_dir`) and the
default judge model name. `default_config()` returns one with defaults
resolved relative to the project root. Scripts read their `--input`/
`--output` defaults from here — **change a default path in one place by
editing this file**, or override per-run with the CLI flags.

### `faa_cxer_ng/io_utils.py`
Two functions, `load_json`/`save_json_atomic`. The atomic save writes to a
`.tmp` file and `os.replace()`s it into place, so a crash mid-write never
leaves a truncated results file — this is what makes `01_run_pipeline.py`
safe to Ctrl-C.

### `faa_cxer_ng/conversation.py`
- `Turn` — a dataclass: `turn_index`, `speaker` ("controller"/"pilot"/
  "unknown"), `text`.
- `ConversationParser.parse(lines)` — splits each `"speaker : text"` line
  on the first `:`, normalizes the speaker label against
  `_CONTROLLER_ALIASES`/`_PILOT_ALIASES` (so "approach", "tower", "ground"
  all become `"controller"`), and falls back to a linguistic heuristic
  (`_guess_speaker`) if there's no recognizable prefix at all. **Add a new
  facility name here** (e.g. a call sign like "Radar" your data uses that
  isn't already in the alias sets) if turns are coming back
  `speaker="unknown"`.
- `resolve_turns_field(record, field=None)` — finds the turn list in a
  conversation record (see §4). This is the one function every other
  "point the pipeline at a new dataset" question routes through.

### `scripts/01_run_pipeline.py`
The entry point. In order: parses CLI args, loads the conversations JSON,
calls `load_model()` (a thin `vllm.LLM(...)` wrapper — the
`compilation_config=0` line is a vLLM-version-compatibility fix, see the
comment inline and `FAA_CxER/faa_cxer/judge/pipeline.py` for the same
issue), wraps it in a `VLLMBackend`, builds a `ConversationErrorPipeline`,
and hands both to a `ConversationAnalysisRunner` to actually run. **This
is where you'd change**: `--tensor-parallel-size` (GPU count),
`--judge-model` (a different HF model id), `--chunk-size` (how many
conversations get analyzed before the next incremental save).

### `faa_cxer_ng/llm_backend.py`
`VLLMBackend.generate_batch(prompts, max_tokens)` — the ONE place that
actually calls `vllm`. Takes a list of prompts, returns a list of
completions in the same order, via a single `llm.generate(prompts, sp)`
call (this is the batching that makes "3 LLM calls total" true). On a GPU
OOM it halves the batch and retries each half recursively down to
`min_batch_size` — unlike FAA_CxER's judge (which shrinks its batch size
permanently across many chunks of a huge dataset), there's no persistent
state to degrade here, since one `run_batch` call already IS the whole
batch; an OOM just means "some prompts in this exact call get retried
smaller." **Change generation settings (temperature, top_p) here**, or swap in a
different backend (e.g. an API-based LLM) by implementing the same
`generate_batch` signature — nothing else in the package imports `vllm`
directly except this file and `scripts/01_run_pipeline.py`'s `load_model`.

### `faa_cxer_ng/pipeline.py`
The orchestrator. Two classes:

- **`ConversationErrorPipeline`** — stateless, does one `run_batch(conversations)`
  call. Internally: parses every conversation's turns, builds every
  stage-1 prompt, sends them all through `backend.generate_batch` in ONE
  call, parses each response with `_parse_context` (falls back to
  `schemas.default_context()` on unparseable/invalid output — and now
  prints the discarded dict when that happens, so a future prompt/schema
  mismatch is easy to diagnose instead of silently returning empty
  results). Then runs `RuleBasedChecker.check()` per conversation (zero
  LLM calls), builds every stage-2 prompt (including any "uncertain" value
  pairs the rule checker escalated) and batches them, then does the same
  for stage 3. `_build_report` assembles everything into the final dict
  and validates it against `schemas.ConversationReport`.
- **`ConversationAnalysisRunner`** — resumable wrapper: loads any existing
  output file, skips conversations already present (matched by `ACN`),
  processes the rest in `chunk_size`-sized batches (still one 3-call
  `run_batch` per chunk — chunking exists so a very large dataset saves
  progress periodically, not to reduce LLM calls further), and calls
  `save_json_atomic` after every chunk.

**If you want to change what happens between stages** (e.g. skip stage 3
entirely, or add a 4th stage), this is the file to edit — `run_batch` is
short and linear on purpose.

### `faa_cxer_ng/prompts/context_extraction.py`
`ContextExtractionPromptBuilder.build(turns)` returns the stage-1 prompt
string: conversation text, a numbered list of every field to extract, and
a fully filled-in example JSON object at the end (LLMs follow a concrete
example far more reliably than a schema description alone). **If you add
a field to `ConversationContext` in `schemas.py`, you need to add it to
both the numbered instructions AND the example JSON here**, or the model
has no signal to produce it. (This file is built with plain string
concatenation, not one f-string/`.format()` call — JSON braces in it must
be single `{`/`}`, not the `{{`/`}}` escaping an f-string would need. This
bit an early version: leftover `{{`/`}}` in the example made the model
copy invalid JSON syntax into every response, silently breaking
extraction — confirmed by rerunning against the real model before and
after the fix: 0-1 rule-confirmed errors with the bug, 8 without it, on
the same 2 conversations.)

### `faa_cxer_ng/schemas.py`
Pydantic models for every stage's output: `CallsignMention`,
`CriticalInstruction`, `Readback`, `ConversationContext` (stage 1's
shape), `DetectedError` (one error, tagged `source: "rule"|"llm"` and
`confidence: "confirmed"|"uncertain"`), `TurnAnalysis`, `ConversationReport`
(the final shape). Every field is optional or has a default — a
partially-malformed LLM response should degrade gracefully, not crash the
whole run. `default_context(...)` builds an empty-but-valid
`ConversationContext` for when stage 1 fails outright.

**To add a new field to what stage 1 extracts**: add it to
`ConversationContext` here, then to the prompt in
`context_extraction.py` (both the instructions and the example), then
use `context["your_field"]` wherever you need it (`rule_checker.py`,
`prompts/semantic_errors.py`, etc.).

### `faa_cxer_ng/json_extractor.py`
`JSONExtractor.extract(raw_text)` — recovers a JSON object/array from raw
LLM text via 5 fallback strategies in order: direct `json.loads`, strip a
markdown fence, scan for a balanced `{...}`/`[...]`, clean up common
mistakes (trailing commas, JS-style comments, single-quoted keys) and
retry, then scan the cleaned text too. Raises `ValueError` only if every
strategy fails. `.as_dict()`/`.as_list()` unwrap common "the model wrapped
its answer in an extra key" patterns (e.g. `{"context": {...}}` when you
asked for just `{...}`). Used identically at all three stages — this is
the only place that has to deal with "the model didn't quite follow the
output-format instructions."

### `faa_cxer_ng/error_taxonomy.py`
The 39-type registry (see §3). Each `ErrorTypeSpec` has a `category`, a
`check_mode` (RULE/HYBRID/LLM), a `default_severity`, and a human
`description`. `RULE_CHECKABLE`/`LLM_JUDGMENT` are derived sets used by
`rule_checker.py` and `prompts/semantic_errors.py` respectively.
`REQUIRED_ELEMENTS_BY_INSTRUCTION_TYPE` is a small checklist (e.g.
`"takeoff_clearance": {"runway"}`) used by the `incomplete_clearance`
rule check. **To add a new error type**: add an `ErrorTypeSpec` to the
`ERROR_TYPES` list; if it's `RULE`/`HYBRID`, also add the actual check to
`rule_checker.py`; if it's `LLM`, it needs no new code — it'll
automatically show up in the stage-2 prompt's category list (built from
`LLM_JUDGMENT` in `prompts/semantic_errors.py`) the next run. **To change
an existing type's severity**, edit its `default_severity` here — nothing
else needs to change, since `rule_checker.py` always calls
`spec_for(error_type).default_severity` rather than hardcoding a level.

### `faa_cxer_ng/normalization.py`
`ATCValueNormalizer` — resolves ICAO phonetics ("niner"→9, "tree"→3) and
number words to a canonical form so a wording difference isn't mistaken
for a real error. Key methods: `digits_from_words` (digit-by-digit
reading — headings, frequencies, squawk codes, flight levels: "tree fife
zero"→"350"), `cardinal_number_from_words` (compound numbers — altitudes/
speeds: "six thousand"→6000; enforces real English number grammar so a
digit-by-digit sequence like "two five zero" is correctly rejected here
rather than misread as 2+5+0=7), `callsign_from_words` (splits an operator
name from its digit sequence: "united one two three" and "united 123"
both → `"united|123"`; phonetic-alphabet words like "Alpha" only count as
tail-number letters *after* a digit appears, because several airline names
— "Delta" being the obvious one — are themselves ICAO phonetic words),
`runway_from_words`. The single entry point everything else calls is
`values_equivalent(expected, actual, field_type)`, returning
`True`/`False`/`None` — `None` means "couldn't confidently parse this,
don't guess." **If a specific real-world value keeps coming back as a
false mismatch or false match**, this is the file to fix — and please add
a test case to `tests/test_normalization.py` covering it, since this
logic has already had one subtle bug caught exactly that way (digit-by-digit
speech vs. cardinal numbers looking similar).

### `faa_cxer_ng/rule_checker.py`
`RuleBasedChecker.check(turns, context)` returns
`(confirmed_errors, uncertain_items)`. One private method per category
group:
- `_check_callsign_mismatches` — walks `context.callsign_mentions` in turn
  order, normalizes each against `primary_callsign`; groups consecutive
  mismatches of the *same* wrong variant into one `callsign_drift` (vs. an
  isolated `callsign_format_error`); separately detects valid vs.
  premature callsign abbreviation.
- `_check_callsign_omitted` — readback turns with no recognizable callsign
  digits at all.
- `_check_instruction_readback_pairs` — the biggest one: for every
  instruction requiring a readback, finds the matching readback, flags
  `missing_readback` if none exists, `roger_only_readback` if it's just an
  acknowledgement, then compares each `key_values` entry against
  `values_extracted` via the normalizer (`_FIELD_CHECK` maps a key name
  like `"runway"` to `(error_type, normalizer_field_type)`) — a confirmed
  mismatch becomes e.g. `runway_mismatch`, an unparseable one goes into
  `uncertain` instead, and any `key_values` entry missing from
  `values_extracted` entirely contributes to `partial_readback`.
- `_check_clearance_completeness` — `incomplete_clearance` (checklist from
  `error_taxonomy.REQUIRED_ELEMENTS_BY_INSTRUCTION_TYPE`),
  `missing_wind_in_clearance`, `clearance_void_time_missed`.
- `_check_emergency` — gated on `context.emergency_declared`:
  `wrong_emergency_squawk` (keyword search for "7700"),
  `emergency_nits_incomplete` (keyword search per NITS element — Nature,
  Intentions, Souls on board, Time/fuel remaining — `_NITS_KEYWORDS`).
- `_check_phraseology` — `_BAD_PHRASES` (a small filler-word blacklist:
  "thank you", "thanks", "no problem") and `_WORD_ORDER_VIOLATIONS` (regex
  patterns for known-bad clearance word order).
- `_check_position_reports` — `missed_position_report`, matching
  `_POSITION_KEYWORDS` against instructions containing "report".

**To add a new rule check**: write a new `_check_*` method following the
same pattern (append `DetectedError` dicts via the `_err(...)` helper,
which fills in `source="rule"`/`confidence="confirmed"` defaults), then
call it from `check()`.

### `faa_cxer_ng/prompts/semantic_errors.py`
`SemanticErrorPromptBuilder.build(turns, context, uncertain_value_pairs)`
— the stage-2 prompt. Explicitly tells the model the mechanical checks
already happened, lists ONLY the `LLM_JUDGMENT` category descriptions
(pulled live from `error_taxonomy.py` — adding a new `LLM`-mode error type
there automatically appears here), and appends any `uncertain` value pairs
the rule checker couldn't resolve, asking the model to judge those
specifically by meaning. Ends with one filled-in example, same reasoning
as the stage-1 prompt.

### `faa_cxer_ng/prompts/synthesis.py`
`SynthesisPromptBuilder.build(turns, context, errors, acn)` — the stage-3
prompt. Explicitly told NOT to invent new errors, only narrate the
already-merged list. Produces `conversation_summary`, `safety_risk_level`,
`operational_status`, per-turn `turn_analysis`, `executive_summary`,
`recommendations`.

### `faa_cxer_ng/aggregate.py`
`ReportAggregator` — takes a list of completed reports (whatever
`01_run_pipeline.py` wrote), computes: `error_type_counts()` (a DataFrame,
one row per error type with its category/check_mode), `source_split()`
(the rule-vs-LLM count that validates the hybrid design), `severity_counts()`,
`safety_risk_distribution()`, `summary_table()` (one row per conversation).
`print_summary()` is what `scripts/02_summarize_results.py` calls.

### `faa_cxer_ng/viz/summary_plots.py`
`SummaryPlotter` — two matplotlib figures: `plot_error_type_frequency`
(horizontal bar chart, colored by check mode — green=rule, orange=hybrid,
blue=llm) and `plot_source_split` (pie chart of the rule-vs-LLM split).

### `scripts/02_summarize_results.py`
Loads a results file, prints `ReportAggregator.print_summary()` +
`summary_table()`, saves both plots. No GPU/vLLM import anywhere in this
script — it's pure post-processing.

### `tests/`
- `test_normalization.py` — every `ATCValueNormalizer` equivalence case,
  including the digit-by-digit-vs-cardinal-number bug this was built to
  avoid.
- `test_rule_checker.py` — one test per `_check_*` category, using
  hand-built `ConversationContext`/`Turn` objects (no LLM involved).
- `test_conversation_field_resolution.py` — `resolve_turns_field`'s
  auto-detect and explicit-override behavior, including the "no
  recognized field" error case.
- `test_json_extractor.py` — each of the 5 recovery strategies.
- `test_error_taxonomy.py` — every type has a valid severity, the
  RULE/HYBRID/LLM sets partition correctly, merged legacy types stay gone.
- `test_pipeline.py` — full wiring against a `FakeBackend` that returns
  canned stage responses: verifies exactly 3 `generate_batch` calls happen
  regardless of conversation count, errors carry the right `source`, and a
  broken/unparseable LLM response degrades to `default_context()` instead
  of crashing.

## 6. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`scripts/01_run_pipeline.py` additionally needs a CUDA GPU host and vLLM
(pinned in `requirements-judge.txt` to the same versions proven to work in
`FAA_CxER` — see that project's README for why they're pinned rather than
left open):

```bash
pip install -r requirements-judge.txt
```

`scripts/02_summarize_results.py` needs neither a GPU nor vLLM — it only
reads the JSON the pipeline already wrote.

## 7. Quickstart

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Sanity check on 2 conversations first
python scripts/01_run_pipeline.py --limit 2

# Full run over data/synthetic_conversations.json — safe to Ctrl-C and
# resume; already-analyzed conversations (by ACN) are skipped on restart
python scripts/01_run_pipeline.py

# A different dataset (see §4 for the --conversation-field flag)
python scripts/01_run_pipeline.py --input data/ROWAN_samples.json --output data/rowan_results.json

# No GPU needed from here on
python scripts/02_summarize_results.py --input data/analysis_results.json
```

`outputs/examples_from_old_pipeline/` has two saved reports from the OLD
`atc_pipeline.py` for reference/comparison — their schema won't exactly
match this pipeline's `ConversationReport` (e.g. no `errors_by_source`,
no rule/LLM `source` tagging), since that's exactly what changed.

## 8. What was intentionally not carried forward

- **The turn-by-turn ref/pred approach** (`_synth_ATC_eval_results.json`,
  `synth_CE_pred.json`, `synth_CE_ref.json`) — abandoned methodology (fake
  ground truth via a corrupted/clean pair, 140s per utterance). Untouched
  in `../Context extraction/` if you need it for comparison.
- **`CER_Summary.py`/`.ipynb`** — superseded by `atc_pipeline.py`, which
  this package itself supersedes.
- Four near-duplicate error types merged away (see `error_taxonomy.py`).

Nothing in `../Context extraction/` or `../FAA_CxER/` was modified to
build this package.

## 9. Tests

```bash
pip install pytest
pytest tests/ -v
```

No GPU or model weights needed for any test — see §5's `tests/` entry for
what each file covers.

## 10. Cookbook — where to make a specific change

| I want to... | Edit |
|---|---|
| Point at a different dataset | `--input`/`--conversation-field` flags on `scripts/01_run_pipeline.py` (§4) |
| Change a default file path | `faa_cxer_ng/config.py` |
| Add a new error type | `faa_cxer_ng/error_taxonomy.py` (+ `rule_checker.py` if RULE/HYBRID) |
| Change an error type's severity | `faa_cxer_ng/error_taxonomy.py` — `default_severity` |
| Fix a false positive/negative on a specific value comparison | `faa_cxer_ng/normalization.py` — add a test case too |
| Add a new deterministic (0-LLM-call) check | `faa_cxer_ng/rule_checker.py` — new `_check_*` method |
| Change what stage 1 extracts | `faa_cxer_ng/schemas.py` (`ConversationContext`) + `faa_cxer_ng/prompts/context_extraction.py` (both instructions AND example) |
| Change the semantic-detection prompt wording | `faa_cxer_ng/prompts/semantic_errors.py` |
| Change the final report's fields | `faa_cxer_ng/schemas.py` (`ConversationReport`) + `faa_cxer_ng/prompts/synthesis.py` |
| Change generation temperature/top_p | `faa_cxer_ng/llm_backend.py` (`VLLMBackend.__init__`) |
| Use a different LLM backend (API instead of local vLLM) | Implement `generate_batch(prompts, max_tokens)` matching `VLLMBackend`'s signature |
| Change GPU count / judge model | `--tensor-parallel-size`/`--judge-model` flags on `scripts/01_run_pipeline.py` |
| Change how often progress saves | `--chunk-size` flag on `scripts/01_run_pipeline.py` |
| Add a new corpus-level stat or plot | `faa_cxer_ng/aggregate.py` (`ReportAggregator`) / `faa_cxer_ng/viz/summary_plots.py` |
