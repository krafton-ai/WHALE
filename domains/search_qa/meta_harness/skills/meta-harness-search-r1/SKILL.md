---
name: meta-harness-search-r1
description: Run one iteration of Search-R1 harness evolution.
---

# Meta-Harness (Search-R1) — Harness Evolution

Run ONE iteration of harness evolution. Do all work in the main session — do
NOT delegate to subagents. Constraints get lost when you delegate.

**You do NOT run benchmarks.** You analyze results + failed trajectories,
prototype changes, and implement new harnesses. The outer loop
(`meta_harness_search_r1.py`) handles benchmarking separately.

## CRITICAL CONSTRAINTS

- You MUST produce the exact number of candidate `harness.py` files the task
  prompt asks for, one per listed slot.
- Each file MUST define `class CandidateEnv(SearchR1Env)`.
- Each candidate is exactly ONE mechanism — do NOT bundle unrelated ideas.
- Do not stop early or claim the current harness is optimal.

### Anti-parameter-tuning rules

The most common failure mode is creating candidates that are just parameter
variants of existing harnesses. Check `evolution_summary.jsonl` for what's
been tried.

**Good candidates change a fundamental mechanism:**

- A new query-rewriting strategy (e.g. LLM-side decomposition, entity
  extraction, embedder-specific prefixing)
- A new `env_response` structure (e.g. dedup-detection nudge,
  retrieval-feedback injection, format-correction loop)
- A new SYSTEM_PROMPT architecture (e.g. ReAct-style explicit plan,
  chain-of-verification, plan-then-execute)
- A new `search` semantics (e.g. score-based re-ranking, follow-up query
  expansion, retrieval-with-evidence-snippets)

**Bad candidates just tune numbers.** If `search()` / `env_response()` /
SYSTEM_PROMPT structure are byte-identical to the base except for constants
or single-word edits, it's a parameter variant. REWRITE with a truly novel
mechanism.

**Combining harnesses is valid.** Take the search reformulation from one
prior harness and the SYSTEM_PROMPT from another; or draw on published
approaches (ReAct, IRCoT, Self-Ask, Search-o1).

**Exploitation axes:** A=SYSTEM_PROMPT structure, B=USER_PROMPT_TEMPLATE,
C=`search()` query rewriting, D=`search()` retrieval params (topk /
max_doc_tokens), E=`env_response` nudge logic, F=stop-condition / turn
budgeting. If the last 3 iterations explored the same axis, pick different
ones.

### Anti-overfitting rules

- **No question-specific hints.** Do not hardcode answers, entities, named
  subjects, or branching on question text.
- **Never mention dataset or subset names in code.** No
  `if "HotpotQA" in question:` branches, no `data_source`-conditional
  logic, no question-pattern hardcoding in SYSTEM_PROMPT /
  USER_PROMPT_TEMPLATE / `search` / `env_response`. Reading `data_source`
  in your *analysis* (Step 1) to group failures is fine; embedding it in
  candidate code is not.
- **General patterns are OK.** Rules like "always consult the retriever
  before producing a final answer" or "normalize whitespace in queries"
  apply broadly — fine.
- **If in doubt, make it more general.**

## CONTEXT

You are evolving a harness for multi-turn factual QA over wiki-18 (≈21M
Wikipedia passages, e5-base-v2 + FAISS, served over local HTTP — NOT a web
search engine). The eval is English factual QA — see `num_examples` in
`val.json` and `data_source` on each trajectory entry for the actual
composition of the current run.

**Read these files for ground truth** — do NOT duplicate facts here:

- `environments/search_r1/base_harness.py` — baseline `SearchR1Env`, default
  `search()`, default `env_response()`, default SYSTEM/USER prompts. Your
  starting point.
- `environments/search_r1/env.py` — `correct_answer()` reward, dataset
  loading, candidate schema validation. Authoritative for reward semantics.
- `configs/tool_config/search_tool_config.yaml` — tool schema seen by the
  verl GRPO trainer. CandidateEnv must match this schema.

Eval set size varies by run; `num_examples` in `val.json` is authoritative.
`data_source` on each trajectory entry identifies which subset that example
came from.

### SCHEMA CONTRACT (DO NOT BREAK)

Your `search()` tool's first parameter MUST be `query_list: list[str]` with
`maxItems=1`. verl's GRPO loader reads the yaml schema; if `CandidateEnv`
breaks this contract, every tool call during training fails with
`TypeError: unexpected keyword argument 'query_list'`, pinning reward to
~0. `env.py:_validate_candidate_search_signature` enforces this at load
time and raises `ValueError` if violated.

verl SearchTool's lenient policy is mirrored in `base_harness.py:search()`:
when the model emits >1 query, keep the first and silently drop the rest.
Maintain this parity if you redefine `search`.

### Retrieval payload — required fields

The retriever server requires this payload shape; missing fields cause
`KeyError: 'document'` server-side:

```python
{
    "queries": [query],
    "topk": topk,                       # function arg, default 3
    "return_scores": True,
    "max_doc_tokens": max_doc_tokens,   # function arg, default 200
    "tokenizer_name": "Qwen/Qwen3.5-2B",
}
```

Both `topk` and `max_doc_tokens` are exposed as `search()` keyword arguments
(see base `search(query_list, topk=3, max_doc_tokens=200)`), so candidate
harnesses can re-tune them axis-D-style by changing the defaults or by
threading per-turn values through `env_response` — the model can also pass
them in its tool call. Whatever values reach the retriever must still be
forwarded in the payload exactly as shown above; do not hardcode `200` or
`3` again in the payload dict.

### Turn budget — `MAX_TURNS` class attribute

`SearchR1Env` (and therefore every `CandidateEnv`) exposes a class
attribute `MAX_TURNS: int = 4` that controls the multi-turn ceiling
(axis F). Override it the same way you'd override `SYSTEM_PROMPT` to give
the harness more search/answer rounds:

```python
class CandidateEnv(SearchR1Env):
    MAX_TURNS = 8           # = base, default 4
    ...
```

The hard ceiling is **16**. `env.py:load_environment` clamps any value
above 16 down to 16 and logs a warning, and `slurm_train.sh` applies the
same clamp before forwarding the value to verl GRPO as
`actor_rollout_ref.rollout.multi_turn.max_assistant_turns`, so the
training and meta-harness eval rollouts always agree on the cap. Asking
for `MAX_TURNS = 24` is therefore safe — the run still proceeds at 16 —
but you do not gain extra depth past 16, so design within that envelope.

The tool_response returned to the model is
`json.dumps({"result": "<stitched docs>"}, ensure_ascii=False)` — NOT a raw
`"Doc N (Title: ...)..."` string. Match this if you redefine `search`.

## Reward semantics (from `env.py:correct_answer`)

- No `<answer>...</answer>` tag → **0.0**
- Tag present but normalized EM check fails → **0.0**
- Tag present, EM match, but `<answer>` count > 10 (spam penalty) → **0.25**
- Tag present, EM match, normal count → **1.0**

There is NO format-only partial credit. `format_reward` is a **separate
logging metric** (1 if any answer tag exists, else 0) — it does NOT
contribute to the reward number. It shows up as a separate column
(`avg_format_reward`) in `evolution_summary.jsonl` for diagnostic use only.

`success_rate` = `avg_reward` = average of `correct_answer` across rollouts.
Pareto frontier ranks by `(avg_success_rate, mean_turn_count)`.

## KEY FILES TO READ

Before proposing, read (relative to your working directory):

1. `logs/accepted_harness.txt` — current best harness name (e.g. `h0`)
2. `harnesses/<accepted>/harness.py` — current harness code
3. `logs/frontier_val.json` — Pareto frontier (success_rate × mean_turn_count)
4. `logs/evolution_summary.jsonl` — past results: one JSON per evaluated
   candidate with success/correct/format rewards, turn count, pareto flag,
   hypothesis, changes, axis, components
5. Latest `logs/iteration_*/comparison.json` snapshots + `report.md` if present
6. `logs/<profile>/<accepted>/<model>/val.json` — `failure_trajectories[]`
   (up to 5 failed rollouts) and `success_trajectories[]` (up to 2). Each
   entry is `{example_id, reward, turn_count, trace}` where **`trace` is a
   single concatenated string** (`[SYSTEM]...[USER]...[ASSISTANT]...
   [TOOL]...`), not a structured dict. Fast to skim, low-fidelity.
7. `logs/<profile>/<accepted>/<model>/trajectories.jsonl` — one JSON line
   per rollout (ALL 256, not capped). Each line:
   `{example_id, reward, correct, stop_condition, turn_count, question,
   answer, harness_trace, completion}` — `harness_trace` is a list of
   per-turn `{turn, action_type, assistant_action, env_feedback}` dicts;
   `completion` is the raw chat messages. Use this for high-fidelity
   per-turn analysis.

## ANALYSIS — Failure mode discovery

Skim `val.json:failure_trajectories[]` for a quick preview (5
concatenated-string traces). Then open `trajectories.jsonl` and read 5–10
failed rollouts end-to-end using the structured `harness_trace` +
`completion` fields. Describe what mechanism actually broke in each —
do NOT pattern-match to a pre-defined catalog. Note the `data_source` of
each failure for analysis purposes: different subsets often fail for
different reasons. Note also that wiki-18 is a 2018 snapshot, so some
"failures" are corpus gaps, not harness flaws.

## WORKFLOW

### Step 0: Post-eval reports (write if missing)

For each `logs/iteration_*/` that has `comparison.json` but no `report.md`,
write `logs/iteration_<NNN>/report.md` (≤ 30 lines) covering:

- Which candidate(s) became accepted, or why none did
- Which axes (A–F) were explored this iteration; outcome
- One concrete takeaway for the next iteration

These accumulate cross-iteration context so the proposer doesn't re-derive
mechanism understanding from raw `evolution_summary.jsonl` every time.

### Step 1: Analyze

Read the accepted `harness.py`, `frontier_val.json`,
`evolution_summary.jsonl`, latest `val.json` / `trajectories.jsonl`. For
each candidate slot, form ONE falsifiable hypothesis tied to a specific
failure mechanism you observed. K slots → K distinct hypotheses on different
axes.

### Step 2: Prototype — MANDATORY

**You MUST prototype your mechanism before writing the final harness.py.**
Do NOT skip this step. Candidates that skip prototyping tend to have bugs.

For each candidate that involves NEW retrieval logic, NEW `env_response`
logic, or NEW query rewriting:

1. Write a test script in `/tmp/` that exercises the core mechanism in
   isolation. Examples:
   - A custom `search()`: call it on 3–5 real questions from
     `trajectories.jsonl` and print the resulting `<tool_response>`.
   - A new `env_response` nudge: simulate a dummy assistant message and
     print what gets injected.
2. Pull real failure cases from `val.json:failure_trajectories[]` to test.
3. Try 2–3 variants and compare before picking the best.
4. Delete `/tmp/` scripts when done.

Pure-SYSTEM_PROMPT-only candidates may skip prototyping.

### Step 3: Implement

For each candidate slot:

1. Copy `harnesses/<accepted>/harness.py` to `harnesses/<name>/harness.py`
   as your starting point. Make targeted changes only.
2. Validate syntax (cheap, catches `SyntaxError`):
   `python -c "import ast; ast.parse(open('harnesses/<name>/harness.py').read()); print('OK')"`
3. Visually re-read the file to confirm `class CandidateEnv(SearchR1Env)`
   exists and the `from environments.search_r1.base_harness import ...`
   line is present.

### Step 4: Self-critique — MANDATORY

Re-read your written `harness.py`. Ask:

- Does this introduce a genuinely NEW mechanism, or is it a parameter
  variant of base / an existing harness in `evolution_summary.jsonl`?
  - Diff is a single numeric constant changed → parameter variant. REWRITE.
  - `search()` / `env_response()` body byte-identical to parent except
    for constants → parameter variant. REWRITE.
  - SYSTEM_PROMPT must be substantively rewritten (not a single-word swap)
    to count as a real change.
- Does it match the SCHEMA CONTRACT (`query_list: list[str]`, `maxItems=1`)?
- Does my `search()` return `json.dumps({"result": ...})` (not raw string)?
- Does my retrieval payload include `return_scores=True`, the **per-call
  `max_doc_tokens`** (forwarded from the `search()` arg, default 200), and
  `tokenizer_name="Qwen/Qwen3.5-2B"`? Hardcoding `200` instead of the
  variable defeats axis-D tunability.
- Does it mention any dataset/subset name, hardcoded entity, or
  question-text condition? (If yes → REWRITE; see Anti-overfitting rules.)

### Step 5: Write pending_eval.json

Write to the working-directory root:

```json
{
  "candidates": [
    {
      "name": "<slot_name>",
      "hypothesis": "<one falsifiable claim about what should improve>",
      "changes": "<specific description of what you changed>",
      "axis": "<A|B|C|D|E|F>",
      "components": ["<short-tag-1>", "<short-tag-2>"]
    }
  ]
}
```

`axis` uses the letter from "Exploitation axes". `components` are short
mechanism-tags you choose for cross-iteration analysis (e.g. naming the
specific technique applied, not naming any dataset / subset / question).

Output line at the end: `CANDIDATES: <name_1>, <name_2>, ...`

## CANDIDATE FILE STRUCTURE

```python
# harnesses/{name}/harness.py
from environments.search_r1.base_harness import SearchR1Env, search

SYSTEM_PROMPT = "..."
USER_PROMPT_TEMPLATE = "... Question: {question}"

class CandidateEnv(SearchR1Env):
    SYSTEM_PROMPT = SYSTEM_PROMPT
    USER_PROMPT_TEMPLATE = USER_PROMPT_TEMPLATE
    TOOLS = [search]

    # Optional: override env_response for custom nudge / dedup / reformat
    # async def env_response(self, messages, state, **kwargs): ...
```

To redefine `search`: see `base_harness.py:search` for the reference (it
already encodes the required payload shape, lenient multi-query clamp, and
JSON-wrapped return). Copy-then-edit; do NOT write `search` from scratch.

To override `env_response`: see `base_harness.py:env_response` for the
signature.
