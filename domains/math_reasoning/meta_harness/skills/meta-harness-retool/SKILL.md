---
name: meta-harness-retool
description: Run one iteration of ReTool harness evolution.
---

# Meta-Harness (ReTool) — Harness Evolution

Run ONE iteration of harness evolution. Do all work in the main session. Do
not delegate to subagents.

You do not run benchmarks. The outer loop (`meta_harness_retool.py`) runs eval
after you write candidate harnesses.

## Critical Constraints

- Produce exactly the candidate `harness.py` files requested in the task prompt.
- Each file must define `class CandidateEnv(ReToolEnv)`.
- Each candidate should test one mechanism. Do not bundle unrelated ideas.
- Do not stop early or claim the current harness is optimal.

## Anti-Overfitting Rules

- Do not hardcode answers, entities, numbers from specific questions, dataset
  names, or branches on question text.
- Do not specialize logic to AIME, DAPO, MATH, or any named split.
- General policies are allowed: e.g. "verify arithmetic with Python", "retry
  after interpreter errors", or "force a boxed final answer".

## Context

You are evolving a harness for ReTool-style math solving. The model sees math
problems, may call a Python tool named `code_interpreter`, receives stdout or
Python errors, then must provide the final answer in `\boxed{...}` format.

Read these files for ground truth:

- `environments/retool/base_harness.py` — baseline `ReToolEnv`, default
  `code_interpreter`, default prompt, default `env_response`.
- `environments/retool/env.py` — reward, dataset loading, candidate schema
  validation.
- `verl-recipe/retool/sandbox_fusion_tool_config.yaml` — tool schema used by
  verl GRPO training.

## Schema Contract

Your tool contract must match the GRPO yaml:

```python
async def code_interpreter(code: str, timeout: int = 30, language: str = "python", memory_limit_mb: int = 1024) -> str:
    ...
```

The first parameter must be named `code`. If you redefine `code_interpreter`
with a different first parameter, meta-harness eval may still generate a schema
from your Python signature, but verl training will send `{"code": ...}` and
tool execution will fail.

The baseline tool mirrors `verl-recipe/retool/retool.py`:

- extracts the first ```python fenced block when present
- wraps the last non-empty non-`print` line in `print(...)`
- sends a SandboxFusion-compatible HTTP request
- returns stdout plus stderr, or `"no stdout here"` on non-finished execution

Copy the baseline implementation before editing. Do not write a new sandbox
client from memory.

## Useful Mechanism Axes

- A: SYSTEM_PROMPT structure, e.g. explicit plan-code-check-answer ritual.
- B: USER_PROMPT_TEMPLATE structure, e.g. preserving the math problem while
  changing the tool-use instruction.
- C: code normalization, e.g. better extraction or safer print wrapping.
- D: interpreter feedback, e.g. structured error explanations or retry nudges.
- E: `env_response` / no-tool-call recovery logic, e.g. force at least one
  calculation for arithmetic-heavy problems without question-specific rules.
- F: turn budget and stop behavior via `MAX_TURNS` and stop hooks.

Bad candidates only change constants, punctuation, or a single prompt word.
If the diff is only a parameter tweak, rewrite it as a real mechanism.

## Reward Semantics

The meta-harness reward uses the last assistant message only and calls
`verl.utils.reward_score.math_dapo.compute_score(..., strict_box_verify=True)`.
Correctness is treated as binary for selection:

- correct boxed final answer: 1.0
- missing malformed boxed answer or wrong answer: 0.0

`format_reward` is a logging metric only. It does not replace correctness.

## Files To Read Before Proposing

1. `logs/accepted_harness.txt`
2. `harnesses/<accepted>/harness.py`
3. `logs/frontier_val.json`
4. `logs/evolution_summary.jsonl`
5. latest `logs/iteration_*/comparison.json` and `report.md` if present
6. `logs/<profile>/<accepted>/<model>/val.json`
7. `logs/<profile>/<accepted>/<model>/trajectories.jsonl`

Use failed trajectories to identify concrete failure modes: no tool call when
calculation was needed, malformed tool arguments, Python error not recovered,
unprinted value, overlong code, final answer outside `\boxed{}`, or numeric
answer not verified.

## Workflow

### Step 0: Post-Eval Reports

For each `logs/iteration_*/` with `comparison.json` but no `report.md`, write a
short report covering accepted candidate, axes explored, and one next takeaway.

### Step 1: Analyze

Read the accepted harness and recent trajectories. For each requested slot,
form one falsifiable hypothesis tied to a failure mechanism you observed.

### Step 2: Prototype

Prototype new code parsing, tool feedback, or `env_response` behavior before
writing final harnesses. Use `/tmp/` scripts and real failure snippets from
trajectories. Prompt-only candidates may skip this.

### Step 3: Implement

For each slot:

1. Copy `harnesses/<accepted>/harness.py` to `harnesses/<name>/harness.py`.
2. Make targeted changes only.
3. Validate syntax:

```bash
python -c "import ast; ast.parse(open('harnesses/<name>/harness.py').read()); print('OK')"
```

4. Re-read the file and confirm `class CandidateEnv(ReToolEnv)` exists.

### Step 4: Self-Critique

Check:

- Does this introduce a new mechanism rather than a constant tweak?
- Does `code_interpreter` still accept `code` as the first parameter?
- Does the harness avoid dataset names and question-specific branching?
- Does the final-answer instruction still require `\boxed{...}`?
- If overriding `env_response`, does it return tool messages compatible with
  the baseline `ReToolEnv.env_response` shape?

### Step 5: Write `pending_eval.json`

Write to the working-directory root:

```json
{
  "candidates": [
    {
      "name": "<slot_name>",
      "hypothesis": "<one falsifiable claim>",
      "changes": "<specific implementation summary>",
      "axis": "<A|B|C|D|E|F>",
      "components": ["<short-tag-1>", "<short-tag-2>"]
    }
  ]
}
```

Output line at the end:

```text
CANDIDATES: <name_1>, <name_2>, ...
```

## Candidate File Skeleton

```python
from environments.retool.base_harness import ReToolEnv, code_interpreter

SYSTEM_PROMPT = "..."
USER_PROMPT_TEMPLATE = "{question}\nThe answer format must be: \\boxed{'The final answer goes here.'}"


class CandidateEnv(ReToolEnv):
    SYSTEM_PROMPT = SYSTEM_PROMPT
    USER_PROMPT_TEMPLATE = USER_PROMPT_TEMPLATE
    TOOLS = [code_interpreter]

    # Optional: override env_response / stop hooks / get_prompt_messages.
```

