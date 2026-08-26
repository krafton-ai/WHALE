# Copyright 2025 Search-R1 Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""LLM-as-a-judge correctness reward for Search-R1.

Drop-in async replacement for ``search_r1_like_qa_em.compute_score``. The
answer-extraction step (last ``<answer>...</answer>`` tag) and the anti-spam
``<answer>``-count guard are kept identical; only the normalize + exact-match
step is swapped for a GPT call.

Wired in by the train script when ``LLM_JUDGE=1`` (it sets
``reward.custom_reward_function.path`` to this file and ``.name=compute_score``).
Without that toggle the default routing in ``verl/utils/reward_score/__init__.py``
runs, so behavior is byte-identical to the EM version.

Env vars:
    LLM_JUDGE_KEY_FILE     path to a file whose first line is the OpenAI key.
                           default: <repo>/configs/llm_judge_key.txt
    LLM_JUDGE_MODEL        OpenAI model. default: gpt-5.4-mini
                           (Use a reasoning-capable model — V11 prompt and
                            max_completion_tokens=256 are sized for those.)
    LLM_JUDGE_CONCURRENCY  asyncio.Semaphore limit per process. default: 64
    LLM_JUDGE_USAGE_LOG    if set, append one JSONL line per call recording
                           {ts, model, prompt_tokens, completion_tokens,
                            reasoning_tokens, total_tokens, verdict, attempts}.
                           Enables exact-cost computation per run without
                           sharing OpenAI dashboard usage with other keys.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import string
import time
from pathlib import Path
from typing import Any

# Inlined from verl/utils/reward_score/search_r1_like_qa_em.py so this module is
# self-contained: meta-harness env.py (in .venv-meta-harness) cannot import
# the verl package, but loads this file directly via importlib.util. The verl
# GRPO trainer (in .venv-searchr1) imports it through verl as before.


_LAST_ASSISTANT_TURN_MARKER = "\nassistant\n"


def _last_assistant_turn(solution_str: str) -> str:
    """Slice the LAST assistant turn out of a serialized multi-turn dialogue.

    Decoded Qwen3.5 trajectories strip the chat template's special tokens
    (<|im_start|>, <|im_end|>) leaving bare role markers. Across all observed
    h29 rollouts the boundary that *introduces* an assistant turn is the
    literal "\\nassistant\\n" — the first assistant turn has no such marker
    (it starts at the head of solution_str), so we fall back to position 0.

    User/tool messages are bounded by patterns like "<answer>user\\n…",
    "</tool_response>\\nuser\\n…" — those are not assistant boundaries and are
    intentionally ignored here. The last assistant turn extends from the last
    "\\nassistant\\n" to the end of solution_str (verl rollouts always
    terminate on an assistant turn under MAX_TURNS budgeting).
    """
    idx = solution_str.rfind(_LAST_ASSISTANT_TURN_MARKER)
    if idx < 0:
        return solution_str
    return solution_str[idx + len(_LAST_ASSISTANT_TURN_MARKER):]


def extract_solution(solution_str: str) -> str | None:
    """Inner-most <answer>...</answer> emitted *by the final assistant turn*, or None.

    Two design choices, both anti-reward-hack:

    1. Restrict the search to the LAST assistant turn (`_last_assistant_turn`).
       This prevents <answer> and </answer> literals embedded in harness-
       injected nudge templates (PREMATURE_COMMIT_NUDGE, EVIDENCE_NUDGE)
       earlier in the trajectory from supplying tags to pair with model
       emissions across the trajectory.

    2. Within the last assistant turn, find the LAST </answer> and pair it
       with the NEAREST preceding <answer> (rfind-based, NOT non-greedy
       regex). For an input like "<answer>foo<answer></answer>" the legacy
       non-greedy regex would capture "foo<answer>", but here we pair the
       second (inner) <answer> with </answer> and capture "" — i.e., a model
       that emits a junk <answer> earlier in the same turn and then a
       properly closed empty <answer></answer> at the very end will be
       scored on the empty payload, not the junk.

    Returns None when:
      - no </answer> in the last assistant turn, OR
      - no <answer> before that </answer> in the last assistant turn.
    `compute_score` short-circuits None → 0.0 without calling the judge.
    """
    last_turn = _last_assistant_turn(solution_str)
    close_pos = last_turn.rfind("</answer>")
    if close_pos < 0:
        return None
    open_pos = last_turn.rfind("<answer>", 0, close_pos)
    if open_pos < 0:
        return None
    return last_turn[open_pos + len("<answer>"):close_pos].strip()


def _normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em_check(prediction: str, golden_answers) -> int:
    """Strict normalized exact-match against any golden answer."""
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    np = _normalize_answer(prediction)
    for g in golden_answers:
        if _normalize_answer(str(g)) == np:
            return 1
    return 0

# V11 prompt — winner of the auto-research sweep documented in
# meta_harness/runs/mh-h18-eval-claude-opus-4-7/llm_judge_experiments.md.
# Compared to the original verl-default prompt, V11 lifts F1 from 0.800 to
# 0.874 on the h18 manual ground-truth labels (precision 0.679 → 0.792,
# recall held at 0.974). False-positive C count (true learning noise) drops
# from 5 to 1 on that benchmark. Pair with gpt-5.4-mini for entity-equivalence
# strictness; the prompt's CORRECT/INCORRECT rules and the model's stricter
# judgment work together.
_PROMPT = (
    "You are an impartial judge for a question-answering task.\n\n"
    "Mark CORRECT when predicted answer refers to the same entity/value/fact\n"
    "as any reference. Acceptable:\n"
    "(1) Surface variation (case, articles, punctuation)\n"
    "(2) Alias / abbreviation / full-vs-short form ('Pete' = 'Peter')\n"
    "(3) Added/removed qualifier ('actor X' = 'X')\n"
    "(4) Less-specific date/place CONTAINED IN reference ('1962' if reference is\n"
    "    'September 10, 1962'; 'Wellington' if reference is 'Wellington, New Zealand')\n"
    "(5) Number/unit form ('4' = 'four', '188 acres' = '188 acre')\n"
    "(6) Different framing of the SAME geographic feature (e.g., 'Gulf of California'\n"
    "    = 'between Baja California and Sonora' — both describe the same body of water)\n"
    "(7) Self-correcting answer — if the prediction explores alternatives but commits\n"
    "    to a final answer that matches the reference, count CORRECT\n"
    "    (e.g., 'X... actually, Y' where Y matches reference)\n\n"
    "Mark INCORRECT for:\n"
    "(a) Different specific person/place/song/team/company/year\n"
    "(b) Numeric/date mismatch where neither contains the other\n"
    "    ('40 miles' vs '35 miles'; '1985' vs '1986')\n"
    "(c) Opposite or contradictory meaning (winner vs loser; 'no one' vs a named entity)\n"
    "(d) Different ENTITY TYPE even if related (city is NOT stadium; block is NOT channel;\n"
    "    episode title is NOT episode number)\n"
    "(e) Failure to commit to an entity (only explanation/disclaimer)\n"
    "(f) Listing multiple distinct candidates joined by separators (slashes \"/\",\n"
    "    pipes \"|\", commas, \"or\", \"either…or\") — even if one candidate matches\n"
    "    the reference. A valid answer commits to a single entity; \"Beijing /\n"
    "    Tokyo / Seoul\" listing the gold among 2+ alternatives is INCORRECT.\n"
    "(g) Verbatim passage or document quote — prediction is a multi-sentence\n"
    "    passage, paragraph, or doc-formatted text (e.g., contains \"Doc N (Title:\n"
    "    ...)\" patterns) that contains the reference answer as a substring but\n"
    "    is not itself a concise answer. A valid answer is the entity/value\n"
    "    alone, not the surrounding evidence prose.\n\n"
    "Question: {question}\n"
    "Reference answer(s): {gold}\n"
    "Predicted answer: {prediction}\n\n"
    "Respond with exactly one word: CORRECT or INCORRECT."
)

# Default key location: <repo_root>/configs/llm_judge_key.txt.
# __file__ = .../Search-R1/verl/utils/reward_score/llm_judge.py
# parents[3] = .../Search-R1
_DEFAULT_KEY_FILE = Path(__file__).resolve().parents[3] / "configs" / "llm_judge_key.txt"

# Process-local singletons. Each Ray reward worker is its own process.
_client: Any = None
_semaphore: asyncio.Semaphore | None = None
_model: str = "gpt-5.4-mini"


def _load_api_key() -> str:
    # 1) explicit env file path, 2) repository default key file, 3) OPENAI_API_KEY.
    candidate_files = [
        os.environ.get("LLM_JUDGE_KEY_FILE"),
        str(_DEFAULT_KEY_FILE),
    ]
    for key_path in candidate_files:
        if not key_path:
            continue
        try:
            with open(key_path) as f:
                key = f.read().strip()
        except OSError:
            continue
        if key:
            return key

    # or its first line if it points at a file path.
    env_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key
    for env_file in (                     os.path.expanduser("~/.env")):
        try:
            with open(env_file) as f:
                for line in f:
                    if line.startswith("OPENAI_API_KEY="):
                        v = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if v:
                            return v
        except OSError:
            continue
    raise RuntimeError(
        "No OpenAI key found (tried LLM_JUDGE_KEY_FILE, repo key.txt, "
        "the LLM_JUDGE_KEY_FILE path, the repository key file, and OPENAI_API_KEY)."
    )


def _init_runtime() -> None:
    """Lazy one-time init. Safe under asyncio (no await inside)."""
    global _client, _semaphore, _model
    if _client is not None:
        return
    from openai import AsyncOpenAI

    _model = os.environ.get("LLM_JUDGE_MODEL", "gpt-5.4-mini")
    concurrency = int(os.environ.get("LLM_JUDGE_CONCURRENCY", "64"))
    _semaphore = asyncio.Semaphore(concurrency)
    _client = AsyncOpenAI(api_key=_load_api_key())


def _format_gold(target: Any) -> str:
    if isinstance(target, str):
        return target
    if hasattr(target, "tolist"):  # numpy array
        target = target.tolist()
    return " | ".join(str(t) for t in target)


def _parse_verdict(text: str) -> int | None:
    """Return 1 (CORRECT), 0 (INCORRECT), or None (unparseable)."""
    m = re.search(r"[A-Za-z]+", text or "")
    if not m:
        return None
    tok = m.group(0).upper()
    if tok == "CORRECT":
        return 1
    if tok == "INCORRECT":
        return 0
    return None


def _record_usage(resp: Any, verdict: int | None, attempt: int) -> None:
    """Append per-call usage to LLM_JUDGE_USAGE_LOG (if set). Best-effort."""
    log_path = os.environ.get("LLM_JUDGE_USAGE_LOG")
    if not log_path:
        return
    try:
        usage = getattr(resp, "usage", None)
        if usage is None:
            return
        details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = (
            getattr(details, "reasoning_tokens", None) if details is not None else None
        )
        rec = {
            "ts": time.time(),
            "model": _model,
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "reasoning_tokens": reasoning_tokens,
            "total_tokens": getattr(usage, "total_tokens", None),
            "verdict": verdict,
            "attempt": attempt,
        }
        # Append-and-newline: small writes, kernel-atomic on POSIX.
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass


async def _call_judge(question: str, gold: str, prediction: str) -> int | None:
    """Returns 1 / 0 on success, None after all retries fail."""
    assert _client is not None and _semaphore is not None
    prompt = _PROMPT.format(question=question, gold=gold, prediction=prediction)
    delay = 1.0
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            async with _semaphore:
                # max_completion_tokens (vs max_tokens=4) — the V11 prompt and
                # gpt-5.x reasoning models can spend a few tokens on hidden
                # reasoning before emitting CORRECT/INCORRECT. 256 leaves
                # headroom while still capping cost at ~$0.0004/call.
                resp = await _client.chat.completions.create(
                    model=_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_completion_tokens=256,
                )
            verdict = _parse_verdict(resp.choices[0].message.content or "")
            _record_usage(resp, verdict, attempt + 1)
            if verdict is not None:
                return verdict
            last_err = RuntimeError(
                f"unparseable judge response: {resp.choices[0].message.content!r}"
            )
        except Exception as e:
            last_err = e
        await asyncio.sleep(delay)
        delay *= 2
    print(f"[llm_judge] giving up after 3 attempts: {last_err}")
    return None


async def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs,
):
    """Async drop-in for ``search_r1_like_qa_em.compute_score``.

    Reward is binary: 1.0 if the judge marks CORRECT, else 0.0. The legacy
    0.25 anti-spam path (≥10 <answer>/</answer> tags ⇒ partial credit) has
    been removed — `extract_solution` now restricts to the last assistant
    turn and the inner-most <answer>...</answer>, which by itself prevents
    the tag-spam reward channel without needing the count-based guard. On
    total API failure, falls back to deterministic EM so a sustained outage
    degrades to the legacy reward rather than collapsing all scores to 0.
    """
    _init_runtime()

    answer = extract_solution(solution_str=solution_str)
    if answer is None:
        return 0.0

    targets = ground_truth["target"]
    question = (extra_info or {}).get("question", "") or ""
    gold_str = _format_gold(targets)

    verdict = await _call_judge(question=question, gold=gold_str, prediction=answer)
    used_fallback = False
    if verdict is None:
        verdict = em_check(answer, targets)
        used_fallback = True

    if random.randint(1, 64) == 1:
        tag = "[llm_judge-fallback-EM]" if used_fallback else "[llm_judge]"
        print("--------------------------------")
        print(f"{tag} Q: {question}")
        print(f"{tag} gold: {gold_str}")
        print(f"{tag} pred: {answer}")
        print(f"{tag} verdict: {verdict}")

    return 1.0 if verdict == 1 else 0.0
