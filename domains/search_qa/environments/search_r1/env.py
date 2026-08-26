"""Search-R1 environment loader for verifiers.

Connects three workflows:
  [1] Meta-harness eval:        load_environment(harness_path="path/to/harness.py")
                                (benchmark.py injects candidate via harness_path kwarg)
  [2] Standalone smoke test:    load_environment()  →  uses SearchR1Env directly
  [3] verl trainer monkey-patch: scripts/nq_hotpotqa/optimal_setting/preamble.py
                                reads the same harness module and applies its
                                SYSTEM_PROMPT / USER_PROMPT_TEMPLATE / env_response
                                to the verl rollout.
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import verifiers as vf
from datasets import Dataset
from verifiers.types import AssistantMessage

logger = logging.getLogger(__name__)


# Hard ceiling on multi-turn search/answer rounds for any candidate harness.
# Candidates may set CandidateEnv.MAX_TURNS to any positive integer; values
# above this cap are silently clamped + a warning is logged so meta-harness
# evolution stays bounded (token budget, wallclock per rollout, KV-cache
# pressure) even when a proposer asks for an aggressive depth. Mirrors the
# slurm_train.sh clamp on the verl side so HARNESS_PATH-set training honors
# the same ceiling.
MAX_TURNS_HARD_CAP: int = 16


# -- reward / metric functions -----------------------------------------------
#
# These mirror verl's *patched* Search-R1 reward function (the post-h29
# reward-hack-fix version) so meta-harness eval scores are directly
# comparable to verl GRPO training reward (`critic/score/mean`). Two changes
# vs the legacy verbatim verl copy:
#
#   1. Extraction is restricted to the LAST assistant message only — earlier
#      assistant emits are ignored. This matches `_last_assistant_turn` in
#      verl/utils/reward_score/llm_judge.py.
#   2. Within the last assistant message we find the inner-most
#      <answer>...</answer> pair via rfind (NOT non-greedy regex), so
#      `<answer>foo<answer></answer>` extracts to "" and a junk earlier
#      <answer> cannot supply the open for a later </answer>.
#   3. Reward is binary {0.0, 1.0} — the legacy >10-answer-tag spam penalty
#      (which awarded 0.25) is removed; design (1)+(2) makes the spam path
#      structurally unreachable.

_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
import string as _string  # noqa: E402  — used by _verl_normalize


def _verl_normalize(s: str) -> str:
    """verl/utils/reward_score/search_r1_like_qa_em.py:normalize_answer"""
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(_string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _verl_em_check(prediction: str, golden_answers) -> int:
    """verl/utils/reward_score/search_r1_like_qa_em.py:em_check"""
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = _verl_normalize(prediction)
    for golden in golden_answers:
        if _verl_normalize(str(golden)) == normalized_prediction:
            return 1
    return 0


def _last_assistant_content(completion: list) -> str:
    """Return the content of the LAST AssistantMessage in `completion`.

    All prior assistant emits are ignored. If no assistant message exists,
    returns the empty string.
    """
    for msg in reversed(completion):
        if not isinstance(msg, AssistantMessage):
            continue
        content = getattr(msg, "content", "") or ""
        if isinstance(content, list):
            content = "\n".join(
                c.get("text", "")
                for c in content
                if isinstance(c, dict) and c.get("type") == "text"
            )
        return str(content)
    return ""


def _inner_most_extract(s: str) -> str | None:
    """Pair the LAST </answer> with the NEAREST preceding <answer> in `s`,
    capturing the text between them. For input "<answer>foo<answer></answer>"
    returns "" (NOT "foo<answer>"). Returns None if either tag is missing.
    Mirrors `extract_solution` in verl/utils/reward_score/llm_judge.py.
    """
    close_pos = s.rfind("</answer>")
    if close_pos < 0:
        return None
    open_pos = s.rfind("<answer>", 0, close_pos)
    if open_pos < 0:
        return None
    return s[open_pos + len("<answer>"):close_pos].strip()


def _assistant_solution_str(completion: list) -> str:
    """DEPRECATED — concatenated every assistant message's content for the
    legacy verbatim-verl reward path. Retained for backwards-compat callers
    (none in this file after the post-h29 patch). New code should use
    `_last_assistant_content(completion)`.
    """
    parts: list[str] = []
    for msg in completion:
        if not isinstance(msg, AssistantMessage):
            continue
        content = getattr(msg, "content", "") or ""
        if isinstance(content, list):
            content = "\n".join(c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text")
        parts.append(str(content))
    return "\n".join(parts)


def correct_answer(completion: list, answer, **kwargs: Any) -> float:
    """Reward — strict normalized EM against the inner-most <answer>...
    </answer> pair in the LAST assistant message. Binary {0.0, 1.0}.

    Returns 0.0 when:
      - the last assistant message lacks any </answer>, OR
      - it lacks any <answer> before that </answer>, OR
      - the extracted answer doesn't normalize-equal any golden answer.
    """
    last = _last_assistant_content(completion)
    extracted = _inner_most_extract(last)
    if extracted is None:
        return 0.0
    golden = answer if isinstance(answer, list) else [answer]
    return 1.0 if _verl_em_check(extracted, golden) else 0.0


def format_reward(completion: list, **kwargs: Any) -> float:
    """Metric: 1 if a closed `<answer>...</answer>` pair exists in the LAST
    assistant message, else 0. Loose check on format compliance — does NOT
    require that the contained answer be correct, only that the model
    actually emitted a balanced answer tag in its final turn.
    """
    last = _last_assistant_content(completion)
    return 1.0 if _ANSWER_TAG_RE.search(last) else 0.0


def correct_reward(completion: list, answer, **kwargs: Any) -> float:
    """Metric: alias of `correct_answer` after the spam-penalty removal.
    Kept as a separate symbol for backwards-compat with logs/configs that
    track `correct_reward` distinct from `correct_answer`."""
    return correct_answer(completion, answer, **kwargs)


# -- LLM-judge reward path ---------------------------------------------------
#
# Opt-in via `LLM_JUDGE=1` env var. When enabled, `load_environment` swaps the
# default `correct_answer` reward (and `correct_reward` metric) for the LLM-as-
# a-judge variant. We share verl's `verl/utils/reward_score/llm_judge.py` so the
# meta-harness reward signal matches the verl GRPO training reward bit-for-bit:
# same V11 prompt, same default model (gpt-5.4-mini), same temperature=0,
# max_completion_tokens=256, asyncio.Semaphore-gated concurrency, 3-retry
# exponential backoff, and EM fallback when all retries fail.
#
# .venv-meta-harness does NOT have the verl package importable, so we load the
# self-contained `llm_judge.py` via importlib (path-based) instead of `from
# verl... import`.

_LLM_JUDGE_PATH = (
    Path(__file__).resolve().parents[2]
    / "verl"
    / "utils"
    / "reward_score"
    / "llm_judge.py"
)
_llm_judge_module: Any = None


def _get_llm_judge():
    """Lazy-load the shared llm_judge module. Process-local singleton."""
    global _llm_judge_module
    if _llm_judge_module is not None:
        return _llm_judge_module
    spec = importlib.util.spec_from_file_location(
        "_search_r1_llm_judge", str(_LLM_JUDGE_PATH)
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load llm_judge from {_LLM_JUDGE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _llm_judge_module = mod
    return mod


def _question_from_prompt(prompt: Any) -> str:
    """Extract the bare question from the first user message of the prompt.
    The harness wraps each question in the long Search-R1 instruction prefix
    (`USER_PROMPT_TEMPLATE`); we strip back to just the question so the judge
    sees only the QA pair."""
    if not isinstance(prompt, list):
        return ""
    for msg in prompt:
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role != "user":
            continue
        content = (
            msg.get("content")
            if isinstance(msg, dict)
            else getattr(msg, "content", "")
        )
        if not isinstance(content, str):
            continue
        # The verl/meta-harness instruction template ends with "Question: ...".
        # Pull the tail after the literal "Question:" if present; else fall back
        # to the whole user content.
        idx = content.rfind("Question:")
        if idx != -1:
            return content[idx + len("Question:") :].strip()
        return content.strip()
    return ""


async def _shared_llm_verdict(state: dict, completion: list, answer, prompt) -> float:
    """Call verl's patched compute_score (V11 prompt + judge + 3-retry + EM
    fallback) once per rollout and cache on `state` so reward and metric
    reuse it.

    `solution_str` passed to the judge is the LAST AssistantMessage content
    only — earlier assistant emits are intentionally dropped so that
    extraction is unambiguous on the model's final turn. The shared
    `compute_score` in verl/utils/reward_score/llm_judge.py then runs
    `_last_assistant_turn` (no `\\nassistant\\n` marker present in this
    single-message slice → fallback returns the whole string) and applies
    inner-most rfind extraction.

    Returns float in {0.0, 1.0} — the legacy 0.25 anti-spam path was
    removed in the post-h29 reward-hack-fix patch.
    """
    cached = state.get("_llm_judge_score") if isinstance(state, dict) else None
    if cached is not None:
        return float(cached)

    judge = _get_llm_judge()
    solution_str = _last_assistant_content(completion)
    golden = answer if isinstance(answer, list) else [answer]
    extra_info = {"question": _question_from_prompt(prompt)}
    score = await judge.compute_score(
        data_source="search_r1_like_qa_em",
        solution_str=solution_str,
        ground_truth={"target": golden},
        extra_info=extra_info,
    )
    score_f = float(score)
    if isinstance(state, dict):
        state["_llm_judge_score"] = score_f
    return score_f


async def correct_answer_llm_judge(
    completion: list, answer, prompt=None, state=None, **kwargs: Any
) -> float:
    """LLM-judge replacement for `correct_answer`. Returns {0.0, 1.0} —
    `compute_score` now removes the legacy 0.25 spam-penalty path."""
    if state is None:
        state = {}
    return await _shared_llm_verdict(state, completion, answer, prompt)


async def correct_reward_llm_judge(
    completion: list, answer, prompt=None, state=None, **kwargs: Any
) -> float:
    """LLM-judge replacement for `correct_reward` metric. Alias of
    `correct_answer_llm_judge` after the spam-penalty removal — kept as a
    separate symbol for backwards-compat with logs/configs."""
    if state is None:
        state = {}
    return await _shared_llm_verdict(state, completion, answer, prompt)


def assistant_tokens(state: dict, **kwargs: Any) -> float:
    """Metric: total assistant-generated tokens across the rollout."""
    usage = state.get("usage") if isinstance(state, dict) else None
    if isinstance(usage, dict):
        return float(usage.get("output_tokens", 0) or 0)
    total = 0
    for step in (state.get("trajectory") or []) if isinstance(state, dict) else []:
        tokens = step.get("tokens") if isinstance(step, dict) else None
        if tokens and tokens.get("completion_ids"):
            total += len(tokens["completion_ids"])
    return float(total)


def tool_tokens(state: dict, **kwargs: Any) -> float:
    """Metric: tool-response tokens added across turns (proxy for retrieval cost)."""
    if not isinstance(state, dict):
        return 0.0
    traj = state.get("trajectory") or []
    total = 0
    for i in range(1, len(traj)):
        prev = traj[i - 1].get("tokens") if isinstance(traj[i - 1], dict) else None
        cur = traj[i].get("tokens") if isinstance(traj[i], dict) else None
        if not (prev and cur):
            continue
        prev_prompt = prev.get("prompt_ids") or []
        prev_comp = prev.get("completion_ids") or []
        cur_prompt = cur.get("prompt_ids") or []
        delta = len(cur_prompt) - len(prev_prompt) - len(prev_comp)
        if delta > 0:
            total += delta
    return float(total)


# -- harness module loading --------------------------------------------------


def _load_module_from_path(path: str | Path):
    """Load a candidate harness file as an anonymous module.

    Also adds the file's grandparent directory (the run_dir, which contains
    `harnesses/`) to sys.path so the candidate may use `from harnesses.hN.harness
    import ...` to reuse helpers from earlier accepted harnesses (PEP 420
    namespace packages — no __init__.py required).
    """
    path = Path(path).resolve()
    # harness file lives at <run_dir>/harnesses/<name>/harness.py
    #   path                = .../harnesses/<name>/harness.py
    #   path.parent         = .../harnesses/<name>
    #   path.parent.parent  = .../harnesses
    #   path.parent.parent.parent = .../<run_dir>   ← what we add to sys.path
    run_dir = path.parent.parent.parent
    if str(run_dir) not in sys.path:
        sys.path.insert(0, str(run_dir))
    module_name = f"sr1env_candidate_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# -- dataset -----------------------------------------------------------------


def _load_dataset_from_parquet(
    path: str | Path,
    user_prompt_template: str,
    num_examples: int = -1,
) -> Dataset:
    """Load eval dataset from a Search-R1 parquet (train.parquet / test_100.parquet
    schema), formatting each row's question through user_prompt_template.

    The candidate's USER_PROMPT_TEMPLATE controls how the bare question
    becomes the first user message — at meta-harness eval time AND, via the
    verl monkey-patch, at training time (1:1 surface).

    Each output row carries:
      question          → full templated user message (verifiers' user-turn input)
      answer            → list[str] of accepted ground-truth strings
      data_source       → bookkeeping; available to reward funcs via kwargs
      raw_question      → bare question (so a custom env_response could re-template)
      extra_info_index  → original parquet row index (debugging / traceability)
    """
    df = pd.read_parquet(path)
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        ei = r["extra_info"]
        question_text = str(ei.get("question", ""))
        templated = (
            user_prompt_template.format(question=question_text)
            if "{question}" in user_prompt_template
            else user_prompt_template
        )
        targets = r["reward_model"]["ground_truth"]["target"]
        if hasattr(targets, "tolist"):
            targets = targets.tolist()
        elif not isinstance(targets, list):
            targets = [str(targets)]
        rows.append(
            {
                "question": templated,
                "answer": [str(t) for t in targets],
                "data_source": str(r["data_source"]),
                "raw_question": question_text,
                "extra_info_index": int(ei.get("index", -1)),
            }
        )
    if num_examples and num_examples > 0:
        rows = rows[:num_examples]
    return Dataset.from_list(rows)


# -- candidate signature contract --------------------------------------------


def _validate_candidate_search_signature(env_class) -> None:
    """Enforce the verl-yaml-schema contract on the candidate's search tool.

    The verl trainer reads its tool schema from
    `configs/tool_config/search_tool_config.yaml` (parameter name
    `query_list`, array of string, maxItems=1). When the model emits a
    tool_call during training it uses that schema. The candidate's `search`
    function therefore MUST accept a `query_list` first parameter — otherwise
    `search(**args)` raises `TypeError: unexpected keyword argument 'query_list'`
    on every call, and the resulting `[tool error]` messages pin reward at ≈0
    (caught in run 29051).

    Verifiers eval auto-generates schema from function signature, so this
    convention also keeps the eval-time and train-time tool API identical.
    Raise loudly here at load time so problems surface immediately rather
    than as silent reward collapse during a multi-hour training run.
    """
    tools = getattr(env_class, "TOOLS", None) or []
    search_fn = next(
        (t for t in tools if getattr(t, "__name__", None) == "search"), None
    )
    if search_fn is None:
        return  # No tool named 'search' to validate.
    sig = inspect.signature(search_fn)
    params = list(sig.parameters.keys())
    if not params or params[0] != "query_list":
        raise ValueError(
            f"CandidateEnv.TOOLS contains 'search' with signature {sig} but "
            f"the first parameter must be named 'query_list' to match the "
            f"verl yaml schema in configs/tool_config/search_tool_config.yaml. "
            f"Without this, candidates work in meta-harness eval (verifiers "
            f"auto-generates schema from signature) but FAIL in verl training "
            f"(yaml schema sends 'query_list', signature receives "
            f"'{params[0] if params else '(none)'}', causing TypeError on "
            f"every tool call). See SKILL.md for the required convention."
        )


# -- environment loader ------------------------------------------------------


DEFAULT_EVAL_DATASET = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "searchR1_processed"
    / "train_meta_harness.parquet"
)


def load_environment(
    env_class=None,
    harness_path: str | None = None,
    num_train_examples: int = 0,
    num_eval_examples: int = -1,
    eval_dataset_path: str | None = None,
    max_turns: int = 10,
    seed: int = 42,
    **kwargs: Any,  # absorbs unused kwargs (e.g. benchmark.py's bignum-era args)
) -> vf.Environment:
    """Load a Search-R1 verifiers environment.

    Resolution priority for env_class:
        env_class arg  >  harness_path → CandidateEnv  >  default SearchR1Env
    """
    if harness_path is not None:
        mod = _load_module_from_path(harness_path)
        if not hasattr(mod, "CandidateEnv"):
            raise AttributeError(f"CandidateEnv not found in {harness_path}")
        env_class = mod.CandidateEnv
    elif env_class is None:
        try:
            from .base_harness import SearchR1Env
        except ImportError:
            from base_harness import SearchR1Env  # standalone module import
        env_class = SearchR1Env

    _validate_candidate_search_signature(env_class)

    system_prompt = getattr(env_class, "SYSTEM_PROMPT", None)
    user_prompt_template = getattr(env_class, "USER_PROMPT_TEMPLATE", "{question}")
    tools = getattr(env_class, "TOOLS", None)

    # Axis F (turn budgeting): candidate's MAX_TURNS class attribute takes
    # precedence over the load_environment argument so harnesses can tune
    # multi-turn depth alongside topk / max_doc_tokens. Clamp to the hard
    # cap (currently 16) when the candidate asks for more — same ceiling
    # applies to the verl HARNESS_PATH-set training path (slurm_train.sh).
    candidate_max_turns = getattr(env_class, "MAX_TURNS", None)
    if isinstance(candidate_max_turns, int) and candidate_max_turns > 0:
        if candidate_max_turns > MAX_TURNS_HARD_CAP:
            logger.warning(
                "[load_environment] %s.MAX_TURNS=%d exceeds hard cap %d; clamping.",
                env_class.__name__, candidate_max_turns, MAX_TURNS_HARD_CAP,
            )
            candidate_max_turns = MAX_TURNS_HARD_CAP
        max_turns = candidate_max_turns

    eval_path = eval_dataset_path or str(DEFAULT_EVAL_DATASET)
    eval_dataset = _load_dataset_from_parquet(
        eval_path, user_prompt_template, num_examples=num_eval_examples
    )
    if num_train_examples and num_train_examples > 0:
        train_dataset = _load_dataset_from_parquet(
            eval_path, user_prompt_template, num_examples=num_train_examples
        )
    else:
        # placeholder; meta-harness sweeps set num_train_examples=0
        train_dataset = eval_dataset

    rubric = vf.Rubric()
    # LLM_JUDGE=1 swaps the EM correctness pipeline for an LLM-as-a-judge
    # exactly mirroring verl GRPO training's reward path (same module,
    # prompt, model, retry, and fallback semantics). Without the toggle,
    # behaviour stays byte-identical to the prior EM-only meta-harness.
    if os.environ.get("LLM_JUDGE", "0") == "1":
        rubric.add_reward_func(correct_answer_llm_judge)
        rubric.add_metric(format_reward)
        rubric.add_metric(correct_reward_llm_judge)
    else:
        rubric.add_reward_func(correct_answer)
        rubric.add_metric(format_reward)
        rubric.add_metric(correct_reward)
    rubric.add_metric(assistant_tokens)
    rubric.add_metric(tool_tokens)

    return env_class(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        system_prompt=system_prompt,
        tools=tools,
        rubric=rubric,
        max_turns=max_turns,
    )


# -- CLI smoke test ----------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import asyncio

    from openai import AsyncOpenAI

    parser = argparse.ArgumentParser(description="Search-R1 env smoke test")
    parser.add_argument("--harness", default=None, help="Path to candidate harness.py")
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--api-base", default="http://localhost:8044/v1")
    parser.add_argument("--num-examples", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    env = load_environment(
        harness_path=args.harness,
        num_eval_examples=args.num_examples,
        max_turns=args.max_turns,
    )
    print(f"Loaded env: {type(env).__name__}")
    print(f"Eval dataset size: {len(env.eval_dataset)}")

    client = AsyncOpenAI(base_url=args.api_base, api_key="EMPTY")

    async def run():
        results = await env.evaluate(
            client=client,
            model=args.model,
            sampling_args={
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            },
            num_examples=args.num_examples,
            max_concurrent=2,
        )
        outputs = results["outputs"]
        rewards = [float(o.get("reward", 0.0)) for o in outputs]
        print(f"\nResults ({len(outputs)} examples):")
        print(f"  avg reward: {sum(rewards) / len(rewards):.3f}")

    asyncio.run(run())
