"""Sweep runner + Pareto frontier utilities for Claude Code-driven BigNum search.

Iterates task_profile x model x seed x harness and stores per-run results at
`logs/<profile>/<harness>/<model_short>_seed<N>/val.json`. Frontier is the
Pareto set over (success_rate up, mean_turn_count down).
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import re
import shutil
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any

import os

from openai import AsyncOpenAI

import verifiers as vf

DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(config_path: Path) -> dict:
    return json.loads(Path(config_path).read_text())


def get_model_short_name(model_id: str) -> str:
    return model_id.split("/")[-1].lower()


# ---------------------------------------------------------------------------
# Harness module loading + evaluation
# ---------------------------------------------------------------------------


def _load_module_from_path(module_path: str | Path):
    path = Path(module_path).resolve()
    module_name = f"cmh_candidate_{path.stem}_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _format_message(msg) -> str:
    if isinstance(msg, str):
        return msg
    role = msg.role if hasattr(msg, "role") else msg.get("role", "?")
    content = msg.content if hasattr(msg, "content") else msg.get("content", "")

    if role == "system":
        return f"[SYSTEM]\n{content}"
    elif role == "user":
        return f"[USER]\n{content}"
    elif role == "assistant":
        parts = ["[ASSISTANT]"]
        if content:
            parts.append(content)
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls is None and isinstance(msg, dict):
            tool_calls = msg.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                if isinstance(tc, str):
                    parts.append(f"  >> tool_call: {tc}")
                    continue
                name = tc.name if hasattr(tc, "name") else tc.get("name", "?")
                raw = (
                    tc.arguments
                    if hasattr(tc, "arguments")
                    else tc.get("arguments", "{}")
                )
                try:
                    args = json.loads(raw) if isinstance(raw, str) else raw
                    args_str = ", ".join(f"{k}={v}" for k, v in args.items())
                except Exception:
                    args_str = str(raw)
                parts.append(f"  >> tool_call: {name}({args_str})")
        return "\n".join(parts)
    elif role == "tool":
        return f"[TOOL]\n{content}"
    return f"[{role.upper()}]\n{content}"


def _clean_tool_calls(tool_calls: Any) -> list[dict] | None:
    if not tool_calls:
        return None
    result = []
    for tc in tool_calls:
        if isinstance(tc, str):
            try:
                result.append(json.loads(tc))
            except Exception:
                result.append({"raw": tc})
        elif isinstance(tc, dict):
            entry = {"name": tc.get("name", ""), "arguments": tc.get("arguments", "{}")}
            if "id" in tc:
                entry["id"] = tc["id"]
            try:
                if isinstance(entry["arguments"], str):
                    entry["arguments"] = json.loads(entry["arguments"])
            except Exception:
                pass
            result.append(entry)
        else:
            entry = {
                "name": getattr(tc, "name", ""),
                "arguments": getattr(tc, "arguments", "{}"),
            }
            if hasattr(tc, "id"):
                entry["id"] = tc.id
            try:
                if isinstance(entry["arguments"], str):
                    entry["arguments"] = json.loads(entry["arguments"])
            except Exception:
                pass
            result.append(entry)
    return result or None


def _clean_message(msg: Any) -> dict:
    role = msg.role if hasattr(msg, "role") else msg.get("role", "")
    content = msg.content if hasattr(msg, "content") else msg.get("content", "")
    tool_calls_raw = getattr(msg, "tool_calls", None)
    if tool_calls_raw is None and isinstance(msg, dict):
        tool_calls_raw = msg.get("tool_calls")
    tool_call_id = getattr(msg, "tool_call_id", None)
    if tool_call_id is None and isinstance(msg, dict):
        tool_call_id = msg.get("tool_call_id")
    out: dict = {"role": role, "content": content}
    tc = _clean_tool_calls(tool_calls_raw)
    if tc:
        out["tool_calls"] = tc
    if tool_call_id:
        out["tool_call_id"] = tool_call_id
    return out


def write_trajectories(outputs: list[dict[str, Any]], output_dir: Path) -> None:
    lines = []
    for o in outputs:
        prompt = o.get("prompt") or []
        question = ""
        for msg in reversed(list(prompt)):
            role = msg.role if hasattr(msg, "role") else msg.get("role", "")
            if role == "user":
                question = (
                    msg.content if hasattr(msg, "content") else msg.get("content", "")
                )
                break

        completion = o.get("completion") or []
        clean_completion = [_clean_message(msg) for msg in completion]

        reward = float(o.get("reward", 0.0))
        entry = {
            "example_id": o.get("example_id"),
            "reward": reward,
            "correct": reward >= 1.0,
            "stop_condition": o.get("stop_condition"),
            "turn_count": o.get("turn_count"),
            "question": question,
            "answer": o.get("answer", ""),
            "harness_trace": o.get("harness_trace", []),
            "completion": clean_completion,
        }
        lines.append(json.dumps(entry))

    (output_dir / "trajectories.jsonl").write_text("\n".join(lines) + "\n")


def _trajectory_text(output: dict[str, Any]) -> str:
    prompt = output.get("prompt") or []
    completion = output.get("completion") or []
    all_messages = list(prompt) + list(completion)

    lines = [_format_message(msg) for msg in all_messages]

    lines.append(f"\nGround truth: {output.get('answer', '')}")
    lines.append(f"Reward: {output.get('reward', 0.0)}")
    metrics = output.get("metrics") or {}
    if metrics:
        lines.append(f"Metrics: {json.dumps(metrics, sort_keys=True)}")
    return "\n".join(lines)


_CORRECT_METRIC_KEYS = ("correct_answer", "correct_answer_llm_judge")


def _correct_metric_value(metrics: dict[str, Any] | None) -> float:
    """Resolve the correctness reward across EM and LLM-judge naming.

    The verl GRPO LLM-judge path registers the reward func as
    ``correct_answer_llm_judge`` (see ``environments/search_r1/env.py``),
    while the default EM path uses ``correct_answer``. Both name a 0/0.25/1.0
    correctness score; ranking and success-rate aggregation should consult
    whichever is present.
    """
    if not metrics:
        return 0.0
    for k in _CORRECT_METRIC_KEYS:
        if k in metrics:
            try:
                return float(metrics[k])
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def summarize_outputs(
    outputs: list[dict[str, Any]],
    metadata: dict[str, Any],
    *,
    max_failures: int = 3,
    max_successes: int = 1,
) -> dict[str, Any]:
    solved = [
        o
        for o in outputs
        if _correct_metric_value(o.get("metrics")) >= 1.0
    ]
    failed = [
        o
        for o in outputs
        if _correct_metric_value(o.get("metrics")) < 1.0
    ]
    failed = sorted(
        failed,
        key=lambda item: (
            float(item.get("reward", 0.0)),
            -int(item.get("turn_count", 0)),
        ),
    )
    successes = sorted(
        solved,
        key=lambda item: (
            -float(item.get("reward", 0.0)),
            int(item.get("turn_count", 0)),
        ),
    )
    turn_counts = [
        int(o.get("turn_count", 0)) for o in outputs if o.get("turn_count") is not None
    ]
    avg_metrics = metadata.get("avg_metrics") or {}
    return {
        "num_examples": len(outputs),
        "avg_reward": metadata.get("avg_reward", 0.0),
        "avg_metrics": avg_metrics,
        "success_rate": _correct_metric_value(avg_metrics),
        "format_rate": avg_metrics.get("format_reward", 0.0),
        "mean_turn_count": mean(turn_counts) if turn_counts else 0.0,
        "solved_examples": len(solved),
        "failed_examples": len(failed),
        "failure_trajectories": [
            {
                "example_id": item.get("example_id"),
                "reward": item.get("reward", 0.0),
                "turn_count": item.get("turn_count", 0),
                "trace": _trajectory_text(item),
            }
            for item in failed[:max_failures]
        ],
        "success_trajectories": [
            {
                "example_id": item.get("example_id"),
                "reward": item.get("reward", 0.0),
                "turn_count": item.get("turn_count", 0),
                "trace": _trajectory_text(item),
            }
            for item in successes[:max_successes]
        ],
    }


class RoundRobinAsyncOpenAI:
    """Round-robin wrapper across N AsyncOpenAI clients.

    Each top-level attribute access (e.g. `client.chat`, `client.models`) is
    dispatched to the next underlying AsyncOpenAI instance, so a sequence of
    calls naturally fans out across replicas.

    Used for Search-R1 meta-harness eval against N vLLM servers (one per GPU)
    when SEARCH_R1_VLLM_URLS=<csv of /v1 endpoints> is set.
    """

    def __init__(self, base_urls: list[str], api_key: str = "EMPTY"):
        from itertools import cycle

        self._clients = [AsyncOpenAI(base_url=u, api_key=api_key) for u in base_urls]
        self._cycle = cycle(range(len(self._clients)))

    def __getattr__(self, name):
        idx = next(self._cycle)
        return getattr(self._clients[idx], name)


def _make_client(
    model: str,
    base_url: str,
    client_type: str = "openai_chat_completions",
    api_key_var: str | None = None,
):
    """Create a verifiers Client (already wrapped) or ClientConfig.

    PyPI verifiers (>=0.1.14) requires `env.evaluate(client=...)` to receive
    either a verifiers `Client` instance or a `ClientConfig`. A bare AsyncOpenAI
    or our RoundRobinAsyncOpenAI raises "Unsupported client type" in
    resolve_client(). Wrap the OpenAI-style instance in
    OpenAIChatCompletionsClient — the base Client.__init__ accepts a non-config
    instance and stores it as self._client, so `client.chat.completions.create`
    is dispatched through our wrapper unchanged.

    Multi-URL behavior: if SEARCH_R1_VLLM_URLS is set with > 1 entry, the
    wrapped instance is RoundRobinAsyncOpenAI which round-robins requests
    across all replicas.
    """
    if client_type == "anthropic_messages":
        return vf.ClientConfig(
            client_type="anthropic_messages",
            api_key_var=api_key_var or "ANTHROPIC_API_KEY",
            api_base_url=base_url,
        )

    # Lazy import to avoid forcing verifiers as a hard dep during basic
    # module import (e.g. during smoke tests of _make_client wiring).
    from verifiers.clients.openai_chat_completions_client import (
        OpenAIChatCompletionsClient,
    )

    api_key = os.getenv(api_key_var) if api_key_var else "EMPTY"

    multi = os.environ.get("SEARCH_R1_VLLM_URLS", "").strip()
    if multi:
        urls = [u.strip() for u in multi.split(",") if u.strip()]
        if len(urls) > 1:
            inner = RoundRobinAsyncOpenAI(urls, api_key=api_key or "EMPTY")
            return OpenAIChatCompletionsClient(inner)
        if len(urls) == 1:
            inner = AsyncOpenAI(base_url=urls[0], api_key=api_key or "EMPTY")
            return OpenAIChatCompletionsClient(inner)

    inner = AsyncOpenAI(base_url=base_url, api_key=api_key or "EMPTY")
    return OpenAIChatCompletionsClient(inner)


async def evaluate_harness_module(
    *,
    harness_path: str | Path,
    task_model: str,
    task_base_url: str,
    output_dir: str | Path,
    num_train_examples: int,
    num_eval_examples: int,
    seed: int,
    digit_choices: list[int] | None,
    system_prompt: str | None = None,
    candidate_harness_path: str | Path | None = None,
    eval_dataset_path: str | None = None,
    client_type: str = "openai_chat_completions",
    api_key_var: str | None = None,
    temperature: float | None = 0.0,
    top_p: float | None = None,
    top_k: int | None = None,
    max_tokens: int | None = 256,
    rollout_n: int | None = 1,
    max_turns: int | None = None,
    max_concurrent: int = 1,
    max_failures: int = 3,
    max_successes: int = 1,
    enable_thinking: bool = False,
    thinking_budget: int | None = None,
    thinking_effort: str | None = None,
    eval_timeout_s: int | float | None = None,
) -> dict[str, Any]:
    module = _load_module_from_path(harness_path)
    if not hasattr(module, "load_environment"):
        raise AttributeError(f"{harness_path} does not define load_environment")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    load_kwargs: dict[str, Any] = {
        "num_train_examples": num_train_examples,
        "num_eval_examples": num_eval_examples,
        "seed": seed,
        "digit_choices": digit_choices,
    }
    if candidate_harness_path is not None:
        load_kwargs["harness_path"] = str(candidate_harness_path)
    elif system_prompt is not None:
        load_kwargs["system_prompt"] = system_prompt
    if eval_dataset_path is not None:
        load_kwargs["eval_dataset_path"] = eval_dataset_path
    if max_turns is not None:
        load_kwargs["max_turns"] = max_turns

    env = module.load_environment(**load_kwargs)
    if max_tokens is not None:
        if hasattr(env, "set_assistant_token_budget"):
            env.set_assistant_token_budget(int(max_tokens))
        elif hasattr(env, "set_max_total_completion_tokens"):
            env.set_max_total_completion_tokens(int(max_tokens))
    client = _make_client(task_model, task_base_url, client_type, api_key_var)
    sampling_args: dict[str, Any] = {}
    if temperature is not None:
        sampling_args["temperature"] = temperature
    if top_p is not None:
        # Newer Anthropic models (Opus 4.7+) return 400 "`top_p` is deprecated
        # for this model" if it's sent. top_p=1.0 (our verl-mirror default) is
        # a no-op anyway, so just strip it for the anthropic_messages path
        # entirely — preserves behaviour for vLLM/OpenAI where top_p is valid.
        if client_type != "anthropic_messages":
            sampling_args["top_p"] = top_p
    if max_tokens is not None:
        sampling_args["max_tokens"] = max_tokens
    if rollout_n is not None and client_type != "anthropic_messages":
        sampling_args["n"] = int(rollout_n)
    extra_body: dict[str, Any] = {}
    if "localhost" in task_base_url or "127.0.0.1" in task_base_url:
        extra_body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        if thinking_budget is not None:
            extra_body["thinking_token_budget"] = thinking_budget
        # top_k matches verl val_kwargs.top_k=20 for the vLLM path. Anthropic
        # rejects top_k when extended thinking is enabled (400 "top_k must be
        # unset when thinking is enabled"), and the OpenAI/Anthropic public
        # APIs don't share top_k semantics with vLLM anyway, so this is
        # gated to the local-vLLM path only.
        if top_k is not None:
            extra_body["top_k"] = top_k
    if extra_body:
        sampling_args["extra_body"] = extra_body
    # Anthropic extended thinking has two modes:
    #   (1) Adaptive (newer models, e.g. Opus 4.7): thinking={"type":"adaptive"}
    #       and output_config={"effort": "low|medium|high|xhigh|max"}. Older
    #       enabled/budget_tokens form returns 400 on these models.
    #   (2) Enabled (older models): thinking={"type":"enabled","budget_tokens":N},
    #       requires max_tokens > N. Some newer models return 400 on this form.
    # The two are mutually exclusive — when thinking_effort is provided we
    # always pick adaptive; otherwise fall back to the budget form if a budget
    # is given. verifiers' AnthropicMessagesClient passes sampling_args dict
    # through to messages.create() via **kwargs, so both `thinking` and
    # `output_config` are forwarded natively.
    if client_type == "anthropic_messages":
        if thinking_effort is not None:
            sampling_args["thinking"] = {"type": "adaptive"}
            sampling_args["output_config"] = {"effort": thinking_effort}
        elif thinking_budget is not None:
            sampling_args["thinking"] = {
                "type": "enabled",
                "budget_tokens": int(thinking_budget),
            }
    eval_coro = env.evaluate(
        client=client,
        model=task_model,
        sampling_args=sampling_args,
        num_examples=num_eval_examples,
        max_concurrent=max_concurrent,
        results_path=output_dir,
        state_columns=["harness_trace", "turn_count", "harness_name"],
        save_results=True,
    )
    if eval_timeout_s is not None and float(eval_timeout_s) > 0:
        results = await asyncio.wait_for(eval_coro, timeout=float(eval_timeout_s))
    else:
        results = await eval_coro
    summary = summarize_outputs(
        results["outputs"],
        results["metadata"],
        max_failures=max_failures,
        max_successes=max_successes,
    )
    summary["harness_path"] = str(Path(harness_path).resolve())
    summary["task_model"] = task_model
    summary["seed"] = seed
    summary["digit_choices"] = digit_choices
    if system_prompt is not None:
        summary["system_prompt"] = system_prompt
    (output_dir / "val.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_trajectories(results["outputs"], output_dir)
    return {"results": results, "summary": summary}


def evaluate_harness_module_sync(**kwargs) -> dict[str, Any]:
    return asyncio.run(evaluate_harness_module(**kwargs))


def _cfg_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed


def _archive_incomplete_output(out_dir: Path, reason: str) -> None:
    """Move partial benchmark outputs aside before retrying a harness.

    A stalled eval can leave results.jsonl/metadata.json without val.json. If the
    next attempt reuses that directory, downstream code can mix partial and fresh
    outputs. Keep the partial files for debugging, but make the retry start from
    a clean output_dir.
    """
    if not out_dir.exists() or (out_dir / "val.json").exists():
        return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    target = out_dir.with_name(f"{out_dir.name}.incomplete-{reason}-{stamp}")
    suffix = 0
    while target.exists():
        suffix += 1
        target = out_dir.with_name(
            f"{out_dir.name}.incomplete-{reason}-{stamp}-{suffix}"
        )
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(out_dir), str(target))
    print(f"    archived incomplete output {out_dir} -> {target}", flush=True)


def _run_eval_job_worker(payload: tuple[str, dict[str, Any], int]) -> tuple[str, bool]:
    desc, kwargs, eval_retries = payload
    print(f"  [bench] {desc}", flush=True)
    out_dir = Path(kwargs["output_dir"])
    ok = False
    last_exc: Exception | None = None
    for attempt in range(1, eval_retries + 2):
        try:
            evaluate_harness_module_sync(**kwargs)
            ok = True
            break
        except Exception as exc:
            last_exc = exc
            label = "timeout" if isinstance(exc, TimeoutError) else "error"
            print(
                f"    FAIL {desc} attempt {attempt}: {type(exc).__name__}: {exc}",
                flush=True,
            )
            _archive_incomplete_output(out_dir, label)
            if attempt <= eval_retries:
                print(f"    retrying {desc} ({attempt}/{eval_retries})", flush=True)
    if ok:
        return (desc, True)
    print(f"    FAIL {desc}: exhausted retries ({last_exc})", flush=True)
    return (desc, False)


# ---------------------------------------------------------------------------
# Results layout + loading
# ---------------------------------------------------------------------------


def run_dir(
    base: Path,
    profile: str,
    harness: str,
    model_short: str,
    seed: int = DEFAULT_SEED,
    enable_thinking: bool = False,
) -> Path:
    suffix = "-think" if enable_thinking else ""
    leaf = (
        f"{model_short}{suffix}"
        if seed == DEFAULT_SEED
        else f"{model_short}{suffix}_seed{seed}"
    )
    return base / profile / harness / leaf


def parse_run_path(base: Path, filepath: Path) -> dict | None:
    try:
        rel = filepath.parent.relative_to(base)
        parts = rel.parts
        if len(parts) != 3:
            return None
        profile, harness, model_leaf = parts
        m = re.match(r"^(.+)_seed(\d+)$", model_leaf)
        if m:
            model_short = m.group(1)
            seed = int(m.group(2))
        else:
            model_short = model_leaf
            seed = DEFAULT_SEED
        return {
            "profile": profile,
            "harness": harness,
            "model_short": model_short,
            "seed": seed,
        }
    except (ValueError, IndexError):
        return None


def load_results(base_dir: Path, filename: str = "val.json") -> dict:
    results: dict = {}
    for filepath in base_dir.rglob(filename):
        parsed = parse_run_path(base_dir, filepath)
        if not parsed:
            continue
        try:
            data = json.loads(filepath.read_text())
            key = (parsed["model_short"], parsed["profile"], parsed["harness"])
            data["_seed"] = parsed["seed"]
            results[key] = data
        except (json.JSONDecodeError, KeyError):
            continue
    return results


# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------


def compute_pareto_frontier(
    points: list[tuple[str, float, float]],
) -> list[tuple[str, float, float]]:
    """Pareto: maximize success_rate, minimize mean_turn_count."""
    sorted_points = sorted(points, key=lambda x: (-x[1], x[2]))
    pareto = []
    min_turns = float("inf")
    for name, acc, turns in sorted_points:
        if turns <= min_turns:
            pareto.append((name, acc, turns))
            min_turns = turns
    return pareto


def print_frontier(
    logs_dir: Path,
    config: dict,
    model_filter: str | None = None,
) -> dict:
    results = load_results(logs_dir, "val.json")
    if not results:
        print("No results found.")
        return {}
    if model_filter:
        results = {k: v for k, v in results.items() if k[0] == model_filter}

    profiles = list(config["task_profiles"].keys())

    # Aggregate across (profile, model, seed) per harness
    by_harness: dict[str, dict[str, list]] = defaultdict(
        lambda: {"success": [], "turns": [], "format": []}
    )
    for (_, _, harness), data in results.items():
        by_harness[harness]["success"].append(float(data.get("success_rate", 0.0)))
        by_harness[harness]["turns"].append(float(data.get("mean_turn_count", 0.0)))
        by_harness[harness]["format"].append(float(data.get("format_rate", 0.0)))

    points = []
    for h, stats in by_harness.items():
        if not stats["success"]:
            continue
        points.append((h, mean(stats["success"]), mean(stats["turns"])))

    pareto = compute_pareto_frontier(points)
    pareto_names = {n for n, _, _ in pareto}

    # Per-profile best
    by_profile: dict[str, list] = defaultdict(list)
    for (model_short, profile, harness), data in results.items():
        by_profile[profile].append(
            {
                "harness": harness,
                "model_short": model_short,
                "success_rate": float(data.get("success_rate", 0.0)),
                "mean_turn_count": float(data.get("mean_turn_count", 0.0)),
            }
        )

    frontier: dict[str, Any] = {}
    for profile in profiles:
        entries = by_profile.get(profile, [])
        if entries:
            best = max(
                entries,
                key=lambda x: (x["success_rate"], -x["mean_turn_count"]),
            )
            frontier[profile] = {
                "best_harness": best["harness"],
                "model_short": best["model_short"],
                "success_rate": round(best["success_rate"], 4),
                "mean_turn_count": round(best["mean_turn_count"], 3),
            }

    frontier["_pareto"] = [
        {
            "harness": n,
            "avg_success_rate": round(a, 4),
            "avg_mean_turn_count": round(t, 3),
        }
        for n, a, t in pareto
    ]

    # Headline "best" by primary axis (success, then turns)
    if points:
        best_name, best_acc, best_turns = sorted(points, key=lambda x: (-x[1], x[2]))[0]
        frontier["_best"] = {
            "harness": best_name,
            "avg_success_rate": round(best_acc, 4),
            "avg_mean_turn_count": round(best_turns, 3),
        }

    frontier_path = logs_dir / "frontier_val.json"
    frontier_path.write_text(json.dumps(frontier, indent=2) + "\n")

    # Print compact table
    print("\n" + "=" * 64)
    print("PARETO FRONTIER (success_rate up, mean_turns down)")
    print("=" * 64)
    print(f"  {'harness':<36} {'acc':>7} {'turns':>8}")
    print(f"  {'-' * 54}")
    sorted_rows = sorted(points, key=lambda x: (-x[1], x[2]))
    for name, acc, turns in sorted_rows:
        marker = " *" if name in pareto_names else "  "
        display = f"{name}{marker}"
        print(f"  {display:<36} {acc * 100:>6.1f}% {turns:>8.2f}")
    print("\nPER-PROFILE BEST:")
    for profile in profiles:
        info = frontier.get(profile)
        if info:
            print(
                f"  {profile:<8} {info['best_harness']}  "
                f"({info['success_rate'] * 100:.1f}%, {info['mean_turn_count']:.2f} turns)"
            )
    print(f"\nWrote {frontier_path}")
    return frontier


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------


def run_sweep(
    config: dict,
    harnesses: list[tuple[str, Path, str | Path | None]],
    logs_dir: Path,
    *,
    force: bool = False,
) -> list[tuple[str, bool]]:
    """Run each (profile, model, seed, harness) combination.

    harnesses: list of (name, harness_path, system_prompt_or_candidate_path)
      - 3rd element str  → system_prompt injected into load_environment (bignum style)
      - 3rd element Path → candidate_harness_path injected as harness_path kwarg (search_qa style)
      - 3rd element None → load_environment() called with no extra args
    Skips runs whose val.json already exists unless force=True.
    Returns (desc, ok) tuples.

    By default up to 3 harness jobs run concurrently; each job uses
    eval.max_concurrent, defaulting to 256, for its own example-level inference
    fanout. Set eval.parallel_harness_evals=1 to preserve sequential harness
    evaluation.
    """
    profiles = config["task_profiles"]
    models = config["models"]
    seeds = config.get("seeds", [DEFAULT_SEED])
    eval_cfg = config.get("eval", {})

    enable_thinking = eval_cfg.get("enable_thinking", False)
    thinking_budget = eval_cfg.get("thinking_budget", None)
    thinking_effort = eval_cfg.get("thinking_effort", None)
    eval_timeout_s = _cfg_int(
        eval_cfg.get("eval_timeout_s", os.environ.get("META_HARNESS_EVAL_TIMEOUT_S")),
        3600,
    )
    eval_retries = _cfg_int(
        eval_cfg.get("eval_retries", os.environ.get("META_HARNESS_EVAL_RETRIES")),
        1,
    )
    if eval_retries is None or eval_retries < 0:
        eval_retries = 0
    parallel_harness_evals = _cfg_int(
        eval_cfg.get(
            "parallel_harness_evals",
            os.environ.get(
                "META_HARNESS_PARALLEL_HARNESS_EVALS",
                os.environ.get("EVAL_PARALLEL_HARNESS_EVALS"),
            ),
        ),
        3,
    )
    if parallel_harness_evals is None or parallel_harness_evals < 1:
        parallel_harness_evals = 1

    jobs: list[tuple[str, dict]] = []
    for profile_name, profile_cfg in profiles.items():
        for model_cfg in models:
            model = model_cfg["model"]
            base_url = model_cfg["api_base"]
            m_client_type = model_cfg.get("client_type", "openai_chat_completions")
            m_api_key_var = model_cfg.get("api_key_var")
            model_short = get_model_short_name(model)
            for seed in seeds:
                for harness_name, harness_path, sys_prompt_or_candidate in harnesses:
                    out_dir = run_dir(
                        logs_dir,
                        profile_name,
                        harness_name,
                        model_short,
                        seed,
                        enable_thinking=enable_thinking,
                    )
                    val_file = out_dir / "val.json"
                    if val_file.exists() and not force:
                        continue
                    if not val_file.exists():
                        _archive_incomplete_output(out_dir, "resume")
                    desc = f"{profile_name}/{harness_name}/{model_short}_seed{seed}"
                    job_kwargs: dict[str, Any] = {
                        "harness_path": harness_path,
                        "task_model": model,
                        "task_base_url": base_url,
                        "output_dir": out_dir,
                        "num_train_examples": eval_cfg.get("num_train_examples", 64),
                        "num_eval_examples": eval_cfg.get("num_eval_examples", 32),
                        "seed": seed,
                        "digit_choices": profile_cfg.get("digit_choices"),
                        "client_type": m_client_type,
                        "api_key_var": m_api_key_var,
                        "temperature": eval_cfg.get("temperature", 0.0),
                        "top_p": eval_cfg.get("top_p"),
                        "top_k": eval_cfg.get("top_k"),
                        "max_tokens": eval_cfg.get("max_tokens", 256),
                        "rollout_n": _cfg_int(
                            eval_cfg.get("n", os.environ.get("EVAL_ROLLOUT_N")),
                            1,
                        ),
                        "max_turns": eval_cfg.get("max_turns"),
                        "max_concurrent": _cfg_int(
                            eval_cfg.get(
                                "max_concurrent",
                                os.environ.get("EVAL_MAX_CONCURRENT"),
                            ),
                            256,
                        ),
                        "max_failures": eval_cfg.get("max_failures_logged", 3),
                        "max_successes": eval_cfg.get("max_successes_logged", 1),
                        "enable_thinking": enable_thinking,
                        "thinking_budget": thinking_budget,
                        "thinking_effort": thinking_effort,
                        "eval_timeout_s": eval_timeout_s,
                    }
                    if isinstance(sys_prompt_or_candidate, Path):
                        job_kwargs["candidate_harness_path"] = sys_prompt_or_candidate
                    elif isinstance(sys_prompt_or_candidate, str):
                        job_kwargs["system_prompt"] = sys_prompt_or_candidate
                    if eval_cfg.get("eval_dataset_path"):
                        job_kwargs["eval_dataset_path"] = eval_cfg["eval_dataset_path"]
                    jobs.append((desc, job_kwargs))

    if parallel_harness_evals <= 1 or len(jobs) <= 1:
        return [_run_eval_job_worker((desc, kwargs, eval_retries)) for desc, kwargs in jobs]

    print(
        f"  [bench] running up to {parallel_harness_evals} harness eval job(s) concurrently",
        flush=True,
    )

    results: list[tuple[str, bool]] = []
    with ProcessPoolExecutor(max_workers=parallel_harness_evals) as executor:
        futures = [
            executor.submit(_run_eval_job_worker, (desc, kwargs, eval_retries))
            for desc, kwargs in jobs
        ]
        for fut in as_completed(futures):
            results.append(fut.result())
    return results


# ---------------------------------------------------------------------------
# CLI (single-harness smoke test)
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harness", required=True, help="Path to harness .py")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "config.json"),
        help="Sweep config JSON",
    )
    parser.add_argument("--logs-dir", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    logs_dir = (
        Path(args.logs_dir) if args.logs_dir else (Path(__file__).parent / "logs")
    )
    logs_dir.mkdir(parents=True, exist_ok=True)
    harness_path = Path(args.harness).resolve()
    harness_name = harness_path.stem
    run_sweep(config, [(harness_name, harness_path, None)], logs_dir, force=args.force)
    print_frontier(logs_dir, config)


if __name__ == "__main__":
    main()
