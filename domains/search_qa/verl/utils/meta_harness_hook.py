"""Search-R1 meta-harness hook helpers for verl.

When env var HARNESS_PATH points to a candidate harness file (produced by the
meta-harness loop), verl loads the candidate's CandidateEnv class and uses its
SYSTEM_PROMPT / USER_PROMPT_TEMPLATE / env_response (+ TOOLS) at training time
so the harness chosen during meta-harness eval is faithfully reproduced inside
the verl rollout.

Two integration points (verl source modifications):

  1. verl/utils/dataset/rl_dataset.py:_build_messages
     - calls `apply_prompt_overrides(messages, example)` to swap the system
       message content to candidate.SYSTEM_PROMPT and the user message content
       to candidate.USER_PROMPT_TEMPLATE.format(question=...)

  2. verl/experimental/agent_loop/tool_agent_loop.py:_handle_processing_tools_state
     - calls `should_dispatch_to_candidate()` first; if True, calls
       `dispatch_env_response(...)` to run candidate.env_response on the
       reconstructed assistant turn, returning ToolMessages that we convert
       back to verl-format dicts.

If HARNESS_PATH is unset, all hooks are no-ops.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ----- candidate caching ----------------------------------------------------

_CACHED: dict[str, Any] = {}


def reset_harness_cache() -> None:
    """Clear process-local CandidateEnv and instance caches.

    The ordinary GRPO path uses one HARNESS_PATH per process, so caching is
    desirable. Batched val-gate evaluation intentionally changes HARNESS_PATH
    inside one long-lived Ray process, so it must reset this cache between
    candidate evaluations.
    """
    _CACHED.clear()


def activate_harness_path(path: str | Path | None):
    """Set the active harness for the current process and refresh prompt envs.

    Returns the loaded CandidateEnv class, or None when path is empty/invalid.
    """
    reset_harness_cache()
    for key in ("_HARNESS_SYSTEM_PROMPT", "_HARNESS_USER_PROMPT_TEMPLATE"):
        os.environ.pop(key, None)

    if path is None or str(path).strip() == "":
        os.environ.pop("HARNESS_PATH", None)
        return None

    os.environ["HARNESS_PATH"] = str(path)
    klass = get_candidate_env_class()
    if klass is not None:
        _cache_prompt_strings(klass)
    return klass


def _load_module_from_path(path: str | Path):
    """Load a candidate harness file as an anonymous module.

    Also adds the file's grandparent directory (the run_dir, which contains
    `harnesses/`) to sys.path so the candidate may use
    `from harnesses.hN.harness import ...` to reuse helpers from earlier
    accepted harnesses (PEP 420 namespace packages — no __init__.py required).
    """
    p = Path(path).resolve()
    # <project>/meta_harness/runs/<name>/harnesses/<n>/harness.py
    #   p.parent.parent.parent = run_dir (so `from harnesses.hN.harness import ...` resolves)
    #   p.parent.parent.parent.parent.parent = project root (so candidate's
    #     `from environments.search_r1.base_harness import ...` resolves
    #     without needing PYTHONPATH=PROJECT_DIR runtime_env propagation —
    #     PYTHONPATH override on Ray workers caused all 8 AgentLoopWorker
    #     actors to cold-import verl from FSx instead of NVMe site-packages,
    #     hanging the trainer for 30+ min).
    run_dir = p.parent.parent.parent
    project_dir = p.parent.parent.parent.parent.parent
    for d in (str(run_dir), str(project_dir)):
        if d not in sys.path:
            sys.path.insert(0, d)
    module_name = f"sr1_meta_harness_candidate_{abs(hash(str(p)))}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, p)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load harness module from {p}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def get_candidate_env_class():
    """Return CandidateEnv class from $HARNESS_PATH, cached. None if unset/invalid."""
    if "candidate_env_class" in _CACHED:
        return _CACHED["candidate_env_class"]
    path = os.environ.get("HARNESS_PATH", "").strip()
    if not path:
        _CACHED["candidate_env_class"] = None
        return None
    try:
        mod = _load_module_from_path(path)
    except Exception as exc:
        logger.warning("[meta_harness_hook] failed to import HARNESS_PATH=%s: %s", path, exc)
        _CACHED["candidate_env_class"] = None
        return None
    klass = getattr(mod, "CandidateEnv", None)
    if klass is None:
        logger.warning("[meta_harness_hook] HARNESS_PATH=%s has no CandidateEnv class", path)
        _CACHED["candidate_env_class"] = None
        return None
    logger.warning("[meta_harness_hook] loaded CandidateEnv from %s", path)
    _CACHED["candidate_env_class"] = klass
    return klass


def get_candidate_instance(max_turns: Optional[int] = None):
    """Return a process-local CandidateEnv() instance for env_response dispatch.

    Bypasses verifiers' ToolEnv constructor (which requires dataset/rubric/etc.)
    by using `__new__`, then manually populates the subset of instance
    attributes that candidate code reads at rollout time:

      - `tools`        : matches `env_response` dispatch loop
      - `system_prompt`: SearchR1Env eval ctor sets this; some candidates read it
      - `max_turns`    : injected from verl's `rollout_config.multi_turn.
                         max_assistant_turns` on first call. Without this,
                         candidates like h10 fall back to a hardcoded default
                         (3) and their budget-warning trigger fires at a
                         different turn than under meta-harness eval (where
                         env.py:load_environment sets max_turns explicitly).

    First call populates the cache and sets max_turns (if provided). Subsequent
    calls return the cached instance regardless of the max_turns arg — the
    instance is process-local and verl's max_assistant_turns is constant across
    a rollout.
    """
    if "candidate_instance" in _CACHED:
        return _CACHED["candidate_instance"]
    klass = get_candidate_env_class()
    if klass is None:
        _CACHED["candidate_instance"] = None
        return None
    # Build a bare instance: bypass __init__ to avoid verifiers ToolEnv init.
    inst = klass.__new__(klass)
    inst.tools = list(getattr(klass, "TOOLS", []) or [])
    inst.system_prompt = getattr(klass, "SYSTEM_PROMPT", None)
    if max_turns is not None:
        inst.max_turns = int(max_turns)
    _CACHED["candidate_instance"] = inst
    return inst


# ----- prompt overrides (Phase 6 hook) --------------------------------------


def _cache_prompt_strings(klass) -> None:
    """Cache CandidateEnv.SYSTEM_PROMPT / USER_PROMPT_TEMPLATE in env vars so
    child processes (HF datasets filter pool, Ray AgentLoopWorker actors) can
    apply the override WITHOUT re-importing the candidate (which drags in the
    full verifiers tree from FSx — caused 30+ min hang during dataset
    construction + actor boot under FSx contention).

    Called once from preamble.py after get_candidate_env_class() succeeds.
    """
    if klass is None:
        return
    sys_prompt = getattr(klass, "SYSTEM_PROMPT", None)
    template = getattr(klass, "USER_PROMPT_TEMPLATE", None)
    if isinstance(sys_prompt, str):
        os.environ["_HARNESS_SYSTEM_PROMPT"] = sys_prompt
    if isinstance(template, str):
        os.environ["_HARNESS_USER_PROMPT_TEMPLATE"] = template


def apply_prompt_overrides(messages: list[dict], example: dict) -> list[dict]:
    """Mutate messages in place: replace system content with the candidate's
    SYSTEM_PROMPT and user content with USER_PROMPT_TEMPLATE.format(question=...).

    Reads strings from os.environ["_HARNESS_SYSTEM_PROMPT" /
    "_HARNESS_USER_PROMPT_TEMPLATE"] populated at preamble time. This avoids
    importing the candidate module in child processes (HF datasets filter pool,
    Ray AgentLoopWorker actors) — the full verifiers transitive tree from FSx
    triggered concurrent import storms that hung the trainer for 30+ minutes
    in run 29044. If the env vars are unset (HARNESS_PATH was never loaded
    on main), this is a fast no-op.

    Question source: example['extra_info']['question'] when present; falls back
    to original user content if the template lacks a {question} slot.
    """
    sys_prompt = os.environ.get("_HARNESS_SYSTEM_PROMPT")
    template = os.environ.get("_HARNESS_USER_PROMPT_TEMPLATE")
    if not sys_prompt and not template:
        return messages

    extra_info = example.get("extra_info") or {}
    question = str(extra_info.get("question", "")) if isinstance(extra_info, dict) else ""

    seen_system = False
    seen_user = False
    for msg in messages:
        role = msg.get("role") if isinstance(msg, dict) else None
        if role == "system" and not seen_system and sys_prompt:
            msg["content"] = sys_prompt
            seen_system = True
        elif role == "user" and not seen_user and template:
            if "{question}" in template:
                msg["content"] = template.format(question=question)
            else:
                msg["content"] = template
            seen_user = True
    return messages


# ----- env_response dispatch (Phase 7 hook) ---------------------------------


def should_dispatch_to_candidate() -> bool:
    """True if HARNESS_PATH points to a CandidateEnv with a real env_response override."""
    klass = get_candidate_env_class()
    if klass is None:
        return False
    # Candidate must define env_response. Inheriting from SearchR1Env counts.
    return getattr(klass, "env_response", None) is not None


# ----- pending-state setup_state dispatch (Phase 10 hook) -------------------
#
# Some candidates (h29 axis-E pre-fetch is the canonical example) implement
# their primary mechanism inside `setup_state` by MUTATING THE INITIAL USER
# MESSAGE. In verifiers eval, `setup_state` runs BEFORE the model's first
# generation, so any mutation flows into the first turn's prompt. Verl's
# `ToolAgentLoop`, however, only invokes `setup_state` via
# `call_candidate_env_response` — which only fires in the PROCESSING_TOOLS
# state, *after* the model has already generated turn 1 against the
# un-augmented prompt. Worse, by then `agent_data.prompt_ids` has been
# tokenized and the dict mutation would have no effect even if it happened.
#
# This hook closes the gap: invoke `setup_state` at PENDING state, before
# `apply_chat_template` runs. We expose `agent_data.messages` as
# `state["prompt"]` so candidates can iterate / mutate the initial user
# message in place — the mutation propagates to `agent_data.messages` and
# then to the tokenized `prompt_ids`.
#
# Strict guards so this is a no-op for everything except candidates that
# explicitly declare a `setup_state` override:
#   * HARNESS_PATH unset                        → no-op
#   * `klass.__name__ == "SearchR1Env"` alias   → no-op (preserves base)
#   * `"setup_state" not in klass.__dict__`     → no-op (lazy path via
#                                                  call_candidate_env_response
#                                                  still runs at PROCESSING_TOOLS,
#                                                  unchanged from pre-fix)
#   * `state["_setup_done"]` already True       → no-op (re-entry safety)
#
# h0 (alias), h21 (no setup_state override), and any HARNESS_PATH=None run
# all hit one of the first three guards. Only candidates with an explicit
# `setup_state` method in their own __dict__ (currently h29) take this
# path. Their existing lazy-path setup_state call (via
# `call_candidate_env_response`) is short-circuited by the `_setup_done`
# flag set here, so setup_state still runs exactly once per rollout.


async def run_candidate_setup_state_if_needed(agent_data) -> None:
    """Invoke `candidate.setup_state(state)` at the pending state when the
    candidate explicitly overrides setup_state. State["prompt"] is set to
    `agent_data.messages` so prompt-mutating candidates (h29 axis-E
    pre-fetch) can rewrite the initial user message in place.

    No-op for: HARNESS_PATH unset, base alias, candidates that don't
    override setup_state in their own __dict__, or trajectories where
    setup_state has already run. See the module-level Phase 10 comment for
    rationale. Any exception inside the candidate is logged and swallowed
    so a buggy setup_state doesn't crash the rollout — the original
    un-augmented prompt is then used (degrades to base-harness behavior
    for that one rollout).
    """
    klass = get_candidate_env_class()
    if klass is None:
        return
    if klass.__name__ == "SearchR1Env":
        return  # base alias case — fall through to base behavior
    if "setup_state" not in klass.__dict__:
        # Candidate inherits setup_state from base (e.g. h21). The base
        # setup_state only initialises bookkeeping keys (turns, turn_count,
        # harness_name, harness_trace) and does not mutate the prompt.
        # Defer to the existing lazy path inside call_candidate_env_response.
        return
    candidate = get_candidate_instance()
    if candidate is None:
        return
    state = get_candidate_state(agent_data)
    if state.get("_setup_done", False):
        return
    setup_fn = getattr(candidate, "setup_state", None)
    if setup_fn is None:
        return
    # Expose agent_data.messages as state["prompt"] so the candidate's
    # setup_state can locate and mutate the initial user message. The
    # mutation propagates to agent_data.messages by-reference because dict
    # entries (`{"role": "user", "content": "..."}`) are shared objects.
    state["prompt"] = agent_data.messages
    try:
        result = await setup_fn(state)
        if isinstance(result, dict):
            state.update(result)
    except Exception as exc:
        logger.warning(
            "[meta_harness_hook] candidate.setup_state raised at pending "
            "state, continuing with un-augmented prompt: %s",
            exc,
        )
    state["_setup_done"] = True


# ----- no_tools_called dispatch (Phase 8 hook) -----------------------------
#
# Eval-side: SearchR1Env declares `@vf.stop async def no_tools_called(state)`
# which verifiers' rollout loop polls each turn — when True, the trajectory
# terminates. Subclasses can override this (mh-full-v4's accepted h2 returns
# False while state['turns'] < 1, forcing the model to search at least once
# before answering).
#
# verl's tool_agent_loop instead unconditionally terminates when the latest
# assistant turn produced no tool_calls. The hook below lets candidates
# whose no_tools_called diverges from the base reclaim the eval-time
# behavior; when the candidate says "don't stop yet", we also mimic
# SearchR1Env.get_prompt_messages by appending the same nudge user message
# it would inject during eval, so the next generation pass sees identical
# context.


# Default nudge text — verbatim copy of the UserMessage SearchR1Env builds
# in base_harness.get_prompt_messages when the latest assistant turn has no
# tool_calls. Keep these strings in sync if base_harness changes.
_NUDGE_TEXT = (
    "Use the search tool to find information, then write your final "
    "answer inside <answer></answer> tags. Example: <answer>42</answer>"
)


def _candidate_overrides_no_tools_called() -> bool:
    """True only when the candidate class declares its own no_tools_called.

    Falls through to verl's default termination semantics when the candidate
    is the base alias (h0 = verbatim copy with `CandidateEnv = SearchR1Env`
    at the bottom) or when the method was inherited unchanged. Only true
    overrides take the dispatch path.

    Implementation note: matches `_candidate_overrides_get_prompt_messages`'s
    structure. We can't rely on `cand_fn is base_fn` identity comparison
    because `_load_module_from_path()` loads `base_harness.py` as a fresh
    module instance (spec_from_file_location), so the candidate's MRO sees
    a different `SearchR1Env` object than `from environments.search_r1...`
    yields — the identity check then spuriously returns True and dispatch
    fires on h0 alias when it shouldn't. We work around the same way as
    the get_prompt_messages function: check the class name (the
    `CandidateEnv = SearchR1Env` alias preserves __name__ as "SearchR1Env")
    and consult __dict__ for a locally-defined method.
    """
    klass = get_candidate_env_class()
    if klass is None:
        return False
    # `CandidateEnv = SearchR1Env` alias case: candidate IS the base class,
    # nothing has been overridden — fall through to verl's default termination.
    if klass.__name__ == "SearchR1Env":
        return False
    # Real subclass: did it define no_tools_called itself (not inherit)?
    return "no_tools_called" in klass.__dict__


def discover_candidate_stop_hooks(candidate) -> list[tuple[str, int]]:
    """Return [(method_name, priority), ...] for every @vf.stop-decorated method
    on the candidate, sorted by priority desc then name asc — same ordering
    verifiers' discover_decorated uses during eval. Two stop hooks are
    deliberately excluded because verl handles them through other paths:

      * `no_tools_called`  — dispatched via should_terminate_on_no_tool_calls
        (which also mirrors the get_prompt_messages nudge), so dispatching
        it here would double-count.
      * `max_turns_reached` — handled by verl's `assistant_turns` counter
        against `max_assistant_turns`; the candidate's MAX_TURNS class attr
        is already forwarded to that counter via slurm_train.sh + env.py.

    All other @vf.stop hooks (base ToolEnv's `has_error`, `prompt_too_long`,
    `max_total_completion_tokens_reached`, `has_final_env_response`, plus
    any candidate-defined custom early-stop) are surfaced here so they take
    effect during training rollouts exactly as during meta-harness eval.
    """
    if candidate is None:
        return []
    import inspect

    SKIP = {"no_tools_called", "max_turns_reached"}
    hooks: list[tuple[str, int]] = []
    for name, method in inspect.getmembers(candidate, predicate=inspect.ismethod):
        if name in SKIP:
            continue
        if not getattr(method, "stop", False):
            continue
        priority = int(getattr(method, "stop_priority", 0))
        hooks.append((name, priority))
    hooks.sort(key=lambda x: (-x[1], x[0]))
    return hooks


async def check_candidate_extra_stop_hooks(agent_data) -> bool:
    """True if any candidate @vf.stop hook (other than no_tools_called /
    max_turns_reached, which are dispatched elsewhere) returns True on the
    current trajectory state. Called by tool_agent_loop after each generation
    turn so candidate-defined early-stop conditions terminate verl rollouts
    the same way they terminate meta-harness eval rollouts.

    Exceptions in any individual hook are logged and treated as "no stop" so
    a buggy candidate doesn't crash the trainer.
    """
    if not should_dispatch_to_candidate():
        return False
    candidate = get_candidate_instance()
    if candidate is None:
        return False
    hooks = discover_candidate_stop_hooks(candidate)
    if not hooks:
        return False
    state = get_candidate_state(agent_data)
    for name, _priority in hooks:
        method = getattr(candidate, name, None)
        if method is None:
            continue
        try:
            result = await method(state)
        except Exception as exc:
            logger.warning(
                "[meta_harness_hook] candidate.%s raised, treating as no-stop: %s",
                name, exc,
            )
            continue
        if bool(result):
            logger.info(
                "[meta_harness_hook] candidate stop hook %s triggered termination",
                name,
            )
            return True
    return False


async def should_terminate_on_no_tool_calls(
    agent_data,
    decode_assistant: Optional[Callable[[], Any]] = None,
) -> tuple[bool, Optional[list[dict]]]:
    """Called by tool_agent_loop when the latest assistant turn produced no
    tool_calls. Returns (should_stop, nudge_messages).

    Returns (True, None) (= verl default) unless:
      * a candidate class is loaded,
      * it overrides no_tools_called relative to base SearchR1Env, AND
      * the override returns False on the current state.

    In that path, returns (False, [nudge_user_msg]) so the caller appends
    a user-role nudge identical to base_harness.get_prompt_messages's, then
    transitions back to GENERATING. Errors at any step fall back to the safe
    default (True, None) and a logged warning.

    `decode_assistant` is an optional async-callable that returns the
    decoded text of the just-generated assistant turn (typically
    `tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)`
    via a thread pool). When supplied — and the caller always supplies it
    from `tool_agent_loop._handle_generating_state` — it lets us populate
    `state["trajectory"]` with the latest assistant turn BEFORE invoking
    `candidate.no_tools_called(state)`. Without that, the candidate's
    gates always early-return False on an empty trajectory and the entire
    h1/h15/h21-style nudge-and-gate contract is silently bypassed.
    Argument is optional + defaults to None so older callers (if any) keep
    working with the pre-fix behavior.
    """
    if not _candidate_overrides_no_tools_called():
        return True, None
    candidate = get_candidate_instance()
    if candidate is None:
        return True, None
    state = get_candidate_state(agent_data)

    # Populate trajectory BEFORE calling candidate.no_tools_called. The
    # decode is run lazily and only once per turn — `get_candidate_nudge_text`
    # (which may run immediately after) reuses the populated trajectory
    # rather than re-decoding.
    decoded_text: Optional[str] = None
    if decode_assistant is not None:
        try:
            decoded_text = await decode_assistant()
        except Exception as exc:
            logger.warning(
                "[meta_harness_hook] decode_assistant raised, "
                "trajectory synthesis disabled for this turn: %s",
                exc,
            )
            decoded_text = None
    state["trajectory"] = _reconstruct_trajectory(
        agent_data,
        decoded_assistant_text=decoded_text,
        tool_calls=getattr(agent_data, "tool_calls", None),
    )

    stop_fn = getattr(candidate, "no_tools_called", None)
    if stop_fn is None:
        return True, None
    try:
        result = await stop_fn(state)
    except Exception as exc:
        logger.warning(
            "[meta_harness_hook] candidate.no_tools_called raised, "
            "falling back to terminate: %s",
            exc,
        )
        return True, None
    if bool(result):
        return True, None
    # Candidate says: don't stop. Mirror SearchR1Env.get_prompt_messages by
    # appending a nudge user message. When the candidate overrides
    # get_prompt_messages with a dynamic nudge (commit / gated / recovery,
    # see Phase 9 hook above), pull its actual nudge text out; otherwise
    # fall back to the verbatim base nudge so unchanged candidates remain
    # byte-identical to pre-hook training.
    nudge_text = await get_candidate_nudge_text(
        agent_data, decode_assistant=decode_assistant
    )
    if not nudge_text:
        nudge_text = _NUDGE_TEXT
    return False, [{"role": "user", "content": nudge_text}]


# Minimal duck-typed message classes — avoid hard verifiers dep at training time.
class _MinimalToolCall:
    __slots__ = ("id", "name", "arguments")

    def __init__(self, id: str, name: str, arguments: str):
        self.id = id
        self.name = name
        self.arguments = arguments


class _MinimalAssistantMessage:
    __slots__ = ("role", "content", "tool_calls")

    def __init__(self, role: str, content: str, tool_calls=None):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls

    def __repr__(self):
        return f"AssistantMessage(content={self.content!r}, tool_calls={self.tool_calls!r})"


class _MinimalGenericMessage:
    __slots__ = ("role", "content", "tool_call_id")

    def __init__(self, role: str, content: str, tool_call_id: Optional[str] = None):
        self.role = role
        self.content = content
        self.tool_call_id = tool_call_id


def _verl_dict_to_verifiers(msg: dict):
    """Convert one verl-format message dict to a duck-typed object that
    matches the attribute surface verifiers / search-qa env_response uses."""
    role = msg.get("role", "")
    content = msg.get("content", "")
    if isinstance(content, list):
        # multi-modal — flatten text bits only
        content = "\n".join(c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text")
    if role == "assistant":
        tool_calls_in = msg.get("tool_calls") or []
        tool_calls = [
            _MinimalToolCall(
                id=tc.get("id", "") or "",
                name=tc.get("name", "") or "",
                arguments=tc.get("arguments", "{}") or "{}",
            )
            for tc in tool_calls_in
        ]
        return _MinimalAssistantMessage(role="assistant", content=content or "", tool_calls=tool_calls or None)
    return _MinimalGenericMessage(role=role, content=content or "", tool_call_id=msg.get("tool_call_id"))


def _verifiers_msg_to_verl_dict(m: Any) -> dict:
    role = getattr(m, "role", "tool")
    content = getattr(m, "content", "") or ""
    out: dict = {"role": role, "content": content}
    tcid = getattr(m, "tool_call_id", None)
    if tcid:
        out["tool_call_id"] = tcid
    tcs = getattr(m, "tool_calls", None)
    if tcs:
        out["tool_calls"] = [
            {"id": getattr(tc, "id", ""), "name": getattr(tc, "name", ""), "arguments": getattr(tc, "arguments", "{}")}
            for tc in tcs
        ]
    return out


def get_candidate_state(agent_data) -> dict:
    """Per-trajectory state dict, stashed on agent_data.extra_fields."""
    if not hasattr(agent_data, "extra_fields") or agent_data.extra_fields is None:
        agent_data.extra_fields = {}
    if "harness_state" not in agent_data.extra_fields:
        agent_data.extra_fields["harness_state"] = {
            "turns": 0,
            "turn_count": 0,
            "harness_name": "search_r1",
            "harness_trace": [],
            "trajectory": [],
        }
    return agent_data.extra_fields["harness_state"]


# ----- get_prompt_messages dispatch (Phase 9 hook) ---------------------------
#
# vf.ToolEnv.get_prompt_messages decides the next-turn prompt at each multi-
# turn boundary. Candidate harnesses override it to inject conditional nudge
# text — every historical override (audit of 48 candidates across
# mh-custom/* and mh-full-v[4-6]/*) follows the same family:
#
#   * Calls super().get_prompt_messages OR replicates the base body, AND
#   * Replaces the hardcoded base nudge with a dynamic alternative
#     (commit_nudge for last turn, gated nudge after specific state, ...)
#
# Zero historical candidate mutates the message list itself (no pop / insert
# / del / slice operations were found). So mirroring the override on the
# verl side boils down to: call candidate.get_prompt_messages, find the
# last user-role message it returns (= the nudge it picked), and substitute
# that text for the hardcoded _NUDGE_TEXT inside should_terminate_on_no_tool_calls.
#
# To call candidate.get_prompt_messages we need a verifiers-style state with
# a usable state["trajectory"]. verl's agent_data only carries the flattened
# message accumulator + assistant_turns counter, so we reconstruct just
# enough trajectory entries to satisfy the two access patterns audited in
# the historical candidates:
#   (a) len(state["trajectory"]) — used for last-turn gating
#   (b) state["trajectory"][-1]["prompt"] / ["completion"][-1] — used to
#       inspect the most recent assistant message + its tool_calls
# Older entries are length-correct stubs; we don't materialise their
# prompts/completions because no historical candidate inspects them.


def _candidate_overrides_get_prompt_messages() -> bool:
    """True only when the candidate class redefines get_prompt_messages
    relative to base SearchR1Env. Implementation note: identity comparison
    against a re-imported `SearchR1Env` is unreliable because
    _load_module_from_path() loads `base_harness.py` as a fresh module
    instance (spec_from_file_location), so the candidate's MRO sees a
    different `SearchR1Env` object than `from environments.search_r1...`
    yields. We work around that by checking the class name (the
    `CandidateEnv = SearchR1Env` alias at the bottom of base_harness.py
    preserves __name__ as "SearchR1Env") and consulting __dict__ for a
    locally-defined method.
    """
    klass = get_candidate_env_class()
    if klass is None:
        return False
    # `CandidateEnv = SearchR1Env` alias case: candidate IS the base class,
    # nothing has been overridden — fall through to verl's default nudge.
    if klass.__name__ == "SearchR1Env":
        return False
    # Real subclass: did it define get_prompt_messages itself (not inherit)?
    return "get_prompt_messages" in klass.__dict__


def _reconstruct_trajectory(
    agent_data,
    decoded_assistant_text: Optional[str] = None,
    tool_calls: Optional[list] = None,
) -> list[dict]:
    """Build a verifiers-style state["trajectory"] from verl agent_data.

    Only the LAST entry carries real (prompt, completion); earlier entries
    are stub dicts. This is sufficient for every historical candidate
    pattern (see audit in the dispatcher comment) and keeps reconstruction
    cost O(1) per call.

    Two modes:

      1. Assistant message PRESENT in `agent_data.messages` — use it
         directly. This fires only in the interaction_config_file path
         (`tool_agent_loop.py:281-285`), where verl explicitly decodes the
         latest assistant turn and appends it to `agent_data.messages`.

      2. Assistant message ABSENT in `agent_data.messages` — synthesize
         one from `decoded_assistant_text` + `tool_calls`. This is the
         common Search-R1 multi-turn path: verl's ToolAgentLoop never
         appends the assistant turn to `agent_data.messages` (only tools
         + nudges accumulate there), so the assistant content has to be
         supplied by the caller. The caller (`should_terminate_on_no_
         tool_calls` / `get_candidate_nudge_text`) gets the text via the
         same `decode_assistant` callable that
         `_handle_processing_tools_state_candidate` already uses.

    Without mode 2, every candidate whose `no_tools_called` or
    `get_prompt_messages` reads `state["trajectory"][-1]` saw an empty
    trajectory in verl rollouts even though meta-harness eval gave them
    the full one — silently bypassing the candidate-specific gate +
    dynamic-nudge contract (h21 grounding gate, h15 collapsed-retrieval
    gate, h1 force-commit gate, …).

    Returns [] when there is no assistant turn to reconstruct (e.g.
    we haven't generated once yet, or the caller passed no synthesis
    inputs and the messages list has no assistant role). The caller
    treats `[]` as "fall back to base behavior".
    """
    n_turns = max(0, int(getattr(agent_data, "assistant_turns", 0)))
    if n_turns <= 0:
        return []
    messages = list(getattr(agent_data, "messages", []) or [])
    last_assistant_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            last_assistant_idx = i
            break

    if last_assistant_idx >= 0:
        # Mode 1: assistant turn is already in agent_data.messages.
        prompt_msgs = [_verl_dict_to_verifiers(m) for m in messages[:last_assistant_idx]]
        completion_msgs = [_verl_dict_to_verifiers(messages[last_assistant_idx])]
    elif decoded_assistant_text is not None:
        # Mode 2: synthesize from response_ids + tool_calls. All accumulated
        # messages (system + user + tool, tool, … + any prior nudges) form
        # the prompt up to this assistant turn.
        prompt_msgs = [_verl_dict_to_verifiers(m) for m in messages]
        synthesized_tool_calls = None
        if tool_calls:
            synthesized_tool_calls = [
                _MinimalToolCall(
                    id=getattr(tc, "id", "") or f"tc_{i}",
                    name=tc.name,
                    arguments=tc.arguments,
                )
                for i, tc in enumerate(tool_calls)
            ]
        synthesized = _MinimalAssistantMessage(
            role="assistant",
            content=decoded_assistant_text or "",
            tool_calls=synthesized_tool_calls,
        )
        completion_msgs = [synthesized]
    else:
        # No assistant in messages AND no synthesis input — caller must
        # accept the empty-trajectory fallback.
        return []

    trajectory = [{"prompt": [], "completion": []} for _ in range(n_turns - 1)]
    trajectory.append({"prompt": prompt_msgs, "completion": completion_msgs})
    return trajectory


async def get_candidate_nudge_text(
    agent_data,
    decode_assistant: Optional[Callable[[], Any]] = None,
) -> Optional[str]:
    """Return the nudge user-message text the candidate would inject when
    the latest assistant turn produced no tool_calls, or None to fall back
    to the hardcoded base nudge (_NUDGE_TEXT).

    Only invoked when candidate.no_tools_called has already returned False
    (= "don't stop yet"). We dispatch candidate.get_prompt_messages to
    reproduce whatever conditional nudge logic the harness has, then pull
    the last user-role message out of the returned list.

    `decode_assistant` (optional): same callable as in
    `should_terminate_on_no_tool_calls`. Used to synthesize the latest
    assistant turn into `state["trajectory"]` when it isn't already there
    (= when the caller is the standalone path that bypasses the
    `should_terminate_on_no_tool_calls` populator). When invoked from
    `should_terminate_on_no_tool_calls` immediately after the trajectory
    has already been populated, the existing trajectory is reused.

    Any failure (no override, candidate raised, no trailing user message)
    yields None so the caller defaults to _NUDGE_TEXT.
    """
    if not _candidate_overrides_get_prompt_messages():
        return None
    candidate = get_candidate_instance()
    if candidate is None:
        return None
    state = get_candidate_state(agent_data)
    if not state.get("trajectory"):
        # Trajectory was not populated by an upstream caller — reconstruct
        # here. The decode happens at most once per turn because
        # should_terminate_on_no_tool_calls always populates first when
        # it's the caller.
        decoded_text: Optional[str] = None
        if decode_assistant is not None:
            try:
                decoded_text = await decode_assistant()
            except Exception as exc:
                logger.warning(
                    "[meta_harness_hook] decode_assistant raised in "
                    "get_candidate_nudge_text, falling back: %s",
                    exc,
                )
                decoded_text = None
        state["trajectory"] = _reconstruct_trajectory(
            agent_data,
            decoded_assistant_text=decoded_text,
            tool_calls=getattr(agent_data, "tool_calls", None),
        )
    if not state["trajectory"]:
        return None
    fn = getattr(candidate, "get_prompt_messages", None)
    if fn is None:
        return None
    try:
        result = await fn(state)
    except Exception as exc:
        logger.warning(
            "[meta_harness_hook] candidate.get_prompt_messages raised, "
            "falling back to default nudge: %s",
            exc,
        )
        return None
    # Pull the last user-role message — that's the candidate's nudge.
    if not result:
        return None
    last = result[-1] if isinstance(result, list) else None
    if last is None:
        return None
    role = getattr(last, "role", None) or (last.get("role") if isinstance(last, dict) else None)
    if role != "user":
        return None
    text = getattr(last, "content", None) or (last.get("content") if isinstance(last, dict) else None)
    if not isinstance(text, str) or not text.strip():
        return None
    return text


async def call_candidate_env_response(
    agent_data,
    decode_assistant: Callable[[], Any],
    max_turns: Optional[int] = None,
) -> Optional[list[dict]]:
    """Reconstruct the latest assistant message (from agent_data.response_ids
    + agent_data.tool_calls), build a verifiers-style message list, and call
    candidate.env_response(messages, state). Convert returned ToolMessages
    back to verl-format dicts.

    `max_turns` is forwarded to `get_candidate_instance()` on the first call
    so that candidates which read `self.max_turns` (e.g. h10's budget-warning
    trigger) see the same value verl's tool_agent_loop enforces as its
    `max_assistant_turns` cap.

    On the first dispatch for a trajectory, we also invoke the candidate's
    `setup_state` so any extra state keys the candidate uses (e.g. h6's
    `recent_queries`) are initialised exactly as in meta-harness eval.

    Returns:
        list[dict] of verl-format tool response messages on success.
        None if the candidate raised — caller should treat as "drop trajectory"
        (per Q3 = malformed-tool-call policy: terminate the rollout).
    """
    candidate = get_candidate_instance(max_turns=max_turns)
    if candidate is None:
        return None

    assistant_text = await decode_assistant()
    verl_assistant_msg = {
        "role": "assistant",
        "content": assistant_text or "",
        "tool_calls": [
            {
                "id": getattr(tc, "id", "") or f"tc_{i}",
                "name": tc.name,
                "arguments": tc.arguments,
            }
            for i, tc in enumerate(agent_data.tool_calls)
        ],
    }
    full_verl_messages = list(agent_data.messages) + [verl_assistant_msg]
    verif_messages = [_verl_dict_to_verifiers(m) for m in full_verl_messages]

    state = get_candidate_state(agent_data)
    # Run setup_state once per trajectory so candidates that subclass
    # SearchR1Env.setup_state (or add new keys like `recent_queries`) get the
    # same initialised state they'd see under verifiers eval.
    if not state.get("_setup_done", False):
        setup_fn = getattr(candidate, "setup_state", None)
        if setup_fn is not None:
            try:
                result = await setup_fn(state)
                if isinstance(result, dict):
                    state.update(result)
            except Exception as exc:
                logger.warning(
                    "[meta_harness_hook] candidate.setup_state raised, "
                    "continuing with default state: %s",
                    exc,
                )
        state["_setup_done"] = True

    try:
        verif_tool_responses = await candidate.env_response(verif_messages, state)
    except Exception as exc:
        logger.warning(
            "[meta_harness_hook] candidate.env_response raised, dropping trajectory: %s",
            exc,
        )
        return None

    if not isinstance(verif_tool_responses, list):
        verif_tool_responses = [verif_tool_responses]
    return [_verifiers_msg_to_verl_dict(m) for m in verif_tool_responses]
