"""Search-R1 base harness: multi-turn wiki-18 retrieval to answer factual questions.

This file is the evolution target for the meta-harness. Proposer candidates
subclass SearchR1Env as CandidateEnv, overriding SYSTEM_PROMPT,
USER_PROMPT_TEMPLATE, TOOLS, and env_response as needed.

Contract:
  - CandidateEnv class name must stay as-is (benchmark.py convention)

Surface notes (matches search-qa coverage):
  - SYSTEM_PROMPT: class attribute → system message at every rollout
  - USER_PROMPT_TEMPLATE: class attribute → format-string applied to each
    dataset row's question to build the first user message. Matches the long
    instruction baked into Search-R1 train.parquet (1:1 reproduction at h0).
    Use "{question}" as the slot.
  - TOOLS: class attribute → list of async tool functions; redefining the
    `search` function entry (with different topk, doc count, etc.) is the
    primary lever for retrieval-side experimentation.
  - env_response: instance method → controls the per-turn loop logic
    (nudges, tool-output reformatting, etc.).
"""

from __future__ import annotations

import itertools
import json
import logging
import os
from typing import Any, cast

import httpx
import verifiers as vf
from verifiers.types import AssistantMessage, ToolMessage, UserMessage
from verifiers.utils.message_utils import concat_messages, maybe_normalize_messages

logger = logging.getLogger(__name__)


# Reproduction of train.parquet's two-message structure: short generic system
# prompt + long user prompt with operating instructions and the bare question.
SYSTEM_PROMPT = "You are a helpful and harmless assistant."

USER_PROMPT_TEMPLATE = (
    "Answer the given question. You must conduct reasoning inside <think> and "
    "</think> first every time you get new information. After reasoning, if you "
    "find you lack some knowledge, you can call a search engine by <tool_call> "
    "query </tool_call> and it will return the top searched results between "
    "<tool_response> and </tool_response>. You can search as many times as your "
    "want. If you find no further external knowledge needed, you can directly "
    "provide the answer inside <answer> and </answer>, without detailed "
    "illustrations. For example, <answer> Beijing </answer>. Question: {question}"
)


# -- retrieval client --------------------------------------------------------


def _resolve_retrieval_urls() -> list[str]:
    """SEARCH_R1_RETRIEVAL_URLS (CSV) → list, fallback single endpoint.

    Mirrors verl/tools/search_tool.py:_resolve_retrieval_urls so meta-harness
    eval can use the same MULTI_RETRIEVE=1 layout as training (8 replicas on
    ports 8000..8007).
    """
    env_val = os.environ.get("SEARCH_R1_RETRIEVAL_URLS", "").strip()
    if env_val:
        urls = [u.strip() for u in env_val.split(",") if u.strip()]
        if urls:
            return urls
    return ["http://127.0.0.1:8000/retrieve"]


_url_cycle: itertools.cycle | None = None


def _next_url() -> str:
    """Round-robin across replicas. Single asyncio loop per process →
    `itertools.cycle` is safe without a lock."""
    global _url_cycle
    if _url_cycle is None:
        _url_cycle = itertools.cycle(_resolve_retrieval_urls())
    return next(_url_cycle)


def _passages_to_string(retrieval_result: list) -> str:
    """Stitch retriever passages into the LLM-facing tool_response.

    Layout matches verl/tools/search_tool.py:_passages_to_string
    ("Doc 1 (Title: ...)\\n{body}\\n").
    """
    out = []
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["contents"]
        title, _, body = content.partition("\n")
        out.append(f"Doc {idx + 1} (Title: {title})\n{body}\n")
    return "\n".join(out).strip()


# -- tool function -----------------------------------------------------------


async def search(
    query_list: list[str], topk: int = 3, max_doc_tokens: int = 200
) -> str:
    """Searches the web for relevant information based on a single query. Returns the top-k passages for that one query.

    Mirrors `verl/tools/search_tool.py:SearchTool.execute()` payload + response
    handling so meta-harness h0 (= a verbatim copy of this file) gives bit-for-bit
    identical model-facing tool_response as the native verl SearchTool path
    (HARNESS_PATH=none / unmodified main branch behavior). Specifically:

      - signature `query_list: list[str]` matches the yaml schema in
        configs/tool_config/search_tool_config.yaml (see env.py validation)
      - request payload includes `return_scores=True`, the per-call
        `max_doc_tokens` (default 200), and `tokenizer_name="Qwen/Qwen3.5-2B"`
        so the retriever server wraps each doc as
        `{"document": ..., "score": ...}` (Tier 1 truncation also applied
        server-side). Without these, the server returns raw doc dicts and
        `_passages_to_string` raises `KeyError: 'document'` (caught in
        run 29144).
      - response is wrapped as `json.dumps({"result": <stitched docs>},
        ensure_ascii=False)` — matches verl SearchTool's tool_response shape.

    Args:
        query_list: A list containing exactly ONE fully-formed semantic query.
            Multiple queries are not accepted; pass only one.
        topk: number of passages to return (default 3).
        max_doc_tokens: per-doc token cap applied server-side before passages
            are returned. Default 200. Raise (e.g. 400) for richer per-doc
            content at the cost of larger tool_response; lower (e.g. 100) to
            pack more retrieval calls under the same context budget.
    """
    # Verl SearchTool's policy: lenient clamp instead of strict error.
    # If model emits >1 query (yaml maxItems=1 doesn't always bind), keep
    # only the first and silently drop the rest. Matches
    # verl/tools/search_tool.py:execute() behaviour exactly so model-facing
    # tool_response stays identical between meta-harness eval and verl
    # training even on malformed multi-query emissions.
    cleaned = [str(q).strip() for q in query_list if str(q).strip()]
    if not cleaned:
        return json.dumps(
            {"result": "Error: 'query_list' is missing, empty, or not a list in parameters."},
            ensure_ascii=False,
        )
    if len(cleaned) > 1:
        logger.info(
            "[search] %d queries received; using only the first one and dropping the rest.",
            len(cleaned),
        )
    query = cleaned[0]

    url = _next_url()
    payload: dict[str, Any] = {
        "queries": [query],
        "topk": topk,
        "return_scores": True,
        "max_doc_tokens": max_doc_tokens,
        "tokenizer_name": "Qwen/Qwen3.5-2B",
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("[search] retriever call failed (%s): %s", url, exc)
            return json.dumps(
                {"result": f"Search error: {exc}"}, ensure_ascii=False
            )

    raw_results = data.get("result") or []
    if not raw_results:
        return json.dumps(
            {"result": "No search results found."}, ensure_ascii=False
        )

    # Multi-query support kept for parity even though maxItems=1 means a
    # single retrieval list — verl SearchTool joins multi-query chunks with
    # "\n---\n", we mirror that.
    chunks = [_passages_to_string(retrieval) for retrieval in raw_results]
    final_result = "\n---\n".join(chunks)
    return json.dumps({"result": final_result}, ensure_ascii=False)


# -- environment -------------------------------------------------------------


class SearchR1Env(vf.ToolEnv):
    """Multi-turn search environment over the local wiki-18 retrieval service."""

    SYSTEM_PROMPT: str = SYSTEM_PROMPT
    USER_PROMPT_TEMPLATE: str = USER_PROMPT_TEMPLATE
    TOOLS: list = [search]
    # Per-candidate ceiling on multi-turn search/answer rounds. Read by
    # env.py:load_environment which takes the candidate's MAX_TURNS over its
    # own max_turns argument (subject to MAX_TURNS_HARD_CAP=16). Mirrors the
    # `topk` / `max_doc_tokens` default pattern so candidate harnesses can
    # tune axis F (turn budgeting) by overriding this single class attribute.
    MAX_TURNS: int = 8
    _assistant_token_budget: int | None = None

    def set_assistant_token_budget(self, max_tokens: int | None) -> None:
        """Cap cumulative assistant-generated tokens across all turns."""
        if max_tokens is None:
            self._assistant_token_budget = None
            return
        max_tokens = int(max_tokens)
        self._assistant_token_budget = max_tokens if max_tokens > 0 else None

    def _get_assistant_token_budget(self) -> int | None:
        if self._assistant_token_budget is not None:
            return self._assistant_token_budget
        raw = os.environ.get("SEARCH_R1_ASSISTANT_TOKEN_BUDGET", "").strip()
        if not raw:
            return None
        try:
            budget = int(raw)
        except ValueError:
            logger.warning("Ignoring invalid SEARCH_R1_ASSISTANT_TOKEN_BUDGET=%r", raw)
            return None
        return budget if budget > 0 else None

    @staticmethod
    def _assistant_tokens_used(state: vf.State) -> int:
        total = 0
        for step in state.get("trajectory") or []:
            tokens = step.get("tokens") if isinstance(step, dict) else None
            completion_ids = tokens.get("completion_ids") if tokens else None
            if completion_ids:
                total += len(completion_ids)
        if total:
            return total

        usage = state.get("usage") if isinstance(state, dict) else None
        if isinstance(usage, dict):
            return int(usage.get("output_tokens", 0) or 0)
        return 0

    async def setup_state(self, state: vf.State, **kwargs: Any) -> vf.State:
        state["turns"] = 0
        state["turn_count"] = 0
        state["harness_name"] = "search_r1"
        state["harness_trace"] = []
        state["assistant_token_budget"] = self._get_assistant_token_budget()
        return state

    @vf.stop
    async def assistant_token_budget_reached(self, state: vf.State) -> bool:
        budget = self._get_assistant_token_budget()
        return budget is not None and self._assistant_tokens_used(state) >= budget

    @vf.stop
    async def no_tools_called(self, state: vf.State) -> bool:
        if not state["trajectory"]:
            return False
        last_msg = state["trajectory"][-1]["completion"][-1]
        return not (hasattr(last_msg, "tool_calls") and last_msg.tool_calls)

    async def get_model_response(
        self,
        state: vf.State,
        prompt: vf.Messages,
        client=None,
        model: str | None = None,
        tool_defs=None,
        sampling_args: dict | None = None,
    ):
        budget = self._get_assistant_token_budget()
        if budget is None:
            return await super().get_model_response(
                state, prompt, client, model, tool_defs, sampling_args
            )

        remaining = budget - self._assistant_tokens_used(state)
        if remaining <= 0:
            state["assistant_token_budget_exhausted"] = True
            state["is_completed"] = True
            return await super().get_model_response(
                state, prompt, client, model, tool_defs, {"max_tokens": 0}
            )

        per_call_args = dict(sampling_args or state.get("sampling_args") or {})
        requested = per_call_args.get("max_tokens")
        if requested is None:
            requested = per_call_args.pop("max_new_tokens", None)
        if requested is not None:
            try:
                remaining = min(remaining, int(requested))
            except (TypeError, ValueError):
                logger.warning("Ignoring non-integer max_tokens=%r", requested)
        per_call_args["max_tokens"] = remaining

        return await super().get_model_response(
            state, prompt, client, model, tool_defs, per_call_args
        )

    async def get_prompt_messages(self, state: vf.State) -> vf.Messages:
        if len(state["trajectory"]) == 0:
            return state["prompt"]

        prev_turn_prompt = state["trajectory"][-1]["prompt"]
        prev_turn_completion = state["trajectory"][-1]["completion"]
        last_msg = prev_turn_completion[-1]

        messages = concat_messages([prev_turn_prompt, prev_turn_completion])

        has_tool_calls = hasattr(last_msg, "tool_calls") and last_msg.tool_calls
        if not has_tool_calls:
            nudge = UserMessage(
                role="user",
                content=(
                    "Use the search tool to find information, then write your final "
                    "answer inside <answer></answer> tags. Example: <answer>42</answer>"
                ),
            )
            return concat_messages([messages, [nudge]])

        env_response = await self.env_response(messages, state)
        env_response = maybe_normalize_messages(env_response, field_name="env_response")
        return concat_messages([messages, env_response])

    def _append_trace(
        self,
        state: vf.State,
        *,
        action: str,
        feedback: str,
        action_type: str,
    ) -> None:
        state["harness_trace"].append(
            {
                "turn": state["turns"],
                "action_type": action_type,
                "assistant_action": action,
                "env_feedback": feedback,
            }
        )
        state["turn_count"] = state["turns"]

    async def env_response(
        self, messages: vf.Messages, state: vf.State, **kwargs: Any
    ) -> vf.Messages:
        last_msg = cast(AssistantMessage, messages[-1])
        state["turns"] = state.get("turns", 0) + 1

        assert last_msg.tool_calls is not None
        tool_messages: list[ToolMessage] = []

        for tool_call in last_msg.tool_calls:
            tool_call_id: str = tool_call.id
            try:
                tool_name: str = tool_call.name
                tool_args: dict = json.loads(tool_call.arguments)
            except Exception as exc:
                err = f"[parse error] {exc}: {tool_call.arguments!r}"
                self._append_trace(
                    state,
                    action=f"[parse error] {tool_call.arguments}",
                    feedback=err,
                    action_type="parse_error",
                )
                tool_messages.append(
                    ToolMessage(role="tool", tool_call_id=tool_call_id, content=err)
                )
                continue

            tool_fn = next(
                (t for t in self.tools if getattr(t, "__name__", "") == tool_name),
                None,
            )
            if tool_fn is None:
                err = f"[unknown tool] {tool_name}"
                self._append_trace(
                    state,
                    action=f"{tool_name}({tool_args})",
                    feedback=err,
                    action_type="unknown_tool",
                )
                tool_messages.append(
                    ToolMessage(role="tool", tool_call_id=tool_call_id, content=err)
                )
                continue

            try:
                result = await tool_fn(**tool_args)
            except Exception as exc:
                err = f"[tool error] {exc}"
                self._append_trace(
                    state,
                    action=f"{tool_name}({tool_args})",
                    feedback=err,
                    action_type="tool_error",
                )
                tool_messages.append(
                    ToolMessage(role="tool", tool_call_id=tool_call_id, content=err)
                )
                continue

            self._append_trace(
                state,
                action=f"{tool_name}({tool_args})",
                feedback=str(result)[:400],
                action_type=tool_name,
            )
            tool_messages.append(
                ToolMessage(role="tool", tool_call_id=tool_call_id, content=str(result))
            )

        return tool_messages


# CandidateEnv alias so that h0 (a verbatim copy of this file) is a valid
# meta-harness candidate without modification — env.py:load_environment looks
# up CandidateEnv on the loaded module. Proposer-generated harnesses overwrite
# this with `class CandidateEnv(SearchR1Env): ...` per the SKILL.md template.
CandidateEnv = SearchR1Env
