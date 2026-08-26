"""ReTool base harness: multi-turn math solving with a Python code interpreter.

This is the evolution target for the ReTool meta-harness. Candidate harnesses
subclass ``ReToolEnv`` as ``CandidateEnv`` and may override prompts, tool
behavior, stop hooks, and ``env_response``. The h0 harness is a verbatim copy of
this file, so the tool response path mirrors ``verl-recipe/retool/retool.py``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Mapping
from typing import Any, cast

import httpx
import verifiers as vf
from verifiers.types import AssistantMessage, ToolMessage, UserMessage
from verifiers.utils.message_utils import concat_messages, maybe_normalize_messages

logger = logging.getLogger(__name__)


ANSWER_FORMAT = "\nThe answer format must be: \\boxed{'The final answer goes here.'}"

SYSTEM_PROMPT: str | None = None
USER_PROMPT_TEMPLATE = "{question}" + ANSWER_FORMAT

_PYTHON_FENCE_RE = re.compile(r"```python(.*?)```", re.DOTALL)


def _sandbox_url() -> str:
    return (
        os.environ.get("RETOOL_SANDBOX_URL")
        or os.environ.get("SANDBOX_URL")
        or "http://localhost:8080/run_code"
    )


def _normalize_code(code: Any) -> str:
    """Match verl-recipe/retool's code extraction and last-expression printing."""
    if not isinstance(code, str):
        code = str(code)
    matches = _PYTHON_FENCE_RE.findall(code)
    if matches:
        code = matches[0].strip()

    lines = code.split("\n")
    for i, line in reversed(list(enumerate(lines))):
        if line == "":
            continue
        if not lines[i].startswith("print"):
            lines[i] = f"print({line})"
        break
    return "\n".join(lines)


async def code_interpreter(
    code: str,
    timeout: int = 30,
    language: str = "python",
    memory_limit_mb: int = 1024,
) -> str:
    """Execute Python through a SandboxFusion-compatible HTTP service.

    The first parameter name must stay ``code`` because the ReTool verl yaml
    schema sends tool-call arguments as ``{"code": ...}``.
    """
    payload = {
        "compile_timeout": timeout,
        "run_timeout": timeout,
        "code": _normalize_code(code),
        "stdin": None,
        "memory_limit_MB": memory_limit_mb,
        "language": language,
        "files": {},
        "fetch_files": [],
    }
    request_timeout = httpx.Timeout(timeout * 2 + 30, connect=10)
    try:
        async with httpx.AsyncClient(timeout=request_timeout) as client:
            response = await client.post(_sandbox_url(), json=payload)
            response.raise_for_status()
            api_response = response.json()
    except Exception as exc:
        logger.warning("[code_interpreter] SandboxFusion request failed: %s", exc)
        return f"SandboxFusion request failed: {exc}"

    run_result = api_response.get("run_result") or {}
    stdout = run_result.get("stdout") or ""
    stderr = run_result.get("stderr") or ""
    if run_result.get("status") == "Finished":
        return stdout + stderr
    return stdout + stderr or "no stdout here"


class ReToolEnv(vf.ToolEnv):
    """Multi-turn math environment over a SandboxFusion code interpreter."""

    SYSTEM_PROMPT: str | None = SYSTEM_PROMPT
    USER_PROMPT_TEMPLATE: str = USER_PROMPT_TEMPLATE
    TOOLS: list = [code_interpreter]
    MAX_TURNS: int = 2
    _assistant_token_budget: int | None = None

    def set_assistant_token_budget(self, max_tokens: int | None) -> None:
        if max_tokens is None:
            self._assistant_token_budget = None
            return
        max_tokens = int(max_tokens)
        self._assistant_token_budget = max_tokens if max_tokens > 0 else None

    def _get_assistant_token_budget(self) -> int | None:
        if self._assistant_token_budget is not None:
            return self._assistant_token_budget
        raw = os.environ.get("RETOOL_ASSISTANT_TOKEN_BUDGET", "").strip()
        if not raw:
            return None
        try:
            budget = int(raw)
        except ValueError:
            logger.warning("Ignoring invalid RETOOL_ASSISTANT_TOKEN_BUDGET=%r", raw)
            return None
        return budget if budget > 0 else None

    @staticmethod
    def _response_completion_tokens(response: Any) -> int:
        usage = getattr(response, "usage", None)
        if usage is None:
            return 0
        value = (
            usage.get("completion_tokens", 0)
            if isinstance(usage, Mapping)
            else getattr(usage, "completion_tokens", 0)
        )
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _message_content_text(msg: Any) -> str:
        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, Mapping):
            content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for part in content:
                if isinstance(part, Mapping):
                    text = part.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
                else:
                    text = getattr(part, "text", None)
                    if isinstance(text, str):
                        chunks.append(text)
            return "\n".join(chunks)
        return ""

    @classmethod
    def _approx_assistant_tokens_from_messages(cls, state: vf.State) -> int:
        """Fallback when the client does not expose usage or token ids."""
        total_chars = 0
        for step in state.get("trajectory") or []:
            completion = step.get("completion") if isinstance(step, Mapping) else None
            for msg in completion or []:
                role = getattr(msg, "role", None)
                if role is None and isinstance(msg, Mapping):
                    role = msg.get("role")
                if role == "assistant":
                    total_chars += len(cls._message_content_text(msg))
        # Qwen tokenization is unavailable inside the MH venv. This fallback is
        # intentionally conservative; vLLM/OpenAI usage is the authoritative path.
        return (total_chars + 2) // 3

    @classmethod
    def _assistant_tokens_used(cls, state: vf.State) -> int:
        total = 0
        for step in state.get("trajectory") or []:
            tokens = step.get("tokens") if isinstance(step, Mapping) else None
            completion_ids = tokens.get("completion_ids") if tokens else None
            if completion_ids:
                total += len(completion_ids)
        if total:
            return total

        for step in state.get("trajectory") or []:
            if not isinstance(step, Mapping):
                continue
            total += cls._response_completion_tokens(step.get("response"))
        if total:
            return total

        usage = state.get("usage") if isinstance(state, dict) else None
        if isinstance(usage, Mapping):
            try:
                used = max(0, int(float(usage.get("output_tokens", 0) or 0)))
                if used:
                    return used
            except (TypeError, ValueError):
                logger.warning("Ignoring invalid state usage output_tokens=%r", usage)

        return cls._approx_assistant_tokens_from_messages(state)

    async def setup_state(self, state: vf.State, **kwargs: Any) -> vf.State:
        state["turns"] = 0
        state["turn_count"] = 0
        state["harness_name"] = "retool"
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
                    "Use the code_interpreter tool for useful calculations or "
                    "verification, then provide the final answer in \\boxed{}."
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


CandidateEnv = ReToolEnv
