# Copyright 2026
#
# chess-puzzle-v0 multi-turn agent loop for verl GRPO/RSFT.

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _repo_root() -> Path:
    return Path(os.environ.get("PROJECT_DIR", Path(__file__).resolve().parents[3])).resolve()


def _default_harness_path() -> str:
    return str(_repo_root() / "environments" / "chess_puzzle" / "base_harness.py")


@register("chess_puzzle_agent")
class ChessPuzzleAgentLoop(AgentLoopBase):
    """Roll out one Lichess puzzle using the shared chess-puzzle runner logic."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        multi_turn = self.rollout_config.multi_turn
        max_assistant_tokens = (
            multi_turn.max_assistant_tokens
            or os.getenv("CHESS_PUZZLE_ASSISTANT_TOKEN_BUDGET")
            or os.getenv("ASSISTANT_TOKEN_BUDGET")
        )
        self.max_assistant_tokens = int(max_assistant_tokens) if max_assistant_tokens else 8129
        raw_policy_max_tokens = os.getenv("CHESS_PUZZLE_POLICY_MAX_TOKENS") or os.getenv("POLICY_MAX_TOKENS")
        self.policy_max_tokens = _as_int(raw_policy_max_tokens, self.max_assistant_tokens)
        raw_default_turns = (
            multi_turn.max_assistant_turns
            or os.getenv("CHESS_PUZZLE_DEFAULT_MAX_TURNS")
            or os.getenv("MAX_TURNS")
        )
        self.max_assistant_turns = _as_int(raw_default_turns, 9)
        raw_turn_cap = os.getenv("CHESS_PUZZLE_MAX_TURNS_CAP") or os.getenv("MAX_TURNS_CAP")
        self.max_turns_cap = _as_int(raw_turn_cap, self.max_assistant_turns * 2)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    def _remaining_assistant_tokens(self, assistant_tokens_used: int) -> int:
        return max(0, int(self.max_assistant_tokens) - int(assistant_tokens_used))

    def _policy_request_id(self) -> str:
        return uuid4().hex

    async def _encode_text(self, text: str) -> list[int]:
        return await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.encode(text, add_special_tokens=False),
        )

    async def _append_nonassistant_messages(
        self,
        *,
        prompt_ids: list[int],
        response_mask: list[int],
        response_logprobs: list[float],
        add_messages: list[dict[str, Any]],
    ) -> bool:
        response_ids = await self.apply_chat_template(add_messages, remove_system_prompt=True)
        if len(response_mask) + len(response_ids) >= self.response_length:
            return False
        prompt_ids += response_ids
        response_mask += [0] * len(response_ids)
        if response_logprobs:
            response_logprobs += [0.0] * len(response_ids)
        return True

    def _cap_sampling(
        self,
        sampling_params: dict[str, Any],
        *,
        response_mask: list[int],
        assistant_tokens_used: int,
    ) -> dict[str, Any] | None:
        remaining_total = self.response_length - len(response_mask)
        if remaining_total <= 0:
            return None
        remaining_assistant = self._remaining_assistant_tokens(assistant_tokens_used)
        if remaining_assistant <= 0:
            return None
        per_call_cap = min(remaining_total, remaining_assistant, self.policy_max_tokens)
        if str(os.getenv("CHESS_PUZZLE_RESPECT_SAMPLING_MAX_TOKENS", "false")).lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            requested = sampling_params.get("max_tokens")
            if requested is None:
                requested = sampling_params.get("max_new_tokens")
            if requested is not None:
                try:
                    per_call_cap = min(per_call_cap, int(requested))
                except (TypeError, ValueError):
                    pass
        if per_call_cap <= 0:
            return None
        capped = dict(sampling_params)
        capped["max_tokens"] = per_call_cap
        return capped

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        from autoharness_chess_puzzle.harness import load_harness
        from autoharness_chess_puzzle.runner import (
            completion_tokens,
            effective_max_turns,
            example_from_mapping,
            initialize_state,
            messages_as_dicts,
            parse_model_action,
            policy_messages_with_harness,
            verify_and_step,
        )

        extra_info = dict(kwargs.get("extra_info") or {})
        harness_path = (
            os.getenv("CHESS_PUZZLE_HARNESS_PATH")
            or os.getenv("HARNESS_PATH")
            or str(extra_info.get("harness_path") or _default_harness_path())
        )
        harness = load_harness(harness_path)
        example = example_from_mapping({"extra_info": extra_info})
        state = initialize_state(example)
        effective_turn_limit = effective_max_turns(harness, self.max_assistant_turns)

        metrics: dict[str, Any] = {
            "num_tool_calls": 0,
            "chess_puzzle_policy_calls": 0,
            "chess_puzzle_assistant_token_budget": self.max_assistant_tokens,
            "chess_puzzle_max_turns_effective": effective_turn_limit,
            "chess_puzzle_max_turns_default": self.max_assistant_turns,
            "chess_puzzle_max_turns_cap": self.max_turns_cap,
        }
        prompt_ids: list[int] = []
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        assistant_turns = 0
        user_turns = 0
        reward = 0.0
        stop_condition = "unknown"
        outcome = "loss"
        event_log: list[dict[str, Any]] = []
        token_budget_exhausted = False

        try:
            initial_messages = messages_as_dicts(policy_messages_with_harness(harness, state))
            prompt_ids = await self.apply_chat_template(initial_messages)
            while True:
                if effective_turn_limit and assistant_turns >= effective_turn_limit:
                    stop_condition = "max_assistant_turns"
                    break
                messages = messages_as_dicts(policy_messages_with_harness(harness, state, state.retry_nudge))
                state.retry_nudge = None
                if assistant_turns > 0:
                    ok = await self._append_nonassistant_messages(
                        prompt_ids=prompt_ids,
                        response_mask=response_mask,
                        response_logprobs=response_logprobs,
                        add_messages=[messages[-1]],
                    )
                    if not ok:
                        stop_condition = "response_length"
                        token_budget_exhausted = True
                        break
                capped = self._cap_sampling(
                    sampling_params,
                    response_mask=response_mask,
                    assistant_tokens_used=state.assistant_tokens_used,
                )
                if capped is None:
                    stop_condition = "assistant_token_budget"
                    token_budget_exhausted = True
                    break

                with simple_timer("generate_sequences", metrics):
                    chat_response = await self.server_manager.chat_completion(
                        request_id=self._policy_request_id(),
                        messages=messages,
                        sampling_params=capped,
                    )
                assistant_text = str(chat_response.get("content", ""))
                record = {"raw_response": assistant_text, "usage": chat_response.get("usage")}
                generated_tokens = completion_tokens(record)
                assistant_ids = await self._encode_text(assistant_text)
                if len(response_mask) + len(assistant_ids) >= self.response_length:
                    stop_condition = "response_length"
                    token_budget_exhausted = True
                    break
                prompt_ids += assistant_ids
                response_mask += [1] * len(assistant_ids)
                state.assistant_tokens_used += generated_tokens
                state.policy_calls += 1
                assistant_turns += 1
                metrics["chess_puzzle_policy_calls"] = state.policy_calls
                metrics["chess_puzzle_assistant_generated_tokens_usage"] = state.assistant_tokens_used
                metrics["chess_puzzle_assistant_generated_tokens_usage_remaining"] = self._remaining_assistant_tokens(
                    state.assistant_tokens_used
                )

                parsed = parse_model_action(harness, assistant_text)
                action_event = {
                    "actor": "assistant",
                    "raw_response": assistant_text,
                    "parsed_action": parsed,
                    "completion_tokens": generated_tokens,
                    "assistant_tokens_used": state.assistant_tokens_used,
                }
                event_log.append(action_event)
                step = verify_and_step(harness=harness, state=state, parsed_action=parsed)
                state.completion_records.extend(
                    [
                        {"role": "assistant", "content": assistant_text},
                        {"role": "tool", "content": step.tool_response},
                    ]
                )
                event_log.append(
                    {
                        "actor": "verifier",
                        "terminal": step.terminal,
                        "outcome": step.outcome,
                        "stop_condition": step.stop_condition,
                        "tool_response": step.tool_response,
                        "info": step.info,
                    }
                )
                metrics["num_tool_calls"] = metrics.get("num_tool_calls", 0) + 1
                ok = await self._append_nonassistant_messages(
                    prompt_ids=prompt_ids,
                    response_mask=response_mask,
                    response_logprobs=response_logprobs,
                    add_messages=[{"role": "user", "content": step.tool_response}],
                )
                if not ok:
                    stop_condition = "response_length"
                    token_budget_exhausted = True
                    break
                user_turns += 1

                if step.retry_nudge:
                    state.retry_nudge = step.retry_nudge
                if step.terminal:
                    reward = step.reward
                    outcome = step.outcome
                    stop_condition = step.stop_condition
                    break
        except Exception as exc:
            logger.warning("chess-puzzle rollout failed: %s", exc)
            reward = 0.0
            outcome = "error"
            stop_condition = "rollout_error"
            event_log.append({"actor": "runner", "error": str(exc)})

        if token_budget_exhausted:
            reward = 0.0
            outcome = "loss"
        solved = 1.0 if reward >= 1.0 else 0.0
        loss = 0.0 if solved else 1.0
        assistant_generated_tokens_tokenized = int(sum(response_mask))
        response_region_tokens_total = int(len(response_mask))
        tool_user_observation_tokens_tokenized = max(
            0,
            response_region_tokens_total - assistant_generated_tokens_tokenized,
        )
        prompt_prefix_tokens = len(prompt_ids) - response_region_tokens_total
        metrics.update(
            {
                "score": solved,
                "acc": solved,
                "correct_answer": solved,
                "win": solved,
                "win_rate": solved,
                "solved": solved,
                "draw": 0.0,
                "loss": loss,
                "chess_puzzle_score": solved,
                "chess_puzzle_correct_moves": len(state.accepted_solver_moves),
                "chess_puzzle_format_retries_used": state.format_retries_used,
                "chess_puzzle_illegal_retries_used": state.illegal_retries_used,
                "chess_puzzle_stop_condition": stop_condition,
                "chess_puzzle_assistant_generated_tokens_usage": state.assistant_tokens_used,
                "chess_puzzle_assistant_generated_tokens_tokenized": assistant_generated_tokens_tokenized,
                "chess_puzzle_response_region_tokens_total": response_region_tokens_total,
                "chess_puzzle_tool_user_observation_tokens_tokenized": tool_user_observation_tokens_tokenized,
                "chess_puzzle_prompt_prefix_tokens": prompt_prefix_tokens,
                "chess_puzzle_max_turns_effective": effective_turn_limit,
                "chess_puzzle_max_turns_default": self.max_assistant_turns,
                "chess_puzzle_max_turns_cap": self.max_turns_cap,
            }
        )

        full_response_ids = prompt_ids[-len(response_mask) :] if response_mask else []
        prompt_prefix_ids = prompt_ids[: len(prompt_ids) - len(response_mask)] if response_mask else prompt_ids
        reward_info = {
            "score": solved,
            "acc": solved,
            "correct_answer": solved,
            "win": solved,
            "win_rate": solved,
            "solved": solved,
            "draw": 0.0,
            "loss": loss,
            "outcome": outcome,
            "stop_condition": stop_condition,
            "assistant_tokens_used": state.assistant_tokens_used,
            "assistant_generated_tokens_usage": state.assistant_tokens_used,
            "assistant_generated_tokens_tokenized": assistant_generated_tokens_tokenized,
            "response_region_tokens_total": response_region_tokens_total,
            "tool_user_observation_tokens_tokenized": tool_user_observation_tokens_tokenized,
            "prompt_prefix_tokens": prompt_prefix_tokens,
            "policy_calls": state.policy_calls,
            "max_turns_effective": effective_turn_limit,
            "max_turns_default": self.max_assistant_turns,
            "max_turns_cap": self.max_turns_cap,
        }
        return AgentLoopOutput(
            prompt_ids=prompt_prefix_ids,
            response_ids=full_response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            num_turns=user_turns + assistant_turns + 1,
            metrics=metrics,
            reward_score=solved,
            extra_fields={
                "reward_extra_info": reward_info,
                "extras": {
                    "task": "chess-puzzle-v0",
                    "example_id": "redacted",
                    "events": event_log[-50:],
                    "answer_redacted": True,
                },
            },
        )
