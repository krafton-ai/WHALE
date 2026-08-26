# Copyright 2026
#
# TextArena-specific multi-turn agent loop for verl GRPO.

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _repo_root() -> Path:
    return Path(os.environ.get("PROJECT_DIR", Path(__file__).resolve().parents[3])).resolve()


def _default_harness_path() -> str:
    return str(_repo_root() / "environments" / "textarena" / "base_harness.py")


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    keep = max(0, max_chars - 80)
    return text[:keep] + "\n...(observation truncated)..."


def _is_chess_observation(text: str) -> bool:
    lower = str(text).lower()
    return "game of chess" in lower or ("current board:" in lower and "a b c d e f g h" in lower)


def _outcome_from_reward(reward: float) -> str:
    if reward > 0:
        return "win"
    if reward < 0:
        return "loss"
    return "draw"


def _score_from_outcome(outcome: str, *, num_players: int) -> float:
    if num_players <= 1:
        return 1.0 if outcome == "win" else 0.0
    if outcome == "win":
        return 1.0
    if outcome == "draw":
        return 0.5
    return 0.0


@register("textarena_agent")
class TextArenaAgentLoop(AgentLoopBase):
    """Roll out one TextArena game inside verl's async agent-loop path.

    Dataset rows supply game metadata through ``extra_info``. The model only
    controls ``ours_player``; other players are advanced with the configured
    opponent policy, currently ``random_legal``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        multi_turn = self.rollout_config.multi_turn
        self.max_user_turns = int(multi_turn.max_user_turns or 0)
        self.max_assistant_turns = int(multi_turn.max_assistant_turns or 0)
        max_assistant_tokens = (
            multi_turn.max_assistant_tokens
            or os.getenv("TEXTARENA_ASSISTANT_TOKEN_BUDGET")
            or os.getenv("TEXTARENA_MAX_TOTAL_RESPONSE_LENGTH")
        )
        self.max_assistant_tokens = int(max_assistant_tokens) if max_assistant_tokens else None
        policy_max_tokens = os.getenv("TEXTARENA_POLICY_MAX_TOKENS")
        self.policy_max_tokens = int(policy_max_tokens) if policy_max_tokens else None
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self.observation_max_chars = _as_int(os.getenv("TEXTARENA_OBSERVATION_MAX_CHARS"), 0)

    def _assistant_tokens_used(self, response_mask: list[int]) -> int:
        return int(sum(1 for value in response_mask if value))

    def _remaining_assistant_tokens(self, response_mask: list[int]) -> int | None:
        if not self.max_assistant_tokens or self.max_assistant_tokens <= 0:
            return None
        return max(0, self.max_assistant_tokens - self._assistant_tokens_used(response_mask))

    def _cap_sampling(self, response_mask: list[int], sampling_params: dict[str, Any]) -> dict[str, Any] | None:
        remaining_total = self.response_length - len(response_mask)
        if remaining_total <= 0:
            return None
        remaining_assistant = self._remaining_assistant_tokens(response_mask)
        if remaining_assistant is None:
            return sampling_params
        if remaining_assistant <= 0:
            return None
        capped = dict(sampling_params)
        per_call_cap = min(remaining_assistant, remaining_total)
        requested = capped.get("max_tokens")
        if requested is None:
            requested = capped.pop("max_new_tokens", None)
        if requested is not None:
            try:
                per_call_cap = min(per_call_cap, int(requested))
            except (TypeError, ValueError):
                logger.warning("Ignoring non-integer max_tokens=%r", requested)
        if self.policy_max_tokens:
            per_call_cap = min(per_call_cap, self.policy_max_tokens)
        if per_call_cap <= 0:
            return None
        capped["max_tokens"] = per_call_cap
        return capped

    def _policy_request_id(self) -> str:
        """Use one fresh routing id per policy call, matching MH HTTP calls.

        The generic async agent loop keeps a sticky outer request id for a whole
        multi-turn conversation. TextArena MH evaluates each move as an
        independent OpenAI-compatible chat completion, so keeping game-level
        stickiness changes vLLM replica routing and therefore stochastic sample
        streams. Prefix caching still keys on prompt content inside vLLM.
        """
        return uuid4().hex

    async def _decode(self, token_ids: list[int]) -> str:
        return await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(token_ids, skip_special_tokens=True),
        )

    async def _encode_assistant_text(self, text: str) -> list[int]:
        return await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.encode(text, add_special_tokens=False),
        )

    async def _append_messages(
        self,
        *,
        prompt_ids: list[int],
        response_mask: list[int],
        response_logprobs: list[float],
        messages: list[dict[str, Any]],
        add_messages: list[dict[str, Any]],
    ) -> bool:
        messages.extend(add_messages)
        response_ids = await self.apply_chat_template(add_messages, remove_system_prompt=True)
        if len(response_mask) + len(response_ids) >= self.response_length:
            return False
        prompt_ids += response_ids
        response_mask += [0] * len(response_ids)
        if response_logprobs:
            response_logprobs += [0.0] * len(response_ids)
        return True

    def _format_policy_messages(
        self,
        *,
        harness,
        raw_observation: str,
        player_id: int,
        turn_count: int,
        remove_available_moves_for_ours: bool,
        retry_nudge: str | None = None,
    ) -> tuple[list[dict[str, Any]], str]:
        from autoharness_textarena.textarena_runner import (
            format_observation_with_harness,
            policy_messages_with_harness,
            remove_available_moves,
        )

        observation = remove_available_moves(raw_observation) if remove_available_moves_for_ours else raw_observation
        observation = format_observation_with_harness(
            harness,
            observation,
            player_id=player_id,
            turn_count=turn_count,
        )
        observation = _truncate(observation, self.observation_max_chars)
        messages, _prompt = policy_messages_with_harness(
            harness,
            player_id=player_id,
            observation=observation,
            retry_nudge=retry_nudge,
        )
        return [{"role": message.role, "content": message.content} for message in messages], observation

    def _select_model_action(
        self,
        *,
        harness,
        raw_observation: str,
        visible_observation: str,
        assistant_text: str,
    ) -> tuple[str | None, dict[str, Any]]:
        from autoharness_textarena.textarena_runner import (
            normalize_chess_action,
            parse_action_with_harness,
        )

        parsed = str(parse_action_with_harness(harness, assistant_text))
        info: dict[str, Any] = {
            "raw_response": assistant_text,
            "parsed_action": parsed,
            "source": "policy",
        }
        if not _is_chess_observation(raw_observation):
            return parsed, info

        legal_by_harness = False
        try:
            legal_by_harness = bool(harness.is_legal_action(raw_observation, parsed))
        except Exception as exc:
            info["harness_error"] = str(exc)
        info["legal_by_harness"] = legal_by_harness
        if legal_by_harness:
            action = normalize_chess_action(parsed)
            info["action"] = action
            return action, info

        info["source"] = "illegal_policy_action"
        info["action"] = parsed
        return None, info

    def _choose_random_legal(
        self,
        *,
        env,
        game_id: str,
        player_id: int,
        raw_observation: str,
        visible_observation: str,
        rng: random.Random,
        turn_count: int,
        ours_player: int,
        trace_events: list[dict[str, Any]],
    ) -> str:
        from autoharness_textarena.textarena_runner import choose_opponent_action

        return choose_opponent_action(
            opponent="random_legal",
            llm=None,
            env=env,
            player_id=player_id,
            raw_observation=raw_observation,
            observation=visible_observation,
            rng=rng,
            trace_path=None,
            trace_context={
                "game_id": game_id,
                "turn_count": turn_count,
                "ours_player": ours_player,
                "mode": "random_legal",
            },
            trace_events=trace_events,
        )

    def _finalize_reward(
        self,
        *,
        env,
        ours_player: int,
        num_players: int,
        invalid_actor: int | None,
        turn_count: int,
        stop_condition: str,
    ) -> tuple[float, dict[str, Any]]:
        from autoharness_textarena.textarena_runner import close_rewards, invalid_players_from_game_info

        rewards, game_info = close_rewards(env)
        if invalid_actor is None:
            invalid_players = invalid_players_from_game_info(game_info)
            if invalid_players:
                invalid_actor = next(iter(invalid_players))

        ours_reward = float(rewards.get(ours_player, 0.0))
        if invalid_actor == ours_player:
            ours_reward = min(ours_reward, -1.0)
        elif invalid_actor is not None and invalid_actor != ours_player:
            ours_reward = max(ours_reward, 1.0)

        outcome = _outcome_from_reward(ours_reward)
        score = _score_from_outcome(outcome, num_players=num_players)
        win = 1.0 if outcome == "win" else 0.0
        draw = 1.0 if outcome == "draw" else 0.0
        loss = 1.0 if outcome == "loss" else 0.0
        return score, {
            "score": score,
            "acc": score,
            "outcome": outcome,
            "win": win,
            "draw": draw,
            "loss": loss,
            "win_rate": win,
            "ours_reward": ours_reward,
            "invalid_actor": invalid_actor,
            "turn_count": turn_count,
            "stop_condition": stop_condition,
            "rewards": rewards,
            "game_info": game_info,
        }

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        from autoharness_textarena.harness import load_harness
        from autoharness_textarena.textarena_runner import (
            illegal_move_retry_nudge,
            make_env,
            observation_to_text,
            remove_available_moves,
            resolve_illegal_move_retry_budget,
            try_propose_action_fallback,
        )

        extra_info = dict(kwargs.get("extra_info") or {})
        game_id = str(extra_info.get("game_id") or os.getenv("TEXTARENA_GAME_ID", "TicTacToe-v0"))
        num_players = _as_int(extra_info.get("num_players"), 2)
        env_seed = _as_int(extra_info.get("env_seed", extra_info.get("seed")), 42)
        ours_player = _as_int(extra_info.get("ours_player"), 0) % max(1, num_players)
        opponent = str(extra_info.get("opponent") or os.getenv("TEXTARENA_OPPONENT", "random_legal"))
        max_game_turns = _as_int(extra_info.get("max_game_turns"), _as_int(os.getenv("TEXTARENA_MAX_GAME_TURNS"), 200))
        remove_available_moves_for_ours = _as_bool(extra_info.get("remove_available_moves_for_ours"), True)
        remove_available_moves_for_opponent = _as_bool(extra_info.get("remove_available_moves_for_opponent"), False)
        retries = _as_int(extra_info.get("retries", os.getenv("TEXTARENA_GAME_RETRIES")), 0)
        harness_path = (
            os.getenv("TEXTARENA_HARNESS_PATH")
            or os.getenv("HARNESS_PATH")
            or str(extra_info.get("harness_path") or _default_harness_path())
        )

        harness = load_harness(harness_path)
        illegal_retry_budget = resolve_illegal_move_retry_budget(harness, retries)
        rng = random.Random(_as_int(extra_info.get("opponent_seed"), env_seed + 17))
        env = make_env(game_id)
        env.reset(num_players=num_players, seed=env_seed)

        raw_messages = list(kwargs.get("raw_prompt") or [])
        messages = [dict(m) for m in raw_messages if isinstance(m, dict) and m.get("role") == "system"]
        metrics: dict[str, Any] = {
            "num_tool_calls": 0,
            "textarena_steps": 0,
            "textarena_ours_turns": 0,
            "textarena_opponent_turns": 0,
        }
        if self.max_assistant_tokens:
            metrics["assistant_token_budget"] = self.max_assistant_tokens
        prompt_ids: list[int] = []
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        assistant_turns = 0
        assistant_generation_attempts = 0
        user_turns = 0
        invalid_actor: int | None = None
        assistant_token_budget_exhausted = False
        token_budget_exhausted = False
        done = False
        last_step_info: dict[str, Any] = {}
        event_log: list[dict[str, Any]] = []

        def current_observation() -> tuple[int, str]:
            player_id, obs = env.get_observation()
            return int(player_id), observation_to_text(obs)

        def step_env(action: str, player_id: int, actor: str) -> None:
            nonlocal done, invalid_actor, last_step_info
            step_done, info = env.step(action=action)
            done = bool(step_done)
            last_step_info = info if isinstance(info, dict) else {}
            metrics["textarena_steps"] = metrics.get("textarena_steps", 0) + 1
            event_log.append(
                {
                    "actor": actor,
                    "player_id": player_id,
                    "action": action,
                    "done": done,
                    "info": last_step_info,
                }
            )
            text = json.dumps(last_step_info, ensure_ascii=False).lower()
            if any(key in text for key in ("invalid", "illegal", "not valid", "error")):
                invalid_actor = player_id
                done = True

        def advance_opponents() -> str:
            nonlocal done, invalid_actor
            summaries: list[str] = []
            while not done:
                player_id, raw_obs = current_observation()
                if player_id == ours_player:
                    break
                if opponent != "random_legal":
                    raise ValueError(f"Unsupported TextArena opponent: {opponent}")
                visible = remove_available_moves(raw_obs) if remove_available_moves_for_opponent else raw_obs
                action = self._choose_random_legal(
                    env=env,
                    game_id=game_id,
                    player_id=player_id,
                    raw_observation=raw_obs,
                    visible_observation=visible,
                    rng=rng,
                    turn_count=int(metrics["textarena_steps"]),
                    ours_player=ours_player,
                    trace_events=event_log,
                )
                step_env(action, player_id, "opponent")
                metrics["textarena_opponent_turns"] = metrics.get("textarena_opponent_turns", 0) + 1
                summaries.append(f"Player {player_id} played {action}.")
            return "\n".join(summaries)

        try:
            advance_opponents()
            if done:
                score, reward_info = self._finalize_reward(
                    env=env,
                    ours_player=ours_player,
                    num_players=num_players,
                    invalid_actor=invalid_actor,
                    turn_count=int(metrics["textarena_steps"]),
                    stop_condition="terminal_before_ours",
                )
                return AgentLoopOutput(
                    prompt_ids=[],
                    response_ids=[],
                    response_mask=[],
                    response_logprobs=None,
                    num_turns=1,
                    metrics=metrics,
                    reward_score=score,
                    extra_fields={"reward_extra_info": reward_info, "extras": {"events": event_log[-20:]}},
                )

            player_id, raw_observation = current_observation()
            turn_messages, visible_observation = self._format_policy_messages(
                harness=harness,
                raw_observation=raw_observation,
                player_id=player_id,
                turn_count=int(metrics["textarena_steps"]),
                remove_available_moves_for_ours=remove_available_moves_for_ours,
            )
            messages = list(turn_messages)
            prompt_ids = await self.apply_chat_template(messages)

            while not done:
                if self.max_assistant_turns and assistant_turns >= self.max_assistant_turns:
                    break
                if max_game_turns and int(metrics["textarena_steps"]) >= max_game_turns:
                    break

                retry_nudge: str | None = None
                retry_used = 0
                action: str | None = None

                while True:
                    if retry_nudge:
                        retry_messages, visible_observation = self._format_policy_messages(
                            harness=harness,
                            raw_observation=raw_observation,
                            player_id=player_id,
                            turn_count=int(metrics["textarena_steps"]),
                            remove_available_moves_for_ours=remove_available_moves_for_ours,
                            retry_nudge=retry_nudge,
                        )
                        ok = await self._append_messages(
                            prompt_ids=prompt_ids,
                            response_mask=response_mask,
                            response_logprobs=response_logprobs,
                            messages=messages,
                            add_messages=[{"role": "user", "content": retry_nudge}],
                        )
                        if not ok:
                            token_budget_exhausted = True
                            break

                    turn_sampling_params = self._cap_sampling(response_mask, sampling_params)
                    if turn_sampling_params is None:
                        if self._remaining_assistant_tokens(response_mask) == 0:
                            assistant_token_budget_exhausted = True
                        else:
                            token_budget_exhausted = True
                        break

                    active_messages = retry_messages if retry_nudge else turn_messages
                    with simple_timer("generate_sequences", metrics):
                        chat_response = await self.server_manager.chat_completion(
                            request_id=self._policy_request_id(),
                            messages=active_messages,
                            sampling_params=dict(turn_sampling_params),
                        )

                    usage = chat_response.get("usage")
                    if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
                        metrics["chat_completion_tokens_last"] = int(usage["completion_tokens"])
                    metrics["num_preempted"] = metrics.get("num_preempted", 0)

                    assistant_generation_attempts += 1
                    assistant_text = str(chat_response.get("content", ""))
                    current_response_ids = await self._encode_assistant_text(assistant_text)
                    prompt_ids += current_response_ids
                    response_mask += [1] * len(current_response_ids)
                    metrics["assistant_tokens"] = self._assistant_tokens_used(response_mask)
                    if self.max_assistant_tokens:
                        metrics["assistant_token_budget"] = self.max_assistant_tokens
                        metrics["assistant_tokens_remaining"] = self._remaining_assistant_tokens(response_mask)

                    action, action_info = self._select_model_action(
                        harness=harness,
                        raw_observation=raw_observation,
                        visible_observation=visible_observation,
                        assistant_text=assistant_text,
                    )
                    action_info["retry_index"] = retry_used
                    action_info["illegal_move_retry_budget"] = illegal_retry_budget
                    event_log.append({"actor": "ours", "player_id": player_id, **action_info})
                    if action is not None:
                        break

                    if retry_used < illegal_retry_budget:
                        retry_used += 1
                        retry_nudge = illegal_move_retry_nudge(str(action_info.get("parsed_action", "")))
                        event_log.append(
                            {
                                "actor": "ours",
                                "player_id": player_id,
                                "source": "illegal_move_retry_nudge",
                                "parsed_action": action_info.get("parsed_action", ""),
                                "retry_nudge": retry_nudge,
                                "illegal_move_retry_used": retry_used,
                                "illegal_move_retry_budget": illegal_retry_budget,
                            }
                        )
                        continue

                    fallback_action, fallback_info = try_propose_action_fallback(
                        harness,
                        observation=visible_observation,
                        verification_observation=raw_observation,
                    )
                    event_log.append(
                        {
                            "actor": "ours",
                            "player_id": player_id,
                            "policy_parsed_action": action_info.get("parsed_action", ""),
                            **fallback_info,
                        }
                    )
                    if fallback_action is not None:
                        action = fallback_action
                        break

                    invalid_actor = player_id
                    done = True
                    event_log.append(
                        {
                            "actor": "ours",
                            "player_id": player_id,
                            "source": "illegal_move_loss",
                            "parsed_action": action_info.get("parsed_action", ""),
                            "illegal_move_retry_used": retry_used,
                            "illegal_move_retry_budget": illegal_retry_budget,
                        }
                    )
                    break

                assistant_turns += 1
                metrics["textarena_ours_turns"] = metrics.get("textarena_ours_turns", 0) + 1
                if token_budget_exhausted or assistant_token_budget_exhausted:
                    break
                if done:
                    break
                if action is None:
                    break

                step_env(action, player_id, "ours")

                if done:
                    break

                advance_opponents()
                if done:
                    break

                player_id, raw_observation = current_observation()
                turn_messages, visible_observation = self._format_policy_messages(
                    harness=harness,
                    raw_observation=raw_observation,
                    player_id=player_id,
                    turn_count=int(metrics["textarena_steps"]),
                    remove_available_moves_for_ours=remove_available_moves_for_ours,
                )
                history_user_messages = [msg for msg in turn_messages if msg.get("role") == "user"]
                add_messages = history_user_messages[-1:] if history_user_messages else turn_messages[-1:]
                ok = await self._append_messages(
                    prompt_ids=prompt_ids,
                    response_mask=response_mask,
                    response_logprobs=response_logprobs,
                    messages=messages,
                    add_messages=add_messages,
                )
                if not ok:
                    token_budget_exhausted = True
                    break
                user_turns += 1

            if done:
                if invalid_actor == ours_player:
                    stop_condition = "invalid_ours"
                elif invalid_actor is not None:
                    stop_condition = "invalid_opponent"
                else:
                    stop_condition = "terminal"
            elif max_game_turns and int(metrics["textarena_steps"]) >= max_game_turns:
                stop_condition = "max_game_turns"
            elif self.max_assistant_turns and assistant_turns >= self.max_assistant_turns:
                stop_condition = "max_assistant_turns"
            elif (
                self._remaining_assistant_tokens(response_mask) is not None
                and self._remaining_assistant_tokens(response_mask) <= 0
            ):
                stop_condition = "assistant_token_budget"
                assistant_token_budget_exhausted = True
            elif len(response_mask) >= self.response_length:
                stop_condition = "response_length"
            elif token_budget_exhausted:
                stop_condition = "token_budget"
            else:
                stop_condition = "token_budget"
            if assistant_token_budget_exhausted and invalid_actor is None:
                invalid_actor = ours_player

            score, reward_info = self._finalize_reward(
                env=env,
                ours_player=ours_player,
                num_players=num_players,
                invalid_actor=invalid_actor,
                turn_count=int(metrics["textarena_steps"]),
                stop_condition=stop_condition,
            )
        except Exception as exc:
            logger.warning("TextArena rollout failed: %s", exc)
            score = 0.0
            reward_info = {
                "score": 0.0,
                "acc": 0.0,
                "outcome": "error",
                "win": 0.0,
                "draw": 0.0,
                "loss": 1.0,
                "win_rate": 0.0,
                "ours_reward": -1.0,
                "error": str(exc),
                "turn_count": int(metrics.get("textarena_steps", 0)),
                "stop_condition": "rollout_error",
            }
            try:
                env.close()
            except Exception:
                pass

        full_response_ids = prompt_ids[-len(response_mask) :] if response_mask else []
        prompt_prefix_ids = prompt_ids[: len(prompt_ids) - len(response_mask)] if response_mask else prompt_ids
        metrics["textarena_score"] = score
        metrics["textarena_invalid_ours"] = 1 if reward_info.get("stop_condition") == "invalid_ours" else 0
        metrics["textarena_assistant_generation_attempts"] = assistant_generation_attempts

        return AgentLoopOutput(
            prompt_ids=prompt_prefix_ids,
            response_ids=full_response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            num_turns=user_turns + assistant_turns + 1,
            metrics=metrics,
            reward_score=score,
            extra_fields={
                "reward_extra_info": reward_info,
                "extras": {
                    "game_id": game_id,
                    "env_seed": env_seed,
                    "ours_player": ours_player,
                    "events": event_log[-20:],
                    "last_step_info": last_step_info,
                },
            },
        )
