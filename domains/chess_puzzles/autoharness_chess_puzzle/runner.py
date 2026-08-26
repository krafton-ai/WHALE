from __future__ import annotations

import json
import os
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

import chess

from autoharness_textarena.config import LLMConfig
from autoharness_textarena.llm import ChatMessage, LLMClient

from .harness import HarnessModule, load_harness

UCI_RE = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)
TRACE_WRITE_LOCK = threading.Lock()
MAX_RETRY_BUDGET = 10
DEFAULT_MAX_TURNS = 9


@dataclass
class PuzzleExample:
    example_id: str
    initial_fen: str
    trigger_move: str
    start_fen: str
    solution_moves: list[str]
    rating: int | None = None
    themes: list[str] = field(default_factory=list)
    source: str = "Lichess/chess-puzzles"


@dataclass
class PuzzleState:
    example: PuzzleExample
    board: chess.Board
    solution_index: int = 0
    last_opponent_move: str | None = None
    accepted_solver_moves: list[str] = field(default_factory=list)
    retry_nudge: str | None = None
    format_retries_used: int = 0
    illegal_retries_used: int = 0
    policy_calls: int = 0
    assistant_tokens_used: int = 0
    completion_records: list[dict[str, Any]] = field(default_factory=list)
    event_log: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class StepResult:
    terminal: bool
    reward: float
    outcome: str
    tool_response: str
    retry_nudge: str | None = None
    stop_condition: str = ""
    info: dict[str, Any] = field(default_factory=dict)


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


def _safe_json_loads(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def normalize_uci(action: str) -> str:
    text = str(action or "").strip().lower()
    match = UCI_RE.search(text)
    return match.group(1).lower() if match else text


def _completion_tokens_from_usage(usage: dict[str, Any] | None) -> int | None:
    if not isinstance(usage, dict):
        return None
    for key in ("completion_tokens", "output_tokens", "generated_tokens", "candidatesTokenCount"):
        value = usage.get(key)
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            pass
    usage_metadata = usage.get("usage_metadata")
    if isinstance(usage_metadata, dict):
        return _completion_tokens_from_usage(usage_metadata)
    return None


def approx_completion_tokens(text: str) -> int:
    text = str(text or "")
    if not text:
        return 0
    return max(1, (len(text) + 2) // 3)


def completion_tokens(record: dict[str, Any]) -> int:
    tokens = _completion_tokens_from_usage(record.get("usage"))
    if tokens is not None:
        return tokens
    return approx_completion_tokens(str(record.get("raw_response", "")))


def assistant_token_budget_from_env(default: int = 8129) -> int:
    for key in ("CHESS_PUZZLE_ASSISTANT_TOKEN_BUDGET", "ASSISTANT_TOKEN_BUDGET"):
        raw = os.getenv(key)
        if raw:
            try:
                value = int(raw)
            except ValueError:
                continue
            if value > 0:
                return value
    return default


def policy_max_tokens_from_env(default: int | None = None) -> int | None:
    for key in ("CHESS_PUZZLE_POLICY_MAX_TOKENS", "POLICY_MAX_TOKENS"):
        raw = os.getenv(key)
        if raw:
            try:
                value = int(raw)
            except ValueError:
                continue
            if value > 0:
                return value
    return default


def retry_budgets(harness: HarnessModule) -> tuple[int, int]:
    format_default = _as_int(os.getenv("CHESS_PUZZLE_FORMAT_RETRIES"), 1)
    illegal_default = _as_int(os.getenv("CHESS_PUZZLE_ILLEGAL_RETRIES"), 1)
    format_budget = harness.format_retry_budget if harness.format_retry_budget is not None else format_default
    illegal_budget = (
        harness.illegal_move_retry_budget
        if harness.illegal_move_retry_budget is not None
        else illegal_default
    )
    return min(MAX_RETRY_BUDGET, max(0, int(format_budget))), min(
        MAX_RETRY_BUDGET, max(0, int(illegal_budget))
    )


def default_max_turns_from_env(default: int = DEFAULT_MAX_TURNS) -> int:
    for key in ("CHESS_PUZZLE_DEFAULT_MAX_TURNS", "DEFAULT_MAX_TURNS", "MAX_TURNS"):
        raw = os.getenv(key)
        if raw:
            value = _as_int(raw, default)
            if value > 0:
                return value
    return default


def max_turns_cap_from_env(default_max_turns: int | None = None) -> int:
    default_turns = int(default_max_turns or default_max_turns_from_env())
    raw = os.getenv("CHESS_PUZZLE_MAX_TURNS_CAP") or os.getenv("MAX_TURNS_CAP")
    cap = _as_int(raw, default_turns * 2) if raw else default_turns * 2
    return max(default_turns, cap)


def effective_max_turns(harness: HarnessModule, default_max_turns: int | None = None) -> int:
    default_turns = int(default_max_turns or default_max_turns_from_env())
    cap = max_turns_cap_from_env(default_turns)
    requested = harness.max_turns if harness.max_turns is not None else default_turns
    return min(cap, max(1, int(requested)))


def _piece_symbol(piece: chess.Piece | None) -> str:
    return piece.symbol() if piece is not None else "."


def board_ascii(board: chess.Board) -> str:
    rows = []
    for rank in range(7, -1, -1):
        cells = [_piece_symbol(board.piece_at(chess.square(file, rank))) for file in range(8)]
        rows.append(f"{rank + 1} " + " ".join(cells))
    rows.append("  a b c d e f g h")
    return "\n".join(rows)


def legal_move_rows(board: chess.Board) -> list[dict[str, str]]:
    rows = []
    for move in board.legal_moves:
        san = board.san(move)
        rows.append({"uci": move.uci(), "san": san})
    return sorted(rows, key=lambda item: item["uci"])


def build_observation(state: PuzzleState) -> str:
    board = state.board
    side = "White" if board.turn == chess.WHITE else "Black"
    rows = [
        "Task: solve the chess puzzle one move at a time.",
        f"Side to move: {side}",
        f"FEN: {board.fen()}",
        "Board:",
        board_ascii(board),
    ]
    if state.last_opponent_move:
        rows.append(f"Last opponent reply: {state.last_opponent_move}")
    if state.accepted_solver_moves:
        rows.append("Your accepted moves so far: " + ", ".join(state.accepted_solver_moves))
    rows.extend(
        [
            f"Puzzle move number: {len(state.accepted_solver_moves) + 1}",
            "Legal moves:",
        ]
    )
    for row in legal_move_rows(board):
        rows.append(f"- [{row['uci']}] SAN={row['san']}")
    return "\n".join(rows)


def format_observation_with_harness(harness: HarnessModule, state: PuzzleState) -> str:
    observation = build_observation(state)
    if harness.format_observation is None:
        return observation
    try:
        return str(
            harness.format_observation(
                observation,
                turn_count=len(state.accepted_solver_moves),
                side_to_move="white" if state.board.turn == chess.WHITE else "black",
                last_opponent_move=state.last_opponent_move,
                max_turns=effective_max_turns(harness),
            )
        )
    except Exception:
        return observation


def policy_messages_with_harness(
    harness: HarnessModule,
    state: PuzzleState,
    retry_nudge: str | None = None,
) -> list[ChatMessage]:
    observation = format_observation_with_harness(harness, state)
    if retry_nudge:
        observation = observation.rstrip() + "\n\nFeedback from verifier:\n" + retry_nudge.strip()
    system = harness.system_prompt or (
        "You solve chess tactics by choosing one legal move at a time. "
        "Return exactly one UCI move in <move></move> tags."
    )
    user_template = harness.user_prompt or (
        "Solve the current chess puzzle position.\n\n{observation}\n\n"
        "Return exactly one move as <move>uci</move>."
    )
    try:
        user = user_template.format(observation=observation)
    except Exception:
        user = user_template + "\n\n" + observation
    messages = [ChatMessage(role="system", content=system)]
    for record in state.completion_records:
        role = str(record.get("role", "user"))
        content = str(record.get("content", ""))
        if not content:
            continue
        if role == "assistant":
            messages.append(ChatMessage(role="assistant", content=content))
        else:
            messages.append(ChatMessage(role="user", content=content))
    messages.append(ChatMessage(role="user", content=user))
    return messages


def messages_as_dicts(messages: list[ChatMessage]) -> list[dict[str, str]]:
    return [{"role": message.role, "content": message.content} for message in messages]


def parse_model_action(harness: HarnessModule, response: str) -> str:
    if harness.parse_action is not None:
        try:
            return normalize_uci(str(harness.parse_action(response)))
        except Exception:
            pass
    text = str(response or "")
    patterns = [
        r"<move>\s*([a-h][1-8][a-h][1-8][qrbn]?)\s*</move>",
        r"\[\s*([a-h][1-8][a-h][1-8][qrbn]?)\s*\]",
        r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b",
    ]
    for pattern in patterns:
        moves: list[str] = []
        for match in re.finditer(pattern, text, re.IGNORECASE):
            move = match.group(1).lower()
            if move not in moves:
                moves.append(move)
        if len(moves) == 1:
            return moves[0]
        if len(moves) > 1:
            return "__ambiguous_multiple_moves__"
    return "__no_move__"


def _format_retry_nudge(parsed: str) -> str:
    return (
        f'The verifier could not parse exactly one UCI move from "{parsed}". '
        "Reply with exactly one move, e.g. <move>e2e4</move>, and no explanation."
    )


def _illegal_retry_nudge(parsed: str) -> str:
    return (
        f'The parsed move "{parsed}" is not legal in the current board. '
        "Choose one move from the visible legal moves and reply only as <move>uci</move>."
    )


def _public_example_id(example: PuzzleExample) -> str:
    return "redacted"


def initialize_state(example: PuzzleExample) -> PuzzleState:
    if example.start_fen:
        board = chess.Board(example.start_fen)
    else:
        board = chess.Board(example.initial_fen)
        trigger = chess.Move.from_uci(example.trigger_move)
        if trigger not in board.legal_moves:
            raise ValueError(f"trigger move is illegal for {example.example_id}: {example.trigger_move}")
        board.push(trigger)
    return PuzzleState(example=example, board=board)


def verify_and_step(
    *,
    harness: HarnessModule,
    state: PuzzleState,
    parsed_action: str,
) -> StepResult:
    format_budget, illegal_budget = retry_budgets(harness)
    parsed = normalize_uci(parsed_action)
    if not re.fullmatch(r"[a-h][1-8][a-h][1-8][qrbn]?", parsed):
        if state.format_retries_used < format_budget:
            state.format_retries_used += 1
            nudge = _format_retry_nudge(parsed)
            return StepResult(
                terminal=False,
                reward=0.0,
                outcome="retry_format",
                tool_response=nudge,
                retry_nudge=nudge,
                stop_condition="format_retry",
                info={"parsed_action": parsed, "format_retries_used": state.format_retries_used},
            )
        return StepResult(
            terminal=True,
            reward=0.0,
            outcome="loss",
            tool_response="Puzzle failed: the response did not contain exactly one parseable UCI move.",
            stop_condition="malformed",
            info={"parsed_action": parsed},
        )

    try:
        move = chess.Move.from_uci(parsed)
    except chess.InvalidMoveError:
        if state.format_retries_used < format_budget:
            state.format_retries_used += 1
            nudge = _format_retry_nudge(parsed)
            return StepResult(
                terminal=False,
                reward=0.0,
                outcome="retry_format",
                tool_response=nudge,
                retry_nudge=nudge,
                stop_condition="format_retry",
                info={"parsed_action": parsed, "format_retries_used": state.format_retries_used},
            )
        return StepResult(
            terminal=True,
            reward=0.0,
            outcome="loss",
            tool_response="Puzzle failed: the response did not contain exactly one valid UCI move.",
            stop_condition="malformed",
            info={"parsed_action": parsed},
        )
    legal = move in state.board.legal_moves
    legal_by_harness = False
    observation = build_observation(state)
    try:
        legal_by_harness = bool(harness.is_legal_action(observation, parsed))
    except Exception as exc:
        state.event_log.append({"source": "harness_is_legal_action_error", "error": str(exc)})
    if not legal or not legal_by_harness:
        if state.illegal_retries_used < illegal_budget:
            state.illegal_retries_used += 1
            nudge = _illegal_retry_nudge(parsed)
            return StepResult(
                terminal=False,
                reward=0.0,
                outcome="retry_illegal",
                tool_response=nudge,
                retry_nudge=nudge,
                stop_condition="illegal_retry",
                info={
                    "parsed_action": parsed,
                    "legal_by_board": legal,
                    "legal_by_harness": legal_by_harness,
                    "illegal_retries_used": state.illegal_retries_used,
                },
            )
        return StepResult(
            terminal=True,
            reward=0.0,
            outcome="loss",
            tool_response="Puzzle failed: the move was illegal in the current board.",
            stop_condition="illegal",
            info={"parsed_action": parsed, "legal_by_board": legal, "legal_by_harness": legal_by_harness},
        )

    expected = state.example.solution_moves[state.solution_index].lower()
    if parsed != expected:
        return StepResult(
            terminal=True,
            reward=0.0,
            outcome="loss",
            tool_response="Puzzle failed: the legal move did not match the puzzle continuation.",
            stop_condition="wrong_move",
            info={"parsed_action": parsed, "legal_by_board": legal, "legal_by_harness": legal_by_harness},
        )

    state.board.push(move)
    state.accepted_solver_moves.append(parsed)
    next_index = state.solution_index + 1
    if next_index >= len(state.example.solution_moves):
        state.solution_index = next_index
        return StepResult(
            terminal=True,
            reward=1.0,
            outcome="solved",
            tool_response="Correct. Puzzle solved.",
            stop_condition="solved",
            info={"parsed_action": parsed, "correct_moves": len(state.accepted_solver_moves)},
        )

    opponent_uci = state.example.solution_moves[next_index].lower()
    opponent_move = chess.Move.from_uci(opponent_uci)
    if opponent_move not in state.board.legal_moves:
        return StepResult(
            terminal=True,
            reward=0.0,
            outcome="error",
            tool_response="Verifier error: stored opponent reply is illegal after the accepted move.",
            stop_condition="invalid_solution_line",
            info={"parsed_action": parsed, "opponent_reply": opponent_uci},
        )
    san = state.board.san(opponent_move)
    state.board.push(opponent_move)
    state.last_opponent_move = opponent_uci
    state.solution_index = next_index + 1
    if state.solution_index >= len(state.example.solution_moves):
        return StepResult(
            terminal=True,
            reward=1.0,
            outcome="solved",
            tool_response=f"Correct. Opponent replied {opponent_uci} ({san}). Puzzle solved.",
            stop_condition="solved",
            info={
                "parsed_action": parsed,
                "opponent_reply": opponent_uci,
                "correct_moves": len(state.accepted_solver_moves),
            },
        )
    return StepResult(
        terminal=False,
        reward=0.0,
        outcome="correct_continue",
        tool_response=f"Correct. Opponent replied {opponent_uci} ({san}). Continue with the next move.",
        retry_nudge=None,
        stop_condition="continue",
        info={
            "parsed_action": parsed,
            "opponent_reply": opponent_uci,
            "correct_moves": len(state.accepted_solver_moves),
        },
    )


def _parse_moves(value: Any) -> list[str]:
    value = _safe_json_loads(value)
    if value is None:
        return []
    if not isinstance(value, str):
        try:
            return [str(move).lower() for move in value if str(move).strip()]
        except TypeError:
            pass
    if isinstance(value, list):
        return [str(move).lower() for move in value if str(move).strip()]
    return [move.lower() for move in re.findall(r"[a-h][1-8][a-h][1-8][qrbn]?", str(value), re.I)]


def _first_present(*values: Any) -> Any:
    for value in values:
        value = _safe_json_loads(value)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        try:
            if not isinstance(value, str) and len(value) == 0:
                continue
        except TypeError:
            pass
        return value
    return None


def _as_list(value: Any) -> list[Any]:
    value = _safe_json_loads(value)
    if value is None:
        return []
    if isinstance(value, str):
        return [item for item in value.split() if item]
    try:
        return [item for item in value if str(item).strip()]
    except TypeError:
        return [value]


def example_from_mapping(row: dict[str, Any]) -> PuzzleExample:
    extra = _safe_json_loads(row.get("extra_info") or {}) or {}
    if not isinstance(extra, dict):
        extra = {}
    raw_moves = _parse_moves(_first_present(extra.get("raw_moves"), row.get("Moves"), row.get("moves"), row.get("answer")))
    solution_moves = _parse_moves(_first_present(extra.get("solution_moves"), row.get("solution_moves")))
    trigger = str(_first_present(extra.get("trigger_move"), row.get("trigger_move"), "") or "").lower()
    if not solution_moves and raw_moves:
        if not trigger:
            trigger = raw_moves[0]
        solution_moves = raw_moves[1:]
    start_fen = str(_first_present(extra.get("start_fen"), row.get("start_fen"), "") or "")
    initial_fen = str(_first_present(extra.get("initial_fen"), row.get("FEN"), row.get("fen"), row.get("prompt"), "") or "")
    if not start_fen and initial_fen and trigger:
        board = chess.Board(initial_fen)
        board.push(chess.Move.from_uci(trigger))
        start_fen = board.fen()
    if not solution_moves:
        raise ValueError("missing solution_moves")
    if len(solution_moves) < 1:
        raise ValueError("solution line is empty")
    return PuzzleExample(
        example_id=str(extra.get("puzzle_id") or row.get("PuzzleId") or row.get("id") or row.get("index") or ""),
        initial_fen=initial_fen,
        trigger_move=trigger,
        start_fen=start_fen,
        solution_moves=solution_moves,
        rating=_as_int(extra.get("rating") or row.get("Rating"), 0) or None,
        themes=[str(item) for item in _as_list(_first_present(extra.get("themes"), row.get("Themes")))],
    )


def _record_event(state: PuzzleState, event: dict[str, Any]) -> None:
    state.event_log.append(event)


def _call_policy(
    *,
    llm: LLMClient,
    messages: list[ChatMessage],
    max_tokens: int,
) -> dict[str, Any]:
    response = llm.complete_response(messages, max_tokens=max_tokens)
    return {"raw_response": response.content, "usage": dict(response.usage or {})}


def run_puzzle_rollout(
    *,
    harness: HarnessModule,
    example: PuzzleExample,
    llm: LLMClient,
    assistant_token_budget: int = 8129,
    policy_max_tokens: int | None = None,
    max_policy_calls: int | None = None,
) -> dict[str, Any]:
    if policy_max_tokens is None and hasattr(llm, "config") and hasattr(llm.config, "max_tokens"):
        llm.config.max_tokens = max(int(getattr(llm.config, "max_tokens", 1)), int(assistant_token_budget))
    state = initialize_state(example)
    default_turns = default_max_turns_from_env()
    turns_cap = max_turns_cap_from_env(default_turns)
    harness_max_turns = effective_max_turns(harness, default_turns)
    max_policy_calls = max_policy_calls or harness_max_turns
    reward = 0.0
    outcome = "loss"
    stop_condition = "unknown"
    while state.policy_calls < max_policy_calls:
        remaining = assistant_token_budget - state.assistant_tokens_used
        if remaining <= 0:
            outcome = "loss"
            stop_condition = "assistant_token_budget"
            break
        max_tokens = max(1, remaining if policy_max_tokens is None else min(policy_max_tokens, remaining))
        messages = policy_messages_with_harness(harness, state, state.retry_nudge)
        state.retry_nudge = None
        record = _call_policy(llm=llm, messages=messages, max_tokens=max_tokens)
        state.policy_calls += 1
        assistant_text = str(record.get("raw_response", ""))
        tokens = completion_tokens(record)
        state.assistant_tokens_used += tokens
        parsed = parse_model_action(harness, assistant_text)
        _record_event(
            state,
            {
                "actor": "assistant",
                "policy_call": state.policy_calls,
                "raw_response": assistant_text,
                "parsed_action": parsed,
                "completion_tokens": tokens,
                "assistant_tokens_used": state.assistant_tokens_used,
            },
        )
        step = verify_and_step(harness=harness, state=state, parsed_action=parsed)
        _record_event(
            state,
            {
                "actor": "verifier",
                "policy_call": state.policy_calls,
                "terminal": step.terminal,
                "outcome": step.outcome,
                "stop_condition": step.stop_condition,
                "tool_response": step.tool_response,
                "info": step.info,
            },
        )
        state.completion_records.extend(
            [
                {"role": "assistant", "content": assistant_text},
                {"role": "tool", "content": step.tool_response},
            ]
        )
        if step.retry_nudge:
            state.retry_nudge = step.retry_nudge
        if step.terminal:
            reward = step.reward
            outcome = step.outcome
            stop_condition = step.stop_condition
            break
    else:
        outcome = "loss"
        stop_condition = "max_policy_calls"

    solved = 1.0 if reward >= 1.0 else 0.0
    loss = 0.0 if solved else 1.0
    metrics = {
        "score": solved,
        "acc": solved,
        "correct_answer": solved,
        "solved": solved,
        "win": solved,
        "win_rate": solved,
        "draw": 0.0,
        "loss": loss,
        "legal_rate": 1.0 if not any(e.get("stop_condition") == "illegal" for e in state.event_log) else 0.0,
        "correct_moves": len(state.accepted_solver_moves),
        "policy_calls": state.policy_calls,
        "assistant_tokens_used": state.assistant_tokens_used,
        "assistant_generated_tokens_usage": state.assistant_tokens_used,
        "max_turns_effective": harness_max_turns,
        "max_turns_default": default_turns,
        "max_turns_cap": turns_cap,
        "format_retries_used": state.format_retries_used,
        "illegal_retries_used": state.illegal_retries_used,
        "turn_count": state.policy_calls,
        "outcome": outcome,
        "stop_condition": stop_condition,
    }
    prompt = messages_as_dicts(policy_messages_with_harness(harness, initialize_state(example)))
    return {
        "example_id": _public_example_id(example),
        "reward": solved,
        "metrics": metrics,
        "prompt": prompt,
        "completion": state.completion_records,
        "harness_trace": state.event_log,
        "turn_count": state.policy_calls,
        "stop_condition": stop_condition,
        "answer": "",
        "answer_redacted": True,
    }


def _make_llm(config: dict[str, Any] | None) -> LLMClient | None:
    if not config:
        return None
    if config.get("provider", "none") in {"", "none", None}:
        return None
    allowed = set(LLMConfig.__dataclass_fields__)
    return LLMClient(LLMConfig(**{key: value for key, value in config.items() if key in allowed}))


def _read_examples(dataset_path: str | Path, *, limit: int | None, seed: int) -> list[PuzzleExample]:
    import pandas as pd

    frame = pd.read_parquet(dataset_path)
    rows = frame.to_dict("records")
    rng = random.Random(seed)
    rng.shuffle(rows)
    if limit is not None and limit > 0:
        rows = rows[:limit]
    return [example_from_mapping(row) for row in rows]


def _clean_message(msg: Any) -> dict[str, Any]:
    if isinstance(msg, dict):
        return {"role": msg.get("role", ""), "content": msg.get("content", "")}
    return {"role": getattr(msg, "role", ""), "content": getattr(msg, "content", "")}


def write_trajectories(outputs: list[dict[str, Any]], output_dir: Path) -> None:
    lines = []
    for output in outputs:
        question = ""
        for msg in reversed(list(output.get("prompt") or [])):
            role = msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "role", "")
            if role == "user":
                question = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
                break
        reward = float(output.get("reward", 0.0))
        lines.append(
            json.dumps(
                {
                    "example_id": output.get("example_id"),
                    "reward": reward,
                    "correct": reward >= 1.0,
                    "stop_condition": output.get("stop_condition"),
                    "turn_count": output.get("turn_count"),
                    "question": question,
                    "answer": "",
                    "answer_redacted": True,
                    "harness_trace": output.get("harness_trace", []),
                    "completion": [_clean_message(msg) for msg in output.get("completion", [])],
                },
                ensure_ascii=False,
            )
        )
    (output_dir / "trajectories.jsonl").write_text("\n".join(lines) + ("\n" if lines else ""))


def summarize_outputs(outputs: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    rewards = [float(output.get("reward", 0.0)) for output in outputs]
    turn_counts = [int(output.get("turn_count", 0)) for output in outputs]
    assistant_tokens = [
        float((output.get("metrics") or {}).get("assistant_tokens_used", 0.0))
        for output in outputs
    ]
    solved = sum(1 for value in rewards if value >= 1.0)
    avg_metrics: dict[str, float] = {}
    metric_keys = set()
    for output in outputs:
        metric_keys.update((output.get("metrics") or {}).keys())
    for key in sorted(metric_keys):
        values = []
        for output in outputs:
            value = (output.get("metrics") or {}).get(key)
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                pass
        if values:
            avg_metrics[key] = mean(values)
    return {
        "num_examples": len(outputs),
        "avg_reward": mean(rewards) if rewards else 0.0,
        "avg_metrics": avg_metrics,
        "success_rate": mean(rewards) if rewards else 0.0,
        "mean_turn_count": mean(turn_counts) if turn_counts else 0.0,
        "mean_assistant_tokens": mean(assistant_tokens) if assistant_tokens else 0.0,
        "solved_examples": solved,
        "failed_examples": len(outputs) - solved,
        "metadata": metadata,
    }


def evaluate_harness(
    *,
    harness_path: str | Path,
    dataset_path: str | Path,
    llm_config: dict[str, Any],
    output_dir: str | Path,
    limit: int = 256,
    seed: int = 42,
    assistant_token_budget: int | None = None,
    policy_max_tokens: int | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    harness = load_harness(harness_path)
    budget = int(assistant_token_budget or assistant_token_budget_from_env())
    llm_config = dict(llm_config)
    llm_config["max_tokens"] = max(int(llm_config.get("max_tokens") or 0), budget)
    llm = _make_llm(llm_config)
    if llm is None:
        raise ValueError("chess puzzle evaluation requires an llm_config")
    per_call = policy_max_tokens if policy_max_tokens is not None else policy_max_tokens_from_env()
    examples = _read_examples(dataset_path, limit=limit, seed=seed)
    max_workers = max(1, min(int(llm_config.get("batch_concurrency") or 1), len(examples)))
    if max_workers == 1:
        outputs = [
            run_puzzle_rollout(
                harness=harness,
                example=example,
                llm=llm,
                assistant_token_budget=budget,
                policy_max_tokens=per_call,
            )
            for example in examples
        ]
    else:
        outputs = [None] * len(examples)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                (
                    idx,
                    pool.submit(
                        run_puzzle_rollout,
                        harness=harness,
                        example=example,
                        llm=llm,
                        assistant_token_budget=budget,
                        policy_max_tokens=per_call,
                    ),
                )
                for idx, example in enumerate(examples)
            ]
            for idx, future in futures:
                outputs[idx] = future.result()
    metadata = {
        "dataset_path": str(dataset_path),
        "limit": len(examples),
        "seed": seed,
        "assistant_token_budget": budget,
        "policy_max_tokens": per_call,
        "max_turns_default": default_max_turns_from_env(),
        "max_turns_cap": max_turns_cap_from_env(),
        "max_turns_effective": effective_max_turns(harness),
        "harness_path": str(harness_path),
        "answer_redacted": True,
    }
    summary = summarize_outputs(outputs, metadata)
    (output / "val.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    write_trajectories(outputs, output)
    trace_path = output / "ChessPuzzle-v0_policy_trace.jsonl"
    with trace_path.open("w") as fh:
        for output_item in outputs:
            for event in output_item.get("harness_trace", []):
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    return summary


def append_trace(path: str | Path | None, event: dict[str, Any]) -> None:
    if not path:
        return
    with TRACE_WRITE_LOCK:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
