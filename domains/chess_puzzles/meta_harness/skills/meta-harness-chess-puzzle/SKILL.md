---
name: meta-harness-chess-puzzle
description: Evolve prompt/parser/observation/retry harnesses for chess-puzzle-v0 without solving chess inside the harness.
---

# Meta-Harness Skill: chess-puzzle-v0

You are improving a harness for `chess-puzzle-v0`, a multi-turn chess tactics
task built from `Lichess/chess-puzzles`.

Run ONE iteration of harness improvement. The outer meta-harness loop evaluates
your candidate harnesses; you do not run benchmarks yourself, do not write
outside the requested candidate harness files, and do not delegate the work.

## Critical Constraints

- Write complete candidate `harness.py` files only.
- Each candidate must define:

```python
def propose_action(board: str) -> str: ...
def is_legal_action(board: str, action: str) -> bool: ...
```

- Optional harness surface:

```python
SYSTEM_PROMPT = "..."
USER_PROMPT = "..."
FORMAT_RETRY_BUDGET = 1
ILLEGAL_MOVE_RETRY_BUDGET = 1
MAX_TURNS = 9

def format_observation(observation: str, **kwargs) -> str: ...
def parse_action(response: str) -> str: ...
```

- `USER_PROMPT` may contain `{observation}`. The runner fills it.
- Each candidate should test exactly one meaningful mechanism, not a bundle of
  unrelated prompt/parser/retry changes.
- Generated code must be safe under
  `autoharness_chess_puzzle.harness.load_harness`.
- Allowed imports are only `__future__`, `collections`, `itertools`, `math`,
  `random`, `re`, and `statistics`.
- No shell commands, no file I/O, no network, no subprocesses, no dynamic import,
  no `eval`, and no `exec`.

## Anti-Cheating Rules

The harness is a communication and validation layer. It is not allowed to solve
the puzzle.

Hard bans:

- Do not import `chess`, use a chess engine, use tablebases, call Stockfish, or
  implement tactical search.
- Do not implement minimax, mate search, capture/check ranking, piece-value
  evaluation, move ordering, or any chess policy that chooses moves.
- Do not hardcode puzzle ids, FENs, solution lines, move sequences, ratings,
  themes, URLs, dataset row positions, or exact validation examples.
- Do not infer hidden answers from metadata, file paths, split names, hashes, or
  other dataset-specific artifacts.
- Do not convert a legal wrong model move into another move. A legal wrong move
  is a true model failure and must remain terminal.
- `propose_action()` must not be the primary policy. Prefer returning `""`.

Permitted behavior:

- Improve how visible information is presented to the model.
- Parse the model response more robustly.
- Validate legality only against the legal moves visibly listed in the
  observation.
- Retry malformed or illegal moves with small, explicit verifier nudges.

## Task Context

Each puzzle row contains a Lichess puzzle line. The environment applies the first
move in `Moves` as the trigger move before the model acts. The hidden solution is
the remaining line, `Moves[1:]`.

At each solver turn the model sees only visible state:

- side to move
- current FEN
- ASCII board
- previous accepted solver moves
- latest opponent reply, if any
- legal moves as visible UCI/SAN rows such as `- [e2e4] SAN=e4`
- verifier feedback after malformed or illegal responses

The model must emit exactly one solver move. If that move is correct, the
environment applies it. If the hidden line then contains an opponent reply, the
environment applies that reply and returns a tool/verifier response asking for
the next solver move. If the solver move is legal but not the hidden puzzle move,
the puzzle fails immediately.

There is no draw. Solved puzzles score 1; all failures score 0.

The assistant-generated token budget is 8129 total tokens per rollout by
default. Tool responses, observations, and verifier nudges are not part of this
assistant-token budget.

## Harness Contract

`SYSTEM_PROMPT` and `USER_PROMPT` define the exact policy prompt used by both
Meta-Harness evaluation and GRPO/RSFT rollout.

`format_observation(observation, **kwargs)` may transform only the visible
observation string. Current keyword arguments include:

- `turn_count`: number of accepted solver moves so far
- `side_to_move`: `"white"` or `"black"`
- `last_opponent_move`: latest opponent UCI reply, or `None`
- `max_turns`: effective assistant policy-call cap for the rollout

`parse_action(response)` should return one UCI move such as `e2e4`, or a
non-UCI sentinel such as `__no_move__` / `__ambiguous_multiple_moves__`. It may
normalize obvious formatting variants such as `<move>e2e4</move>`, `[e2e4]`,
or a single bare UCI token. Reject multiple distinct moves as ambiguous.

`is_legal_action(board, action)` receives the visible observation string and the
parsed action. It may only check membership in the visible legal move list. It
must not reconstruct the board or search for good moves.

`FORMAT_RETRY_BUDGET` and `ILLEGAL_MOVE_RETRY_BUDGET` are small nonnegative
integers. The runner clamps each retry budget to a maximum of 10. Retries are
for malformed or illegal actions only. Legal wrong moves are not retried.

`MAX_TURNS` is an optional positive integer controlling the maximum number of
assistant policy calls in one puzzle rollout, including retries. The base
harness default is 9, resolved from the largest solution continuation length in
the current train/test/MH validation splits. The runner clamps generated
harness values to at most 18, exactly twice that base default. Increasing this
cap can let the model spend more retry/continuation calls, but it must not be
used to hide answer-specific logic or bypass model choice.

## Useful Mechanism Axes

Focus on harness mechanisms that can help the model use visible state without
embedding chess search.

- Prompt architecture: final-only instruction, exact XML move format, no
  explanation, side-to-move reminders, one-move-at-a-time framing.
- Observation formatting: compact visible board/FEN/legal move presentation,
  clearer section ordering, accepted-move history, latest opponent reply, and
  legal move aliases already present in the observation.
- Parser robustness: last valid `<move>...</move>`, bracketed UCI, bare UCI,
  promotion suffix case normalization, ambiguity rejection, and safe sentinels.
- Retry feedback: short verifier messages for malformed or illegal output,
  preserving the current board and visible legal moves.
- History compaction: keep prior accepted moves and opponent replies visible in
  a concise way so long puzzle lines do not drown out the current legal move set.
- Edge-case handling: promotions, castling notation only when a visible UCI/SAN
  alias supports it, whitespace, Markdown fences, and repeated move mentions.

Weak mechanisms:

- Changing only punctuation, capitalization, or one adjective.
- Increasing retry budgets without changing what feedback says.
- Adding examples that look like dataset rows.
- Adding chess heuristics, even if they seem harmless.

## Files To Read Before Proposing

Use the run directory artifacts to understand previous failures and accepted
harness behavior:

- `logs/accepted_harness.txt`
- `harnesses/<accepted>/harness.py`
- `logs/frontier_val.json`
- `logs/evolution_summary.jsonl`
- latest per-candidate comparison or proposal report, if present
- latest `*_policy_trace.jsonl`, `*_trajectories.jsonl`, or validation JSON
  artifacts, if present

## Workflow

1. Identify the current accepted harness and read it.
2. Inspect recent failed and successful trajectories. Separate failures caused
   by malformed output, illegal output, legal wrong moves, token budget, and
   parser ambiguity.
3. Choose one concrete mechanism likely to improve solved rate or reduce
   avoidable malformed/illegal failures.
4. Write candidate harnesses with minimal, auditable changes.
5. Self-critique each candidate against the anti-cheating rules and the schema
   contract before finishing.

## Candidate Skeleton

```python
import re

SYSTEM_PROMPT = """You solve chess tactics one move at a time.
Use only the visible board position and legal moves. Return exactly one UCI move
inside <move></move> tags."""

USER_PROMPT = """Solve the current chess puzzle position.

{observation}

Return exactly one move as <move>uci</move> and no explanation."""

FORMAT_RETRY_BUDGET = 1
ILLEGAL_MOVE_RETRY_BUDGET = 1
MAX_TURNS = 9


def propose_action(board: str) -> str:
    return ""


def is_legal_action(board: str, action: str) -> bool:
    moves = set(m.group(1).lower() for m in re.finditer(
        r"\[\s*([a-h][1-8][a-h][1-8][qrbn]?)\s*\]", str(board), re.I
    ))
    action = str(action or "").strip().lower()
    return bool(re.fullmatch(r"[a-h][1-8][a-h][1-8][qrbn]?", action)) and action in moves
```
