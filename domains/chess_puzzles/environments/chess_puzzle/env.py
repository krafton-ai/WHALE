"""Meta-Harness evaluation adapter for chess-puzzle-v0."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from autoharness_chess_puzzle.runner import evaluate_harness as _evaluate_harness


def evaluate_harness(
    *,
    harness_path: str | Path,
    dataset_path: str | Path,
    llm_config: dict[str, Any],
    output_dir: str | Path,
    limit: int = 256,
    seed: int = 42,
    assistant_token_budget: int = 8129,
    policy_max_tokens: int | None = None,
    **_: Any,
) -> dict[str, Any]:
    return _evaluate_harness(
        harness_path=harness_path,
        dataset_path=dataset_path,
        llm_config=llm_config,
        output_dir=output_dir,
        limit=limit,
        seed=seed,
        assistant_token_budget=assistant_token_budget,
        policy_max_tokens=policy_max_tokens,
    )
