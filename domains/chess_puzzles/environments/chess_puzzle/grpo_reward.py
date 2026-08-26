from __future__ import annotations

from typing import Any


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    del data_source, solution_str, ground_truth, kwargs
    rollout_scores = extra_info.get("rollout_reward_scores") or {}
    try:
        score = float(rollout_scores.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    return {
        "score": score,
        "acc": score,
        "correct_answer": score,
        "win": score,
        "solved": score,
        "outcome": rollout_scores.get("outcome", ""),
    }

