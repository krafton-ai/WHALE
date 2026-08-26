#!/usr/bin/env python3
"""Resolve the default chess-puzzle MAX_TURNS from parquet splits."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


UCI_RE = re.compile(r"[a-h][1-8][a-h][1-8][qrbn]?", re.I)


def _safe_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _moves(value: Any) -> list[str]:
    value = _safe_json(value)
    if value is None:
        return []
    if isinstance(value, str):
        return [m.lower() for m in UCI_RE.findall(value)]
    try:
        return [str(m).lower() for m in value if str(m).strip()]
    except TypeError:
        return []


def _is_present(value: Any) -> bool:
    value = _safe_json(value)
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    try:
        return len(value) > 0
    except TypeError:
        return True


def _first_present(*values: Any) -> Any:
    for value in values:
        if _is_present(value):
            return value
    return None


def _solution_len(row: dict[str, Any]) -> int:
    extra = _safe_json(row.get("extra_info") or {})
    if not isinstance(extra, dict):
        extra = {}
    solution = _moves(_first_present(extra.get("solution_moves"), row.get("solution_moves")))
    if solution:
        return len(solution)
    raw_moves = _moves(_first_present(extra.get("raw_moves"), row.get("Moves"), row.get("moves"), row.get("answer")))
    return max(0, len(raw_moves) - 1)


def parquet_max_solution_len(path: Path) -> int:
    import pandas as pd

    frame = pd.read_parquet(path)
    if frame.empty:
        return 0
    return max(_solution_len(row) for row in frame.to_dict("records"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--fallback", type=int, default=9)
    args = parser.parse_args()

    maxima = []
    for raw in args.files:
        if not raw or raw.startswith("[") or "," in raw:
            continue
        path = Path(raw)
        if path.exists():
            maxima.append(parquet_max_solution_len(path))
    print(max(maxima) if maxima else int(args.fallback))


if __name__ == "__main__":
    main()
