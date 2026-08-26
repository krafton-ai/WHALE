from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import chess
import pandas as pd


DEFAULT_DATASET = "Lichess/chess-puzzles"
SPLIT_SIZES = {"train": 16384, "test": 1024, "mh_val": 256}


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _moves(value: Any) -> list[str]:
    if isinstance(value, str):
        return [item.lower() for item in value.split() if item.strip()]
    if isinstance(value, list):
        return [str(item).lower() for item in value if str(item).strip()]
    return []


def _themes(value: Any) -> list[str]:
    if isinstance(value, str):
        return [item for item in value.split() if item]
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return []


def _position_key(fen: str) -> str:
    parts = fen.split()
    if len(parts) >= 2:
        return parts[0] + " " + parts[1]
    return fen


def _initial_prompt(extra_info: dict[str, Any]) -> list[dict[str, str]]:
    try:
        from autoharness_chess_puzzle.harness import load_harness
        from autoharness_chess_puzzle.runner import (
            example_from_mapping,
            initialize_state,
            messages_as_dicts,
            policy_messages_with_harness,
        )

        repo_root = Path(__file__).resolve().parents[1]
        harness = load_harness(repo_root / "environments" / "chess_puzzle" / "base_harness.py")
        example = example_from_mapping({"extra_info": extra_info})
        return messages_as_dicts(policy_messages_with_harness(harness, initialize_state(example)))
    except Exception:
        return [
            {
                "role": "system",
                "content": "You solve chess puzzles by returning one UCI move at a time.",
            },
            {
                "role": "user",
                "content": "Solve the current chess puzzle position. Return exactly one UCI move.",
            },
        ]


def convert_lichess_row(row: dict[str, Any]) -> dict[str, Any] | None:
    initial_fen = str(row.get("FEN") or row.get("fen") or "").strip()
    raw_moves = _moves(row.get("Moves") or row.get("moves"))
    if not initial_fen or len(raw_moves) < 2:
        return None
    try:
        board = chess.Board(initial_fen)
        trigger_move = chess.Move.from_uci(raw_moves[0])
        if trigger_move not in board.legal_moves:
            return None
        board.push(trigger_move)
        start_fen = board.fen()
        solution_moves = raw_moves[1:]
        verify_board = chess.Board(start_fen)
        for move_uci in solution_moves:
            move = chess.Move.from_uci(move_uci)
            if move not in verify_board.legal_moves:
                return None
            verify_board.push(move)
    except Exception:
        return None

    puzzle_id = str(row.get("PuzzleId") or row.get("id") or stable_hash(initial_fen + " " + " ".join(raw_moves))[:16])
    rating = row.get("Rating") or row.get("rating")
    try:
        rating_int = int(rating)
    except (TypeError, ValueError):
        rating_int = None
    themes = _themes(row.get("Themes") or row.get("themes"))
    extra_info = {
        "task": "chess-puzzle-v0",
        "source": DEFAULT_DATASET,
        "puzzle_id": puzzle_id,
        "initial_fen": initial_fen,
        "trigger_move": raw_moves[0],
        "start_fen": start_fen,
        "solution_moves": solution_moves,
        "raw_moves": raw_moves,
        "rating": rating_int,
        "themes": themes,
        "position_key": _position_key(start_fen),
    }
    return {
        "data_source": "chess-puzzle-v0",
        "prompt": _initial_prompt(extra_info),
        "ability": "chess_puzzle",
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": extra_info,
    }


def _split_for_key(key: str, sizes: dict[str, int]) -> str:
    total = sum(sizes.values())
    bucket = int(stable_hash(key)[:12], 16) % total
    cursor = 0
    for name in ("mh_val", "test", "train"):
        cursor += sizes[name]
        if bucket < cursor:
            return name
    return "train"


def _split_path(output_dir: Path, split: str, size: int) -> Path:
    if split == "train":
        return output_dir / f"lichess_puzzles_train_{size}.parquet"
    if split == "test":
        return output_dir / f"lichess_puzzles_test_{size}.parquet"
    return output_dir / f"lichess_puzzles_mh_val_{size}.parquet"


def _decode_extra_info(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def _load_existing_split(path: Path, expected_size: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    frame = pd.read_parquet(path)
    if len(frame) != expected_size:
        return []
    return frame.to_dict("records")


def build_splits(
    *,
    output_dir: Path,
    dataset_name: str = DEFAULT_DATASET,
    train_size: int = 16384,
    test_size: int = 1024,
    mh_val_size: int = 256,
    seed: int = 42,
    max_scan: int = 2_000_000,
    min_rating: int | None = None,
    max_rating: int | None = None,
    min_solution_plies: int = 1,
    max_solution_plies: int = 9,
    preserve_existing_eval_splits: bool = True,
) -> dict[str, Path]:
    from datasets import load_dataset

    sizes = {"train": train_size, "test": test_size, "mh_val": mh_val_size}
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in sizes}
    seen_positions: set[str] = set()
    output_dir.mkdir(parents=True, exist_ok=True)
    preserved: dict[str, int] = {}
    if preserve_existing_eval_splits:
        for split in ("test", "mh_val"):
            existing = _load_existing_split(_split_path(output_dir, split, sizes[split]), sizes[split])
            if not existing:
                continue
            rows[split] = existing
            preserved[split] = len(existing)
            for record in existing:
                position_key = str(_decode_extra_info(record.get("extra_info")).get("position_key") or "")
                if position_key:
                    seen_positions.add(position_key)

    stream = load_dataset(dataset_name, split="train", streaming=True)
    scanned = 0
    for raw in stream:
        scanned += 1
        if scanned > max_scan:
            break
        converted = convert_lichess_row(dict(raw))
        if converted is None:
            continue
        extra = converted["extra_info"]
        rating = extra.get("rating")
        if min_rating is not None and rating is not None and int(rating) < min_rating:
            continue
        if max_rating is not None and rating is not None and int(rating) > max_rating:
            continue
        plies = len(extra["solution_moves"])
        if plies < min_solution_plies or plies > max_solution_plies:
            continue
        position_key = str(extra["position_key"])
        if position_key in seen_positions:
            continue
        split_key = stable_hash(position_key + f":{seed}")
        split = _split_for_key(split_key, sizes)
        if len(rows[split]) >= sizes[split]:
            for fallback in ("mh_val", "test", "train"):
                if len(rows[fallback]) < sizes[fallback]:
                    split = fallback
                    break
            else:
                break
        seen_positions.add(position_key)
        converted["extra_info"]["split"] = split
        rows[split].append(converted)
        if all(len(rows[name]) >= sizes[name] for name in sizes):
            break

    missing = {name: sizes[name] - len(values) for name, values in rows.items() if len(values) < sizes[name]}
    if missing:
        raise RuntimeError(f"could not fill requested splits after scanning {scanned} rows: {missing}")

    paths = {
        "train": _split_path(output_dir, "train", train_size),
        "test": _split_path(output_dir, "test", test_size),
        "mh_val": _split_path(output_dir, "mh_val", mh_val_size),
    }
    for name, path in paths.items():
        frame = pd.DataFrame(rows[name])
        frame.to_parquet(path, index=False)
    manifest = {
        "dataset": dataset_name,
        "seed": seed,
        "sizes": {name: len(values) for name, values in rows.items()},
        "paths": {name: str(path) for name, path in paths.items()},
        "max_scan": max_scan,
        "filters": {
            "min_rating": min_rating,
            "max_rating": max_rating,
            "min_solution_plies": min_solution_plies,
            "max_solution_plies": max_solution_plies,
        },
        "preserved_existing_eval_splits": preserved,
        "disjoint_key": "start_fen piece-placement plus active color",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/chess_puzzle"))
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--train-size", type=int, default=16384)
    parser.add_argument("--test-size", type=int, default=1024)
    parser.add_argument("--mh-val-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-scan", type=int, default=2_000_000)
    parser.add_argument("--min-rating", type=int, default=None)
    parser.add_argument("--max-rating", type=int, default=None)
    parser.add_argument("--min-solution-plies", type=int, default=1)
    parser.add_argument("--max-solution-plies", type=int, default=9)
    parser.add_argument(
        "--no-preserve-existing-eval-splits",
        action="store_true",
        help="Rebuild test and MH validation splits instead of preserving existing files of the requested sizes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = build_splits(
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        train_size=args.train_size,
        test_size=args.test_size,
        mh_val_size=args.mh_val_size,
        seed=args.seed,
        max_scan=args.max_scan,
        min_rating=args.min_rating,
        max_rating=args.max_rating,
        min_solution_plies=args.min_solution_plies,
        max_solution_plies=args.max_solution_plies,
        preserve_existing_eval_splits=not args.no_preserve_existing_eval_splits,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
