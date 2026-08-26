#!/usr/bin/env python3
"""Resolve epoch-based training frequencies to trainer step counts."""

from __future__ import annotations

import argparse
import ast
import math
import os
from pathlib import Path
from typing import Iterable


def _parse_files(value: str) -> list[str]:
    value = value.strip()
    if not value:
        return []
    if value.startswith("["):
        parsed = ast.literal_eval(value)
        return [str(item) for item in parsed]
    return [part.strip() for part in value.split(",") if part.strip()]


def _count_parquet(path: Path) -> int:
    import pyarrow.parquet as pq

    if path.is_file():
        return int(pq.ParquetFile(path).metadata.num_rows)
    total = 0
    for item in sorted(path.glob("*.parquet")):
        total += int(pq.ParquetFile(item).metadata.num_rows)
    if total <= 0:
        raise ValueError(f"no parquet rows found under {path}")
    return total


def _count_dataset(path_or_id: str) -> int:
    override = os.environ.get("TRAIN_DATASET_SIZE", "").strip()
    if override:
        return int(override)

    path = Path(path_or_id)
    if path.exists():
        if path.is_file() or any(path.glob("*.parquet")):
            return _count_parquet(path)
        try:
            from datasets import load_from_disk

            return len(load_from_disk(str(path)))
        except Exception:
            pass

    from datasets import load_dataset

    return len(load_dataset(path_or_id, split="train"))


def _count_rows(files: Iterable[str]) -> int:
    total = 0
    for item in files:
        total += _count_dataset(item)
    if total <= 0:
        raise ValueError("training dataset row count resolved to zero")
    return total


def _steps_for_fraction(steps_per_epoch: int, fraction: str | None) -> int | None:
    if fraction is None or str(fraction).strip() == "":
        return None
    value = float(str(fraction).strip().removesuffix("epoch").removesuffix("ep"))
    return max(1, int(math.ceil(steps_per_epoch * value)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-files", required=True)
    parser.add_argument("--train-batch-size", required=True, type=int)
    parser.add_argument("--total-epochs", default=None)
    parser.add_argument("--save-epochs", default=None)
    parser.add_argument("--test-epochs", default=None)
    parser.add_argument("--round-epochs", default=None)
    args = parser.parse_args()

    rows = _count_rows(_parse_files(args.train_files))
    steps_per_epoch = max(1, int(math.ceil(rows / args.train_batch_size)))

    print(f"TRAIN_DATASET_ROWS={rows}")
    print(f"STEPS_PER_EPOCH={steps_per_epoch}")
    if args.total_epochs not in {None, ""}:
        total_steps = max(1, int(math.ceil(steps_per_epoch * float(args.total_epochs))))
        print(f"TOTAL_STEPS={total_steps}")
    save_steps = _steps_for_fraction(steps_per_epoch, args.save_epochs)
    if save_steps is not None:
        print(f"SAVE_FREQ_STEPS={save_steps}")
    test_steps = _steps_for_fraction(steps_per_epoch, args.test_epochs)
    if test_steps is not None:
        print(f"TEST_FREQ_STEPS={test_steps}")
    round_steps = _steps_for_fraction(steps_per_epoch, args.round_epochs)
    if round_steps is not None:
        print(f"ROUND_GRPO_STEPS={round_steps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
