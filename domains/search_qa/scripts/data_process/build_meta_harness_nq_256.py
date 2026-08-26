#!/usr/bin/env python3
"""Build the default meta-harness eval set as 256 NQ train rows.

Protocol:
  1. Keep the current 128 NQ meta-harness rows.
  2. Sample 128 additional rows from the GRPO NQ train set with seed 42.
  3. Require every row to be a subset of the GRPO NQ train set and require no
     duplicate structural/question keys.

The output schema matches the Search-R1 parquet schema.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "searchR1_processed"
NQ_SOURCE = "searchR1_nq"


def _extra(row: pd.Series) -> dict:
    value = row.get("extra_info")
    return value if isinstance(value, dict) else {}


def _key(row: pd.Series) -> tuple[str, int | None, str]:
    extra = _extra(row)
    idx = extra.get("index")
    try:
        idx = int(idx)
    except (TypeError, ValueError):
        idx = None
    return (str(row.get("data_source")), idx, str(extra.get("question", "")))


def _question(row: pd.Series) -> str:
    return str(_extra(row).get("question", ""))


def _load_base_nq(base_nq_path: Path, fallback_path: Path, expected: int) -> pd.DataFrame:
    if base_nq_path.exists():
        base = pd.read_parquet(base_nq_path)
    elif fallback_path.exists():
        base = pd.read_parquet(fallback_path)
        base = base[base["data_source"] == NQ_SOURCE].head(expected).copy()
    else:
        raise SystemExit(
            f"missing base NQ rows: {base_nq_path} and fallback {fallback_path}"
        )

    base = base[base["data_source"] == NQ_SOURCE].reset_index(drop=True)
    if len(base) != expected:
        raise SystemExit(f"expected {expected} base NQ rows, found {len(base)}")
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-nq", default=str(DATA_DIR / "train_nq_single.parquet"))
    parser.add_argument("--base-nq", default=str(DATA_DIR / "mh_nq_single_128.parquet"))
    parser.add_argument("--fallback", default=str(DATA_DIR / "train_meta_harness_256.parquet"))
    parser.add_argument("--out", default=str(DATA_DIR / "train_meta_harness_nq_256.parquet"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-count", type=int, default=128)
    parser.add_argument("--total", type=int, default=256)
    args = parser.parse_args()

    train = pd.read_parquet(args.train_nq)
    train = train[train["data_source"] == NQ_SOURCE].reset_index(drop=True)
    if len(train) == 0:
        raise SystemExit(f"no NQ rows found in {args.train_nq}")

    base = _load_base_nq(Path(args.base_nq), Path(args.fallback), args.base_count)
    needed = args.total - len(base)
    if needed < 0:
        raise SystemExit(f"requested total {args.total} is smaller than base rows {len(base)}")

    train_keys = {_key(row) for _, row in train.iterrows()}
    base_keys = {_key(row) for _, row in base.iterrows()}
    base_questions = {_question(row) for _, row in base.iterrows()}
    if len(base_keys) != len(base):
        raise SystemExit("base NQ rows contain duplicate keys")
    if len(base_questions) != len(base):
        raise SystemExit("base NQ rows contain duplicate questions")
    if not base_keys <= train_keys:
        raise SystemExit(
            f"{len(base_keys - train_keys)} base NQ rows are not in the GRPO NQ train set"
        )

    candidates = train.copy()
    candidates["_key"] = candidates.apply(_key, axis=1)
    candidates["_question"] = candidates.apply(_question, axis=1)
    candidates = candidates[
        ~candidates["_key"].isin(base_keys)
        & ~candidates["_question"].isin(base_questions)
    ].drop(columns=["_key", "_question"])
    if len(candidates) < needed:
        raise SystemExit(f"only {len(candidates)} candidates remain; need {needed}")

    sampled = candidates.sample(n=needed, random_state=args.seed).copy()
    out = pd.concat([base, sampled], ignore_index=True)
    out_keys = {_key(row) for _, row in out.iterrows()}
    out_questions = {_question(row) for _, row in out.iterrows()}

    if len(out) != args.total:
        raise SystemExit(f"expected {args.total} output rows, got {len(out)}")
    if len(out_keys) != len(out):
        raise SystemExit("output contains duplicate keys")
    if len(out_questions) != len(out):
        raise SystemExit("output contains duplicate questions")
    if set(out["data_source"].unique()) != {NQ_SOURCE}:
        raise SystemExit(f"output is not NQ-only: {out['data_source'].value_counts()}")
    if not out_keys <= train_keys:
        raise SystemExit("output contains rows outside the GRPO NQ train set")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    print(f"wrote {out_path} rows={len(out)} seed={args.seed}")
    print(out["data_source"].value_counts().to_string())


if __name__ == "__main__":
    main()
