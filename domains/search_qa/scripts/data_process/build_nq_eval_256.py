#!/usr/bin/env python3
"""Build an NQ-only 256-row evaluation parquet.

Protocol:
  1. Keep the 100 NQ rows already present in test_100.parquet.
  2. Sample 156 additional NQ rows from test.parquet with seed 42, excluding
     the retained 100 by both (data_source, extra_info.index) and question text.

The output schema matches the source Search-R1 parquet schema.
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


def _index_key(row: pd.Series) -> tuple[str, int | None]:
    extra = _extra(row)
    idx = extra.get("index")
    try:
        idx = int(idx)
    except (TypeError, ValueError):
        idx = None
    return (str(row.get("data_source")), idx)


def _question_key(row: pd.Series) -> str:
    return str(_extra(row).get("question", ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-100", default=str(DATA_DIR / "test_100.parquet"))
    parser.add_argument("--test-full", default=str(DATA_DIR / "test.parquet"))
    parser.add_argument("--out", default=str(DATA_DIR / "test_nq_256.parquet"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total", type=int, default=256)
    args = parser.parse_args()

    test_100 = pd.read_parquet(args.test_100)
    test_full = pd.read_parquet(args.test_full)

    base = test_100[test_100["data_source"] == NQ_SOURCE].copy()
    if len(base) != 100:
        raise SystemExit(f"expected 100 NQ rows in {args.test_100}, found {len(base)}")

    need = args.total - len(base)
    if need < 0:
        raise SystemExit(f"requested total {args.total} is smaller than base NQ rows {len(base)}")

    candidates = test_full[test_full["data_source"] == NQ_SOURCE].copy()
    base_index_keys = set(base.apply(_index_key, axis=1))
    base_question_keys = set(base.apply(_question_key, axis=1))

    candidates["_index_key"] = candidates.apply(_index_key, axis=1)
    candidates["_question_key"] = candidates.apply(_question_key, axis=1)
    candidates = candidates[
        ~candidates["_index_key"].isin(base_index_keys)
        & ~candidates["_question_key"].isin(base_question_keys)
    ].drop(columns=["_index_key", "_question_key"])

    if len(candidates) < need:
        raise SystemExit(
            f"only {len(candidates)} eligible NQ candidates remain; need {need}"
        )

    sampled = candidates.sample(n=need, random_state=args.seed).copy()
    out = pd.concat([base, sampled], ignore_index=True)
    if len(out) != args.total:
        raise SystemExit(f"expected {args.total} output rows, got {len(out)}")
    if set(out["data_source"].unique()) != {NQ_SOURCE}:
        raise SystemExit(f"output is not NQ-only: {out['data_source'].value_counts()}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    print(f"wrote {out_path} rows={len(out)} seed={args.seed}")
    print(out["data_source"].value_counts().to_string())


if __name__ == "__main__":
    main()
