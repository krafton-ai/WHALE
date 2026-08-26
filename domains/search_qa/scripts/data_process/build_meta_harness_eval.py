#!/usr/bin/env python3
"""Build the meta-harness evaluation set from train.parquet's held-out portion.

Sampling protocol:
  1. Held-out = rows in train.parquet whose key does NOT appear in
     train_filtered.parquet under EITHER of two keys:
       - (data_source, extra_info.index)   ← integer-pair structural key
       - extra_info.question               ← question-text safety net
     A row is excluded if either key matches; this is conservative on purpose
     to avoid any leakage of train_filtered samples into meta-harness eval.
  2. From the held-out rows, sample N per data_source (default 30).
     Sampling is seeded (default 42) so the same train.parquet+train_filtered
     pair always yields the same train_meta_harness.parquet.

Output schema is identical to train.parquet (data_source, prompt, ability,
reward_model, extra_info, metadata) so downstream verl/verifiers loaders
treat it as a drop-in substitute.

Usage:
    python scripts/data_process/build_meta_harness_eval.py
    python scripts/data_process/build_meta_harness_eval.py --per-source 50 --seed 7
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "searchR1_processed"


def _index_key(row: pd.Series) -> tuple[str, int]:
    return (row["data_source"], int(row["extra_info"]["index"]))


def _question_key(row: pd.Series) -> str:
    return str(row["extra_info"]["question"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", default=str(DATA_DIR / "train.parquet"))
    ap.add_argument("--filtered", default=str(DATA_DIR / "train_filtered.parquet"))
    ap.add_argument("--out", default=str(DATA_DIR / "train_meta_harness.parquet"))
    ap.add_argument("--per-source", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train = pd.read_parquet(args.train)
    filtered = pd.read_parquet(args.filtered)
    print(f"[builder] train={len(train):,}  filtered={len(filtered):,}")

    # Build exclusion sets from filtered.
    filt_idx_keys = set(filtered.apply(_index_key, axis=1))
    filt_q_keys = set(filtered.apply(_question_key, axis=1))
    print(
        f"[builder] exclusion keys: {len(filt_idx_keys):,} (data_source,index) "
        f"pairs, {len(filt_q_keys):,} unique questions"
    )

    train["_idx_key"] = train.apply(_index_key, axis=1)
    train["_q_key"] = train.apply(_question_key, axis=1)
    held_out = train[
        ~train["_idx_key"].isin(filt_idx_keys)
        & ~train["_q_key"].isin(filt_q_keys)
    ].copy()
    print(f"[builder] held-out rows after dual exclusion: {len(held_out):,}")
    print("[builder] held-out per data_source:")
    for ds, n in held_out["data_source"].value_counts().items():
        print(f"           {ds}: {n:,}")

    sampled_frames = []
    for ds, group in held_out.groupby("data_source", sort=True):
        n = min(args.per_source, len(group))
        if n < args.per_source:
            print(
                f"[builder] WARNING: {ds} has only {len(group):,} held-out rows; "
                f"sampling {n} (< requested {args.per_source})"
            )
        s = group.sample(n=n, random_state=args.seed).copy()
        sampled_frames.append(s)
        print(f"[builder] sampled {n} from {ds}")

    out = pd.concat(sampled_frames, ignore_index=True)
    out = out.drop(columns=["_idx_key", "_q_key"])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"[builder] wrote {out_path} ({len(out):,} rows)")
    print("[builder] output per data_source:")
    for ds, n in out["data_source"].value_counts().items():
        print(f"           {ds}: {n:,}")


if __name__ == "__main__":
    main()
