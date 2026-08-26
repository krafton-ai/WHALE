#!/usr/bin/env python3
"""Build the final held-out test set as source test minus GRPO validation rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _row_key(row: pd.Series) -> str:
    extra = row.get("extra_info") or {}
    question = ""
    index = None
    if isinstance(extra, dict):
        question = str(extra.get("question") or "")
        index = extra.get("index")
    reward_model = row.get("reward_model") or {}
    target = None
    if isinstance(reward_model, dict):
        target = (reward_model.get("ground_truth") or {}).get("target")
    return json.dumps(
        {
            "data_source": row.get("data_source"),
            "index": _jsonable(index),
            "question": question,
            "target": _jsonable(target),
        },
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Full test parquet")
    parser.add_argument("--exclude", required=True, help="GRPO validation parquet")
    parser.add_argument("--out", required=True, help="Output parquet path")
    parser.add_argument(
        "--source-data-source",
        default=None,
        help="Optional data_source value to keep from --source before excluding validation rows.",
    )
    parser.add_argument(
        "--require-excluded",
        action="store_true",
        help="Fail if no validation rows matched the source test set.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.source)
    exclude = Path(args.exclude)
    out = Path(args.out)

    raw_src_df = pd.read_parquet(source)
    if args.source_data_source:
        src_df = raw_src_df.loc[
            raw_src_df["data_source"] == args.source_data_source
        ].reset_index(drop=True)
    else:
        src_df = raw_src_df
    exc_df = pd.read_parquet(exclude)

    exc_keys = set(exc_df.apply(_row_key, axis=1).tolist())
    src_keys = src_df.apply(_row_key, axis=1)
    keep_mask = ~src_keys.isin(exc_keys)
    out_df = src_df.loc[keep_mask].reset_index(drop=True)

    removed = int((~keep_mask).sum())
    if args.require_excluded and removed == 0:
        raise SystemExit(
            f"no rows from {exclude} matched source {source}; refusing to write {out}"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out, index=False)
    print(
        json.dumps(
            {
                "source": str(source),
                "exclude": str(exclude),
                "out": str(out),
                "source_data_source": args.source_data_source,
                "raw_source_rows": int(len(raw_src_df)),
                "source_rows": int(len(src_df)),
                "exclude_rows": int(len(exc_df)),
                "removed_rows": removed,
                "out_rows": int(len(out_df)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
