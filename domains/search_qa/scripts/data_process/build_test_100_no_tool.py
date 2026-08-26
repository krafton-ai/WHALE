# Copyright 2025 Search-R1 Contributors
"""Rebuild ``test_100.parquet`` into ``test_100_no_tool.parquet`` with a
simplified prompt that does NOT mention tool calling. Used for the no-tool
ablation eval. Same 700 samples (same row order, same data_sources) as
``test_100.parquet`` so the comparison is apples-to-apples.

System prompt:  "You are a helpful and harmless assistant."
User prompt:    "Answer the given question. You must conduct reasoning inside
                <think> and </think>. You should provide the answer inside
                <answer> and </answer>, without detailed illustrations.
                For example, <answer> Beijing </answer>. Question: {q}"
"""
from __future__ import annotations

import os

import pandas as pd

SRC = os.environ.get("SRC_PARQUET", "data/searchR1_processed/test_100.parquet")
DST = os.environ.get("DST_PARQUET", "data/searchR1_processed/test_100_no_tool.parquet")

USER_PREFIX = (
    "Answer the given question. You must conduct reasoning inside <think> and "
    "</think>. You should provide the answer inside <answer> and </answer>, "
    "without detailed illustrations. For example, <answer> Beijing </answer>. "
    "Question: "
)
SYSTEM_CONTENT = "You are a helpful and harmless assistant."


def rebuild_prompt(row: pd.Series) -> list[dict]:
    """Replace the prompt with the no-tool version, preserving the question."""
    extra = row.get("extra_info") or {}
    question = (extra.get("question") if isinstance(extra, dict) else None) or ""
    user_content = USER_PREFIX + question
    return [
        {"role": "system", "content": SYSTEM_CONTENT},
        {"role": "user", "content": user_content},
    ]


def main() -> None:
    if not os.path.exists(SRC):
        raise SystemExit(f"missing source parquet: {SRC}")

    df = pd.read_parquet(SRC).reset_index(drop=True)
    df["prompt"] = df.apply(rebuild_prompt, axis=1)

    # extra_info.need_tools_kwargs / tools_kwargs are leftovers from the tool-calling
    # data prep; the no-tool eval will not use them but we drop the flag so single
    # turn agent does not warn about missing tools_kwargs.
    def _strip_tools_kwargs(extra):
        if not isinstance(extra, dict):
            return extra
        out = dict(extra)
        out["need_tools_kwargs"] = False
        # Keep the original tools_kwargs struct so pyarrow can preserve schema;
        # single_turn_agent will simply ignore it.
        return out

    df["extra_info"] = df["extra_info"].apply(_strip_tools_kwargs)

    df.to_parquet(DST, index=False)
    print(f"wrote {DST}  rows={len(df)}")
    print(df["data_source"].value_counts())
    print()
    print("--- example prompt[0] ---")
    for msg in df.iloc[0]["prompt"]:
        print(f"[{msg['role']}] {msg['content'][:300]}")


if __name__ == "__main__":
    main()
