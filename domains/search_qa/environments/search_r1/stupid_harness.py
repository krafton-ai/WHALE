"""Minimal Search-R1 base harness for the stupid-harness experiment.

This file is intentionally separate from base_harness.py so t4/t8 baseline
experiments cannot accidentally inherit the t2/k1 prompt/retrieval changes.

Experiment contract:
  - MAX_TURNS = 2
  - retrieval returns exactly top-1 passage, regardless of model-supplied topk
  - prompt is minimal but still preserves the <answer>...</answer> reward format
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from environments.search_r1.base_harness import (
    SearchR1Env as _BaseSearchR1Env,
    _next_url,
    _passages_to_string,
    logger,
)


SYSTEM_PROMPT = "Answer factual questions using search when needed."

USER_PROMPT_TEMPLATE = (
    "Question: {question}\n"
    "Use the search tool if needed. Put the final answer only inside "
    "<answer>...</answer>."
)


async def search(
    query_list: list[str], topk: int = 1, max_doc_tokens: int = 200
) -> str:
    """Searches for relevant information and returns exactly one passage.

    Args:
        query_list: A list containing exactly one semantic query.
        topk: Ignored. The experiment hard-forces topk=1.
        max_doc_tokens: Per-doc token cap applied by the retriever service.
    """
    cleaned = [str(q).strip() for q in query_list if str(q).strip()]
    if not cleaned:
        return json.dumps(
            {"result": "Error: 'query_list' is missing, empty, or not a list in parameters."},
            ensure_ascii=False,
        )
    if len(cleaned) > 1:
        logger.info(
            "[stupid_search] %d queries received; using only the first one.",
            len(cleaned),
        )
    query = cleaned[0]

    try:
        max_doc_tokens = int(max_doc_tokens)
    except (TypeError, ValueError):
        max_doc_tokens = 200
    if max_doc_tokens <= 0:
        max_doc_tokens = 200

    url = _next_url()
    payload: dict[str, Any] = {
        "queries": [query],
        "topk": 1,
        "return_scores": True,
        "max_doc_tokens": max_doc_tokens,
        "tokenizer_name": "Qwen/Qwen3.5-2B",
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("[stupid_search] retriever call failed (%s): %s", url, exc)
            return json.dumps({"result": f"Search error: {exc}"}, ensure_ascii=False)

    raw_results = data.get("result") or []
    if not raw_results:
        return json.dumps({"result": "No search results found."}, ensure_ascii=False)

    chunks = [_passages_to_string(retrieval) for retrieval in raw_results]
    final_result = "\n---\n".join(chunks)
    return json.dumps({"result": final_result}, ensure_ascii=False)


class SearchR1Env(_BaseSearchR1Env):
    SYSTEM_PROMPT: str = SYSTEM_PROMPT
    USER_PROMPT_TEMPLATE: str = USER_PROMPT_TEMPLATE
    TOOLS: list = [search]
    MAX_TURNS: int = 2


CandidateEnv = SearchR1Env
