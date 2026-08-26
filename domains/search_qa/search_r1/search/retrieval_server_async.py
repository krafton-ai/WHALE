"""Async + window-batching variant of retrieval_server.py.

Why:
    The synchronous retrieve endpoint forces every concurrent /retrieve call
    to do its own single-query FAISS search even when many calls are arriving
    within the same few milliseconds. With our verl agent_loop dispatching
    one HTTP request per (trajectory, turn), a single training step issues
    ~1500 individual queries. Each call carries Ray-RPC + HTTP overhead, and
    the FAISS GPU search is only ~2.5 ms/query at batch=64 vs ~16-50 ms at
    batch=1 — i.e. throughput could be ~10x higher if we coalesced.

    upstream Search-R1 sidesteps this by issuing one batched /retrieve per
    rollout turn (every trajectory's <search>q</search> sent together). Our
    agent_loop is per-trajectory async and won't naturally batch on the
    client side without invasive changes to verl. Server-side batching is
    the cheap fix: we accept individual requests but coalesce them inside
    the server's event loop into one FAISS call.

How:
    * Each incoming /retrieve request enqueues its queries (as
      (query, topk, return_score, future) tuples) onto a single asyncio
      Queue, then awaits the futures.
    * A long-running BatchScheduler task pulls items off the queue: it
      blocks for the first item, then collects more for up to
      ``window_ms`` (or until ``max_batch`` items accumulate), then runs
      ``retriever.batch_search`` once on the combined query list (offloaded
      to a thread to keep the asyncio loop responsive).
    * Each future is resolved with that caller's slice of the results.

Compatibility:
    * Same /retrieve endpoint contract, same JSON request/response schema
      as retrieval_server.py.
    * Existing SearchTool / SearchExecutionWorker clients require zero
      change.
    * Falls back transparently to single-query behaviour when only one
      caller is in flight.

Knobs (env vars):
    RETRIEVE_BATCH_WINDOW_MS  (default 5)   -- how long to wait for more
                                               queries before firing the
                                               FAISS call once we have at
                                               least one query in hand.
    RETRIEVE_BATCH_MAX_SIZE   (default 256) -- hard cap on coalesced batch.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# Make the project root importable when this file is invoked as a script via
# `python search_r1/search/retrieval_server_async.py`. Without this, the
# script directory is the only entry on sys.path and the absolute import
# below would raise ModuleNotFoundError.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Reuse retriever construction + classes from the sync server module so we
# don't duplicate index/encoder loading code.
from search_r1.search.retrieval_server import Config, get_retriever  # noqa: E402

logger = logging.getLogger("retrieval_server_async")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Server-side per-doc tokenizer truncation. Doing the truncate here means
# clients (SearchTool) don't have to load HF tokenizer in every actor
# process and don't burn GIL time per call — they just stitch pre-trimmed
# strings into the LLM context. Lazy module-level cache, one entry per name.
# ---------------------------------------------------------------------------
_TOKENIZER_CACHE: dict[str, Any] = {}


def _get_tokenizer(name: str):
    tok = _TOKENIZER_CACHE.get(name)
    if tok is None:
        from transformers import AutoTokenizer  # local import to avoid hard dep at module load
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        _TOKENIZER_CACHE[name] = tok
    return tok


def _truncate(text: str, max_tokens: int, tokenizer_name: str) -> str:
    if not max_tokens or not tokenizer_name:
        return text
    try:
        tok = _get_tokenizer(tokenizer_name)
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) <= max_tokens:
            return text
        return tok.decode(ids[:max_tokens])
    except Exception as e:  # noqa: BLE001
        logger.warning("truncate fell back due to: %s", e)
        return text


def _truncate_pairs_in_place(pairs, max_tokens: int, tokenizer_name: str):
    """Mutate each doc's `contents` field in place to be at most max_tokens tokens.

    `pairs` is a list of (docs_list, scores_list) tuples. Each doc is a dict
    with at least a `contents` key (wiki-18 jsonl shape).
    """
    for docs, _scores in pairs:
        for d in docs:
            if isinstance(d, dict) and "contents" in d:
                d["contents"] = _truncate(d["contents"], max_tokens, tokenizer_name)
    return pairs


# ---------------------------------------------------------------------------
# Request schema (extends retrieval_server.py with optional truncation hints)
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    queries: list[str]
    topk: Optional[int] = None
    return_scores: bool = False
    # Server-side truncation knobs (Tier 1). When both set, the server caps
    # each doc's `contents` to its first `max_doc_tokens` tokens under
    # `tokenizer_name` BEFORE returning. Clients pass these so we don't have
    # to duplicate the tokenizer in every actor process.
    max_doc_tokens: Optional[int] = None
    tokenizer_name: Optional[str] = None


# ---------------------------------------------------------------------------
# Window-based batch scheduler
# ---------------------------------------------------------------------------
@dataclass
class _PendingQuery:
    query: str
    topk: int
    future: asyncio.Future  # resolved with (docs_list, scores_list_or_None)


class BatchScheduler:
    """Coalesce concurrent retrieve calls into a single FAISS batch search.

    Each enqueued query carries its own ``topk``. We always run FAISS with
    the maximum requested topk in the batch and slice each caller's result
    down to its own ``topk``; the cost of the larger search is negligible
    compared to per-call dispatch.
    """

    def __init__(self, retriever, window_ms: int = 5, max_batch: int = 256) -> None:
        self.retriever = retriever
        self.window_s = window_ms / 1000.0
        self.max_batch = max_batch
        self._queue: asyncio.Queue[_PendingQuery] = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None
        self._stats_lock = asyncio.Lock()
        self.batches_run = 0
        self.queries_served = 0
        self.last_log = time.time()

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="batch_scheduler")
            logger.info("BatchScheduler started (window=%.0f ms, max_batch=%d)",
                        self.window_s * 1000, self.max_batch)

    async def submit(self, query: str, topk: int) -> _PendingQuery:
        loop = asyncio.get_running_loop()
        item = _PendingQuery(query=query, topk=topk, future=loop.create_future())
        await self._queue.put(item)
        return item

    async def _run(self) -> None:
        while True:
            try:
                # Block until at least one query arrives.
                first: _PendingQuery = await self._queue.get()
                pending: list[_PendingQuery] = [first]

                # Collect more for up to window_ms, or until max_batch reached.
                deadline = time.monotonic() + self.window_s
                while len(pending) < self.max_batch:
                    timeout = deadline - time.monotonic()
                    if timeout <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                        pending.append(item)
                    except asyncio.TimeoutError:
                        break

                # Drop any futures that callers gave up on (cancelled).
                live = [it for it in pending if not it.future.cancelled()]
                if not live:
                    continue

                # All queries share the same FAISS call with num=max(topk).
                k_per = max(it.topk for it in live)
                queries = [it.query for it in live]

                t0 = time.monotonic()
                try:
                    # batch_search is sync C++/CUDA → offload to a thread so
                    # our asyncio loop stays free to enqueue more requests.
                    docs, scores = await asyncio.to_thread(
                        self.retriever.batch_search,
                        query_list=queries,
                        num=k_per,
                        return_score=True,
                    )
                    elapsed_ms = (time.monotonic() - t0) * 1000
                    for i, it in enumerate(live):
                        if it.future.done():
                            continue
                        # Slice each caller's result to its requested topk.
                        it.future.set_result((docs[i][: it.topk], scores[i][: it.topk]))

                    async with self._stats_lock:
                        self.batches_run += 1
                        self.queries_served += len(live)
                        if time.time() - self.last_log >= 30:
                            logger.info(
                                "batches=%d queries=%d avg_batch=%.1f last_batch=%d last_ms=%.1f",
                                self.batches_run,
                                self.queries_served,
                                self.queries_served / max(1, self.batches_run),
                                len(live),
                                elapsed_ms,
                            )
                            self.last_log = time.time()
                except Exception as e:  # noqa: BLE001
                    logger.exception("batch_search failed: %s", e)
                    for it in live:
                        if not it.future.done():
                            it.future.set_exception(e)
            except asyncio.CancelledError:
                logger.info("BatchScheduler cancelled")
                raise
            except Exception:  # noqa: BLE001
                # Make sure the loop never dies silently — log and continue.
                logger.exception("BatchScheduler iteration crashed; continuing")
                await asyncio.sleep(0.1)


# ---------------------------------------------------------------------------
# FastAPI app — drop-in replacement for retrieval_server.py
# ---------------------------------------------------------------------------
app = FastAPI()
retriever = None  # populated in __main__
scheduler: Optional[BatchScheduler] = None
default_topk: int = 3


@app.on_event("startup")
async def _startup() -> None:
    assert scheduler is not None
    scheduler.start()


@app.post("/retrieve")
async def retrieve_endpoint(request: QueryRequest) -> dict[str, Any]:
    """Same I/O contract as retrieval_server.py:retrieve_endpoint, but async
    and routed through BatchScheduler so concurrent calls share one FAISS
    GPU search.

    When the request includes `max_doc_tokens` + `tokenizer_name`, this
    server applies the per-doc truncation server-side (Tier 1) so clients
    don't need to host an HF tokenizer in their actor processes.
    """
    assert scheduler is not None
    topk = request.topk or default_topk

    # Submit each query individually so coalescing works across requests.
    items = [await scheduler.submit(q, topk) for q in request.queries]
    pairs = await asyncio.gather(*(it.future for it in items))

    # Tier 1: per-doc truncation. Run in a worker thread because tokenizer
    # encode/decode is GIL-heavy and we don't want to stall the asyncio loop
    # while ~K docs (K = sum of topks across the batch) are tokenized.
    if request.max_doc_tokens and request.tokenizer_name:
        pairs = await asyncio.to_thread(
            _truncate_pairs_in_place, pairs, request.max_doc_tokens, request.tokenizer_name
        )

    resp = []
    for docs, scores in pairs:
        if request.return_scores:
            resp.append([{"document": d, "score": s} for d, s in zip(docs, scores)])
        else:
            resp.append(list(docs))
    return {"result": resp}


# ---------------------------------------------------------------------------
# Entrypoint (mirrors retrieval_server.py's argparse)
# ---------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the async-batching local FAISS retriever.")
    parser.add_argument("--index_path", type=str, required=True)
    parser.add_argument("--corpus_path", type=str, required=True)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--retriever_name", type=str, default="e5")
    parser.add_argument("--retriever_model", type=str, default="intfloat/e5-base-v2")
    parser.add_argument("--faiss_gpu", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    return parser


if __name__ == "__main__":
    args = _build_argparser().parse_args()

    cfg = Config(
        retrieval_method=args.retriever_name,
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        retrieval_topk=args.topk,
        faiss_gpu=args.faiss_gpu,
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method="mean",
        retrieval_query_max_length=256,
        retrieval_use_fp16=True,
        retrieval_batch_size=512,
    )

    retriever = get_retriever(cfg)
    default_topk = args.topk
    scheduler = BatchScheduler(
        retriever=retriever,
        window_ms=int(os.environ.get("RETRIEVE_BATCH_WINDOW_MS", "5")),
        max_batch=int(os.environ.get("RETRIEVE_BATCH_MAX_SIZE", "256")),
    )

    logger.info(
        "ready: index=%s corpus=%s topk=%d gpu=%s port=%d window_ms=%d max_batch=%d",
        args.index_path,
        args.corpus_path,
        args.topk,
        args.faiss_gpu,
        args.port,
        scheduler.window_s * 1000,
        scheduler.max_batch,
    )
    uvicorn.run(app, host=args.host, port=args.port)
