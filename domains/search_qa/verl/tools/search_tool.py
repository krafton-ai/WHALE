# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SearchTool — async httpx variant.

Tier 2 of the throughput plan: drop the Ray-actor + 32-thread `SearchExecutionWorker`
pool that funneled every trajectory's retrieve through one Python process,
and instead let each AgentLoopWorker's asyncio loop drive its own httpx
calls directly. AgentLoopWorker actors are separate processes (so GIL is
per-worker), and within each, asyncio gives us hundreds of concurrent
in-flight HTTP calls without thread contention.

Tier 1 (server-side per-doc tokenizer truncation) lives in
`search_r1/search/retrieval_server_async.py`. Combined, the client's
per-call work is now: serialize JSON → await POST → parse JSON → stitch
already-trimmed strings. No tokenizer load, no Ray RPC, no thread pool.
"""

import asyncio
import json
import logging
import os
from typing import Any, Optional
from uuid import uuid4

import httpx

from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# Codes the server may return that we want to retry on (transient).
_RETRYABLE_STATUS = {500, 502, 503, 504}


def _passages_to_string(retrieval_result) -> str:
    """Stitch already-truncated docs into the LLM-facing tool_response.

    The retrieval server (Tier 1) trims each doc's `contents` to
    `max_doc_tokens` before returning, so this function does NO tokenizer
    work — purely string concatenation, fast and GIL-cheap.
    """
    out = []
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["contents"]
        title = content.split("\n", 1)[0]
        body = content.split("\n", 1)[1] if "\n" in content else ""
        out.append(f"Doc {idx + 1} (Title: {title})\n{body}\n")
    return "\n".join(out).strip()


class SearchTool(BaseTool):
    """Search tool that calls the retrieval service via direct async HTTP.

    Each AgentLoopWorker process gets its own SearchTool instance and one
    shared httpx.AsyncClient (lazy-initialized on first use, scoped to the
    asyncio loop running in that worker).

    Configuration keys (search_tool_config.yaml):
        retrieval_service_url: required. URL of the /retrieve endpoint.
        topk: number of docs per query (default 3).
        timeout: per-attempt HTTP timeout in seconds (default 30).
        max_retries: number of retry attempts on transient errors (default 3).
        retry_initial_delay: backoff base in seconds (default 0.5).
        max_doc_tokens: per-doc truncation cap. Forwarded to server.
        tokenizer_name: HF tokenizer used for the cap. Forwarded to server.
        max_connections: httpx connection pool size (default 512).
        max_keepalive_connections: pool keepalive (default 128).
    """

    # Per-process singletons. Set lazily on first execute() call. Multiple
    # AgentLoopWorker processes each have their own copies; within a single
    # process, all SearchTool instances share the client.
    _async_client: Optional[httpx.AsyncClient] = None
    # Cache for the URL we picked once for this process — avoids reparsing
    # the actor name on every call.
    _resolved_url: Optional[str] = None

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict: dict[str, dict[str, Any]] = {}

        # Multi-retriever support. Priority for resolving the URL list:
        #   1. env SEARCH_R1_RETRIEVAL_URLS  (CSV — set by slurm_train.sh in
        #      MULTI_RETRIEVE=1 mode; e.g. "http://127.0.0.1:8000/retrieve,
        #      http://127.0.0.1:8001/retrieve, ...")
        #   2. config['retrieval_service_urls']  (yaml list)
        #   3. config['retrieval_service_url']   (legacy single)
        # When the list has > 1 entry, each AgentLoopWorker process pins to
        # ONE URL based on its actor index, so each retriever instance sees
        # a stable subset of trajectories — predictable load balance with no
        # per-call routing logic.
        self._retrieval_urls: list[str] = self._resolve_retrieval_urls(config)
        assert self._retrieval_urls, (
            "SearchTool requires at least one retrieval URL. Set "
            "SEARCH_R1_RETRIEVAL_URLS env, config['retrieval_service_urls'], "
            "or config['retrieval_service_url']."
        )
        # Backward compat: keep .retrieval_service_url as the first entry so
        # any external code reading this attr still works.
        self.retrieval_service_url = self._retrieval_urls[0]

        self.topk = config.get("topk", 3)
        self.timeout = config.get("timeout", 30)
        self.max_retries = config.get("max_retries", 3)
        self.retry_initial_delay = config.get("retry_initial_delay", 0.5)
        # Forwarded to the server for Tier 1 truncation. Client does not
        # itself load a tokenizer.
        self.max_doc_tokens = config.get("max_doc_tokens", 200)
        self.tokenizer_name = config.get("tokenizer_name", "Qwen/Qwen3.5-2B")
        self._max_connections = config.get("max_connections", 512)
        self._max_keepalive = config.get("max_keepalive_connections", 128)

        # Note: legacy `num_workers` / `rate_limit` / `enable_global_rate_limit`
        # entries in the YAML are tolerated but ignored — the actor pool /
        # token bucket is gone.
        for legacy_key in ("num_workers", "rate_limit", "enable_global_rate_limit"):
            if legacy_key in config:
                logger.info(f"[SearchTool] Ignoring legacy config key '{legacy_key}' (async-httpx variant).")

        logger.info(
            f"Initialized SearchTool (async-httpx, {len(self._retrieval_urls)} URL(s)) with config: {config}"
        )

    @staticmethod
    def _resolve_retrieval_urls(config: dict) -> list[str]:
        env_csv = os.environ.get("SEARCH_R1_RETRIEVAL_URLS", "").strip()
        if env_csv:
            urls = [u.strip() for u in env_csv.split(",") if u.strip()]
            if urls:
                return urls
        yaml_list = config.get("retrieval_service_urls")
        if yaml_list:
            return list(yaml_list)
        single = config.get("retrieval_service_url")
        return [single] if single else []

    def _pick_url(self) -> str:
        """Select which retriever URL this process talks to.

        Resolution is cached per-process so the choice is stable across
        every call inside a given AgentLoopWorker actor: each retriever
        process sees a deterministic subset of trajectories.
        """
        if SearchTool._resolved_url is not None:
            return SearchTool._resolved_url

        if len(self._retrieval_urls) == 1:
            SearchTool._resolved_url = self._retrieval_urls[0]
            return SearchTool._resolved_url

        # Multi-URL: pin by actor index when available, else fall back to a
        # stable hash of the actor name (or a random pick on the bare
        # process). All three cases give us "this process always talks to
        # the same retriever".
        actor_idx = self._extract_actor_index()
        n = len(self._retrieval_urls)
        SearchTool._resolved_url = self._retrieval_urls[actor_idx % n]
        logger.info(
            f"[SearchTool] actor_idx={actor_idx} → routing to {SearchTool._resolved_url} "
            f"(of {n} retrievers)"
        )
        return SearchTool._resolved_url

    @staticmethod
    def _extract_actor_index() -> int:
        """Best-effort: parse the integer suffix from this actor's name.

        verl names AgentLoopWorker actors `agent_loop_worker_{i}_{hex8}`.
        On a non-actor process or if the name format ever changes, fall
        back to a stable hash of whatever name we can find.
        """
        try:
            import ray
            name = ray.get_runtime_context().get_actor_name() or ""
        except Exception:
            name = ""
        if name:
            parts = name.split("_")
            for j, p in enumerate(parts):
                if p == "worker" and j + 1 < len(parts):
                    try:
                        return int(parts[j + 1])
                    except ValueError:
                        break
            # Stable fallback: hash the name → integer.
            return abs(hash(name))
        # Last resort: pid-based stability so at least we don't pick a
        # different URL on every call.
        return os.getpid()

    @classmethod
    def _get_client(cls, max_connections: int, max_keepalive: int) -> httpx.AsyncClient:
        if cls._async_client is None:
            cls._async_client = httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=max_connections,
                    max_keepalive_connections=max_keepalive,
                ),
                timeout=httpx.Timeout(60.0, connect=10.0),
                http2=False,
            )
        return cls._async_client

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the OpenAI tool schema."""
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a tool instance.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            instance_id, tool_creation_response.
        """
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "reward": [],
        }
        return instance_id, ToolResponse()

    async def _post_with_retry(self, payload: dict) -> tuple[Optional[dict], Optional[str]]:
        client = self._get_client(self._max_connections, self._max_keepalive)
        url = self._pick_url()
        last_err: Optional[str] = None
        for attempt in range(self.max_retries):
            try:
                resp = await client.post(url, json=payload, timeout=self.timeout)
                if resp.status_code in _RETRYABLE_STATUS:
                    last_err = f"server {resp.status_code} (attempt {attempt + 1}/{self.max_retries})"
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_initial_delay * (attempt + 1))
                    continue
                resp.raise_for_status()
                return resp.json(), None
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError) as e:
                last_err = f"network {type(e).__name__}: {e}"
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_initial_delay * (attempt + 1))
                continue
            except httpx.HTTPStatusError as e:
                # 4xx: don't retry — model emitted bad query the server rejected.
                last_err = f"http {e.response.status_code}: {e}"
                break
            except Exception as e:  # noqa: BLE001
                last_err = f"unexpected {type(e).__name__}: {e}"
                break
        return None, last_err

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute the search tool.

        Args:
            instance_id: The instance ID of the tool
            parameters: Tool parameters containing query_list and optional timeout

        Returns:
            tool_response: The response str of the tool.
            tool_reward_score: The step reward score of the tool.
            tool_metrics: The metrics of the tool.
        """
        query_list_from_params = parameters.get("query_list")

        # Lenient parsing: untrained models often emit query_list as a string
        # rather than a JSON list. Try to recover when feasible:
        #   "[\"q1\", \"q2\"]"   -> json.loads -> ["q1", "q2"]
        #   "single query"       -> wrap as ["single query"]
        #   malformed JSON       -> wrap raw string as one element
        if isinstance(query_list_from_params, str):
            s = query_list_from_params.strip()
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    query_list_from_params = [str(x) for x in parsed if x is not None]
                else:
                    query_list_from_params = [s]
            except (ValueError, json.JSONDecodeError):
                query_list_from_params = [s] if s else None

        if not query_list_from_params or not isinstance(query_list_from_params, list):
            error_msg = "Error: 'query_list' is missing, empty, or not a list in parameters."
            logger.error(f"[SearchTool] {error_msg} Received parameters: {parameters}")
            return ToolResponse(text=json.dumps({"result": error_msg})), 0.0, {}

        # Drop empty entries; if everything's empty, surface the error.
        query_list_from_params = [str(q).strip() for q in query_list_from_params if str(q).strip()]
        if not query_list_from_params:
            error_msg = "Error: 'query_list' is missing, empty, or not a list in parameters."
            logger.error(f"[SearchTool] {error_msg} Received parameters: {parameters}")
            return ToolResponse(text=json.dumps({"result": error_msg})), 0.0, {}

        # Hard cap: only the first query is used. The tool schema declares
        # maxItems=1 but model may still emit multiple — we clamp here so
        # the LLM context always receives exactly topk docs (one query × topk),
        # never N×topk. Drop the rest silently.
        if len(query_list_from_params) > 1:
            logger.info(
                f"[SearchTool] {len(query_list_from_params)} queries received; "
                f"using only the first one and dropping the rest."
            )
            query_list_from_params = query_list_from_params[:1]

        # Per-call topk / max_doc_tokens. The tool_schema in
        # configs/tool_config/search_tool_config.yaml exposes both as optional
        # integer fields with bounded ranges (topk 1..10, max_doc_tokens
        # 50..500), and meta-harness base_harness.py search() takes them as
        # keyword args with the same defaults — keeping verl SearchTool
        # symmetric ensures HARNESS_PATH-set and HARNESS_PATH-unset rollouts
        # both honor whatever values the model emits, falling back to the
        # yaml config (self.topk / self.max_doc_tokens) when the model
        # omits them. Untrained models occasionally serialize integers as
        # strings — coerce defensively.
        def _coerce_int(val: Any, fallback: int) -> int:
            if val is None:
                return fallback
            try:
                return int(val)
            except (TypeError, ValueError):
                logger.info(
                    f"[SearchTool] non-integer override for tool param ({val!r}); "
                    f"falling back to {fallback}."
                )
                return fallback

        topk_value = _coerce_int(parameters.get("topk"), self.topk)
        max_doc_tokens_value = _coerce_int(
            parameters.get("max_doc_tokens"), self.max_doc_tokens
        )

        payload = {
            "queries": query_list_from_params,
            "topk": topk_value,
            "return_scores": True,
            "max_doc_tokens": max_doc_tokens_value,
            "tokenizer_name": self.tokenizer_name,
        }

        try:
            api_response, error_msg = await self._post_with_retry(payload)
        except Exception as e:  # noqa: BLE001
            error_result = json.dumps({"result": f"Search execution failed: {e}"})
            logger.error(f"[SearchTool] Execution failed: {e}")
            return ToolResponse(text=error_result), 0.0, {"error": str(e)}

        if api_response is None:
            error_result = json.dumps({"result": f"Search error: {error_msg}"}, ensure_ascii=False)
            self._instance_dict[instance_id]["reward"].append(error_result.strip())
            return ToolResponse(text=error_result), 0.0, {"error": error_msg or "no response"}

        raw_results = api_response.get("result", [])
        if not raw_results:
            no_result_text = json.dumps({"result": "No search results found."}, ensure_ascii=False)
            self._instance_dict[instance_id]["reward"].append(no_result_text.strip())
            return ToolResponse(text=no_result_text), 0.0, {"status": "no_results", "total_results": 0}

        pretty_chunks = []
        total_results = 0
        for retrieval in raw_results:
            pretty_chunks.append(_passages_to_string(retrieval))
            total_results += len(retrieval) if isinstance(retrieval, list) else 1

        final_result = "\n---\n".join(pretty_chunks)
        result_text = json.dumps({"result": final_result}, ensure_ascii=False)
        self._instance_dict[instance_id]["reward"].append(result_text.strip())

        metrics = {
            "query_count": len(query_list_from_params),
            "status": "success",
            "total_results": total_results,
            "api_request_error": None,
        }
        return ToolResponse(text=result_text), 0.0, metrics

    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
