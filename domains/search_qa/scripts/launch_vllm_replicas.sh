#!/usr/bin/env bash
# Launch N vLLM OpenAI-compatible servers (one per GPU). For meta-harness eval,
# pairs with retrieval_launch_multi.sh — each GPU runs one retriever shard AND
# one vLLM replica (Qwen3.5-2B is ~4.5 GB so it fits comfortably alongside the
# ~30 GB fp16 wiki-18 FAISS upload).
#
# Layout when called from scripts/run_meta_harness.py with replicas=8:
#   GPU 0  port 8044  Qwen/Qwen3.5-2B   ─┐ each replica is a complete vLLM
#   GPU 1  port 8045  Qwen/Qwen3.5-2B   │ instance with its own KV cache;
#   GPU 2  port 8046  Qwen/Qwen3.5-2B   │ meta-harness eval round-robins
#   GPU 3  port 8047  Qwen/Qwen3.5-2B   ├─ across them via the
#   GPU 4  port 8048  Qwen/Qwen3.5-2B   │ SEARCH_R1_VLLM_URLS env (read by
#   GPU 5  port 8049  Qwen/Qwen3.5-2B   │ meta_harness/benchmark.py).
#   GPU 6  port 8050  Qwen/Qwen3.5-2B   │
#   GPU 7  port 8051  Qwen/Qwen3.5-2B   ─┘
#
# Env knobs:
#   REPLICAS         default 8
#   BASE_PORT        default 8044
#   GPU_OFFSET       default 0          (first GPU index to bind)
#   MODEL            default Qwen/Qwen3.5-2B
#   TOOL_CALL_PARSER default qwen3_coder  (Qwen3.5 chat template emits XML)
#   GPU_MEM_UTIL     default 0.75       (vLLM weights + KV cache budget; leaves
#                                        ~30 GB headroom on H200 for the
#                                        co-located retriever FAISS fp16 footprint)
#   MAX_MODEL_LEN    default 8192
#   ENABLE_CHUNKED_PREFILL default true
#   ENABLE_PREFIX_CACHING  default true
#   EXTRA_ARGS       default = "--enforce-eager --language-model-only
#                                --gdn-prefill-backend triton
#                                --trust-remote-code
#                                --enable-chunked-prefill
#                                --enable-prefix-caching"
#                                (kept in lockstep with verl GRPO rollout
#                                engine flags in train_grpo_qwen3_5_2b_paper.sh
#                                — override to add/remove flags as a single
#                                space-separated string. NOTE:
#                                `--reasoning-parser qwen3` was REMOVED so the
#                                vLLM HTTP server returns `<think>...</think>`
#                                inline in `content` instead of splitting it
#                                into `reasoning_content`. verl has no
#                                reasoning_parser concept (raw generated
#                                tokens are preserved verbatim in
#                                prompt_ids), so removing it here makes the
#                                two stacks behave identically: verifiers
#                                stores the full content unchanged, and the
#                                Qwen3.5 chat template's fallback path
#                                (`</think>` in content → split via string
#                                ops at template rendering time) reconstructs
#                                the same bytes verl preserves natively.
#                                Re-enable only if you specifically want the
#                                API-level reasoning_content split.)
#   VLLM_BIN         default vllm       (plain `vllm serve` from the activated
#                                        venv; override to .venv-searchr1/bin/vllm
#                                        if you don't want to source the venv)
#
# This script expects to be run with .venv-searchr1 active (or VLLM_BIN pointing
# at that venv's vllm binary). It does NOT use `vf-vllm` because that script
# lives in .venv-meta-harness via the verifiers package — the meta-harness
# orchestrator and the vLLM workers live in different venvs by design.

set -euo pipefail

REPLICAS="${REPLICAS:-8}"
BASE_PORT="${BASE_PORT:-8044}"
GPU_OFFSET="${GPU_OFFSET:-0}"
MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.75}"
# 32768 = verl train_grpo_qwen3_5_2b_paper.sh's actor_rollout_ref.rollout
# .max_model_len. Keep meta-harness eval's vLLM identical to verl's so KV
# slot allocation / context budget match.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
# --enforce-eager is REQUIRED on H200 (sm_90) with our vllm 0.20.1+cu129 wheel
# (Inductor autotune generates kernel images incompatible with H200 → CUDA
# error 209 device kernel image invalid). verl uses enforce_eager=true via
# Hydra; standalone `vllm serve` needs the flag passed explicitly.
#
# --language-model-only skips vision encoder allocation. Qwen3.5-2B's
# architecture is Qwen3_5ForConditionalGeneration (multimodal vision-language);
# Search-R1's text-only input never triggers the vision path so the encoder
# would just consume KV-cache budget for nothing.
#
# --reasoning-parser qwen3 is DELIBERATELY OMITTED. The reasoning parser
# splits <think>...</think> blocks out of `content` into a separate
# `reasoning_content` API field. verl has no equivalent — it preserves the
# raw generated tokens (including <think>...</think>) verbatim in
# prompt_ids. When the reasoning parser is on, verifiers stores
# reasoning_content separately and the next-turn request includes both
# `content` and `reasoning_content`; the Qwen3.5 chat template reassembles
# them via the `if message.reasoning_content is string` branch. When the
# reasoning parser is OFF (this default), verifiers stores the full
# <think>...</think> in `content`; the template's fallback branch
# (`</think>` in content → string split) recovers the same structure. Both
# paths render byte-identical bytes back into the prompt, but the OFF path
# is simpler and matches verl's native raw-token behavior 1:1.
#
# --enable-chunked-prefill / --enable-prefix-caching default ON here to match
# the RSFT/GRPO rollout launcher defaults. Set ENABLE_CHUNKED_PREFILL=false or
# ENABLE_PREFIX_CACHING=false to omit either flag for a controlled comparison.
#
# --gdn-prefill-backend triton mirrors verl GRPO's
# +actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend=triton
# override (used for the Qwen3.5 gated-delta-net linear-attention layers; the
# CUDA default kernel and triton kernel can produce different prefill SSM
# states, which makes meta-harness eval scores drift from verl rollout scores).
#
# --trust-remote-code mirrors verl GRPO's actor_rollout_ref.model.trust_remote_code=true.
# Qwen3.5-2B uses a custom modeling file (Qwen3_5ForConditionalGeneration) that
# vLLM needs to load via remote code; without this flag the server can fall
# back to a different code path than verl's embedded engine.
#
# Override by setting EXTRA_ARGS to a value that adds/removes flags.
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-true}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-true}"
DEFAULT_EXTRA_ARGS="--enforce-eager --language-model-only --gdn-prefill-backend triton --trust-remote-code"
if [[ "${ENABLE_CHUNKED_PREFILL,,}" == "true" ]]; then
    DEFAULT_EXTRA_ARGS+=" --enable-chunked-prefill"
fi
if [[ "${ENABLE_PREFIX_CACHING,,}" == "true" ]]; then
    DEFAULT_EXTRA_ARGS+=" --enable-prefix-caching"
fi
EXTRA_ARGS="${EXTRA_ARGS:-$DEFAULT_EXTRA_ARGS}"
VLLM_BIN="${VLLM_BIN:-vllm}"

echo "[multi-vllm] replicas=$REPLICAS BASE_PORT=$BASE_PORT GPU_OFFSET=$GPU_OFFSET"
echo "[multi-vllm] MODEL=$MODEL TOOL_CALL_PARSER=$TOOL_CALL_PARSER"
echo "[multi-vllm] GPU_MEM_UTIL=$GPU_MEM_UTIL MAX_MODEL_LEN=$MAX_MODEL_LEN"
echo "[multi-vllm] ENABLE_CHUNKED_PREFILL=$ENABLE_CHUNKED_PREFILL"
echo "[multi-vllm] ENABLE_PREFIX_CACHING=$ENABLE_PREFIX_CACHING"
echo "[multi-vllm] EXTRA_ARGS=$EXTRA_ARGS"
echo "[multi-vllm] VLLM_BIN=$VLLM_BIN"

pids=()
cleanup() {
    local rc="$?"
    if (( ${#pids[@]} )); then
        echo "[multi-vllm] terminating ${#pids[@]} vllm processes" >&2
        for p in "${pids[@]}"; do
            kill -TERM "$p" 2>/dev/null || true
        done
        wait 2>/dev/null || true
    fi
    exit "$rc"
}
trap cleanup INT TERM EXIT

for i in $(seq 0 $((REPLICAS - 1))); do
    port=$((BASE_PORT + i))
    gpu=$((GPU_OFFSET + i))

    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="$gpu" \
        "$VLLM_BIN" serve "$MODEL" \
            --enable-auto-tool-choice \
            --tool-call-parser "$TOOL_CALL_PARSER" \
            --port "$port" \
            --gpu-memory-utilization "$GPU_MEM_UTIL" \
            --max-model-len "$MAX_MODEL_LEN" \
            $EXTRA_ARGS &
    pids+=("$!")
    echo "[multi-vllm] launched replica_$i pid=${pids[-1]} GPU=$gpu port=$port"
done

echo "[multi-vllm] all $REPLICAS replicas launched; waiting (pids=${pids[*]})"
wait
