#!/usr/bin/env bash
# Search-R1 GRPO training for Qwen/Qwen3.5-2B with multi-turn tool calling.
#
# Forked from train_grpo_qwen3_1_7b_paper.sh. Generation hyperparameters
# (temperature, top_p, top_k, assistant-token budget, max_assistant_turns,
# enable_thinking, MICRO sizes) are kept identical to the 1.7B paper config
# so that meta-harness eval, GRPO rollout, and GRPO validation all sample
# from the same distribution.
#
# Differences vs 1.7B paper, with reasons:
#
#   * BASE_MODEL                 Qwen/Qwen3.5-2B   (1.7B → 3.5-2B)
#   * RUN_NAME                   search-r1-grpo-qwen3.5-2b
#   * multi_turn.format          qwen3_coder       (Qwen3.5 chat template emits
#                                                  XML <tool_call><function=…>
#                                                  <parameter=…> tags. The 1.7B
#                                                  paper used hermes (JSON in
#                                                  <tool_call>). Wrong parser =
#                                                  every assistant tool_call is
#                                                  silently un-parseable.)
#   * ENABLE_CHUNKED_PREFILL     true              (Qwen3.5-2B's text backbone
#                                                  is hybrid attention: 18
#                                                  linear_attention + 6
#                                                  full_attention layers, every
#                                                  4th layer full. linear layers
#                                                  use a Mamba/SSM-style state,
#                                                  not a classical KV cache.
#                                                  vLLM's chunked_prefill /
#                                                  prefix_caching paths assume
#                                                  per-layer KV slots; the
#                                                  official Qwen3.5 vLLM recipe
#                                                  is silent on chunked_prefill
#                                                  and flags
#                                                  "prefix caching for Mamba
#                                                  cache 'align' mode is
#                                                  currently experimental".
#                                                  This run family now keeps
#                                                  both ON for rollout speed.)
#   * ENABLE_PREFIX_CACHING      true              (same reason as above)
#
# Qwen3.5-2B model facts (HF Qwen/Qwen3.5-2B):
#   - architectures: ["Qwen3_5ForConditionalGeneration"]
#     (multimodal vision-language; vision encoder is small, ~100M params, kept
#      resident but unused for text-only Search-R1 input — accepted overhead.)
#   - model_type: qwen3_5 (text submodel: qwen3_5_text)
#   - vocab_size: 248,320  (vs 1.7B's 151,936 — softmax peak grows ~1.63×)
#   - max_position_embeddings: 262,144 (well above our MAX_MODEL_LEN=10k)
#   - hybrid attention layer_types: 18 linear + 6 full, full every 4th layer
#   - mrope (multimodal RoPE), partial_rotary_factor=0.25
#   - tokenizer vocab includes vision tokens; <tool_call>=248058 (single token)
#   - enable_thinking flag semantics are INVERTED vs Qwen3:
#         Qwen3-1.7B: enable_thinking=false → suppresses thinking (default ON)
#         Qwen3.5-2B: enable_thinking=true  → activates thinking  (default OFF)
#     so the literal flag value (true) is identical across both; only the
#     downstream effect changes.
#
# Defaults are tuned for **H200 SXM 141GB** (FSDP + vLLM colocate, single node,
# retriever on GPU 0,1 + training on GPU 2-7). For A100 40GB or smaller cards,
# override the memory knobs back to their conservative values:
#   FSDP_PARAM_OFFLOAD=True FSDP_OPTIM_OFFLOAD=True GPU_MEM_UTIL=0.55 \
#   PPO_MICRO_PER_GPU=1 ROLLOUT_LOG_PROB_MICRO_PER_GPU=1 REF_MICRO_PER_GPU=1 \
#   bash scripts/nq_hotpotqa/train_grpo_qwen3_5_2b_paper.sh
#
# Prerequisites:
#   1. Retriever up on GPU 0,1 (single-retrieve mode) OR GPU 0-7 (MULTI_RETRIEVE=1)
#       CUDA_VISIBLE_DEVICES=0,1 bash retrieval_launch.sh   (in .venv-retriever)
#   2. Activate searchr1 venv
#       source .venv-searchr1/bin/activate
#   3. Run this script (defaults tuned for 8x H200-141GB)
#       bash scripts/nq_hotpotqa/train_grpo_qwen3_5_2b_paper.sh
#
# Multi-retriever colocate mode (feat/multi-retrieve):
#   When MULTI_RETRIEVE=1 is set, slurm_train.sh runs 8 retriever shards
#   colocated on GPU 0-7 alongside the trainer. Suggested invocation:
#     N_GPUS_PER_NODE=8 GPU_MEM_UTIL=0.55 TRAIN_BATCH_SIZE=64 \
#         PPO_MINI_BATCH_SIZE=64 MULTI_RETRIEVE=1 sbatch slurm_train.sh
#   This script auto-bumps the defaults below when MULTI_RETRIEVE=1 is set
#   AND the corresponding env var has not been set explicitly. Override
#   any individual knob to opt out of the bump for that knob.
#
# Override knobs (env vars) — same shape as 1.7B paper script:
#   CUDA_VISIBLE_DEVICES  default 2,3,4,5,6,7        (retriever owns 0,1)
#   N_GPUS_PER_NODE       default 6
#   BASE_MODEL            default Qwen/Qwen3.5-2B
#   RUN_NAME              default search-r1-grpo-qwen3.5-2b
#   EXPERIMENT_NAME       (back-compat alias)
#   OUTPUT_ROOT           default outputs
#   GPU_MEM_UTIL          default 0.80
#   ROLLOUT_N             default 8
#   TRAIN_BATCH_SIZE      default 64
#   PPO_MINI_BATCH_SIZE   default 64
#   PPO_MICRO_PER_GPU     default 8     (vocab 1.63× larger than 1.7B; softmax
#                                        peak ≈ 112 GB at MICRO=8, fits 141 GB
#                                        H200. Drop to 4 if OOM.)
#   ROLLOUT_LOG_PROB_MICRO_PER_GPU default 8
#   REF_MICRO_PER_GPU     default 8
#   MAX_RESP_LENGTH       default 8192  (cumulative assistant-token budget;
#                                        excludes tool/user response tokens)
#   MAX_TOTAL_RESPONSE_LENGTH default MAX_RESP_LENGTH + 1024 * MAX_ASSISTANT_TURNS
#                                        (separate tensor/window safety cap;
#                                         includes assistant + tool/user tokens)
#   MAX_PROMPT_LENGTH     default 2048
#   MAX_MODEL_LEN         default 32768
#   MAX_NUM_BATCHED_TOKENS default 16384
#   MAX_ASSISTANT_TURNS   default selected HARNESS_PATH MAX_TURNS, otherwise 4
#   MAX_TOOL_RESPONSE_LEN default 99999
#   FSDP_PARAM_OFFLOAD    default False
#   FSDP_OPTIM_OFFLOAD    default False
#   ENABLE_PREFIX_CACHING default true
#   ENABLE_CHUNKED_PREFILL default true
#   USE_KL_LOSS           default False
#   CLIP_RATIO_LOW        default 0.2
#   CLIP_RATIO_HIGH       default 0.28
#   DUMP_ROLLOUTS         default true
#   DUMP_VAL_ROLLOUTS     default true
#   WANDB_API_KEY         default ""    (set non-empty to enable wandb)
#   WANDB_PROJECT         default search_r1_train
#   LLM_JUDGE             default 0
#   LLM_JUDGE_KEY_FILE    default $PROJECT_DIR/configs/llm_judge_key.txt
#   LLM_JUDGE_MODEL       default gpt-5.4-mini
#   LLM_JUDGE_CONCURRENCY default 64
#   TRAINER_SAVE_FREQ     default 7   (0.1 epoch for 4480 NQ rows / batch 64)
#   TRAINER_TEST_FREQ     default 7   (0.1 epoch for 4480 NQ rows / batch 64)
#   TRAINER_MAX_KEEP      default null  (keep all checkpoints; integer for rotation)
#   TRAINER_TOTAL_EPOCHS  default 4
#   TRAIN_FILES           default data/searchR1_processed/train_nq_single.parquet
#   VAL_FILES             default data/searchR1_processed/test_100.parquet (nq256 deprecated)
#   RESUME_MODE           default disable   (verl only recognises "disable" /
#                                            "auto" / "resume_path"; booleans
#                                            silently fall through.)

set -euo pipefail

# Multi-retriever colocate mode: bump defaults that only make sense with 8 GPUs
# AND were not already set explicitly by the caller. Each knob is bumped only
# if it is unset/empty, so any explicit `KEY=val ...` invocation opts out of
# that one bump while keeping the others.
if [[ "${MULTI_RETRIEVE:-0}" == "1" ]]; then
    : "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
    : "${N_GPUS_PER_NODE:=8}"
    # Per-GPU memory budget on H200 (141 GB):
    #   * Retriever FAISS Flat fp16  ~ 30.5 GB
    #   * e5 encoder fp16             ~  0.2 GB
    #   * vLLM (weights + KV) @ 0.55  ~ 78   GB
    #     (Qwen3.5-2B weights ~4.5 GB bf16 + vision encoder ~0.2 GB +
    #      KV/state for hybrid layers; the linear layers use SSM state
    #      instead of classical KV cache, so KV footprint is somewhat
    #      lower per-token than a pure-attention Qwen3 of similar size.)
    #   * FSDP active step            ~ 12   GB (slightly higher than 1.7B
    #                                            due to vocab 248k softmax)
    #   * Headroom                    ~ 20   GB
    : "${GPU_MEM_UTIL:=0.55}"
    : "${TRAIN_BATCH_SIZE:=64}"
    : "${PPO_MINI_BATCH_SIZE:=64}"
    : "${AGENT_NUM_WORKERS:=8}"
fi

: "${CUDA_VISIBLE_DEVICES:=2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CONFIG_PATH="$PROJECT_DIR/configs"
DATA_DIR="$PROJECT_DIR/data/searchR1_processed"

export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-2B}"
# RUN_NAME controls wandb run name AND output directory:
#   $OUTPUT_ROOT/$RUN_NAME/{global_step_N, logs, rollouts, val_rollouts}
export RUN_NAME="${RUN_NAME:-${EXPERIMENT_NAME:-search-r1-grpo-qwen3.5-2b}}"
export EXPERIMENT_NAME="$RUN_NAME"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs}"

N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-6}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"
ROLLOUT_N="${ROLLOUT_N:-8}"
VAL_ROLLOUT_N="${VAL_ROLLOUT_N:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
# H200 default. resp=2048 → ~2.5 GB activation per sample. Qwen3.5-2B vocab
# is 248,320 (1.63× larger than Qwen3-1.7B's 151,936), so softmax peak at
# MICRO=8 ≈ 56 GB, fits 141 GB. A100 40GB: drop to 1.
PPO_MICRO_PER_GPU="${PPO_MICRO_PER_GPU:-8}"
# Forward-only log-prob recompute on rollouts. Decoupled from PPO_MICRO.
ROLLOUT_LOG_PROB_MICRO_PER_GPU="${ROLLOUT_LOG_PROB_MICRO_PER_GPU:-8}"
REF_MICRO_PER_GPU="${REF_MICRO_PER_GPU:-8}"
MAX_RESP_LENGTH="${MAX_RESP_LENGTH:-8192}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
# 32k context window. Generous depth budget: with resp=2048 + parallel
# tool_responses (≤ MAX_PARALLEL_CALLS × topk × max_doc_tokens) the loop
# can run up to MAX_TURNS=16 before approaching the 32768 ceiling.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-4}"
if [[ -n "${HARNESS_PATH:-}" ]]; then
    [[ -f "$HARNESS_PATH" ]] || { echo "[train_grpo] ERROR: HARNESS_PATH '$HARNESS_PATH' missing" >&2; exit 1; }
    RESOLVED_MAX_ASSISTANT_TURNS="$(python3 "$PROJECT_DIR/scripts/resolve_harness_max_turns.py" "$HARNESS_PATH")" || {
        echo "[train_grpo] ERROR: could not resolve MAX_TURNS from HARNESS_PATH=$HARNESS_PATH" >&2
        exit 1
    }
    if (( RESOLVED_MAX_ASSISTANT_TURNS <= 0 )); then
        echo "[train_grpo] ERROR: resolved MAX_TURNS must be positive, got $RESOLVED_MAX_ASSISTANT_TURNS ($HARNESS_PATH)" >&2
        exit 1
    fi
    if (( RESOLVED_MAX_ASSISTANT_TURNS > 16 )); then
        echo "[train_grpo] WARNING: resolved MAX_TURNS=$RESOLVED_MAX_ASSISTANT_TURNS > 16 cap; clamping to 16" >&2
        RESOLVED_MAX_ASSISTANT_TURNS=16
    fi
    MAX_ASSISTANT_TURNS="$RESOLVED_MAX_ASSISTANT_TURNS"
fi
# Per-doc cap in SearchTool already bounds the response; verl's 256-char
# tool_response truncation would slash the whole tool_response down to ~64
# tokens (middle-truncation) and starve the model. Disable it.
MAX_TOOL_RESPONSE_LEN="${MAX_TOOL_RESPONSE_LEN:-99999}"
# Parallel tool_call cap (verl side). Default 1 = first tool_call only,
# rest silently dropped. We set the cap to 999 (effectively no cap) to
# mirror meta-harness base_harness.env_response which iterates every
# tool_call the model emits in a single turn. Keeps HARNESS_PATH-set
# verl training and meta-harness eval generation-equivalent.
MAX_PARALLEL_CALLS="${MAX_PARALLEL_CALLS:-999}"
MAX_TOTAL_RESPONSE_LENGTH="${MAX_TOTAL_RESPONSE_LENGTH:-$((MAX_RESP_LENGTH + MAX_ASSISTANT_TURNS * 1024))}"
if (( MAX_TOTAL_RESPONSE_LENGTH < MAX_RESP_LENGTH )); then
    echo "[train] ERROR: MAX_TOTAL_RESPONSE_LENGTH ($MAX_TOTAL_RESPONSE_LENGTH) must be >= MAX_RESP_LENGTH ($MAX_RESP_LENGTH)" >&2
    exit 1
fi
FSDP_PARAM_OFFLOAD="${FSDP_PARAM_OFFLOAD:-False}"
FSDP_OPTIM_OFFLOAD="${FSDP_OPTIM_OFFLOAD:-False}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-true}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-true}"

# ---- algorithm knobs ----
USE_KL_LOSS="${USE_KL_LOSS:-False}"
CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.2}"
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.28}"

# ---- logging / dump knobs ----
# Export WANDB_API_KEY=<WANDB_API_KEY> before invoking; do not hardcode secrets here.
WANDB_API_KEY="${WANDB_API_KEY:-}"
WANDB_PROJECT="${WANDB_PROJECT:-search_r1_train}"

# ---- reward knobs ----
LLM_JUDGE="${LLM_JUDGE:-0}"
LLM_JUDGE_KEY_FILE="${LLM_JUDGE_KEY_FILE:-$PROJECT_DIR/configs/llm_judge_key.txt}"
LLM_JUDGE_MODEL="${LLM_JUDGE_MODEL:-gpt-5.4-mini}"
LLM_JUDGE_CONCURRENCY="${LLM_JUDGE_CONCURRENCY:-64}"

DUMP_ROLLOUTS="${DUMP_ROLLOUTS:-true}"
DUMP_VAL_ROLLOUTS="${DUMP_VAL_ROLLOUTS:-true}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-}"
VAL_DATA_DIR="${VAL_DATA_DIR:-}"

TRAINER_SAVE_FREQ="${TRAINER_SAVE_FREQ:-7}"
TRAINER_TEST_FREQ="${TRAINER_TEST_FREQ:-7}"
TRAINER_MAX_KEEP="${TRAINER_MAX_KEEP:-null}"
TRAINER_TOTAL_EPOCHS="${TRAINER_TOTAL_EPOCHS:-4}"
TRAIN_FILES="${TRAIN_FILES:-$DATA_DIR/train_nq_single.parquet}"
# nq256 is deprecated — test_100 (7 datasets x 100 = 700 rows) is ALWAYS the default val set.
VAL_FILES="${VAL_FILES:-$DATA_DIR/test_100.parquet}"
RESUME_MODE="${RESUME_MODE:-disable}"

DEFAULT_NQ_VAL_FILE="$DATA_DIR/test_nq_256.parquet"
if [[ "$VAL_FILES" == "$DEFAULT_NQ_VAL_FILE" && ! -f "$VAL_FILES" ]]; then
    echo "[train] $VAL_FILES missing; building NQ-only 256-row eval set with seed 42"
    NQ_BUILDER_PY="$PROJECT_DIR/.venv-searchr1-fla/bin/python"
    [[ -x "$NQ_BUILDER_PY" ]] || NQ_BUILDER_PY=python3
    "$NQ_BUILDER_PY" "$PROJECT_DIR/scripts/data_process/build_nq_eval_256.py" --out "$VAL_FILES"
fi

RUN_DIR="$OUTPUT_ROOT/$RUN_NAME"
CKPT_DIR="$RUN_DIR"
LOG_DIR="$RUN_DIR/logs"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-$RUN_DIR/rollouts}"
VAL_DATA_DIR="${VAL_DATA_DIR:-$RUN_DIR/val_rollouts}"
mkdir -p "$CKPT_DIR" "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/${RUN_NAME}_${TS}.log"

echo "[train_grpo_qwen3_5_2b_paper] starting"
echo "  CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "  N_GPUS_PER_NODE      = $N_GPUS_PER_NODE"
echo "  BASE_MODEL           = $BASE_MODEL"
echo "  RUN_NAME             = $RUN_NAME"
echo "  RUN_DIR              = $RUN_DIR"
echo "  LOG_FILE             = $LOG_FILE"
echo "  GPU_MEM_UTIL         = $GPU_MEM_UTIL"
echo "  ROLLOUT_N            = $ROLLOUT_N"
echo "  VAL_ROLLOUT_N        = $VAL_ROLLOUT_N"
echo "  TRAIN_BATCH_SIZE     = $TRAIN_BATCH_SIZE"
echo "  PPO_MINI_BATCH_SIZE  = $PPO_MINI_BATCH_SIZE"
echo "  PPO_MICRO_PER_GPU    = $PPO_MICRO_PER_GPU"
echo "  ROLLOUT_LOG_PROB_MICRO_PER_GPU = $ROLLOUT_LOG_PROB_MICRO_PER_GPU"
echo "  MAX_RESP_LENGTH      = $MAX_RESP_LENGTH (assistant-token budget)"
echo "  MAX_TOTAL_RESPONSE_LENGTH = $MAX_TOTAL_RESPONSE_LENGTH (assistant + tool/user tensor window)"
echo "  MAX_PROMPT_LENGTH    = $MAX_PROMPT_LENGTH"
echo "  MAX_MODEL_LEN        = $MAX_MODEL_LEN"
echo "  MAX_NUM_BATCHED_TOKENS = $MAX_NUM_BATCHED_TOKENS"
echo "  MAX_ASSISTANT_TURNS  = $MAX_ASSISTANT_TURNS"
echo "  ENABLE_CHUNKED_PREFILL = $ENABLE_CHUNKED_PREFILL"
echo "  ENABLE_PREFIX_CACHING  = $ENABLE_PREFIX_CACHING"
echo "  TRAINER_SAVE_FREQ    = $TRAINER_SAVE_FREQ"
echo "  TRAINER_TEST_FREQ    = $TRAINER_TEST_FREQ"
echo "  TRAINER_MAX_KEEP     = $TRAINER_MAX_KEEP"
echo "  TRAINER_TOTAL_EPOCHS = $TRAINER_TOTAL_EPOCHS"
echo "  RESUME_MODE          = $RESUME_MODE"
echo "  VAL_FILES            = $VAL_FILES"
echo "  USE_KL_LOSS          = $USE_KL_LOSS"
echo "  CLIP_RATIO_LOW       = $CLIP_RATIO_LOW"
echo "  CLIP_RATIO_HIGH      = $CLIP_RATIO_HIGH"
echo "  LLM_JUDGE            = $LLM_JUDGE"

# ---- assemble conditional Hydra overrides ---------------------------------
EXTRA_OVERRIDES=()

if [[ "$LLM_JUDGE" == "1" ]]; then
    if [[ ! -f "$LLM_JUDGE_KEY_FILE" ]]; then
        echo "[train] ERROR: LLM_JUDGE=1 but key file '$LLM_JUDGE_KEY_FILE' is missing." >&2
        exit 1
    fi
    EXTRA_OVERRIDES+=("reward.custom_reward_function.path=$PROJECT_DIR/verl/utils/reward_score/llm_judge.py")
    EXTRA_OVERRIDES+=("reward.custom_reward_function.name=compute_score")
    export LLM_JUDGE_KEY_FILE
    export LLM_JUDGE_MODEL
    export LLM_JUDGE_CONCURRENCY
    echo "  llm_judge            = ENABLED (model=$LLM_JUDGE_MODEL, concurrency=$LLM_JUDGE_CONCURRENCY, key=$LLM_JUDGE_KEY_FILE)"
else
    echo "  llm_judge            = disabled (EM scoring)"
fi

if [[ -n "$WANDB_API_KEY" ]]; then
    export WANDB_API_KEY
    EXTRA_OVERRIDES+=("trainer.logger=[console,wandb]")
    EXTRA_OVERRIDES+=("trainer.project_name=$WANDB_PROJECT")
    echo "  wandb                = ENABLED (project=$WANDB_PROJECT, run=$RUN_NAME)"
else
    EXTRA_OVERRIDES+=("trainer.logger=[]")
    EXTRA_OVERRIDES+=("trainer.project_name=$WANDB_PROJECT")
    echo "  wandb                = disabled (set WANDB_API_KEY to enable)"
fi

if [[ "$DUMP_ROLLOUTS" == "true" ]]; then
    EXTRA_OVERRIDES+=("trainer.rollout_data_dir=$ROLLOUT_DATA_DIR")
    mkdir -p "$ROLLOUT_DATA_DIR"
    echo "  DUMP_ROLLOUTS        = true   ($ROLLOUT_DATA_DIR)"
else
    echo "  DUMP_ROLLOUTS        = false"
fi
if [[ "$DUMP_VAL_ROLLOUTS" == "true" ]]; then
    EXTRA_OVERRIDES+=("trainer.validation_data_dir=$VAL_DATA_DIR")
    mkdir -p "$VAL_DATA_DIR"
    echo "  DUMP_VAL_ROLLOUTS    = true   ($VAL_DATA_DIR)"
else
    echo "  DUMP_VAL_ROLLOUTS    = false"
fi

# Trainer entrypoint goes through optimal_setting/preamble.py (Ray prestart cap +
# log noise patches); see 1.7B paper script header for rationale.
PYTHONUNBUFFERED=1 python3 "$PROJECT_DIR/scripts/nq_hotpotqa/optimal_setting/preamble.py" \
    --config-path="$CONFIG_PATH" \
    --config-name='search_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.val_batch_size="$TRAIN_BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_TOTAL_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=8 \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.trust_remote_code=true \
    +data.apply_chat_template_kwargs.enable_thinking=true \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.trust_remote_code=true \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_PER_GPU" \
    actor_rollout_ref.actor.use_kl_loss="$USE_KL_LOSS" \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_low="$CLIP_RATIO_LOW" \
    actor_rollout_ref.actor.clip_ratio_high="$CLIP_RATIO_HIGH" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload="$FSDP_PARAM_OFFLOAD" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="$FSDP_OPTIM_OFFLOAD" \
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_NUM_BATCHED_TOKENS" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$ROLLOUT_LOG_PROB_MICRO_PER_GPU" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM_UTIL" \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$MAX_ASSISTANT_TURNS" \
    actor_rollout_ref.rollout.multi_turn.max_assistant_tokens="$MAX_RESP_LENGTH" \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length="$MAX_TOOL_RESPONSE_LEN" \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls="$MAX_PARALLEL_CALLS" \
    actor_rollout_ref.rollout.multi_turn.format=qwen3_coder \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.agent.num_workers="${AGENT_NUM_WORKERS:-4}" \
    reward.num_workers=4 \
    actor_rollout_ref.rollout.do_sample=true \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.n="$VAL_ROLLOUT_N" \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.enable_chunked_prefill="$ENABLE_CHUNKED_PREFILL" \
    actor_rollout_ref.rollout.enable_prefix_caching="$ENABLE_PREFIX_CACHING" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$CONFIG_PATH/tool_config/search_tool_config.yaml" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$REF_MICRO_PER_GPU" \
    actor_rollout_ref.ref.fsdp_config.param_offload="$FSDP_PARAM_OFFLOAD" \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_only=false \
    trainer.val_before_train=true \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
    trainer.nnodes=1 \
    trainer.save_freq="$TRAINER_SAVE_FREQ" \
    trainer.test_freq="$TRAINER_TEST_FREQ" \
    trainer.max_actor_ckpt_to_keep="$TRAINER_MAX_KEEP" \
    trainer.total_epochs="$TRAINER_TOTAL_EPOCHS" \
    trainer.resume_mode="$RESUME_MODE" \
    "${EXTRA_OVERRIDES[@]}" \
    "$@" 2>&1 | tee "$LOG_FILE"
