#!/usr/bin/env bash
# ReTool GRPO for Qwen/Qwen3.5-2B.
#
# This follows verl-recipe/retool's GRPO path, with Qwen3.5-specific runtime
# changes carried over from the working Search-R1 Qwen3.5-2B setup:
#   - tool-call parser: qwen3_coder, not hermes
#   - enable_thinking=true in the chat template
#   - chunked prefill and prefix caching disabled by default
#   - trust_remote_code=true and SDPA attention override

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"

RETOOL_RECIPE_DIR="${RETOOL_RECIPE_DIR:-$PROJECT_DIR/verl-recipe/retool}"
[[ -f "$RETOOL_RECIPE_DIR/retool.py" ]] || {
    echo "[retool_grpo] ERROR: missing $RETOOL_RECIPE_DIR/retool.py. Clone verl-recipe first." >&2
    exit 1
}
RETOOL_RECIPE_PATCH="${RETOOL_RECIPE_PATCH:-$PROJECT_DIR/patches/verl-recipe-retool-local.patch}"
if [[ "${APPLY_RETOOL_RECIPE_PATCH:-auto}" != "false" && -f "$RETOOL_RECIPE_PATCH" ]] &&
    ! grep -q "class CustomSandboxFusionTool(BaseTool)" "$RETOOL_RECIPE_DIR/retool.py"; then
    RECIPE_GIT_ROOT="$(git -C "$RETOOL_RECIPE_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
    if [[ -z "$RECIPE_GIT_ROOT" ]]; then
        echo "[retool_grpo] ERROR: ReTool recipe patch is needed, but $RETOOL_RECIPE_DIR is not in a git worktree" >&2
        exit 1
    fi
    echo "[retool_grpo] applying local ReTool recipe patch: $RETOOL_RECIPE_PATCH"
    git -C "$RECIPE_GIT_ROOT" apply "$RETOOL_RECIPE_PATCH"
fi
if ! grep -q "class CustomSandboxFusionTool(BaseTool)" "$RETOOL_RECIPE_DIR/retool.py"; then
    echo "[retool_grpo] ERROR: ReTool recipe patch is missing. Set APPLY_RETOOL_RECIPE_PATCH=false only with a compatible recipe." >&2
    exit 1
fi

# Direct holder-launched runs bypass submit wrappers, so recover the same W&B
# account used by the Search-R1 training script when the environment lacks it.
SEARCH_R1_WANDB_SCRIPT="${SEARCH_R1_WANDB_SCRIPT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../search_qa" && pwd)/scripts/nq_hotpotqa/train_grpo_qwen3_5_2b_paper.sh}"
if [[ -z "${WANDB_API_KEY:-}" && -r "$SEARCH_R1_WANDB_SCRIPT" ]]; then
    WANDB_API_KEY="$(
        python3 - "$SEARCH_R1_WANDB_SCRIPT" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8")
for line in text.splitlines():
    if not line.startswith("WANDB_API_KEY="):
        continue
    match = re.search(r"WANDB_API_KEY:-([^}]+)", line)
    if match:
        print(match.group(1).strip("\"'"))
    break
PY
    )"
    if [[ -n "$WANDB_API_KEY" ]]; then
        export WANDB_API_KEY
    fi
fi
if [[ -v WANDB_RUN_ID && -z "$WANDB_RUN_ID" ]]; then
    unset WANDB_RUN_ID
fi
if [[ -v WANDB_RESUME && -z "$WANDB_RESUME" ]]; then
    unset WANDB_RESUME
fi

# The recipe code imports recipe.retool.* and its yaml paths use recipe/retool/*.
# Keep the upstream sparse clone intact and expose it through the expected path.
mkdir -p "$PROJECT_DIR/recipe"
ln -sfn "$RETOOL_RECIPE_DIR" "$PROJECT_DIR/recipe/retool"

# Use the local copied/patched verl tree. It contains the meta-harness hooks
# needed for HARNESS_PATH prompt/tool dispatch parity.
VERL_SOURCE_DIR="${VERL_SOURCE_DIR:-$PROJECT_DIR}"
if [[ -d "$VERL_SOURCE_DIR/verl" ]]; then
    export PYTHONPATH="$PROJECT_DIR:$VERL_SOURCE_DIR:${PYTHONPATH:-}"
else
    export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
fi

# Direct holder-launched runs bypass slurm_retool_train.sh. Keep the same
# H200/vLLM guard here so vLLM does not try the unavailable cp312 DeepGEMM
# extension from the cu129 wheel in our Python 3.10 environment.
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"
export VLLM_USE_DEEP_GEMM_E8M0="${VLLM_USE_DEEP_GEMM_E8M0:-0}"
export VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES="${VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES:-0}"

as_hydra_list() {
    local value="$1"
    if [[ "$value" == \[* ]]; then
        printf '%s\n' "$value"
        return
    fi
    IFS=',' read -r -a parts <<< "$value"
    local out="["
    local first=1
    for raw in "${parts[@]}"; do
        local item
        item="$(echo "$raw" | xargs)"
        [[ -z "$item" ]] && continue
        if (( first )); then
            out+="'$item'"
            first=0
        else
            out+=",'$item'"
        fi
    done
    out+="]"
    printf '%s\n' "$out"
}

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-2B}"
RUN_NAME="${RUN_NAME:-retool-grpo-qwen3.5-2b-dapo17k-dedup-r8192-bs256-gmem055-ppo2-logprob4-ref4-mnbt16384}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-$RUN_NAME}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs}"
RUN_DIR="$OUTPUT_ROOT/$RUN_NAME"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$RUN_DIR" "$LOG_DIR"
LOG_FILE="$LOG_DIR/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

# Use the deduplicated local DAPO train set by default. The public HF train
# split currently contains the 17,917 examples repeated 100 times.
TRAIN_FILES="${TRAIN_FILES:-$PROJECT_DIR/data/retool/dapo_math_17k_dedup.parquet}"
VAL_FILES="${VAL_FILES:-yentinglin/aime_2025,Maxwell-Jia/AIME_2024}"
TRAIN_FILES_HYDRA="${TRAIN_FILES_HYDRA:-$(as_hydra_list "$TRAIN_FILES")}"
VAL_FILES_HYDRA="${VAL_FILES_HYDRA:-$(as_hydra_list "$VAL_FILES")}"

SANDBOX_NUM_WORKERS="${SANDBOX_NUM_WORKERS:-128}"
SANDBOX_RATE_LIMIT="${SANDBOX_RATE_LIMIT:-128}"
SANDBOX_DEFAULT_TIMEOUT="${SANDBOX_DEFAULT_TIMEOUT:-30}"
SANDBOX_MEMORY_LIMIT_MB="${SANDBOX_MEMORY_LIMIT_MB:-1024}"
TOOL_CONFIG_PATH="${TOOL_CONFIG_PATH:-$PROJECT_DIR/recipe/retool/sandbox_fusion_tool_config.yaml}"
if [[ -n "${SANDBOX_URL:-}" && -z "${TOOL_CONFIG_PATH_EXPLICIT:-}" ]]; then
    RUNTIME_TOOL_CONFIG="$RUN_DIR/sandbox_fusion_tool_config.yaml"
    python3 - "$TOOL_CONFIG_PATH" "$RUNTIME_TOOL_CONFIG" "$SANDBOX_URL" \
        "${SANDBOX_NUM_WORKERS:-}" \
        "${SANDBOX_RATE_LIMIT:-}" \
        "${SANDBOX_DEFAULT_TIMEOUT:-}" \
        "${SANDBOX_MEMORY_LIMIT_MB:-}" <<'PY'
import sys

src, dst, url, num_workers, rate_limit, default_timeout, memory_limit_mb = sys.argv[1:]
overrides = {
    "num_workers": num_workers,
    "rate_limit": rate_limit,
    "default_timeout": default_timeout,
    "memory_limit_mb": memory_limit_mb,
}
overrides = {key: value for key, value in overrides.items() if value}
text = open(src, encoding="utf-8").read()
lines = []
for line in text.splitlines():
    if "sandbox_fusion_url:" in line:
        indent = line.split("sandbox_fusion_url:", 1)[0]
        lines.append(f'{indent}sandbox_fusion_url: "{url}"')
    elif line.lstrip().split(":", 1)[0] in overrides:
        key = line.lstrip().split(":", 1)[0]
        indent = line[: len(line) - len(line.lstrip())]
        lines.append(f"{indent}{key}: {overrides[key]}")
    else:
        lines.append(line)
open(dst, "w", encoding="utf-8").write("\n".join(lines) + "\n")
PY
    TOOL_CONFIG_PATH="$RUNTIME_TOOL_CONFIG"
fi

N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
NNODES="${NNODES:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.55}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
ROLLOUT_N="${ROLLOUT_N:-8}"
VAL_ROLLOUT_N="${VAL_ROLLOUT_N:-8}"
AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-32}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
PPO_MICRO_PER_GPU="${PPO_MICRO_PER_GPU:-2}"
ROLLOUT_LOG_PROB_MICRO_PER_GPU="${ROLLOUT_LOG_PROB_MICRO_PER_GPU:-4}"
REF_MICRO_PER_GPU="${REF_MICRO_PER_GPU:-4}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESP_LENGTH="${MAX_RESP_LENGTH:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_TURNS="${MAX_TURNS:-2}"
MAX_TOOL_RESPONSE_LEN="${MAX_TOOL_RESPONSE_LEN:-4096}"
MAX_PARALLEL_CALLS="${MAX_PARALLEL_CALLS:-1}"
HARNESS_PATH="${HARNESS_PATH:-$PROJECT_DIR/environments/retool/base_harness.py}"
if [[ -n "$HARNESS_PATH" ]]; then
    if [[ "$HARNESS_PATH" != /* ]]; then
        HARNESS_PATH="$PROJECT_DIR/$HARNESS_PATH"
    fi
    [[ -f "$HARNESS_PATH" ]] || {
        echo "[retool_grpo] ERROR: HARNESS_PATH does not exist: $HARNESS_PATH" >&2
        exit 1
    }
    HARNESS_DIR="$(cd "$(dirname "$HARNESS_PATH")" && pwd)"
    HARNESS_PATH="$HARNESS_DIR/$(basename "$HARNESS_PATH")"
    RESOLVED_MAX_TURNS="$(PROJECT_DIR="$PROJECT_DIR" python3 "$PROJECT_DIR/scripts/resolve_harness_max_turns.py" "$HARNESS_PATH")"
    if (( RESOLVED_MAX_TURNS > 16 )); then
        echo "[retool_grpo] HARNESS_PATH MAX_TURNS=$RESOLVED_MAX_TURNS exceeds hard cap 16; clamping"
        RESOLVED_MAX_TURNS=16
    fi
    MAX_TURNS="$RESOLVED_MAX_TURNS"
fi
MAX_TOTAL_RESPONSE_LENGTH="${MAX_TOTAL_RESPONSE_LENGTH:-$MAX_RESP_LENGTH}"
if (( MAX_TOTAL_RESPONSE_LENGTH < MAX_RESP_LENGTH )); then
    echo "[retool_grpo] ERROR: MAX_TOTAL_RESPONSE_LENGTH ($MAX_TOTAL_RESPONSE_LENGTH) must be >= MAX_RESP_LENGTH ($MAX_RESP_LENGTH)" >&2
    exit 1
fi
MIN_DYNAMIC_TOKEN_LEN_PER_GPU=$((MAX_PROMPT_LENGTH + MAX_TOTAL_RESPONSE_LENGTH))
ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU="${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-$MIN_DYNAMIC_TOKEN_LEN_PER_GPU}"
ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-$ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU}"
REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-$ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU}"
for pair in \
    "ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=$ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU" \
    "ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=$ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU" \
    "REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=$REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU"; do
    value="${pair#*=}"
    if (( value < MIN_DYNAMIC_TOKEN_LEN_PER_GPU )); then
        echo "[retool_grpo] ERROR: ${pair%%=*} ($value) must be >= MAX_PROMPT_LENGTH + MAX_TOTAL_RESPONSE_LENGTH ($MIN_DYNAMIC_TOKEN_LEN_PER_GPU) for dynamic batching" >&2
        exit 1
    fi
done
export RETOOL_ASSISTANT_TOKEN_BUDGET="$MAX_TOTAL_RESPONSE_LENGTH"

ACTOR_LR="${ACTOR_LR:-1e-7}"
USE_KL_LOSS="${USE_KL_LOSS:-True}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.01}"
KL_LOSS_TYPE="${KL_LOSS_TYPE:-low_var_kl}"
ENTROPY_COEFF="${ENTROPY_COEFF:-0.001}"
CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.2}"
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.28}"
ACTOR_USE_DYNAMIC_BSZ="${ACTOR_USE_DYNAMIC_BSZ:-False}"
ROLLOUT_LOG_PROB_USE_DYNAMIC_BSZ="${ROLLOUT_LOG_PROB_USE_DYNAMIC_BSZ:-False}"
REF_LOG_PROB_USE_DYNAMIC_BSZ="${REF_LOG_PROB_USE_DYNAMIC_BSZ:-False}"
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-True}"
ENABLE_GRADIENT_CHECKPOINTING="${ENABLE_GRADIENT_CHECKPOINTING:-False}"
FSDP_PARAM_OFFLOAD="${FSDP_PARAM_OFFLOAD:-False}"
FSDP_OPTIM_OFFLOAD="${FSDP_OPTIM_OFFLOAD:-False}"

ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-false}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-false}"
ENFORCE_EAGER="${ENFORCE_EAGER:-true}"
GDN_PREFILL_BACKEND="${GDN_PREFILL_BACKEND:-triton}"
VLLM_DISTRIBUTED_EXECUTOR_BACKEND="${VLLM_DISTRIBUTED_EXECUTOR_BACKEND:-uni}"

ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
ROLLOUT_TOP_K="${ROLLOUT_TOP_K:-20}"

TRAINER_SAVE_FREQ="${TRAINER_SAVE_FREQ:-auto}"
TRAINER_TEST_FREQ="${TRAINER_TEST_FREQ:-auto}"
TRAINER_SAVE_FREQ_EPOCHS="${TRAINER_SAVE_FREQ_EPOCHS:-0.1}"
TRAINER_TEST_FREQ_EPOCHS="${TRAINER_TEST_FREQ_EPOCHS:-0.1}"
TRAINER_MAX_KEEP="${TRAINER_MAX_KEEP:-null}"
TRAINER_TOTAL_EPOCHS="${TRAINER_TOTAL_EPOCHS:-2}"
RESUME_MODE="${RESUME_MODE:-auto}"
TRAINER_VAL_BEFORE_TRAIN="${TRAINER_VAL_BEFORE_TRAIN:-True}"
DATA_SHUFFLE="${DATA_SHUFFLE:-true}"
DATA_SEED="${DATA_SEED:-}"

[[ "$TRAINER_SAVE_FREQ" == "auto" ]] && TRAINER_SAVE_FREQ=""
[[ "$TRAINER_TEST_FREQ" == "auto" ]] && TRAINER_TEST_FREQ=""
EPOCH_RESOLVE_LOG=""
if [[ -z "$TRAINER_SAVE_FREQ" || -z "$TRAINER_TEST_FREQ" ]]; then
    while IFS='=' read -r key value; do
        case "$key" in
            TRAIN_DATASET_ROWS|STEPS_PER_EPOCH)
                EPOCH_RESOLVE_LOG+="$key=$value "
                ;;
            SAVE_FREQ_STEPS)
                [[ -z "$TRAINER_SAVE_FREQ" ]] && TRAINER_SAVE_FREQ="$value"
                ;;
            TEST_FREQ_STEPS)
                [[ -z "$TRAINER_TEST_FREQ" ]] && TRAINER_TEST_FREQ="$value"
                ;;
        esac
    done < <(
        python3 "$PROJECT_DIR/scripts/resolve_epoch_steps.py" \
            --train-files "$TRAIN_FILES_HYDRA" \
            --train-batch-size "$TRAIN_BATCH_SIZE" \
            --save-epochs "$TRAINER_SAVE_FREQ_EPOCHS" \
            --test-epochs "$TRAINER_TEST_FREQ_EPOCHS"
    )
fi

WANDB_PROJECT="${WANDB_PROJECT:-agenticrl_retool_train}"
TRAINER_LOGGER="${TRAINER_LOGGER:-[console,wandb]}"
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    TRAINER_LOGGER="${TRAINER_LOGGER_IF_NO_WANDB:-[console]}"
fi

ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-}"
VAL_DATA_DIR="${VAL_DATA_DIR:-}"
DUMP_ROLLOUTS="${DUMP_ROLLOUTS:-true}"
DUMP_VAL_ROLLOUTS="${DUMP_VAL_ROLLOUTS:-true}"
EXTRA_OVERRIDES=()
if [[ -n "$DATA_SEED" ]]; then
    EXTRA_OVERRIDES+=("data.seed=$DATA_SEED")
fi
if [[ -n "$HARNESS_PATH" ]]; then
    export HARNESS_PATH
    EXTRA_OVERRIDES+=("+ray_kwargs.ray_init.runtime_env.env_vars.HARNESS_PATH=$HARNESS_PATH")
    EXTRA_OVERRIDES+=("+ray_kwargs.ray_init.runtime_env.env_vars.RETOOL_ASSISTANT_TOKEN_BUDGET='$RETOOL_ASSISTANT_TOKEN_BUDGET'")
    if [[ -n "${SANDBOX_URL:-}" ]]; then
        EXTRA_OVERRIDES+=("+ray_kwargs.ray_init.runtime_env.env_vars.SANDBOX_URL=$SANDBOX_URL")
        EXTRA_OVERRIDES+=("+ray_kwargs.ray_init.runtime_env.env_vars.RETOOL_SANDBOX_URL=$SANDBOX_URL")
    fi
fi
if [[ "$DUMP_ROLLOUTS" == "true" ]]; then
    ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-$RUN_DIR/rollouts}"
    mkdir -p "$ROLLOUT_DATA_DIR"
    EXTRA_OVERRIDES+=("trainer.rollout_data_dir=$ROLLOUT_DATA_DIR")
fi
if [[ "$DUMP_VAL_ROLLOUTS" == "true" ]]; then
    VAL_DATA_DIR="${VAL_DATA_DIR:-$RUN_DIR/val_rollouts}"
    mkdir -p "$VAL_DATA_DIR"
    EXTRA_OVERRIDES+=("trainer.validation_data_dir=$VAL_DATA_DIR")
fi

if [[ -n "$GDN_PREFILL_BACKEND" ]]; then
    EXTRA_OVERRIDES+=("+actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend=$GDN_PREFILL_BACKEND")
fi
if [[ -n "$VLLM_DISTRIBUTED_EXECUTOR_BACKEND" ]]; then
    EXTRA_OVERRIDES+=("+actor_rollout_ref.rollout.engine_kwargs.vllm.distributed_executor_backend=$VLLM_DISTRIBUTED_EXECUTOR_BACKEND")
fi
if [[ "$TRAINER_MAX_KEEP" != "null" ]]; then
    EXTRA_OVERRIDES+=("trainer.max_actor_ckpt_to_keep=$TRAINER_MAX_KEEP")
fi

echo "[retool_grpo] starting"
echo "  BASE_MODEL=$BASE_MODEL"
echo "  RUN_DIR=$RUN_DIR"
echo "  TRAIN_FILES=$TRAIN_FILES_HYDRA"
echo "  VAL_FILES=$VAL_FILES_HYDRA"
echo "  TOOL_CONFIG_PATH=$TOOL_CONFIG_PATH"
if [[ -n "${SANDBOX_URL:-}" ]]; then
    echo "  SANDBOX_URL=$SANDBOX_URL"
    echo "  SANDBOX_NUM_WORKERS=${SANDBOX_NUM_WORKERS:-recipe-default} SANDBOX_RATE_LIMIT=${SANDBOX_RATE_LIMIT:-recipe-default}"
fi
echo "  N_GPUS_PER_NODE=$N_GPUS_PER_NODE GPU_MEM_UTIL=$GPU_MEM_UTIL"
echo "  ROLLOUT_N=$ROLLOUT_N VAL_ROLLOUT_N=$VAL_ROLLOUT_N"
echo "  AGENT_NUM_WORKERS=$AGENT_NUM_WORKERS"
echo "  TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE=$PPO_MINI_BATCH_SIZE"
echo "  PPO_MICRO_PER_GPU=$PPO_MICRO_PER_GPU ROLLOUT_LOG_PROB_MICRO_PER_GPU=$ROLLOUT_LOG_PROB_MICRO_PER_GPU REF_MICRO_PER_GPU=$REF_MICRO_PER_GPU"
echo "  TRAINER_TOTAL_EPOCHS=$TRAINER_TOTAL_EPOCHS SAVE_FREQ_STEPS=$TRAINER_SAVE_FREQ TEST_FREQ_STEPS=$TRAINER_TEST_FREQ ${EPOCH_RESOLVE_LOG:-}"
echo "  RESUME_MODE=$RESUME_MODE VAL_BEFORE_TRAIN=$TRAINER_VAL_BEFORE_TRAIN"
echo "  DATA_SHUFFLE=$DATA_SHUFFLE DATA_SEED=${DATA_SEED:-<default>}"
echo "  MAX_PROMPT_LENGTH=$MAX_PROMPT_LENGTH MAX_RESP_LENGTH=$MAX_RESP_LENGTH MAX_TOTAL_RESPONSE_LENGTH=$MAX_TOTAL_RESPONSE_LENGTH MAX_MODEL_LEN=$MAX_MODEL_LEN"
if [[ -n "$HARNESS_PATH" ]]; then
    echo "  HARNESS_PATH=$HARNESS_PATH"
    echo "  HARNESS_MAX_TURNS=$MAX_TURNS"
fi
echo "  MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS"
echo "  dynamic_bsz actor=$ACTOR_USE_DYNAMIC_BSZ rollout_log_prob=$ROLLOUT_LOG_PROB_USE_DYNAMIC_BSZ ref_log_prob=$REF_LOG_PROB_USE_DYNAMIC_BSZ"
echo "  dynamic_token_caps actor=$ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU rollout_log_prob=$ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU ref_log_prob=$REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU"
if [[ -d "$VERL_SOURCE_DIR/verl" ]]; then
    echo "  VERL_SOURCE_DIR=$VERL_SOURCE_DIR"
fi
echo "  vllm_distributed_executor_backend=$VLLM_DISTRIBUTED_EXECUTOR_BACKEND"
echo "  qwen3.5: format=qwen3_coder enable_thinking=true chunked_prefill=$ENABLE_CHUNKED_PREFILL prefix_caching=$ENABLE_PREFIX_CACHING"

REWARD_FUNCTION_PATH="${REWARD_FUNCTION_PATH:-$PROJECT_DIR/environments/retool/grpo_binary_reward.py}"
REWARD_FUNCTION_NAME="${REWARD_FUNCTION_NAME:-compute_score}"
echo "  reward_function=$REWARD_FUNCTION_PATH:$REWARD_FUNCTION_NAME"

export RETOOL_PATCH_VERL_RUNTIME="${RETOOL_PATCH_VERL_RUNTIME:-1}"
PYTHONUNBUFFERED=1 python3 "$PROJECT_DIR/scripts/retool_preamble.py" \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files="$TRAIN_FILES_HYDRA" \
    data.val_files="$VAL_FILES_HYDRA" \
    data.return_raw_chat=True \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.val_batch_size="$TRAIN_BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_TOTAL_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle="$DATA_SHUFFLE" \
    data.custom_cls.path="$PROJECT_DIR/recipe/retool/retool.py" \
    data.custom_cls.name=CustomRLHFDataset \
    data.trust_remote_code=true \
    +data.apply_chat_template_kwargs.enable_thinking=true \
    custom_reward_function.path="$REWARD_FUNCTION_PATH" \
    custom_reward_function.name="$REWARD_FUNCTION_NAME" \
    actor_rollout_ref.model.path="$BASE_MODEL" \
    actor_rollout_ref.model.trust_remote_code=true \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.use_remove_padding="$USE_REMOVE_PADDING" \
    actor_rollout_ref.model.enable_gradient_checkpointing="$ENABLE_GRADIENT_CHECKPOINTING" \
    actor_rollout_ref.actor.use_kl_loss="$USE_KL_LOSS" \
    actor_rollout_ref.actor.kl_loss_coef="$KL_LOSS_COEF" \
    actor_rollout_ref.actor.kl_loss_type="$KL_LOSS_TYPE" \
    actor_rollout_ref.actor.entropy_coeff="$ENTROPY_COEFF" \
    actor_rollout_ref.actor.clip_ratio_low="$CLIP_RATIO_LOW" \
    actor_rollout_ref.actor.clip_ratio_high="$CLIP_RATIO_HIGH" \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.optim.lr="$ACTOR_LR" \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0 \
    actor_rollout_ref.actor.use_dynamic_bsz="$ACTOR_USE_DYNAMIC_BSZ" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU" \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_PER_GPU" \
    actor_rollout_ref.actor.fsdp_config.param_offload="$FSDP_PARAM_OFFLOAD" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="$FSDP_OPTIM_OFFLOAD" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM_UTIL" \
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_NUM_BATCHED_TOKENS" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$ROLLOUT_LOG_PROB_MICRO_PER_GPU" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="$ROLLOUT_LOG_PROB_USE_DYNAMIC_BSZ" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU" \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.do_sample=true \
    actor_rollout_ref.rollout.temperature="$ROLLOUT_TEMPERATURE" \
    actor_rollout_ref.rollout.top_p="$ROLLOUT_TOP_P" \
    actor_rollout_ref.rollout.top_k="$ROLLOUT_TOP_K" \
    actor_rollout_ref.rollout.enforce_eager="$ENFORCE_EAGER" \
    actor_rollout_ref.rollout.enable_chunked_prefill="$ENABLE_CHUNKED_PREFILL" \
    actor_rollout_ref.rollout.enable_prefix_caching="$ENABLE_PREFIX_CACHING" \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns="$MAX_TURNS" \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$MAX_TURNS" \
    actor_rollout_ref.rollout.multi_turn.max_assistant_tokens="$MAX_RESP_LENGTH" \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length="$MAX_TOOL_RESPONSE_LEN" \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls="$MAX_PARALLEL_CALLS" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG_PATH" \
    actor_rollout_ref.rollout.multi_turn.format=qwen3_coder \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.agent.num_workers="$AGENT_NUM_WORKERS" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.temperature="$ROLLOUT_TEMPERATURE" \
    actor_rollout_ref.rollout.val_kwargs.top_p="$ROLLOUT_TOP_P" \
    actor_rollout_ref.rollout.val_kwargs.top_k="$ROLLOUT_TOP_K" \
    actor_rollout_ref.rollout.val_kwargs.n="$VAL_ROLLOUT_N" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$REF_MICRO_PER_GPU" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz="$REF_LOG_PROB_USE_DYNAMIC_BSZ" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU" \
    actor_rollout_ref.ref.fsdp_config.param_offload="$FSDP_PARAM_OFFLOAD" \
    reward.num_workers="${REWARD_NUM_WORKERS:-4}" \
    trainer.logger="$TRAINER_LOGGER" \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
    trainer.nnodes="$NNODES" \
    trainer.val_before_train="$TRAINER_VAL_BEFORE_TRAIN" \
    trainer.log_val_generations="${LOG_VAL_GENERATIONS:-20}" \
    trainer.default_local_dir="$RUN_DIR" \
    trainer.save_freq="$TRAINER_SAVE_FREQ" \
    trainer.test_freq="$TRAINER_TEST_FREQ" \
    trainer.total_epochs="$TRAINER_TOTAL_EPOCHS" \
    trainer.resume_mode="$RESUME_MODE" \
    "${EXTRA_OVERRIDES[@]}" \
    "$@" 2>&1 | tee "$LOG_FILE"
