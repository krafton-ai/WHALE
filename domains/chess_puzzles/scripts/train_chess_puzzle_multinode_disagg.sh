#!/usr/bin/env bash
# chess-puzzle-v0 GRPO/online-RSFT with disaggregated FSDP trainer and standalone vLLM rollout.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"

LOCAL_DEPS="${CHESS_PUZZLE_DEPS_DIR:-$PROJECT_DIR/.deps-chess-puzzle}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-}"
BASE_PYTHONPATH="$PROJECT_DIR"
if [[ -d "$LOCAL_DEPS" ]]; then
    BASE_PYTHONPATH="$BASE_PYTHONPATH:$LOCAL_DEPS"
fi
if [[ -n "$EXTRA_PYTHONPATH" && -d "$EXTRA_PYTHONPATH" ]]; then
    export PYTHONPATH="$BASE_PYTHONPATH:$EXTRA_PYTHONPATH:${PYTHONPATH:-}"
else
    export PYTHONPATH="$BASE_PYTHONPATH:${PYTHONPATH:-}"
fi
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"
export VLLM_USE_DEEP_GEMM_E8M0="${VLLM_USE_DEEP_GEMM_E8M0:-0}"
export VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES="${VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES:-0}"
export VERL_VLLM_GENERATE_TIMEOUT_S="${VERL_VLLM_GENERATE_TIMEOUT_S:-600}"
export VERL_TASK_RUNNER_START_TIMEOUT_S="${VERL_TASK_RUNNER_START_TIMEOUT_S:-1200}"
export RAY_COLLECTIVE_NCCL_ID_TIMEOUT_S="${RAY_COLLECTIVE_NCCL_ID_TIMEOUT_S:-1200}"
export VERL_RAY_COLLECTIVE_NCCL_ID_TIMEOUT_S="${VERL_RAY_COLLECTIVE_NCCL_ID_TIMEOUT_S:-$RAY_COLLECTIVE_NCCL_ID_TIMEOUT_S}"
export RAY_worker_register_timeout_seconds="${RAY_worker_register_timeout_seconds:-300}"
export VERL_AGENT_LOOP_WORKER_STARTUP_BATCH_SIZE="${VERL_AGENT_LOOP_WORKER_STARTUP_BATCH_SIZE:-16}"
export VERL_AGENT_LOOP_WORKER_STARTUP_TIMEOUT_S="${VERL_AGENT_LOOP_WORKER_STARTUP_TIMEOUT_S:-300}"
RAY_WAIT_REGISTER_CENTER_TIMEOUT_S="${RAY_WAIT_REGISTER_CENTER_TIMEOUT_S:-1200}"

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

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-4B}"
CHESS_PUZZLE_ALGO="${CHESS_PUZZLE_ALGO:-grpo}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"
RUN_NAME="${RUN_NAME:-chess-puzzle-${CHESS_PUZZLE_ALGO}-disagg-zero1-qwen3.5-4b-r16384-bs256-2n-tr1x4-roll2x6-${RUN_STAMP}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-$RUN_NAME}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs}"
RUN_DIR="$OUTPUT_ROOT/$RUN_NAME"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$RUN_DIR" "$LOG_DIR"
LOG_FILE="$LOG_DIR/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

TRAIN_FILES="${TRAIN_FILES:-$PROJECT_DIR/data/chess_puzzle/lichess_puzzles_train_16384.parquet}"
VAL_FILES="${VAL_FILES:-$PROJECT_DIR/data/chess_puzzle/lichess_puzzles_test_256.parquet}"
TRAIN_FILES_HYDRA="${TRAIN_FILES_HYDRA:-$(as_hydra_list "$TRAIN_FILES")}"
VAL_FILES_HYDRA="${VAL_FILES_HYDRA:-$(as_hydra_list "$VAL_FILES")}"
for path in "$TRAIN_FILES" "$VAL_FILES"; do
    if [[ "$path" != *,* && "$path" != \[* && ! -f "$path" ]]; then
        echo "[chess_puzzle_disagg] ERROR: missing parquet: $path" >&2
        exit 1
    fi
done

MH_VAL_FILES_FOR_DEFAULT="${MH_VAL_FILES:-$PROJECT_DIR/data/chess_puzzle/lichess_puzzles_mh_val_256.parquet}"
DEFAULT_MAX_TURNS="${CHESS_PUZZLE_DEFAULT_MAX_TURNS:-}"
if [[ -z "$DEFAULT_MAX_TURNS" ]]; then
    DEFAULT_MAX_TURNS="$(python3 "$PROJECT_DIR/scripts/chess_puzzle_max_turns.py" \
        "$TRAIN_FILES" "$VAL_FILES" "$MH_VAL_FILES_FOR_DEFAULT" --fallback 9 2>/dev/null || true)"
fi
DEFAULT_MAX_TURNS="${DEFAULT_MAX_TURNS:-9}"

# Trainer: 1 node x 4 GPU. Standalone rollout/vLLM: 2 nodes x 6 GPU.
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
NNODES="${NNODES:-1}"
ROLLOUT_NNODES="${ROLLOUT_NNODES:-2}"
ROLLOUT_GPUS_PER_NODE="${ROLLOUT_GPUS_PER_NODE:-6}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
ROLLOUT_UPDATE_WEIGHTS_BUCKET_MB="${ROLLOUT_UPDATE_WEIGHTS_BUCKET_MB:-3072}"

ROLLOUT_N="${ROLLOUT_N:-8}"
VAL_ROLLOUT_N="${VAL_ROLLOUT_N:-8}"
AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-128}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
PPO_MICRO_PER_GPU="${PPO_MICRO_PER_GPU:-1}"
ROLLOUT_LOG_PROB_MICRO_PER_GPU="${ROLLOUT_LOG_PROB_MICRO_PER_GPU:-1}"
REF_MICRO_PER_GPU="${REF_MICRO_PER_GPU:-1}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_TOTAL_RESPONSE_LENGTH="${MAX_TOTAL_RESPONSE_LENGTH:-16384}"
ASSISTANT_TOKEN_BUDGET="${ASSISTANT_TOKEN_BUDGET:-8129}"
POLICY_MAX_TOKENS="${POLICY_MAX_TOKENS:-$ASSISTANT_TOKEN_BUDGET}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_TURNS="${MAX_TURNS:-$DEFAULT_MAX_TURNS}"
MAX_TURNS_CAP="${MAX_TURNS_CAP:-$((DEFAULT_MAX_TURNS * 2))}"
CHESS_PUZZLE_ENABLE_THINKING="${CHESS_PUZZLE_ENABLE_THINKING:-true}"
FORMAT_RETRIES="${FORMAT_RETRIES:-1}"
ILLEGAL_RETRIES="${ILLEGAL_RETRIES:-1}"

HARNESS_PATH="${HARNESS_PATH:-$PROJECT_DIR/environments/chess_puzzle/base_harness.py}"
if [[ "$HARNESS_PATH" != /* ]]; then
    HARNESS_PATH="$PROJECT_DIR/$HARNESS_PATH"
fi
[[ -f "$HARNESS_PATH" ]] || {
    echo "[chess_puzzle_disagg] ERROR: HARNESS_PATH does not exist: $HARNESS_PATH" >&2
    exit 1
}
export HARNESS_PATH
export CHESS_PUZZLE_HARNESS_PATH="$HARNESS_PATH"
export CHESS_PUZZLE_ASSISTANT_TOKEN_BUDGET="$ASSISTANT_TOKEN_BUDGET"
export CHESS_PUZZLE_POLICY_MAX_TOKENS="$POLICY_MAX_TOKENS"
export CHESS_PUZZLE_FORMAT_RETRIES="$FORMAT_RETRIES"
export CHESS_PUZZLE_ILLEGAL_RETRIES="$ILLEGAL_RETRIES"
export CHESS_PUZZLE_DEFAULT_MAX_TURNS="$DEFAULT_MAX_TURNS"
export CHESS_PUZZLE_MAX_TURNS_CAP="$MAX_TURNS_CAP"

MIN_DYNAMIC_TOKEN_LEN_PER_GPU=$((MAX_PROMPT_LENGTH + MAX_TOTAL_RESPONSE_LENGTH))
ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU="${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-$MIN_DYNAMIC_TOKEN_LEN_PER_GPU}"
ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-$ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU}"
REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-$ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU}"

if [[ "$CHESS_PUZZLE_ALGO" == "rsft" ]]; then
    ACTOR_LR="${ACTOR_LR:-1e-7}"
    USE_KL_LOSS="${USE_KL_LOSS:-False}"
    KL_LOSS_COEF="${KL_LOSS_COEF:-0.0}"
    ENTROPY_COEFF="${ENTROPY_COEFF:-0.0}"
    ENABLE_ONLINE_RSFT="${ENABLE_ONLINE_RSFT:-true}"
else
    ACTOR_LR="${ACTOR_LR:-1e-7}"
    USE_KL_LOSS="${USE_KL_LOSS:-True}"
    KL_LOSS_COEF="${KL_LOSS_COEF:-0.01}"
    ENTROPY_COEFF="${ENTROPY_COEFF:-0.001}"
    ENABLE_ONLINE_RSFT="${ENABLE_ONLINE_RSFT:-false}"
fi
KL_LOSS_TYPE="${KL_LOSS_TYPE:-low_var_kl}"
CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.2}"
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.28}"
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-True}"
ENABLE_GRADIENT_CHECKPOINTING="${ENABLE_GRADIENT_CHECKPOINTING:-True}"
FSDP_PARAM_OFFLOAD="${FSDP_PARAM_OFFLOAD:-False}"
FSDP_OPTIM_OFFLOAD="${FSDP_OPTIM_OFFLOAD:-False}"
ACTOR_USE_DYNAMIC_BSZ="${ACTOR_USE_DYNAMIC_BSZ:-False}"
ROLLOUT_LOG_PROB_USE_DYNAMIC_BSZ="${ROLLOUT_LOG_PROB_USE_DYNAMIC_BSZ:-False}"
REF_LOG_PROB_USE_DYNAMIC_BSZ="${REF_LOG_PROB_USE_DYNAMIC_BSZ:-False}"

ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-false}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-true}"
ENFORCE_EAGER="${ENFORCE_EAGER:-true}"
GDN_PREFILL_BACKEND="${GDN_PREFILL_BACKEND:-triton}"
VLLM_DISTRIBUTED_EXECUTOR_BACKEND="${VLLM_DISTRIBUTED_EXECUTOR_BACKEND:-uni}"

ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
ROLLOUT_TOP_K="${ROLLOUT_TOP_K:-20}"

TRAINER_SAVE_FREQ="${TRAINER_SAVE_FREQ:-7}"
TRAINER_TEST_FREQ="${TRAINER_TEST_FREQ:-7}"
TRAINER_MAX_KEEP="${TRAINER_MAX_KEEP:-null}"
TRAINER_TOTAL_EPOCHS="${TRAINER_TOTAL_EPOCHS:-4}"
RESUME_MODE="${RESUME_MODE:-auto}"
TRAINER_VAL_BEFORE_TRAIN="${TRAINER_VAL_BEFORE_TRAIN:-True}"
DATA_SHUFFLE="${DATA_SHUFFLE:-true}"
DATA_SEED="${DATA_SEED:-42}"
WANDB_PROJECT="${WANDB_PROJECT:-agenticrl_chess_puzzle_train}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
TRAINER_LOGGER="${TRAINER_LOGGER:-[console,wandb]}"
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    TRAINER_LOGGER="${TRAINER_LOGGER_IF_NO_WANDB:-[console]}"
fi
export WANDB_PROJECT
if [[ -n "$WANDB_ENTITY" ]]; then
    export WANDB_ENTITY
fi

RSFT_SFT_EPOCHS="${RSFT_SFT_EPOCHS:-1}"
RSFT_SFT_MINI_BATCH_SIZE="${RSFT_SFT_MINI_BATCH_SIZE:-64}"
RSFT_SFT_MICRO_PER_GPU="${RSFT_SFT_MICRO_PER_GPU:-1}"
RSFT_SCORE_THRESHOLD="${RSFT_SCORE_THRESHOLD:-0.5}"

ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-}"
VAL_DATA_DIR="${VAL_DATA_DIR:-}"
DUMP_ROLLOUTS="${DUMP_ROLLOUTS:-true}"
DUMP_VAL_ROLLOUTS="${DUMP_VAL_ROLLOUTS:-true}"

EXTRA_OVERRIDES=()
EXTRA_OVERRIDES+=("data.seed=$DATA_SEED")
EXTRA_OVERRIDES+=("+ray_kwargs.ray_init.runtime_env.env_vars.CHESS_PUZZLE_HARNESS_PATH=$CHESS_PUZZLE_HARNESS_PATH")
EXTRA_OVERRIDES+=("+ray_kwargs.ray_init.runtime_env.env_vars.CHESS_PUZZLE_ASSISTANT_TOKEN_BUDGET='$CHESS_PUZZLE_ASSISTANT_TOKEN_BUDGET'")
EXTRA_OVERRIDES+=("+ray_kwargs.ray_init.runtime_env.env_vars.CHESS_PUZZLE_POLICY_MAX_TOKENS='$CHESS_PUZZLE_POLICY_MAX_TOKENS'")
EXTRA_OVERRIDES+=("+ray_kwargs.ray_init.runtime_env.env_vars.CHESS_PUZZLE_FORMAT_RETRIES='$CHESS_PUZZLE_FORMAT_RETRIES'")
EXTRA_OVERRIDES+=("+ray_kwargs.ray_init.runtime_env.env_vars.CHESS_PUZZLE_ILLEGAL_RETRIES='$CHESS_PUZZLE_ILLEGAL_RETRIES'")
EXTRA_OVERRIDES+=("+ray_kwargs.ray_init.runtime_env.env_vars.CHESS_PUZZLE_DEFAULT_MAX_TURNS='$CHESS_PUZZLE_DEFAULT_MAX_TURNS'")
EXTRA_OVERRIDES+=("+ray_kwargs.ray_init.runtime_env.env_vars.CHESS_PUZZLE_MAX_TURNS_CAP='$CHESS_PUZZLE_MAX_TURNS_CAP'")
EXTRA_OVERRIDES+=("+ray_kwargs.ray_init.runtime_env.env_vars.WANDB_PROJECT='$WANDB_PROJECT'")
EXTRA_OVERRIDES+=("+ray_kwargs.ray_init.runtime_env.env_vars.VERL_VLLM_GENERATE_TIMEOUT_S='$VERL_VLLM_GENERATE_TIMEOUT_S'")
EXTRA_OVERRIDES+=("+ray_kwargs.ray_init.runtime_env.env_vars.VERL_TASK_RUNNER_START_TIMEOUT_S='$VERL_TASK_RUNNER_START_TIMEOUT_S'")
EXTRA_OVERRIDES+=("+ray_kwargs.ray_init.runtime_env.env_vars.RAY_COLLECTIVE_NCCL_ID_TIMEOUT_S='$RAY_COLLECTIVE_NCCL_ID_TIMEOUT_S'")
EXTRA_OVERRIDES+=("+ray_kwargs.ray_init.runtime_env.env_vars.VERL_RAY_COLLECTIVE_NCCL_ID_TIMEOUT_S='$VERL_RAY_COLLECTIVE_NCCL_ID_TIMEOUT_S'")
EXTRA_OVERRIDES+=("+ray_kwargs.ray_init.runtime_env.env_vars.RAY_worker_register_timeout_seconds='$RAY_worker_register_timeout_seconds'")
EXTRA_OVERRIDES+=("+ray_kwargs.ray_init.runtime_env.env_vars.VERL_AGENT_LOOP_WORKER_STARTUP_BATCH_SIZE='$VERL_AGENT_LOOP_WORKER_STARTUP_BATCH_SIZE'")
EXTRA_OVERRIDES+=("+ray_kwargs.ray_init.runtime_env.env_vars.VERL_AGENT_LOOP_WORKER_STARTUP_TIMEOUT_S='$VERL_AGENT_LOOP_WORKER_STARTUP_TIMEOUT_S'")
EXTRA_OVERRIDES+=("+ray_kwargs.ray_init.runtime_env.env_vars.HF_HOME='${HF_HOME:-$PROJECT_DIR/.cache/huggingface}'")
EXTRA_OVERRIDES+=("+ray_kwargs.ray_init.runtime_env.env_vars.TRANSFORMERS_OFFLINE='${TRANSFORMERS_OFFLINE:-1}'")
EXTRA_OVERRIDES+=("+ray_kwargs.ray_init.runtime_env.env_vars.HF_HUB_OFFLINE='${HF_HUB_OFFLINE:-1}'")
EXTRA_OVERRIDES+=("+ray_kwargs.ray_init.runtime_env.env_vars.HF_DATASETS_OFFLINE='${HF_DATASETS_OFFLINE:-1}'")
EXTRA_OVERRIDES+=("+ray_kwargs.ray_init.runtime_env.env_vars.HF_HUB_DISABLE_TELEMETRY='${HF_HUB_DISABLE_TELEMETRY:-1}'")
EXTRA_OVERRIDES+=("+ray_kwargs.ray_init.runtime_env.env_vars.ROCR_VISIBLE_DEVICES=")
EXTRA_OVERRIDES+=("+ray_kwargs.ray_init.runtime_env.env_vars.HIP_VISIBLE_DEVICES=")
if [[ -n "${CHESS_PUZZLE_RSFT_WORKER_TRACEBACK_S:-}" ]]; then
    EXTRA_OVERRIDES+=("+ray_kwargs.ray_init.runtime_env.env_vars.CHESS_PUZZLE_RSFT_WORKER_TRACEBACK_S='${CHESS_PUZZLE_RSFT_WORKER_TRACEBACK_S}'")
fi
if [[ -n "$WANDB_ENTITY" ]]; then
    EXTRA_OVERRIDES+=("+ray_kwargs.ray_init.runtime_env.env_vars.WANDB_ENTITY='$WANDB_ENTITY'")
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
if [[ "$ENABLE_ONLINE_RSFT" == "true" ]]; then
    EXTRA_OVERRIDES+=("+trainer.online_rsft.enable=True")
    EXTRA_OVERRIDES+=("+trainer.online_rsft.sft_epochs=$RSFT_SFT_EPOCHS")
    EXTRA_OVERRIDES+=("+trainer.online_rsft.sft_mini_batch_size=$RSFT_SFT_MINI_BATCH_SIZE")
    EXTRA_OVERRIDES+=("+trainer.online_rsft.sft_micro_batch_size_per_gpu=$RSFT_SFT_MICRO_PER_GPU")
    EXTRA_OVERRIDES+=("+trainer.online_rsft.score_threshold=$RSFT_SCORE_THRESHOLD")
fi

echo "[chess_puzzle_disagg] starting"
echo "  algo=$CHESS_PUZZLE_ALGO BASE_MODEL=$BASE_MODEL"
echo "  RUN_DIR=$RUN_DIR"
echo "  TRAIN_FILES=$TRAIN_FILES_HYDRA"
echo "  VAL_FILES=$VAL_FILES_HYDRA"
echo "  HARNESS_PATH=$HARNESS_PATH"
echo "  trainer=${NNODES}x${N_GPUS_PER_NODE} rollout=${ROLLOUT_NNODES}x${ROLLOUT_GPUS_PER_NODE} GPU_MEM_UTIL=$GPU_MEM_UTIL"
echo "  trainer_zero_stage=1 optimizer_state_sharding_only=true"
echo "  ROLLOUT_N=$ROLLOUT_N VAL_ROLLOUT_N=$VAL_ROLLOUT_N AGENT_NUM_WORKERS=$AGENT_NUM_WORKERS"
echo "  TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE=$PPO_MINI_BATCH_SIZE SFT_MINI=$RSFT_SFT_MINI_BATCH_SIZE SFT_MICRO=$RSFT_SFT_MICRO_PER_GPU"
echo "  MAX_TOTAL_RESPONSE_LENGTH=$MAX_TOTAL_RESPONSE_LENGTH ASSISTANT_TOKEN_BUDGET=$ASSISTANT_TOKEN_BUDGET POLICY_MAX_TOKENS=$POLICY_MAX_TOKENS MAX_MODEL_LEN=$MAX_MODEL_LEN"
echo "  MAX_TURNS=$MAX_TURNS default_max_turns=$DEFAULT_MAX_TURNS max_turns_cap=$MAX_TURNS_CAP format_retries=$FORMAT_RETRIES illegal_retries=$ILLEGAL_RETRIES thinking=$CHESS_PUZZLE_ENABLE_THINKING"
echo "  prefix_caching=$ENABLE_PREFIX_CACHING chunked_prefill=$ENABLE_CHUNKED_PREFILL max_batched_tokens=$MAX_NUM_BATCHED_TOKENS max_num_seqs=$MAX_NUM_SEQS"
echo "  save_freq=$TRAINER_SAVE_FREQ test_freq=$TRAINER_TEST_FREQ total_epochs=$TRAINER_TOTAL_EPOCHS"
echo "  ray_wait_register_center_timeout=$RAY_WAIT_REGISTER_CENTER_TIMEOUT_S task_runner_timeout=$VERL_TASK_RUNNER_START_TIMEOUT_S ray_collective_nccl_id_timeout=$VERL_RAY_COLLECTIVE_NCCL_ID_TIMEOUT_S"
echo "  ray_worker_register_timeout=$RAY_worker_register_timeout_seconds agent_worker_startup_batch=$VERL_AGENT_LOOP_WORKER_STARTUP_BATCH_SIZE agent_worker_startup_timeout=$VERL_AGENT_LOOP_WORKER_STARTUP_TIMEOUT_S"
echo "  logger=$TRAINER_LOGGER project=$WANDB_PROJECT"
if [[ -n "$WANDB_ENTITY" ]]; then
    echo "  wandb_entity=$WANDB_ENTITY"
fi

PYTHONUNBUFFERED=1 python3 "$PROJECT_DIR/scripts/chess_puzzle_preamble_disagg_rsft.py" \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    critic.enable=False \
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
    data.trust_remote_code=true \
    +data.apply_chat_template_kwargs.enable_thinking="$CHESS_PUZZLE_ENABLE_THINKING" \
    custom_reward_function.path="$PROJECT_DIR/environments/chess_puzzle/grpo_reward.py" \
    custom_reward_function.name=compute_score \
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
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=False \
    actor_rollout_ref.actor.fsdp_config.param_offload="$FSDP_PARAM_OFFLOAD" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="$FSDP_OPTIM_OFFLOAD" \
    actor_rollout_ref.actor.checkpoint.save_contents=[model,extra] \
    actor_rollout_ref.actor.checkpoint.load_contents=[model,extra] \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.nnodes="$ROLLOUT_NNODES" \
    actor_rollout_ref.rollout.n_gpus_per_node="$ROLLOUT_GPUS_PER_NODE" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.data_parallel_size=1 \
    actor_rollout_ref.rollout.pipeline_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM_UTIL" \
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_NUM_BATCHED_TOKENS" \
    actor_rollout_ref.rollout.max_num_seqs="$MAX_NUM_SEQS" \
    actor_rollout_ref.rollout.load_format=auto \
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="$ROLLOUT_UPDATE_WEIGHTS_BUCKET_MB" \
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
    actor_rollout_ref.rollout.multi_turn.max_assistant_tokens="$ASSISTANT_TOKEN_BUDGET" \
    actor_rollout_ref.rollout.agent.default_agent_loop=chess_puzzle_agent \
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
    reward.num_workers="${REWARD_NUM_WORKERS:-8}" \
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
    trainer.ray_wait_register_center_timeout="$RAY_WAIT_REGISTER_CENTER_TIMEOUT_S" \
    +trainer.disaggregated_rsft.zero_stage=1 \
    "${EXTRA_OVERRIDES[@]}" \
    "$@" 2>&1 | tee "$LOG_FILE"
