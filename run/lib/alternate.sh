#!/usr/bin/env bash
# Alternating weight update and harness search.
#
# Sourced by the per-domain launchers in domains/<domain>/run/. Required:
#
#   DOMAIN_DIR      absolute path of domains/<domain>
#   TRAIN_SCRIPT    training driver; receives Hydra overrides on its command line
#   MH_MODULE       python -m target implementing the harness search
#   MH_CONFIG       config JSON passed to the harness search as --config
#   BASE_HARNESS    harness h0 the first cycle starts from
#   RUN_NAME        run identity; also the checkpoint directory name
#   TOTAL_STEPS     total weight-update steps for the run
#   ROUND_STEPS     weight-update steps per cycle (the budget E)
#
# Optional, with the defaults used for the paper:
#   ROUND_MH_ITERS=5            harness-search iterations per cycle; the paper's
#                               I = 6 counts the evaluation of the incoming
#                               harness as iteration 1
#   PROPOSALS_PER_ITER=3        candidate harnesses proposed per iteration (M)
#   PROPOSER_MODEL, PROPOSER_EFFORT, PROPOSER_TIMEOUT
#   ADAPTIVE=1                  end each phase by a patience rule instead
#   MH_MIN_ITERS=5, MH_PATIENCE=2, MH_MAX_ITERS=15
#   MAX_ROUNDS=30               safety cap on the number of cycles
#   RSFT_SFT_EPOCHS=1, RSFT_SFT_MINI_BATCH_SIZE=64,
#   RSFT_SFT_MICRO_PER_GPU=2, RSFT_SCORE_THRESHOLD=0.5
#   CKPT_ROOT                   defaults to domains/<domain>/outputs

set -euo pipefail

: "${DOMAIN_DIR:?DOMAIN_DIR must be set}"
: "${TRAIN_SCRIPT:?TRAIN_SCRIPT must be set}"
: "${MH_MODULE:?MH_MODULE must be set}"
: "${BASE_HARNESS:?BASE_HARNESS must be set}"
: "${RUN_NAME:?RUN_NAME must be set}"
: "${TOTAL_STEPS:?TOTAL_STEPS must be set}"
: "${ROUND_STEPS:?ROUND_STEPS must be set}"

CKPT_ROOT="${CKPT_ROOT:-$DOMAIN_DIR/outputs}"
MH_CONFIG="${MH_CONFIG:-}"
ROUND_MH_ITERS="${ROUND_MH_ITERS:-5}"
PROPOSALS_PER_ITER="${PROPOSALS_PER_ITER:-3}"
PROPOSER_MODEL="${PROPOSER_MODEL:-claude-opus-4-7}"
PROPOSER_EFFORT="${PROPOSER_EFFORT:-max}"
PROPOSER_TIMEOUT="${PROPOSER_TIMEOUT:-2400}"

ADAPTIVE="${ADAPTIVE:-0}"
MH_MIN_ITERS="${MH_MIN_ITERS:-5}"
MH_PATIENCE="${MH_PATIENCE:-2}"
MH_MAX_ITERS="${MH_MAX_ITERS:-15}"
MAX_ROUNDS="${MAX_ROUNDS:-30}"

RSFT_SFT_EPOCHS="${RSFT_SFT_EPOCHS:-1}"
RSFT_SFT_MINI_BATCH_SIZE="${RSFT_SFT_MINI_BATCH_SIZE:-64}"
RSFT_SFT_MICRO_PER_GPU="${RSFT_SFT_MICRO_PER_GPU:-2}"
RSFT_SCORE_THRESHOLD="${RSFT_SCORE_THRESHOLD:-0.5}"

RUN_DIR="$CKPT_ROOT/$RUN_NAME"
MH_RUNS_ROOT="$DOMAIN_DIR/meta_harness/runs"
LOG="$RUN_DIR/alternation.log"
mkdir -p "$RUN_DIR" "$MH_RUNS_ROOT"

# Log to stderr so that a function can still return a value on stdout.
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG" >&2; }

# Weight-update phase. verl is told where to stop with a Hydra override, and the
# harness is passed both as an env var and through the Ray runtime env so the
# rollout workers see it.
run_weight_phase() {
  local harness="$1" target="$2"
  [[ -f "$harness" ]] || { log "ERROR: harness not found: $harness"; exit 3; }
  log "  weight-update phase -> step $target (harness=$harness)"
  (
    export HARNESS_PATH="$harness"
    export RUN_NAME
    bash "$TRAIN_SCRIPT" \
      "+ray_kwargs.ray_init.runtime_env.env_vars.HARNESS_PATH=$harness" \
      trainer.total_training_steps="$target" \
      +trainer.online_rsft.enable=True \
      +trainer.online_rsft.sft_epochs="$RSFT_SFT_EPOCHS" \
      +trainer.online_rsft.sft_mini_batch_size="$RSFT_SFT_MINI_BATCH_SIZE" \
      +trainer.online_rsft.sft_micro_batch_size_per_gpu="$RSFT_SFT_MICRO_PER_GPU" \
      +trainer.online_rsft.score_threshold="$RSFT_SCORE_THRESHOLD"
  )
}

# Harness-search phase. The incoming harness becomes h0 of this search and the
# checkpoint just produced becomes the model the candidates are scored against.
# Prints the accepted harness path on stdout.
run_harness_phase() {
  local round="$1" harness_in="$2" ckpt_dir="$3"
  local mh_run="mh-$RUN_NAME-round$round"
  local mh_dir="$MH_RUNS_ROOT/$mh_run"
  local accepted="$mh_dir/logs/accepted_harness.txt"

  if [[ -f "$accepted" && -f "$mh_dir/harnesses/$(cat "$accepted")/harness.py" ]]; then
    log "  harness-search phase $round already complete, reusing $(cat "$accepted")"
  else
    local flags=(--run-name "$mh_run"
                 --proposals-per-iter "$PROPOSALS_PER_ITER"
                 --proposer-model "$PROPOSER_MODEL"
                 --proposer-effort "$PROPOSER_EFFORT"
                 --propose-timeout "$PROPOSER_TIMEOUT")
    [[ -n "$MH_CONFIG" ]] && flags+=(--config "$MH_CONFIG")
    if [[ "$ADAPTIVE" == "1" ]]; then
      flags+=(--iterations "$MH_MAX_ITERS"
              --early-stop-min-iters "$MH_MIN_ITERS"
              --early-stop-patience "$MH_PATIENCE")
    else
      flags+=(--iterations "$ROUND_MH_ITERS")
    fi

    log "  harness-search phase $round (h0=$harness_in, model=$ckpt_dir)"
    (
      cd "$DOMAIN_DIR"
      export BASELINE_HARNESS_OVERRIDE="$harness_in"
      export VLLM_MODEL="$ckpt_dir"
      export RUN_NAME="$mh_run"
      export ITERATIONS="$ROUND_MH_ITERS"
      PYTHONUNBUFFERED=1 python -m "$MH_MODULE" "${flags[@]}" >&2
    )
  fi

  [[ -f "$accepted" ]] || { log "ERROR: no accepted harness in $accepted"; exit 4; }
  local winner winner_path
  winner="$(cat "$accepted")"
  winner_path="$mh_dir/harnesses/$winner/harness.py"
  [[ -f "$winner_path" ]] || { log "ERROR: accepted harness missing: $winner_path"; exit 4; }
  log "  accepted harness: $winner"
  printf '%s' "$winner_path"
}

main() {
  local harness="$BASE_HARNESS"
  local cumulative=0
  local round=0

  log "=== $RUN_NAME | total $TOTAL_STEPS steps | E=$ROUND_STEPS steps | I=$((ROUND_MH_ITERS + 1)) | adaptive=$ADAPTIVE ==="

  while (( cumulative < TOTAL_STEPS && round < MAX_ROUNDS )); do
    round=$(( round + 1 ))
    cumulative=$(( cumulative + ROUND_STEPS ))
    (( cumulative > TOTAL_STEPS )) && cumulative=$TOTAL_STEPS
    local ckpt="$RUN_DIR/global_step_$cumulative"

    log "--- cycle $round (target step $cumulative) ---"
    if [[ -d "$ckpt/actor" ]]; then
      log "  checkpoint already present, skipping the weight-update phase"
    else
      run_weight_phase "$harness" "$cumulative"
      [[ -d "$ckpt/actor" ]] || { log "ERROR: expected checkpoint $ckpt/actor"; exit 3; }
    fi

    if (( cumulative >= TOTAL_STEPS )); then
      log "  budget exhausted after the weight-update phase; run complete"
      break
    fi

    harness="$(run_harness_phase "$round" "$harness" "$ckpt")"
  done

  log "=== done: $round cycles, final harness $harness ==="
}

main "$@"
