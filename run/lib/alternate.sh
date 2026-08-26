#!/usr/bin/env bash
# Alternating weight update and harness search.
#
# Sourced by the per-domain launchers in domains/<domain>/run/. The caller must
# define:
#
#   DOMAIN_DIR          absolute path of domains/<domain>
#   TRAIN_SCRIPT        training driver invoked for one weight-update phase
#   MH_MODULE           python -m target implementing the harness search
#   BASE_HARNESS        harness the first cycle starts from
#   RUN_NAME            name of this run; also the checkpoint directory name
#   CKPT_ROOT           where weight-update checkpoints are written
#   TOTAL_STEPS         total weight-update steps across the run
#   ROUND_STEPS         weight-update steps per cycle (the budget E)
#   ROUND_MH_ITERS      harness-search iterations per cycle (I minus the
#                       evaluation of the incoming harness)
#
# Optional:
#   ADAPTIVE=1                 stop each phase by a patience rule instead of the
#                              fixed budgets above
#   ES_WINDOW, ES_PATIENCE     weight-update phase rule, in steps
#   MH_MIN_ITERS, MH_PATIENCE  harness-search phase rule, in iterations
#   MH_MAX_ITERS               cap on a single harness-search phase
#   MAX_ROUNDS                 cap on the number of cycles
#   PROPOSALS_PER_ITER         candidate harnesses proposed per iteration (M)

set -euo pipefail

: "${DOMAIN_DIR:?DOMAIN_DIR must be set}"
: "${TRAIN_SCRIPT:?TRAIN_SCRIPT must be set}"
: "${MH_MODULE:?MH_MODULE must be set}"
: "${BASE_HARNESS:?BASE_HARNESS must be set}"
: "${RUN_NAME:?RUN_NAME must be set}"

CKPT_ROOT="${CKPT_ROOT:-$DOMAIN_DIR/outputs}"
TOTAL_STEPS="${TOTAL_STEPS:?TOTAL_STEPS must be set}"
ROUND_STEPS="${ROUND_STEPS:?ROUND_STEPS must be set}"
ROUND_MH_ITERS="${ROUND_MH_ITERS:-5}"
PROPOSALS_PER_ITER="${PROPOSALS_PER_ITER:-3}"

ADAPTIVE="${ADAPTIVE:-0}"
MH_MIN_ITERS="${MH_MIN_ITERS:-5}"
MH_PATIENCE="${MH_PATIENCE:-2}"
MH_MAX_ITERS="${MH_MAX_ITERS:-15}"
MAX_ROUNDS="${MAX_ROUNDS:-30}"

RUN_DIR="$CKPT_ROOT/$RUN_NAME"
MH_RUNS_ROOT="$DOMAIN_DIR/meta_harness/runs"
LOG="$RUN_DIR/alternation.log"
mkdir -p "$RUN_DIR" "$MH_RUNS_ROOT"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# Weight-update phase: train from the current harness up to a cumulative step.
run_weight_phase() {
  local harness="$1" target="$2"
  [[ -f "$harness" ]] || { log "ERROR: harness not found: $harness"; exit 3; }
  log "  weight-update phase -> step $target (harness=$harness)"
  HARNESS_PATH="$harness" \
  RUN_NAME="$RUN_NAME" \
  TRAINER_TOTAL_STEPS="$target" \
    bash "$TRAIN_SCRIPT"
}

# Harness search phase: propose, evaluate and accept, then report the winner.
run_harness_phase() {
  local round="$1" harness_in="$2"
  local mh_run="mh-$RUN_NAME-round$round"
  local mh_dir="$MH_RUNS_ROOT/$mh_run"
  local flags=(
    --run-name "$mh_run"
    --proposals-per-iter "$PROPOSALS_PER_ITER"
  )
  if [[ "$ADAPTIVE" == "1" ]]; then
    flags+=(--iterations "$MH_MAX_ITERS"
            --early-stop-min-iters "$MH_MIN_ITERS"
            --early-stop-patience "$MH_PATIENCE")
  else
    flags+=(--iterations "$ROUND_MH_ITERS")
  fi

  log "  harness-search phase (run=$mh_run, incoming harness=$harness_in)"
  ( cd "$DOMAIN_DIR" && BASE_HARNESS="$harness_in" \
      python -m "$MH_MODULE" "${flags[@]}" )

  local accepted="$mh_dir/logs/accepted_harness.txt"
  [[ -f "$accepted" ]] || { log "ERROR: no accepted harness recorded in $accepted"; exit 4; }
  local winner
  winner="$(cat "$accepted")"
  local winner_path="$mh_dir/harnesses/$winner/harness.py"
  [[ -f "$winner_path" ]] || { log "ERROR: accepted harness missing: $winner_path"; exit 4; }
  log "  accepted harness: $winner"
  printf '%s' "$winner_path"
}

main() {
  local harness="$BASE_HARNESS"
  local cumulative=0
  local round=0

  log "=== $RUN_NAME | total steps $TOTAL_STEPS | E=$ROUND_STEPS steps | adaptive=$ADAPTIVE ==="

  while (( cumulative < TOTAL_STEPS && round < MAX_ROUNDS )); do
    round=$(( round + 1 ))
    cumulative=$(( cumulative + ROUND_STEPS ))
    (( cumulative > TOTAL_STEPS )) && cumulative=$TOTAL_STEPS

    log "--- cycle $round ---"
    if [[ -d "$RUN_DIR/global_step_$cumulative/actor" ]]; then
      log "  checkpoint for step $cumulative already present, skipping the weight-update phase"
    else
      run_weight_phase "$harness" "$cumulative"
      [[ -d "$RUN_DIR/global_step_$cumulative/actor" ]] || {
        log "ERROR: expected checkpoint $RUN_DIR/global_step_$cumulative/actor"; exit 3; }
    fi

    if (( cumulative >= TOTAL_STEPS )); then
      log "  budget exhausted after the weight-update phase; run complete"
      break
    fi

    harness="$(run_harness_phase "$round" "$harness")"
  done

  log "=== done: $round cycles, final harness $harness ==="
}

main "$@"
