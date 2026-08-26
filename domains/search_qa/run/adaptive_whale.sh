#!/usr/bin/env bash
# Adaptive WHALE: each phase ends by a patience rule on its training signal.
#
# Start the retrieval server first; see docs/reproducing.md.
set -euo pipefail
DOMAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$DOMAIN_DIR:${PYTHONPATH:-}"

export DOMAIN_DIR
export TRAIN_SCRIPT="$DOMAIN_DIR/scripts/nq_hotpotqa/train_grpo_qwen3_5_2b_paper.sh"
export MH_MODULE="meta_harness.meta_harness_search_r1"
export BASE_HARNESS="${BASE_HARNESS:-$DOMAIN_DIR/environments/search_r1/base_harness.py}"
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-2B}"
export TOTAL_STEPS="${TOTAL_STEPS:-296}"   # 4 epochs at 74 steps/epoch
export RUN_NAME="${RUN_NAME:-adaptive-whale}"
export ADAPTIVE=1
export ROUND_STEPS="${ROUND_STEPS:-15}"      # minimum weight-update phase, 0.2 epoch
export MH_MIN_ITERS="${MH_MIN_ITERS:-5}"
export MH_PATIENCE="${MH_PATIENCE:-2}"
export MH_MAX_ITERS="${MH_MAX_ITERS:-15}"
export MAX_ROUNDS="${MAX_ROUNDS:-30}"
exec bash "$DOMAIN_DIR/../../run/lib/alternate.sh"
