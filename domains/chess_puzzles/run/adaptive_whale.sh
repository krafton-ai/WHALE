#!/usr/bin/env bash
# Adaptive WHALE: each phase ends by a patience rule on its training signal.
#
# No external service is required for this domain.
set -euo pipefail
DOMAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$DOMAIN_DIR:${PYTHONPATH:-}"

export DOMAIN_DIR
export TRAIN_SCRIPT="$DOMAIN_DIR/scripts/train_chess_puzzle_multinode_disagg.sh"
export MH_MODULE="meta_harness.meta_harness_chess_puzzle"
export BASE_HARNESS="${BASE_HARNESS:-$DOMAIN_DIR/environments/chess_puzzle/base_harness.py}"
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-4B}"
export MH_CONFIG="$DOMAIN_DIR/meta_harness/config-chess-puzzle.json"
export TOTAL_STEPS="${TOTAL_STEPS:-256}"   # 4 epochs at 64 steps/epoch
export RUN_NAME="${RUN_NAME:-adaptive-whale}"
export ADAPTIVE=1
export ROUND_STEPS="${ROUND_STEPS:-13}"      # minimum weight-update phase, 0.2 epoch
export MH_MIN_ITERS="${MH_MIN_ITERS:-5}"
export MH_PATIENCE="${MH_PATIENCE:-2}"
export MH_MAX_ITERS="${MH_MAX_ITERS:-15}"
export MAX_ROUNDS="${MAX_ROUNDS:-30}"
exec bash "$DOMAIN_DIR/../../run/lib/alternate.sh"
