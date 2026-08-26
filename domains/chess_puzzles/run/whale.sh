#!/usr/bin/env bash
# WHALE: alternating weight updates and harness search on a fixed schedule.
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
export RUN_NAME="${RUN_NAME:-whale}"
export TOTAL_STEPS="${TOTAL_STEPS:-256}"
export ROUND_STEPS="${ROUND_STEPS:-39}"        # E = 0.6 epoch
export ROUND_MH_ITERS="${ROUND_MH_ITERS:-5}"   # I = 6 counting the incoming harness
exec bash "$DOMAIN_DIR/../../run/lib/alternate.sh"
