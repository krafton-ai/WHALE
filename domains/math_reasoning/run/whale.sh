#!/usr/bin/env bash
# WHALE: alternating weight updates and harness search on a fixed schedule.
#
# Start the code-execution sandbox first (see docs/reproducing.md).
set -euo pipefail
DOMAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$DOMAIN_DIR:${PYTHONPATH:-}"

export DOMAIN_DIR
export TRAIN_SCRIPT="$DOMAIN_DIR/scripts/train_grpo_qwen3_5_2b_retool.sh"
export MH_MODULE="meta_harness.meta_harness_retool"
export BASE_HARNESS="${BASE_HARNESS:-$DOMAIN_DIR/environments/retool/base_harness.py}"
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-2B}"
export RUN_NAME="${RUN_NAME:-whale}"
export TOTAL_STEPS="${TOTAL_STEPS:-420}"
export ROUND_STEPS="${ROUND_STEPS:-42}"        # E = 0.6 epoch
export ROUND_MH_ITERS="${ROUND_MH_ITERS:-5}"   # I = 6 counting the incoming harness
exec bash "$DOMAIN_DIR/../../run/lib/alternate.sh"
