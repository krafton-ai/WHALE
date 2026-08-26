#!/usr/bin/env bash
# Weight-only baseline: RSFT under the fixed base harness.
#
# Start the code-execution sandbox first (see docs/reproducing.md).
set -euo pipefail
DOMAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$DOMAIN_DIR:${PYTHONPATH:-}"

RUN_NAME="${RUN_NAME:-rsft-only}"
export HARNESS_PATH="${HARNESS_PATH:-$DOMAIN_DIR/environments/retool/base_harness.py}"
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-2B}"
export TRAINER_TOTAL_STEPS="${TRAINER_TOTAL_STEPS:-420}"
export RUN_NAME
bash "$DOMAIN_DIR/scripts/train_grpo_qwen3_5_2b_retool.sh"
