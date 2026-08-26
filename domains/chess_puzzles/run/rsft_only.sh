#!/usr/bin/env bash
# Weight-only baseline: RSFT under the fixed base harness.
#
# No external service is required for this domain.
set -euo pipefail
DOMAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$DOMAIN_DIR:${PYTHONPATH:-}"

RUN_NAME="${RUN_NAME:-rsft-only}"
export HARNESS_PATH="${HARNESS_PATH:-$DOMAIN_DIR/environments/chess_puzzle/base_harness.py}"
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-4B}"
export TRAINER_TOTAL_STEPS="${TRAINER_TOTAL_STEPS:-256}"
export RUN_NAME
bash "$DOMAIN_DIR/scripts/train_chess_puzzle_multinode_disagg.sh"
