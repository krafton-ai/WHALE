#!/usr/bin/env bash
# Weight-only baseline: RSFT under the fixed base harness.
#
# Start the retriever first (see docs/reproducing.md).
set -euo pipefail
DOMAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$DOMAIN_DIR:${PYTHONPATH:-}"

RUN_NAME="${RUN_NAME:-rsft-only}"
export HARNESS_PATH="${HARNESS_PATH:-$DOMAIN_DIR/environments/search_r1/base_harness.py}"
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-2B}"
export TRAINER_TOTAL_STEPS="${TRAINER_TOTAL_STEPS:-296}"
export RUN_NAME
bash "$DOMAIN_DIR/scripts/nq_hotpotqa/train_grpo_qwen3_5_2b_paper.sh"
