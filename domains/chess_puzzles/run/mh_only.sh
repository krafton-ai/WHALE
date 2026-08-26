#!/usr/bin/env bash
# Harness-only baseline: harness search over the frozen base model.
#
# No external service is required for this domain.
set -euo pipefail
DOMAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$DOMAIN_DIR:${PYTHONPATH:-}"

RUN_NAME="${RUN_NAME:-mh-only}"
ITERATIONS="${ITERATIONS:-40}"
PROPOSALS_PER_ITER="${PROPOSALS_PER_ITER:-3}"
export BASE_HARNESS="${BASE_HARNESS:-$DOMAIN_DIR/environments/chess_puzzle/base_harness.py}"
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-4B}"
cd "$DOMAIN_DIR"
python -m meta_harness.meta_harness_chess_puzzle \
  --run-name "$RUN_NAME" \
  --iterations "$ITERATIONS" \
  --proposals-per-iter "$PROPOSALS_PER_ITER"
