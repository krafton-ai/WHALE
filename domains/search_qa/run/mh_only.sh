#!/usr/bin/env bash
# Harness-only baseline: harness search over the frozen base model.
#
# Start the retrieval server first; see docs/reproducing.md.
set -euo pipefail
DOMAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$DOMAIN_DIR:${PYTHONPATH:-}"

export RUN_NAME="${RUN_NAME:-mh-only}"
export BASELINE_HARNESS_OVERRIDE="${BASELINE_HARNESS_OVERRIDE:-$DOMAIN_DIR/environments/search_r1/stupid_harness.py}"
export VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3.5-2B}"
ITERATIONS="${ITERATIONS:-40}"
cd "$DOMAIN_DIR"
python -m meta_harness.meta_harness_search_r1 \
  --run-name "$RUN_NAME" \
  --iterations "$ITERATIONS" \
  --proposals-per-iter "${PROPOSALS_PER_ITER:-3}" \
  --proposer-model "${PROPOSER_MODEL:-claude-opus-4-7}" \
  --proposer-effort "${PROPOSER_EFFORT:-max}" \
  --propose-timeout "${PROPOSER_TIMEOUT:-2400}" \
  ${MH_CONFIG:+--config "$MH_CONFIG"}
