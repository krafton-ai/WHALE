#!/usr/bin/env bash
# Weight-only baseline: RSFT under the fixed base harness h0.
#
# Start the retrieval server first; see docs/reproducing.md.
set -euo pipefail
DOMAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$DOMAIN_DIR:${PYTHONPATH:-}"

export RUN_NAME="${RUN_NAME:-rsft-only}"
export HARNESS_PATH="${HARNESS_PATH:-$DOMAIN_DIR/environments/search_r1/stupid_harness.py}"
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-2B}"
TOTAL_STEPS="${TOTAL_STEPS:-296}"   # 4 epochs at 74 steps/epoch
bash "$DOMAIN_DIR/scripts/nq_hotpotqa/train_grpo_qwen3_5_2b_paper.sh" \
  trainer.total_training_steps="$TOTAL_STEPS" \
  +trainer.online_rsft.enable=True \
  +trainer.online_rsft.sft_epochs="${RSFT_SFT_EPOCHS:-1}" \
  +trainer.online_rsft.sft_mini_batch_size="${RSFT_SFT_MINI_BATCH_SIZE:-64}" \
  +trainer.online_rsft.sft_micro_batch_size_per_gpu="${RSFT_SFT_MICRO_PER_GPU:-2}" \
  +trainer.online_rsft.score_threshold="${RSFT_SCORE_THRESHOLD:-0.5}"
