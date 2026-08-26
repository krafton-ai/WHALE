#!/usr/bin/env bash
# Start or reuse the self-hosted SandboxFusion job for one ReTool run.

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_NAME="${RUN_NAME:-retool-grpo-qwen3.5-2b-r8192-bs128}"

SANDBOX_PARTITION="${SANDBOX_PARTITION:-m7i-cpu2}"
SANDBOX_CPUS_PER_TASK="${SANDBOX_CPUS_PER_TASK:-16}"
SANDBOX_PORT="${SANDBOX_PORT:-8080}"
SANDBOX_TIME="${SANDBOX_TIME:-48:00:00}"
SANDBOX_JOB_NAME="${SANDBOX_JOB_NAME:-retool-sandbox}"
SANDBOX_IMAGE="${SANDBOX_IMAGE:-volcengine/sandbox-fusion:server-20250609}"
SANDBOX_PULL_IMAGE="${SANDBOX_PULL_IMAGE:-true}"
SANDBOX_WAIT_TIMEOUT="${SANDBOX_WAIT_TIMEOUT:-1800}"
SANDBOX_DOCKER_CPUS="${SANDBOX_DOCKER_CPUS:-$SANDBOX_CPUS_PER_TASK}"
SANDBOX_DOCKER_MEMORY="${SANDBOX_DOCKER_MEMORY:-56g}"
REQUESTED_SANDBOX_JOB_NAME="$SANDBOX_JOB_NAME"

STATE_DIR="$REPO/outputs/sandbox/$RUN_NAME"
ENV_FILE="$STATE_DIR/sandbox.env"
SUBMIT_FILE="$STATE_DIR/sandbox.submit.env"
mkdir -p "$REPO/outputs/slurm" "$STATE_DIR"

wait_for_env_file() {
    local job_id="$1"
    local deadline
    deadline=$(( $(date +%s) + SANDBOX_WAIT_TIMEOUT ))
    while [[ ! -s "$ENV_FILE" ]]; do
        if [[ -z "$(squeue -j "$job_id" -h -o '%T' 2>/dev/null)" ]]; then
            echo "[cpu_sandbox] ERROR: sandbox job $job_id is no longer in queue before readiness" >&2
            echo "[cpu_sandbox] check outputs/slurm/${SANDBOX_JOB_NAME}-${job_id}.err" >&2
            exit 1
        fi
        if (( $(date +%s) >= deadline )); then
            echo "[cpu_sandbox] ERROR: timed out waiting for sandbox readiness" >&2
            echo "[cpu_sandbox] sandbox job is still $job_id; inspect or cancel that job explicitly if needed" >&2
            exit 1
        fi
        sleep 5
    done
}

smoke_test_sandbox() {
    local url="$1"
    python3 - "$url" <<'PY'
import json
import sys
import urllib.request

url = sys.argv[1]
payload = {"code": "print(1 + 1)", "language": "python"}
req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json", "Accept": "application/json"},
)
with urllib.request.urlopen(req, timeout=10) as r:
    result = json.loads(r.read().decode())
stdout = (result.get("run_result") or {}).get("stdout", "")
assert stdout.strip() == "2", result
PY
}

if [[ -s "$ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    SANDBOX_JOB_NAME="$REQUESTED_SANDBOX_JOB_NAME"
    if [[ -n "${SANDBOX_JOB_ID:-}" && -n "${SANDBOX_URL:-}" ]]; then
        job_line="$(squeue -j "$SANDBOX_JOB_ID" -h -o '%u|%T|%j' 2>/dev/null || true)"
        if [[ -n "$job_line" ]]; then
            job_user="${job_line%%|*}"
            rest="${job_line#*|}"
            job_state="${rest%%|*}"
            job_name="${rest#*|}"
            if [[ "$job_user" == "$USER" && "$job_name" == "$SANDBOX_JOB_NAME" && "$job_state" == "RUNNING" ]]; then
                if smoke_test_sandbox "$SANDBOX_URL"; then
                    echo "[cpu_sandbox] reusing $SANDBOX_URL from job $SANDBOX_JOB_ID"
                    exit 0
                fi
            fi
        fi
    fi
fi

if [[ -s "$SUBMIT_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$SUBMIT_FILE"
    SANDBOX_JOB_NAME="$REQUESTED_SANDBOX_JOB_NAME"
    if [[ -n "${SANDBOX_JOB_ID:-}" ]]; then
        job_line="$(squeue -j "$SANDBOX_JOB_ID" -h -o '%u|%T|%j' 2>/dev/null || true)"
        if [[ -n "$job_line" ]]; then
            job_user="${job_line%%|*}"
            job_name="${job_line##*|}"
            if [[ "$job_user" == "$USER" && "$job_name" == "$SANDBOX_JOB_NAME" ]]; then
                echo "[cpu_sandbox] waiting for existing submitted sandbox job $SANDBOX_JOB_ID"
                wait_for_env_file "$SANDBOX_JOB_ID"
                # shellcheck disable=SC1090
                source "$ENV_FILE"
                echo "[cpu_sandbox] sandbox ready: $SANDBOX_URL"
                exit 0
            fi
        fi
    fi
fi

rm -f "$ENV_FILE" "$STATE_DIR/sandbox.ready"

echo "[cpu_sandbox] submitting SandboxFusion on $SANDBOX_PARTITION"
SANDBOX_JOB_ID="$(
    sbatch --parsable \
        --job-name="$SANDBOX_JOB_NAME" \
        --partition="$SANDBOX_PARTITION" \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task="$SANDBOX_CPUS_PER_TASK" \
        --mem=0 \
        --exclusive \
        --time="$SANDBOX_TIME" \
        --export=ALL,PROJECT_DIR="$REPO",RUN_NAME="$RUN_NAME",SANDBOX_PORT="$SANDBOX_PORT",SANDBOX_IMAGE="$SANDBOX_IMAGE",SANDBOX_PULL_IMAGE="$SANDBOX_PULL_IMAGE",SANDBOX_DOCKER_CPUS="$SANDBOX_DOCKER_CPUS",SANDBOX_DOCKER_MEMORY="$SANDBOX_DOCKER_MEMORY" \
        "$REPO/scripts/slurm_sandbox_fusion_cpu.sh"
)"
{
    printf 'export SANDBOX_JOB_ID=%q\n' "$SANDBOX_JOB_ID"
    printf 'export SANDBOX_RUN_NAME=%q\n' "$RUN_NAME"
    printf 'export SANDBOX_JOB_NAME=%q\n' "$SANDBOX_JOB_NAME"
} > "$SUBMIT_FILE"
echo "[cpu_sandbox] sandbox job id: $SANDBOX_JOB_ID"
echo "[cpu_sandbox] waiting for $ENV_FILE"

wait_for_env_file "$SANDBOX_JOB_ID"

# shellcheck disable=SC1090
source "$ENV_FILE"
echo "[cpu_sandbox] sandbox ready: $SANDBOX_URL"
