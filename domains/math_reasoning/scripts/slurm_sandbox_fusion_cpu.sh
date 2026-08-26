#!/usr/bin/env bash
# Run a self-hosted SandboxFusion server on one CPU node.
#
# The H200 training job should use SANDBOX_MODE=external and the URL emitted by
# this job in outputs/sandbox/$RUN_NAME/sandbox.env.

#SBATCH --job-name=retool-sandbox
#SBATCH --partition=CPU_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"

RUN_NAME="${RUN_NAME:-retool-grpo-qwen3.5-2b-r8192-bs128}"
SANDBOX_IMAGE="${SANDBOX_IMAGE:-volcengine/sandbox-fusion:server-20250609}"
SANDBOX_PORT="${SANDBOX_PORT:-8080}"
SANDBOX_BIND_HOST="${SANDBOX_BIND_HOST:-0.0.0.0}"
SANDBOX_PULL_IMAGE="${SANDBOX_PULL_IMAGE:-true}"
SANDBOX_READY_ATTEMPTS="${SANDBOX_READY_ATTEMPTS:-180}"
SANDBOX_DOCKER_CPUS="${SANDBOX_DOCKER_CPUS:-${SLURM_CPUS_PER_TASK:-16}}"
SANDBOX_DOCKER_MEMORY="${SANDBOX_DOCKER_MEMORY:-56g}"

SANDBOX_RUN_DIR="$PROJECT_DIR/outputs/sandbox/$RUN_NAME"
SANDBOX_LOG_DIR="$PROJECT_DIR/outputs/slurm"
mkdir -p "$SANDBOX_RUN_DIR" "$SANDBOX_LOG_DIR"

ENV_FILE="$SANDBOX_RUN_DIR/sandbox.env"
READY_FILE="$SANDBOX_RUN_DIR/sandbox.ready"
rm -f "$ENV_FILE" "$READY_FILE"

NODE_HOST="${SLURMD_NODENAME:-$(hostname -f 2>/dev/null || hostname)}"
SANDBOX_URL="http://${NODE_HOST}:${SANDBOX_PORT}/run_code"
CONTAINER_NAME="retool-sandbox-${USER}-${SLURM_JOB_ID:-$$}"
DOCKER_LOG="$SANDBOX_LOG_DIR/${SLURM_JOB_NAME:-retool-sandbox}-${SLURM_JOB_ID:-local}.docker.log"

cleanup() {
    rm -f "$ENV_FILE" "$READY_FILE"
    if command -v docker >/dev/null 2>&1; then
        docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

echo "[sandbox_slurm] node=$NODE_HOST job=${SLURM_JOB_ID:-local}"
echo "[sandbox_slurm] image=$SANDBOX_IMAGE"
echo "[sandbox_slurm] url=$SANDBOX_URL"

command -v docker >/dev/null 2>&1 || {
    echo "[sandbox_slurm] ERROR: docker is required on the CPU node" >&2
    exit 1
}
docker info >/dev/null

if [[ "$SANDBOX_PULL_IMAGE" == "true" ]]; then
    echo "[sandbox_slurm] pulling $SANDBOX_IMAGE"
    docker pull "$SANDBOX_IMAGE"
fi

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
echo "[sandbox_slurm] starting Docker container $CONTAINER_NAME"
docker run --rm \
    --name "$CONTAINER_NAME" \
    --cpus "$SANDBOX_DOCKER_CPUS" \
    --memory "$SANDBOX_DOCKER_MEMORY" \
    -e HOST=0.0.0.0 \
    -e PORT=8080 \
    -p "${SANDBOX_BIND_HOST}:${SANDBOX_PORT}:8080" \
    "$SANDBOX_IMAGE" > "$DOCKER_LOG" 2>&1 &
DOCKER_PID=$!

for i in $(seq 1 "$SANDBOX_READY_ATTEMPTS"); do
    if python3 - <<PY
import json, urllib.request

payload = {"code": "print(1 + 1)", "language": "python"}
req = urllib.request.Request(
    "$SANDBOX_URL",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json", "Accept": "application/json"},
)
with urllib.request.urlopen(req, timeout=10) as r:
    result = json.loads(r.read().decode())
stdout = (result.get("run_result") or {}).get("stdout", "")
assert stdout.strip() == "2", result
PY
    then
        {
            printf 'export SANDBOX_URL=%q\n' "$SANDBOX_URL"
            printf 'export SANDBOX_NODE=%q\n' "$NODE_HOST"
            printf 'export SANDBOX_JOB_ID=%q\n' "${SLURM_JOB_ID:-}"
            printf 'export SANDBOX_IMAGE=%q\n' "$SANDBOX_IMAGE"
            printf 'export SANDBOX_PORT=%q\n' "$SANDBOX_PORT"
        } > "${ENV_FILE}.tmp"
        mv "${ENV_FILE}.tmp" "$ENV_FILE"
        date -u +"%Y-%m-%dT%H:%M:%SZ" > "$READY_FILE"
        echo "[sandbox_slurm] readiness smoke test passed"
        echo "[sandbox_slurm] wrote $ENV_FILE"
        break
    fi

    if ! kill -0 "$DOCKER_PID" 2>/dev/null; then
        echo "[sandbox_slurm] ERROR: Docker container exited before becoming ready" >&2
        sed 's/^/[sandbox_docker] /' "$DOCKER_LOG" >&2 || true
        exit 1
    fi

    if (( i == SANDBOX_READY_ATTEMPTS )); then
        echo "[sandbox_slurm] ERROR: SandboxFusion did not become ready" >&2
        sed 's/^/[sandbox_docker] /' "$DOCKER_LOG" >&2 || true
        exit 1
    fi
    sleep 2
done

echo "[sandbox_slurm] serving until this Slurm job is cancelled or reaches its time limit"
wait "$DOCKER_PID"
