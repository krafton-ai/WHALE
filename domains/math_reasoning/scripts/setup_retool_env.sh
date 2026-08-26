#!/usr/bin/env bash
# Build a local training venv for ReTool GRPO on Qwen/Qwen3.5-2B.
#
# Default mode installs this repo's copied/patched verl tree because it
# contains the ReTool meta-harness hooks and Qwen3.5 runtime settings. Set
# INSTALL_VERL_MODE=searchr1 only for emergency fallback against the historical
# Search-R1 environment, or pypi for the public recipe pin.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY_VERSION="${PY_VERSION:-3.10}"
VENV_NAME="${VENV_NAME:-.venv-retool-qwen35}"
INSTALL_VERL_MODE="${INSTALL_VERL_MODE:-local}" # local | searchr1 | pypi
SEARCH_R1_REPO="${SEARCH_R1_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../search_qa" && pwd)}"

UV_HOME="$PROJECT_DIR/.uv"
export UV_PYTHON_INSTALL_DIR="$UV_HOME/python"
export UV_CACHE_DIR="$UV_HOME/cache"
export UV_TOOL_DIR="$UV_HOME/tools"
export UV_DATA_DIR="$UV_HOME/data"
export PATH="$UV_HOME/bin:$PATH"

cd "$PROJECT_DIR"

mkdir -p "$UV_HOME"/{bin,python,cache,tools,data}
if [[ ! -x "$UV_HOME/bin/uv" ]]; then
    echo "[setup_retool_env] installing uv into $UV_HOME/bin"
    curl -LsSf https://astral.sh/uv/install.sh | \
        env UV_INSTALL_DIR="$UV_HOME/bin" UV_NO_MODIFY_PATH=1 sh
fi

cat > "$UV_HOME/env.sh" <<EOF
export UV_PYTHON_INSTALL_DIR="$UV_HOME/python"
export UV_CACHE_DIR="$UV_HOME/cache"
export UV_TOOL_DIR="$UV_HOME/tools"
export UV_DATA_DIR="$UV_HOME/data"
case ":\$PATH:" in
    *":$UV_HOME/bin:"*) ;;
    *) export PATH="$UV_HOME/bin:\$PATH" ;;
esac
EOF

uv python install "$PY_VERSION"

echo "[setup_retool_env] recreating $VENV_NAME"
rm -rf "$VENV_NAME"
uv venv "$VENV_NAME" --python "$PY_VERSION"
export VIRTUAL_ENV="$PROJECT_DIR/$VENV_NAME"

# H200 cluster currently uses NVIDIA driver 12.8. Keep the torch/vLLM pair
# aligned with the working Search-R1 Qwen3.5 stack.
TORCH_CUDA_BUILD="${TORCH_CUDA_BUILD:-cu128}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/${TORCH_CUDA_BUILD}}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.26.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.11.0}"
VLLM_WHEEL="${VLLM_WHEEL:-https://github.com/vllm-project/vllm/releases/download/v0.20.1/vllm-0.20.1+cu129-cp38-abi3-manylinux_2_31_x86_64.whl}"

uv pip install \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}" \
    --index-url "$TORCH_INDEX_URL"

case "$INSTALL_VERL_MODE" in
    local)
        [[ -f "$PROJECT_DIR/pyproject.toml" && -d "$PROJECT_DIR/verl" ]] || {
            echo "[setup_retool_env] ERROR: local verl package metadata/tree missing" >&2
            exit 1
        }
        echo "[setup_retool_env] installing verl from $PROJECT_DIR"
        uv pip install "$PROJECT_DIR"
        if [[ -f "$PROJECT_DIR/requirements.txt" ]]; then
            uv pip install -r "$PROJECT_DIR/requirements.txt"
        fi
        ;;
    searchr1)
        [[ -d "$SEARCH_R1_REPO/verl" ]] || {
            echo "[setup_retool_env] ERROR: SEARCH_R1_REPO=$SEARCH_R1_REPO does not look like a verl repo" >&2
            exit 1
        }
        echo "[setup_retool_env] installing verl from $SEARCH_R1_REPO"
        uv pip install "$SEARCH_R1_REPO"
        if [[ -f "$SEARCH_R1_REPO/requirements.txt" ]]; then
            uv pip install -r "$SEARCH_R1_REPO/requirements.txt"
        fi
        ;;
    pypi)
        echo "[setup_retool_env] installing verl==0.6.1"
        uv pip install "verl==0.6.1"
        ;;
    *)
        echo "[setup_retool_env] ERROR: INSTALL_VERL_MODE must be local, searchr1, or pypi" >&2
        exit 1
        ;;
esac

uv pip install "$VLLM_WHEEL" flashinfer-python qwen_vl_utils
uv pip install datasets pandas pyarrow omegaconf regex multiprocess pebble timeout_decorator python-dateutil requests verifiers openai anthropic

python - <<'PY'
import importlib

for name in ["torch", "vllm", "transformers", "datasets", "verl"]:
    mod = importlib.import_module(name)
    print(f"[setup_retool_env] import ok: {name} ({getattr(mod, '__version__', 'unknown')})")

try:
    from verl.experimental.agent_loop.tool_parser import ToolParser
    ToolParser.get_tool_parser("qwen3_coder", tokenizer=None)
except Exception as exc:
    print(f"[setup_retool_env] qwen3_coder parser smoke warning: {exc}")
else:
    print("[setup_retool_env] qwen3_coder parser is registered")
PY

unset VIRTUAL_ENV
echo "[setup_retool_env] done. Activate with: source $PROJECT_DIR/$VENV_NAME/bin/activate"
