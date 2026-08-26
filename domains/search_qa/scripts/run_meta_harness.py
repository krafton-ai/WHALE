#!/usr/bin/env python3
"""tmux launcher for Search-R1 meta-harness.

Reads a TOML config, generates the JSON config that meta_harness_search_r1.py
consumes, then spawns a detached tmux session with three panes:

  pane 0 (top):    vLLM stack    -- runs N vf-vllm replicas (one per GPU)
  pane 1 (middle): Retriever     -- runs N e5+FAISS retriever shards
  pane 2 (bottom): Training      -- waits for both stacks, then runs the loop

Usage:
    uv run python scripts/run_meta_harness.py @ configs/meta_harness/searchr1.toml
    uv run python scripts/run_meta_harness.py configs/meta_harness/searchr1.toml --iterations 5

CLI overrides take precedence over the TOML values:
    --session NAME        tmux session name
    --cwd PATH            working directory all panes start in
    --run-name NAME       meta_harness/runs/<run_name>/
    --iterations N        number of evolution iterations
    --fresh / --no-fresh  delete (or keep) existing runs/<run_name>/
    --propose-only        skip benchmark sweep (smoke test)
    --no-attach           build the tmux session but don't attach
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path


# -- shell helpers ------------------------------------------------------------


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, **kwargs)


def tmux_exists() -> bool:
    try:
        subprocess.run(["tmux", "-V"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return True


def tmux_session_exists(session: str) -> bool:
    return (
        subprocess.run(
            ["tmux", "has-session", "-t", session], capture_output=True
        ).returncode
        == 0
    )


# -- config loading + override -----------------------------------------------


def load_toml(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> tuple[str, Path, str]:
    session = args.session or cfg["run"]["session"]
    cwd = Path(args.cwd or cfg["run"]["cwd"]).expanduser().resolve()
    run_name = args.run_name or cfg["run"]["run_name"]

    if args.iterations is not None:
        cfg["meta"]["iterations"] = args.iterations
    if args.fresh is not None:
        cfg["meta"]["fresh"] = args.fresh
    if args.propose_only is not None:
        cfg["meta"]["propose_only"] = args.propose_only

    return session, cwd, run_name


# -- meta_harness JSON generation --------------------------------------------


def write_meta_harness_config_json(cfg: dict, run_name: str, cwd: Path) -> Path:
    """Translate TOML → JSON schema that meta_harness_search_r1.py expects.

    The meta-harness loop only needs to know about ONE 'model entry' (it picks
    api_base from there for AsyncOpenAI construction). When we run with N
    vLLM replicas, we still write the first replica's URL here as a fallback;
    the launcher additionally exports SEARCH_R1_VLLM_URLS so benchmark.py's
    _make_client picks up the full list and round-robins across them.
    """
    vllm = cfg["vllm"]
    eval_cfg = cfg["eval"]
    proposer = cfg["proposer"]

    base_url = f"http://localhost:{vllm['base_port']}/v1"
    model_entry: dict = {
        "model": vllm["model"],
        "api_base": base_url,
        "client_type": vllm.get("client_type", "openai_chat_completions"),
    }
    if vllm.get("api_key_var"):
        model_entry["api_key_var"] = vllm["api_key_var"]

    eval_path = eval_cfg["eval_dataset_path"]
    eval_path_abs = (
        eval_path if os.path.isabs(eval_path) else str((cwd / eval_path).resolve())
    )

    eval_block = {
        "num_train_examples": eval_cfg["num_train_examples"],
        "num_eval_examples": eval_cfg["num_eval_examples"],
        "eval_dataset_path": eval_path_abs,
        "temperature": eval_cfg["temperature"],
        "max_tokens": eval_cfg["max_tokens"],
        "max_turns": eval_cfg.get("max_turns", 10),
        "max_concurrent": eval_cfg.get("max_concurrent", 256),
        "parallel_harness_evals": eval_cfg.get("parallel_harness_evals", 3),
        "n": eval_cfg.get("n", 1),
        "enable_thinking": eval_cfg.get("enable_thinking", False),
        "max_failures_logged": eval_cfg.get("max_failures_logged", 5),
        "max_successes_logged": eval_cfg.get("max_successes_logged", 2),
    }
    # Pass top_p / top_k through only when set in the TOML. benchmark.py treats
    # missing keys as "don't send the field", which matches OpenAI/Anthropic
    # client defaults. We keep these optional rather than mandatory so older
    # TOMLs (without these keys) continue to work; new TOMLs that want
    # lockstep-with-verl behaviour set top_p=1.0 and top_k=20.
    if "top_p" in eval_cfg:
        eval_block["top_p"] = eval_cfg["top_p"]
    if "top_k" in eval_cfg:
        eval_block["top_k"] = eval_cfg["top_k"]

    json_cfg = {
        "task_profiles": {"default": {}},
        "models": [model_entry],
        "seeds": eval_cfg["seeds"],
        "eval": eval_block,
        "meta": {
            "num_candidates": proposer.get("num_candidates", 1),
            "proposer_model": proposer["model"],
            "proposer_effort": proposer["effort"],
            "propose_timeout_s": proposer["propose_timeout"],
        },
    }

    out_dir = cwd / "meta_harness" / "runs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "config.json"
    json_path.write_text(json.dumps(json_cfg, indent=2) + "\n")
    return json_path


# -- command builders --------------------------------------------------------


def build_vllm_cmd(cfg: dict) -> str:
    """Single command line that launches N vLLM replicas via launch_vllm_replicas.sh.

    Sources .venv-searchr1 first so the bundled `vllm` CLI is on PATH (the
    launch script invokes `vllm serve`, which lives there alongside verl).
    """
    vllm = cfg["vllm"]
    extra = vllm.get("extra_args") or []
    extra_str = " ".join(extra)
    env_parts = [
        f"REPLICAS={vllm.get('replicas', 8)}",
        f"BASE_PORT={vllm['base_port']}",
        f"GPU_OFFSET={vllm.get('gpu_offset', 0)}",
        f"MODEL={vllm['model']}",
        f"TOOL_CALL_PARSER={vllm.get('tool_call_parser', 'hermes')}",
        f"GPU_MEM_UTIL={vllm.get('gpu_memory_utilization', 0.45)}",
        f"MAX_MODEL_LEN={vllm.get('max_model_len', 8192)}",
    ]
    if extra_str:
        env_parts.append(f'EXTRA_ARGS="{extra_str}"')
    env_str = " ".join(env_parts)
    return (
        "source .venv-searchr1/bin/activate && "
        f"{env_str} bash scripts/launch_vllm_replicas.sh"
    )


def build_retriever_cmd(cfg: dict) -> str:
    """Single command line that launches N retriever replicas via retrieval_launch_multi.sh."""
    retr = cfg.get("retriever", {})
    env_parts = [
        f"TOTAL_SHARDS={retr.get('replicas', 8)}",
        f"BASE_PORT={retr.get('base_port', 8000)}",
        f"GPU_OFFSET={retr.get('gpu_offset', 0)}",
        f"FAISS_GPU={retr.get('faiss_gpu', 1)}",
        f"TOPK={retr.get('topk', 3)}",
    ]
    env_str = " ".join(env_parts)
    return (
        "source .venv-retriever/bin/activate && "
        f"{env_str} bash retrieval_launch_multi.sh"
    )


def build_meta_harness_cmd(cfg: dict, run_name: str, json_path_rel: Path) -> str:
    vllm = cfg["vllm"]
    retr = cfg.get("retriever", {})
    meta = cfg["meta"]
    proposer = cfg["proposer"]

    n_vllm = vllm.get("replicas", 8)
    base_vllm_port = vllm["base_port"]
    n_retr = retr.get("replicas", 8)
    base_retr_port = retr.get("base_port", 8000)
    timeout = vllm.get("ready_timeout_s", 600)

    # Build SEARCH_R1_VLLM_URLS / SEARCH_R1_RETRIEVAL_URLS so benchmark.py +
    # base_harness.py round-robin across all replicas.
    vllm_urls = ",".join(
        f"http://localhost:{base_vllm_port + i}/v1" for i in range(n_vllm)
    )
    retr_urls = ",".join(
        f"http://localhost:{base_retr_port + i}/retrieve" for i in range(n_retr)
    )

    # Wait for ALL replicas to come up (both stacks).
    vllm_wait = " ".join(
        f"until curl -fs http://localhost:{base_vllm_port + i}/v1/models "
        f">/dev/null 2>&1; do sleep 2; done;"
        for i in range(n_vllm)
    )
    retr_ping = (
        '{"queries":["ping"],"topk":1,"return_scores":false}'
    )
    retr_wait = " ".join(
        f"until curl -fs -X POST http://localhost:{base_retr_port + i}/retrieve "
        f"-H 'Content-Type: application/json' -d '{retr_ping}' "
        f">/dev/null 2>&1; do sleep 2; done;"
        for i in range(n_retr)
    )
    wait_cmd = (
        f"echo '[launcher] waiting for {n_vllm} vLLMs + {n_retr} retrievers "
        f"(timeout {timeout}s)'; "
        f"timeout {timeout} bash -c '{retr_wait} {vllm_wait}'"
    )

    flags = [
        f"--run-name {run_name}",
        f"--iterations {meta['iterations']}",
        f"--config {json_path_rel}",
        f"--proposer-model {proposer['model']}",
        f"--proposer-effort {proposer['effort']}",
        f"--propose-timeout {proposer['propose_timeout']}",
    ]
    if meta.get("fresh"):
        flags.append("--fresh")
    if meta.get("skip_baseline"):
        flags.append("--skip-baseline")
    if meta.get("propose_only"):
        flags.append("--propose-only")
    if proposer.get("use_api_key"):
        flags.append("--use-api-key")

    env_prefix = (
        f"SEARCH_R1_VLLM_URLS={vllm_urls} "
        f"SEARCH_R1_RETRIEVAL_URLS={retr_urls}"
    )
    # The loop itself runs under .venv-meta-harness (verifiers + claude_wrapper
    # + openai). vLLM and retriever are served from the other two venvs in the
    # sibling tmux panes.
    cmd = (
        "source .venv-meta-harness/bin/activate && "
        f"{env_prefix} python -m meta_harness.meta_harness_search_r1 "
        + " ".join(flags)
    )
    return f"{wait_cmd} && {cmd}"


# -- tmux orchestration ------------------------------------------------------


def create_tmux(
    session: str,
    cwd: Path,
    vllm_cmd: str,
    retriever_cmd: str,
    meta_cmd: str,
    window: str = "MetaHarness",
) -> None:
    target = f"{session}:{window}"

    _run(["tmux", "new-session", "-d", "-s", session, "-n", window, "-c", str(cwd)])
    # split: pane 0 (top) → split off pane 1 below; pane 1 → split off pane 2 below
    _run(["tmux", "split-window", "-v", "-t", f"{target}.0", "-c", str(cwd)])
    _run(["tmux", "split-window", "-v", "-t", f"{target}.1", "-c", str(cwd)])
    _run(["tmux", "select-layout", "-t", target, "even-vertical"])

    _run(["tmux", "set-option", "-t", session, "-g", "pane-border-status", "top"])
    _run(["tmux", "select-pane", "-t", f"{target}.0", "-T", "vLLM"])
    _run(["tmux", "select-pane", "-t", f"{target}.1", "-T", "Retriever"])
    _run(["tmux", "select-pane", "-t", f"{target}.2", "-T", "Training"])

    _run(["tmux", "send-keys", "-t", f"{target}.0", vllm_cmd, "C-m"])
    _run(["tmux", "send-keys", "-t", f"{target}.1", retriever_cmd, "C-m"])
    _run(["tmux", "send-keys", "-t", f"{target}.2", meta_cmd, "C-m"])

    _run(["tmux", "select-window", "-t", target])
    _run(["tmux", "select-pane", "-t", f"{target}.2"])


# -- CLI ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    argv = sys.argv[1:]
    if argv and argv[0] == "@":
        # visual separator only — see the launch example in the docstring
        argv = argv[1:]

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", type=Path, help="Path to TOML config")
    parser.add_argument("--session", default=None)
    parser.add_argument("--cwd", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--fresh", dest="fresh", action="store_true", default=None)
    parser.add_argument("--no-fresh", dest="fresh", action="store_false")
    parser.add_argument("--propose-only", action="store_true", default=None)
    parser.add_argument("--no-attach", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if not args.config.exists():
        sys.exit(f"config not found: {args.config}")

    cfg = load_toml(args.config)
    session, cwd, run_name = apply_cli_overrides(cfg, args)

    if not tmux_exists():
        sys.exit("tmux is not installed.")
    if tmux_session_exists(session):
        sys.exit(
            f"tmux session '{session}' already exists. "
            f"Kill it first: tmux kill-session -t {session}"
        )
    if not cwd.exists():
        sys.exit(f"cwd does not exist: {cwd}")

    json_path = write_meta_harness_config_json(cfg, run_name, cwd)
    json_path_rel = json_path.relative_to(cwd)

    vllm_cmd = build_vllm_cmd(cfg)
    retriever_cmd = build_retriever_cmd(cfg)
    meta_cmd = build_meta_harness_cmd(cfg, run_name, json_path_rel)

    print(f"[launcher] session    : {session}")
    print(f"[launcher] cwd        : {cwd}")
    print(f"[launcher] run_name   : {run_name}")
    print(f"[launcher] config     : {json_path_rel}")
    print(f"[launcher] vllm       : {vllm_cmd}")
    print(f"[launcher] retriever  : {retriever_cmd}")
    print(f"[launcher] training   : {meta_cmd}")

    create_tmux(session, cwd, vllm_cmd, retriever_cmd, meta_cmd)

    if args.no_attach:
        print(f"[launcher] tmux session '{session}' created. Attach with:")
        print(f"           tmux attach -t {session}")
        return

    os.execvp("tmux", ["tmux", "attach-session", "-t", session])


if __name__ == "__main__":
    main()
