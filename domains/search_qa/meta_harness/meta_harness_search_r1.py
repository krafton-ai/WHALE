#!/usr/bin/env python3
"""Agent-driven meta-harness loop for the Search-R1 tool-use task (wiki-18 retrieval).

Full harness evolution: the proposer generates the entire harness.py file
(class CandidateEnv, SYSTEM_PROMPT, TOOLS, env_response) each iteration.

Each run is isolated under runs/<run_name>/:
  harnesses/h0/harness.py   ← baseline copied from environments/search_r1/base_harness.py
  harnesses/h1/harness.py   ← first proposer candidate
  logs/                     ← eval results, trajectories, frontier
  pending_eval.json         ← proposer output
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from meta_harness import claude_wrapper
    from meta_harness.benchmark import (
        get_model_short_name,
        load_config,
        load_results,
        print_frontier,
        run_sweep,
    )
else:
    from . import claude_wrapper
    from .benchmark import (
        get_model_short_name,
        load_config,
        load_results,
        print_frontier,
        run_sweep,
    )

EXPERIMENT_DIR = Path(__file__).parent
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = EXPERIMENT_DIR / "config-search-r1.json"
BASE_ENV_PY = REPO_ROOT / "environments" / "search_r1" / "env.py"
BASELINE_HARNESS = Path(
    os.environ.get(
        "BASELINE_HARNESS_OVERRIDE",
        str(REPO_ROOT / "environments" / "search_r1" / "base_harness.py"),
    )
)
SKILL_DIR = EXPERIMENT_DIR / "skills" / "meta-harness-search-r1"
PROMPT_ONLY_SKILL_DIR = EXPERIMENT_DIR / "skills" / "meta-harness-search-r1-promptonly"
RUNS_DIR = EXPERIMENT_DIR / "runs"

# Prompt-only ablation mode (off by default; existing runs are unaffected).
# When enabled via --prompt-only, the proposer skill switches to the
# prompt-only contract and every candidate must match h0 at the AST level
# except for the SYSTEM_PROMPT / USER_PROMPT_TEMPLATE assignments.
PROMPT_ONLY = False

PROPOSER_ALLOWED_TOOLS = ["Read", "Glob", "Grep", "Write", "Edit", "Bash"]

_interrupted = False


# -- ANSI helpers -------------------------------------------------------------

_USE_COLOR = sys.stdout.isatty()


def _c(code, text):
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def _bold(t):
    return _c("1", t)


def _dim(t):
    return _c("2", t)


def _green(t):
    return _c("32", t)


def _red(t):
    return _c("31", t)


def _yellow(t):
    return _c("33", t)


def _cyan(t):
    return _c("36", t)


def _ts():
    return _dim(datetime.now().strftime("[%H:%M:%S]"))


def _elapsed(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def _pct(val):
    s = f"{val * 100:.1f}%" if val <= 1.0 else f"{val:.1f}%"
    if val <= 1.0:
        val = val * 100
    if val >= 60:
        return _green(s)
    if val >= 40:
        return _yellow(s)
    return _red(s)


def _handle_signal(signum, frame):
    global _interrupted
    _interrupted = True
    print("\nInterrupted, finishing current step...", flush=True)


# -- Run directory helpers ----------------------------------------------------


def get_run_dir(run_name: str) -> Path:
    return RUNS_DIR / run_name


def ensure_run_layout(run_dir: Path) -> None:
    for d in [
        run_dir / "harnesses",
        run_dir / "logs" / "claude_sessions",
    ]:
        d.mkdir(parents=True, exist_ok=True)


def init_h0(run_dir: Path) -> None:
    """Copy baseline base_harness.py as h0/harness.py."""
    h0_dir = run_dir / "harnesses" / "h0"
    h0_dir.mkdir(parents=True, exist_ok=True)
    target = h0_dir / "harness.py"
    if not target.exists():
        shutil.copy(BASELINE_HARNESS, target)


def next_harness_name(run_dir: Path) -> str:
    """Return the next h<N> name."""
    existing = list((run_dir / "harnesses").glob("h*"))
    nums = []
    for p in existing:
        try:
            nums.append(int(p.name[1:]))
        except ValueError:
            continue
    return f"h{max(nums) + 1}" if nums else "h0"


def get_accepted_harness(run_dir: Path) -> str:
    """Return the name of the currently accepted harness (e.g. 'h0')."""
    accepted_file = run_dir / "logs" / "accepted_harness.txt"
    if accepted_file.exists():
        name = accepted_file.read_text().strip()
        if (run_dir / "harnesses" / name / "harness.py").exists():
            return name
    return "h0"


def set_accepted_harness(run_dir: Path, name: str) -> None:
    (run_dir / "logs" / "accepted_harness.txt").write_text(name + "\n")


def read_harness_path(run_dir: Path, harness_name: str) -> Path:
    return run_dir / "harnesses" / harness_name / "harness.py"


def evolution_summary_tail(run_dir: Path, limit: int = 20) -> str:
    path = run_dir / "logs" / "evolution_summary.jsonl"
    if not path.exists():
        return ""
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    return "\n".join(lines[-limit:])


def append_evolution_summary(run_dir: Path, rows: list[dict]) -> None:
    path = run_dir / "logs" / "evolution_summary.jsonl"
    with open(path, "a") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


# -- Candidate validation -----------------------------------------------------

_PROMPT_ONLY_NAMES = {"SYSTEM_PROMPT", "USER_PROMPT_TEMPLATE"}


def _prompt_only_normalized_dump(source: str) -> str:
    """AST dump with prompt-constant RHS and docstrings masked out.

    Two harnesses whose dumps match differ at most in SYSTEM_PROMPT /
    USER_PROMPT_TEMPLATE values and in docstrings; comments never reach the
    AST. Everything else (imports, search(), MAX_TURNS, env_response, ...)
    stays comparable verbatim.
    """
    import ast

    tree = ast.parse(source)

    class _Mask(ast.NodeTransformer):
        def visit_Assign(self, node):
            self.generic_visit(node)
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if names and set(names) <= _PROMPT_ONLY_NAMES:
                node.value = ast.Constant(value="__PROMPT__")
            return node

        def visit_AnnAssign(self, node):
            self.generic_visit(node)
            if (
                isinstance(node.target, ast.Name)
                and node.target.id in _PROMPT_ONLY_NAMES
                and node.value is not None
            ):
                # Normalize to a plain Assign so `X: str = ...` and `X = ...`
                # compare equal for the prompt constants.
                return ast.Assign(
                    targets=[ast.Name(id=node.target.id, ctx=ast.Store())],
                    value=ast.Constant(value="__PROMPT__"),
                    type_comment=None,
                )
            return node

    tree = _Mask().visit(tree)

    def _strip_docstrings(node):
        body = getattr(node, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]
        for child in ast.iter_child_nodes(node):
            _strip_docstrings(child)

    _strip_docstrings(tree)
    return ast.dump(tree)


def validate_prompt_only(run_dir: Path, harness_name: str) -> None:
    """Reject candidates that differ from h0 outside the prompt constants."""
    baseline_path = run_dir / "harnesses" / "h0" / "harness.py"
    candidate_path = run_dir / "harnesses" / harness_name / "harness.py"
    baseline_dump = _prompt_only_normalized_dump(baseline_path.read_text())
    candidate_dump = _prompt_only_normalized_dump(candidate_path.read_text())
    if baseline_dump != candidate_dump:
        raise ValueError(
            f"{harness_name} violates the prompt-only constraint: code other "
            "than SYSTEM_PROMPT / USER_PROMPT_TEMPLATE differs from h0. "
            "Inline all prompt text into those two constants and leave every "
            "other statement byte-identical to h0."
        )


def validate_candidate(run_dir: Path, harness_name: str) -> None:
    """Check that harness.py exists and contains class CandidateEnv."""
    harness_path = run_dir / "harnesses" / harness_name / "harness.py"
    if not harness_path.exists():
        raise ValueError(f"harness.py not found at {harness_path}")
    content = harness_path.read_text()
    if not content.strip():
        raise ValueError(f"harness.py is empty at {harness_path}")
    # Prompt-only candidates mirror h0, which may expose CandidateEnv as an
    # alias (`CandidateEnv = SearchR1Env`); accept that form only in
    # prompt-only mode so default behavior is unchanged.
    if "class CandidateEnv" not in content and not (
        PROMPT_ONLY and "CandidateEnv =" in content
    ):
        raise ValueError(
            f"class CandidateEnv not found in {harness_path}. "
            "Proposer must define CandidateEnv(SearchR1Env)."
        )
    if PROMPT_ONLY:
        validate_prompt_only(run_dir, harness_name)


# -- Proposer -----------------------------------------------------------------


def render_task_prompt(iteration: int, next_names: list[str]) -> str:
    """Build the prompt for the proposer Claude session.

    Kept slim (paper-style). Diversity / mechanism rules live in SKILL.md
    so they don't drift between task prompt and skill instructions.
    """
    k = len(next_names)
    slot_lines = "\n".join(f"  - harnesses/{n}/harness.py" for n in next_names)
    return (
        f"Run iteration {iteration} of the Search-R1 harness evolution loop.\n\n"
        f"## Run directories (relative to your working directory)\n"
        f"- harnesses: harnesses/\n"
        f"- logs: logs/\n"
        f"- pending_eval.json: write to the working-directory root\n\n"
        f"## Output\n"
        f"Produce {k} new harness.py file(s):\n{slot_lines}\n"
        f"Write `pending_eval.json` with {k} candidate entries (same order).\n\n"
        f"Follow the `meta-harness-search-r1` skill as your operating procedure."
    )


def propose_claude(
    *,
    task_prompt: str,
    iteration: int,
    run_dir: Path,
    proposer_model: str,
    proposer_effort: str,
    timeout_seconds: int,
    use_api_key: bool,
) -> claude_wrapper.SessionResult:
    os.environ.pop("CLAUDECODE", None)
    saved_key = None
    if not use_api_key:
        saved_key = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        result = claude_wrapper.run(
            prompt=task_prompt,
            model=proposer_model,
            allowed_tools=PROPOSER_ALLOWED_TOOLS,
            skills=[str(PROMPT_ONLY_SKILL_DIR if PROMPT_ONLY else SKILL_DIR)],
            cwd=str(run_dir),
            log_dir=str(run_dir / "logs" / "claude_sessions"),
            name=f"iter{iteration:03d}",
            timeout_seconds=timeout_seconds,
            effort=proposer_effort,
        )
    finally:
        if saved_key is not None:
            os.environ["ANTHROPIC_API_KEY"] = saved_key
    return result


# -- Acceptance / frontier update --------------------------------------------


def pick_accepted_from_frontier(
    frontier: dict, run_dir: Path
) -> tuple[str | None, str | None]:
    """Pick the top Pareto point by success_rate, tie-break by fewer turns."""
    pareto = frontier.get("_pareto", [])
    if not pareto:
        return None, None
    top = sorted(
        pareto,
        key=lambda p: (-p["avg_success_rate"], p["avg_mean_turn_count"]),
    )[0]
    name = top["harness"]
    harness_path = run_dir / "harnesses" / name / "harness.py"
    if harness_path.exists():
        return name, name
    return name, None


# -- Sweep helpers ------------------------------------------------------------


def build_harness_list(
    run_dir: Path, harness_names: list[str]
) -> list[tuple[str, Path, Path]]:
    """Build (name, base_env_py, candidate_harness_path) tuples."""
    return [
        (name, BASE_ENV_PY, read_harness_path(run_dir, name)) for name in harness_names
    ]


# -- Main loop ----------------------------------------------------------------


def run_evolve(args: argparse.Namespace) -> None:
    config_path = Path(args.config) if args.config else CONFIG_PATH
    config = load_config(config_path)
    meta_cfg = config.get("meta", {})
    proposer_model = args.proposer_model or meta_cfg.get(
        "proposer_model", "claude-opus-4-7"
    )
    proposer_effort = args.proposer_effort or meta_cfg.get("proposer_effort", "max")
    propose_timeout = args.propose_timeout or meta_cfg.get("propose_timeout_s", 2400)
    proposals_per_iter = (
        args.proposals_per_iter
        if args.proposals_per_iter is not None
        else meta_cfg.get("proposals_per_iter", 1)
    )
    if proposals_per_iter < 1:
        proposals_per_iter = 1

    run_dir = get_run_dir(args.run_name)

    if args.fresh and run_dir.exists():
        shutil.rmtree(run_dir)

    ensure_run_layout(run_dir)
    init_h0(run_dir)
    set_accepted_harness(run_dir, get_accepted_harness(run_dir))

    logs_dir = run_dir / "logs"
    pending_eval = run_dir / "pending_eval.json"

    profiles = list(config["task_profiles"].keys())
    models = [get_model_short_name(m["model"]) for m in config["models"]]
    print(
        f"{_ts()} {_bold('Claude Meta-Harness (Search-R1) — Full Harness Evolution')}  "
        f"run={args.run_name}  iters={args.iterations}  proposals/iter={proposals_per_iter}  "
        f"profiles={profiles}  models={models}"
    )

    # -- Phase 0: baseline sweep (always runs; no opt-out) -------------------
    # Every meta-harness run starts by evaluating h0 (= a verbatim copy of
    # environments/search_r1/base_harness.py). With env.py:correct_answer now
    # mirroring verl's compute_score bit-for-bit, the h0 success_rate is
    # directly comparable to verl GRPO training's `critic/score/mean`. Knowing
    # this baseline before exploring candidates is a hard prerequisite — we
    # removed the `--skip-baseline` opt-out to enforce it.
    if args.propose_only:
        print(f"{_ts()} {_yellow('propose-only mode: skipping all benchmark sweeps')}")
    else:
        print(f"\n{_ts()} {_bold('Phase 0: Baseline sweep (h0) — always runs')}")
        harness_list = build_harness_list(run_dir, ["h0"])
        run_sweep(config, harness_list, logs_dir)
        print_frontier(logs_dir, config)

    es_min_iters = max(int(getattr(args, "early_stop_min_iters", 0) or 0), 0)
    es_patience = max(int(getattr(args, "early_stop_patience", 0) or 0), 1)
    es_best_score = float("-inf")
    es_best_iter = 0
    # -- Phase 1..N: evolution -----------------------------------------------
    for i in range(args.iterations):
        if _interrupted:
            print("Interrupted.")
            break
        iteration = _next_iteration_number(logs_dir)
        iter_start = time.time()

        current_harness = get_accepted_harness(run_dir)
        # Reserve K consecutive harness names up front (h{base}..h{base+K-1}).
        # The proposer writes all K in one Claude Code session; the names are
        # surfaced in the task prompt so the proposer knows where each goes.
        base_next = next_harness_name(run_dir)
        base_num = int(base_next[1:])
        next_names = [f"h{base_num + j}" for j in range(proposals_per_iter)]
        iteration_dir = logs_dir / f"iteration_{iteration:03d}"
        iteration_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"\n{_ts()} {_bold(f'Iteration {iteration}')} "
            f"({i + 1}/{args.iterations})  "
            f"accepted={_cyan(current_harness)}  "
            f"next={_cyan(','.join(next_names))}"
        )
        print("-" * 64)

        task_prompt = render_task_prompt(iteration, next_names)
        (iteration_dir / "proposer_prompt.txt").write_text(task_prompt)

        if pending_eval.exists():
            pending_eval.unlink()

        # Single Claude Code session that produces K harnesses (paper's
        # mechanism: SKILL + task prompt instruct the proposer to write K
        # files and a single pending_eval.json with K candidate entries).
        propose_start = time.time()
        print(
            f"  {_ts()} {_cyan('proposing')} {proposals_per_iter} candidate(s) "
            f"via Claude Code...",
            flush=True,
        )
        result = propose_claude(
            task_prompt=task_prompt,
            iteration=iteration,
            run_dir=run_dir,
            proposer_model=proposer_model,
            proposer_effort=proposer_effort,
            timeout_seconds=propose_timeout,
            use_api_key=args.use_api_key,
        )
        propose_time = time.time() - propose_start
        result.show()

        if result.exit_code != 0:
            print(f"  {_red('FAIL')} proposer exit={result.exit_code}")
            continue
        if not pending_eval.exists():
            print(
                f"  {_red('FAIL')} no pending_eval.json after {_elapsed(propose_time)}"
            )
            continue

        shutil.copy(pending_eval, iteration_dir / "pending_eval.json")

        candidates = json.loads(pending_eval.read_text()).get("candidates", [])
        print(
            f"  {_ts()} proposed {_bold(str(len(candidates)))} candidate(s) in "
            f"{_elapsed(propose_time)}"
        )
        for ci, c in enumerate(candidates):
            hyp = c.get("hypothesis", "")
            print(f"    {ci + 1}. {_bold(c['name'])}: {hyp[:80]}")

        # Validate
        valid_candidates = []
        prompt_only_rejects = []
        for c in candidates:
            name = c["name"]
            try:
                validate_candidate(run_dir, name)
                valid_candidates.append(c)
                print(f"    {_green('OK')} {name}")
            except Exception as exc:
                print(f"    {_red('FAIL')} {name}: {exc}")
                if PROMPT_ONLY:
                    prompt_only_rejects.append(
                        {
                            "iteration": iteration,
                            "harness": name,
                            "status": "rejected_prompt_only",
                            "reason": str(exc)[:300],
                            "hypothesis": c.get("hypothesis", ""),
                            "axis": c.get("axis", "?"),
                        }
                    )
        # Record prompt-only rejections so later proposer sessions see the
        # constraint violation in the archive (all-invalid iterations are
        # already recorded below through the existing path).
        if PROMPT_ONLY and prompt_only_rejects and valid_candidates:
            append_evolution_summary(run_dir, prompt_only_rejects)

        if not valid_candidates:
            print(f"  {_red('0 valid')} candidates, skipping iteration")
            append_evolution_summary(
                run_dir,
                [
                    {
                        "iteration": iteration,
                        "harness": c.get("name"),
                        "status": "invalid",
                        "hypothesis": c.get("hypothesis", ""),
                        "axis": c.get("axis", "?"),
                    }
                    for c in candidates
                ],
            )
            continue

        # Benchmark
        bench_start = time.time()
        if args.propose_only:
            print(
                f"  {_ts()} {_yellow('propose-only: skipping benchmark sweep; candidates validated only')}"
            )
            frontier = {}
            bench_time = 0.0
        else:
            print(
                f"  {_ts()} {_cyan('benchmarking')} {len(valid_candidates)} "
                f"candidate(s) across profiles/models/seeds"
            )
            harness_names = [c["name"] for c in valid_candidates]
            harness_list = build_harness_list(run_dir, harness_names)
            run_sweep(config, harness_list, logs_dir)
            bench_time = time.time() - bench_start
            frontier = print_frontier(logs_dir, config)

        # Pull per-harness aggregates for the evolution summary
        results = load_results(logs_dir, "val.json")
        summary_rows = []
        for c in valid_candidates:
            name = c["name"]
            vals = [v for (_, _, h), v in results.items() if h == name]
            accs = [float(v.get("success_rate", 0.0)) for v in vals]
            turns = [float(v.get("mean_turn_count", 0.0)) for v in vals]
            # LLM-judge env.py registers the metric as ``correct_reward_llm_judge``;
            # legacy EM env.py uses ``correct_reward``. Prefer whichever is present.
            corrects = [
                float(
                    (v.get("avg_metrics") or {}).get("correct_reward_llm_judge")
                    or (v.get("avg_metrics") or {}).get("correct_reward", 0.0)
                )
                for v in vals
            ]
            formats = [
                float((v.get("avg_metrics") or {}).get("format_reward", 0.0))
                for v in vals
            ]
            avg_acc = sum(accs) / len(accs) if accs else 0.0
            avg_turns = sum(turns) / len(turns) if turns else 0.0
            avg_correct = sum(corrects) / len(corrects) if corrects else 0.0
            avg_format = sum(formats) / len(formats) if formats else 0.0
            on_pareto = any(p["harness"] == name for p in frontier.get("_pareto", []))
            row = {
                "iteration": iteration,
                "harness": name,
                "status": "evaluated",
                "avg_success_rate": round(avg_acc, 4),
                "avg_correct_reward": round(avg_correct, 4),
                "avg_format_reward": round(avg_format, 4),
                "avg_mean_turn_count": round(avg_turns, 3),
                "on_pareto": on_pareto,
                "hypothesis": c.get("hypothesis", ""),
                "changes": c.get("changes", ""),
                # Paper-style axis/components: persisted when the proposer
                # provided them (see SKILL.md), so cross-iteration analysis
                # can see which mechanism axes have been explored.
                "axis": c.get("axis", "?"),
            }
            if "components" in c:
                row["components"] = c["components"]
            summary_rows.append(row)
        append_evolution_summary(run_dir, summary_rows)

        comparison = {
            "iteration": iteration,
            "baseline_harness": current_harness,
            "candidates": summary_rows,
            "frontier": frontier,
        }
        (iteration_dir / "comparison.json").write_text(
            json.dumps(comparison, indent=2) + "\n"
        )

        # Accept: top Pareto point globally
        new_name, resolved = pick_accepted_from_frontier(frontier, run_dir)
        if resolved is not None and resolved != current_harness:
            set_accepted_harness(run_dir, resolved)
            print(f"  {_green('NEW ACCEPTED')} -> {new_name}")
        else:
            print(f"  {_dim('accepted harness unchanged')}")

        wall_time = time.time() - iter_start
        print(
            "  "
            + _dim(
                f"timing: propose={_elapsed(propose_time)} "
                f"bench={_elapsed(bench_time)} "
                f"total={_elapsed(wall_time)}"
            )
        )

        if es_min_iters > 0 and not args.propose_only:
            _es_top = max(
                (float(p.get("avg_success_rate", 0.0)) for p in ((frontier or {}).get("_pareto") or [])),
                default=None,
            )
            if _es_top is not None and _es_top > es_best_score:
                es_best_score = _es_top
                es_best_iter = iteration
            if iteration >= es_min_iters and iteration - es_best_iter >= es_patience:
                (logs_dir / "early_stop.json").write_text(
                    json.dumps(
                        {
                            "stopped_after_iteration": iteration,
                            "best_score": es_best_score,
                            "best_iteration": es_best_iter,
                            "min_iters": es_min_iters,
                            "patience": es_patience,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                print(
                    f"{_ts()} patience early stop after iteration {iteration} "
                    f"(best {es_best_score:.4f} at iteration {es_best_iter})"
                )
                break


def _next_iteration_number(logs_dir: Path) -> int:
    existing = list(logs_dir.glob("iteration_*"))
    nums = []
    for p in existing:
        try:
            nums.append(int(p.name.split("_")[-1]))
        except ValueError:
            continue
    return (max(nums) + 1) if nums else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-name", required=True, help="Name for this run (e.g. exp01)"
    )
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--early-stop-min-iters",
        type=int,
        default=0,
        help="Patience-based early stop: minimum iterations before stopping (0 disables).",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=2,
        help="Stop after this many iterations without a new global-best training score.",
    )
    parser.add_argument(
        "--proposals-per-iter", type=int, default=None,
        help="Sequential proposer calls per iteration (default 1). Each call "
        "advances the next harness name and appends its candidate(s) to the "
        "iteration's pool; validation/benchmark runs on the merged pool.",
    )
    parser.add_argument("--proposer-model", default=None)
    parser.add_argument("--proposer-effort", default=None, help="low|medium|high|max")
    parser.add_argument("--propose-timeout", type=int, default=None, help="seconds")
    parser.add_argument(
        "--use-api-key",
        action="store_true",
        help="Keep ANTHROPIC_API_KEY env var (default: strip so claude CLI uses subscription)",
    )
    parser.add_argument(
        "--propose-only",
        action="store_true",
        help="Only run the proposer + candidate validation; skip all sweeps",
    )
    parser.add_argument(
        "--prompt-only",
        action="store_true",
        help="Restrict the search space to SYSTEM_PROMPT / USER_PROMPT_TEMPLATE: "
        "use the prompt-only proposer skill and reject any candidate whose "
        "non-prompt AST differs from h0 (default: off; existing behavior)",
    )
    parser.add_argument(
        "--fresh", action="store_true", help="Delete existing run and start fresh"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config JSON (default: config-search-r1.json in script dir)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global PROMPT_ONLY
    PROMPT_ONLY = bool(getattr(args, "prompt_only", False))
    if PROMPT_ONLY:
        print(f"{_ts()} prompt-only mode: skill={PROMPT_ONLY_SKILL_DIR}")
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    run_evolve(args)


if __name__ == "__main__":
    main()
