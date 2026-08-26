#!/usr/bin/env python3
"""Meta-Harness loop for chess-puzzle-v0 harness synthesis."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from meta_harness import claude_wrapper
    from meta_harness.chess_puzzle_benchmark import load_config, print_frontier, run_sweep
else:
    from . import claude_wrapper
    from .chess_puzzle_benchmark import load_config, print_frontier, run_sweep

from autoharness_chess_puzzle.harness import load_harness


EXPERIMENT_DIR = Path(__file__).parent
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = EXPERIMENT_DIR / "config-chess-puzzle.json"
BASELINE_HARNESS = REPO_ROOT / "environments" / "chess_puzzle" / "base_harness.py"
SKILL_DIR = EXPERIMENT_DIR / "skills" / "meta-harness-chess-puzzle"
PROMPT_ONLY_SKILL_DIR = EXPERIMENT_DIR / "skills" / "meta-harness-chess-puzzle-promptonly"
RUNS_DIR = EXPERIMENT_DIR / "runs"

# Prompt-only ablation mode (off by default; existing runs are unaffected).
# When enabled via --prompt-only, the proposer skill switches to the
# prompt-only contract and every candidate must match h0 at the AST level
# except for the SYSTEM_PROMPT / USER_PROMPT assignments.
PROMPT_ONLY = False

PROPOSER_ALLOWED_TOOLS = ["Read", "Glob", "Grep", "Write", "Edit"]
PROPOSER_SYSTEM_PROMPT = (
    "You are generating chess-puzzle-v0 Meta-Harness candidates inside one run "
    "directory. Do not run shell commands. Do not write to /tmp or to any "
    "absolute path outside the current working directory. The only files you "
    "may write are harnesses/<slot>/harness.py, pending_eval.json at the "
    "working-directory root, and optional logs/iteration_*/report.md summaries. "
    "The harness must never reveal or infer hidden puzzle solutions."
)


def _ts() -> str:
    return time.strftime("[%H:%M:%S]")


def get_run_dir(run_name: str) -> Path:
    return RUNS_DIR / run_name


def ensure_run_layout(run_dir: Path) -> None:
    for path in [run_dir / "harnesses", run_dir / "logs" / "claude_sessions"]:
        path.mkdir(parents=True, exist_ok=True)


def init_h0(run_dir: Path, baseline_harness: Path) -> None:
    target_dir = run_dir / "harnesses" / "h0"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "harness.py"
    if not target.exists():
        shutil.copy(baseline_harness, target)


def baseline_harness() -> Path:
    override = os.environ.get("BASELINE_HARNESS_OVERRIDE", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return BASELINE_HARNESS


def next_harness_names(run_dir: Path, count: int) -> list[str]:
    nums = []
    for path in (run_dir / "harnesses").glob("h*"):
        try:
            nums.append(int(path.name[1:]))
        except ValueError:
            continue
    start = max(nums) + 1 if nums else 0
    return [f"h{start + i}" for i in range(count)]


def accepted_file(run_dir: Path) -> Path:
    return run_dir / "logs" / "accepted_harness.txt"


def get_accepted_harness(run_dir: Path) -> str:
    path = accepted_file(run_dir)
    if path.exists():
        name = path.read_text().strip()
        if (run_dir / "harnesses" / name / "harness.py").exists():
            return name
    return "h0"


def set_accepted_harness(run_dir: Path, name: str) -> None:
    accepted_file(run_dir).write_text(name + "\n")


_PROMPT_ONLY_NAMES = {"SYSTEM_PROMPT", "USER_PROMPT"}


def _prompt_only_normalized_dump(source: str) -> str:
    """AST dump with prompt-constant RHS and docstrings masked out.

    Two harnesses whose dumps match differ at most in SYSTEM_PROMPT /
    USER_PROMPT values and in docstrings; comments never reach the AST.
    Everything else (parser, retry budgets, MAX_TURNS, helper functions,
    observation formatting, ...) stays comparable verbatim.
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


def validate_prompt_only(run_dir: Path, name: str) -> None:
    """Reject candidates that differ from h0 outside the prompt constants."""
    baseline_path = run_dir / "harnesses" / "h0" / "harness.py"
    candidate_path = run_dir / "harnesses" / name / "harness.py"
    baseline_dump = _prompt_only_normalized_dump(baseline_path.read_text())
    candidate_dump = _prompt_only_normalized_dump(candidate_path.read_text())
    if baseline_dump != candidate_dump:
        raise ValueError(
            f"{name} violates the prompt-only constraint: code other than "
            "SYSTEM_PROMPT / USER_PROMPT differs from h0. Inline all prompt "
            "text into those two constants and leave every other statement "
            "byte-identical to h0."
        )


def validate_candidate(run_dir: Path, name: str) -> None:
    load_harness(run_dir / "harnesses" / name / "harness.py")
    if PROMPT_ONLY:
        validate_prompt_only(run_dir, name)


def recover_pending_candidates(run_dir: Path, names: list[str]) -> list[dict[str, Any]]:
    candidates = []
    for name in names:
        if not (run_dir / "harnesses" / name / "harness.py").exists():
            return []
        candidates.append(
            {
                "name": name,
                "axis": "recovered",
                "hypothesis": "Recovered from proposer-created harness.py without pending_eval.json.",
                "changes": "The proposer wrote the harness file but exited before metadata was written.",
            }
        )
    return candidates


def build_harness_list(run_dir: Path, names: list[str]) -> list[tuple[str, Path]]:
    return [(name, run_dir / "harnesses" / name / "harness.py") for name in names]


def render_task_prompt(iteration: int, next_names: list[str]) -> str:
    slot_lines = "\n".join(f"  - harnesses/{name}/harness.py" for name in next_names)
    return (
        f"Run iteration {iteration} of the chess-puzzle-v0 AutoHarness evolution loop.\n\n"
        "## Run directories\n"
        "- harnesses: harnesses/\n"
        "- logs: logs/\n"
        "- pending_eval.json: write to the working-directory root\n\n"
        "## Filesystem boundary\n"
        "You may write only inside this working directory. Do not write to /tmp, "
        "home directories, parent directories, or any absolute path outside this "
        "run directory. Do not create scratch files.\n\n"
        "## Output\n"
        f"Produce {len(next_names)} complete harness.py file(s):\n{slot_lines}\n"
        f"Write pending_eval.json with {len(next_names)} candidate entries in the same order.\n\n"
        "Follow the provided chess-puzzle-v0 skill strictly, especially the anti-cheating rules."
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
):
    saved_key = None
    if not use_api_key:
        saved_key = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        return claude_wrapper.run(
            prompt=task_prompt,
            model=proposer_model,
            allowed_tools=PROPOSER_ALLOWED_TOOLS,
            system_prompt=PROPOSER_SYSTEM_PROMPT,
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


def _proposer_retry_count() -> int:
    raw = os.environ.get("CHESS_PUZZLE_PROPOSER_RETRIES", "5")
    try:
        return max(1, int(raw))
    except ValueError:
        return 5


def _proposer_retry_backoff_s() -> float:
    raw = os.environ.get("CHESS_PUZZLE_PROPOSER_RETRY_BACKOFF_S", "120")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 120.0


def _proposer_wrote_candidates(run_dir: Path, names: list[str]) -> bool:
    if (run_dir / "pending_eval.json").exists():
        return True
    return any((run_dir / "harnesses" / name / "harness.py").exists() for name in names)


def _is_transient_proposer_failure(result: Any) -> bool:
    if int(getattr(result, "exit_code", 0) or 0) == 0:
        return False
    haystack = f"{getattr(result, 'text', '')}\n{getattr(result, 'stderr', '')}".lower()
    transient_markers = (
        "api error: overloaded",
        "overloaded",
        "rate limit",
        "rate_limit",
        "temporarily unavailable",
        "service unavailable",
        "internal server error",
        "connection reset",
        "timed out",
        "timeout",
    )
    return any(marker in haystack for marker in transient_markers)


def propose_claude_with_retries(
    *,
    task_prompt: str,
    iteration: int,
    run_dir: Path,
    next_names: list[str],
    proposer_model: str,
    proposer_effort: str,
    timeout_seconds: int,
    use_api_key: bool,
):
    max_attempts = _proposer_retry_count()
    backoff_s = _proposer_retry_backoff_s()
    last_result = None
    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            print(
                f"{_ts()} proposer retry {attempt}/{max_attempts} "
                f"after {backoff_s:g}s backoff"
            )
            time.sleep(backoff_s)
        last_result = propose_claude(
            task_prompt=task_prompt,
            iteration=iteration,
            run_dir=run_dir,
            proposer_model=proposer_model,
            proposer_effort=proposer_effort,
            timeout_seconds=timeout_seconds,
            use_api_key=use_api_key,
        )
        if int(getattr(last_result, "exit_code", 0) or 0) == 0:
            return last_result
        if _proposer_wrote_candidates(run_dir, next_names):
            return last_result
        if not _is_transient_proposer_failure(last_result):
            return last_result
        last_result.show()
    return last_result


def pick_accepted(frontier: dict[str, Any], run_dir: Path) -> str:
    best = frontier.get("_best") or {}
    name = best.get("harness") or "h0"
    if (run_dir / "harnesses" / name / "harness.py").exists():
        return name
    return get_accepted_harness(run_dir)


def append_evolution_summary(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    path = run_dir / "logs" / "evolution_summary.jsonl"
    with path.open("a") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def candidate_summary_rows(
    iteration: int,
    candidates: list[dict[str, Any]],
    logs_dir: Path,
    frontier: dict[str, Any],
) -> list[dict[str, Any]]:
    pareto = {item["harness"] for item in frontier.get("_pareto", [])}
    rows = []
    for item in candidates:
        name = item["name"]
        vals = list(logs_dir.glob(f"*/{name}/*/val.json"))
        scores = []
        turns = []
        rewards = []
        for val in vals:
            data = json.loads(val.read_text())
            scores.append(float(data.get("success_rate", 0.0)))
            turns.append(float(data.get("mean_turn_count", 0.0)))
            rewards.append(float(data.get("avg_reward", 0.0)))
        rows.append(
            {
                "iteration": iteration,
                "harness": name,
                "status": "evaluated",
                "avg_success_rate": round(sum(scores) / len(scores), 4) if scores else 0.0,
                "avg_mean_turn_count": round(sum(turns) / len(turns), 3) if turns else 0.0,
                "avg_reward": round(sum(rewards) / len(rewards), 4) if rewards else 0.0,
                "on_pareto": name in pareto,
                "hypothesis": item.get("hypothesis", ""),
                "changes": item.get("changes", ""),
                "axis": item.get("axis", "?"),
            }
        )
    return rows


def find_early_stop_candidate(
    rows: list[dict[str, Any]],
    valid_names: list[str],
    threshold: float,
) -> dict[str, Any] | None:
    if threshold <= 0:
        return None
    valid = set(valid_names)
    hits = [
        row
        for row in rows
        if row.get("harness") in valid and float(row.get("avg_success_rate", 0.0)) >= threshold
    ]
    if not hits:
        return None
    return sorted(
        hits,
        key=lambda row: (
            -float(row.get("avg_success_rate", 0.0)),
            float(row.get("avg_mean_turn_count", float("inf"))),
            str(row.get("harness", "")),
        ),
    )[0]


def run_evolve(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config) if args.config else CONFIG_PATH)
    meta_cfg = config.get("meta", {})
    proposer_model = args.proposer_model or meta_cfg.get("proposer_model", "claude-opus-4-7")
    proposer_effort = args.proposer_effort or meta_cfg.get("proposer_effort", "max")
    propose_timeout = args.propose_timeout or meta_cfg.get("propose_timeout_s", 2400)
    proposals_per_iter = args.proposals_per_iter or meta_cfg.get("proposals_per_iter", 1)
    early_stop_success_rate = float(
        args.early_stop_success_rate
        if args.early_stop_success_rate is not None
        else meta_cfg.get("early_stop_success_rate", 1.0)
    )

    run_dir = get_run_dir(args.run_name)
    if args.fresh and run_dir.exists():
        shutil.rmtree(run_dir)
    ensure_run_layout(run_dir)
    baseline = baseline_harness()
    if not baseline.exists():
        raise FileNotFoundError(f"baseline harness does not exist: {baseline}")
    init_h0(run_dir, baseline)
    set_accepted_harness(run_dir, get_accepted_harness(run_dir))

    logs_dir = run_dir / "logs"
    print(
        f"{_ts()} Meta-Harness (chess-puzzle-v0) run={args.run_name} "
        f"iters={args.iterations} proposals/iter={proposals_per_iter} "
        f"early_stop_success_rate={early_stop_success_rate:g}"
    )
    print(f"{_ts()} baseline_harness={baseline}")
    print(f"{_ts()} skill_dir={SKILL_DIR}")

    print(f"\n{_ts()} Phase 0: baseline h0 sweep")
    run_sweep(config, build_harness_list(run_dir, ["h0"]), logs_dir, force=args.force)
    frontier = print_frontier(logs_dir, config)
    set_accepted_harness(run_dir, pick_accepted(frontier, run_dir))

    start_iteration = int(getattr(args, "start_iteration", 1) or 1)
    es_min_iters = max(int(getattr(args, "early_stop_min_iters", 0) or 0), 0)
    es_patience = max(int(getattr(args, "early_stop_patience", 0) or 0), 1)
    es_best_score = float("-inf")
    es_best_iter = 0
    for iteration in range(start_iteration, start_iteration + int(args.iterations)):
        current = get_accepted_harness(run_dir)
        next_names = next_harness_names(run_dir, int(proposals_per_iter))
        task_prompt = render_task_prompt(iteration, next_names)
        iter_dir = logs_dir / f"iteration_{iteration:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        (iter_dir / "proposer_prompt.txt").write_text(task_prompt)
        print(f"\n{_ts()} Iteration {iteration}: accepted={current}, slots={next_names}")

        if args.eval_only:
            candidates = [
                {"name": name}
                for name in next_names
                if (run_dir / "harnesses" / name / "harness.py").exists()
            ]
        else:
            result = propose_claude_with_retries(
                task_prompt=task_prompt,
                iteration=iteration,
                run_dir=run_dir,
                next_names=next_names,
                proposer_model=proposer_model,
                proposer_effort=proposer_effort,
                timeout_seconds=int(propose_timeout),
                use_api_key=args.use_api_key,
            )
            result.show()
            pending_path = run_dir / "pending_eval.json"
            if not pending_path.exists():
                candidates = recover_pending_candidates(run_dir, next_names)
                if not candidates:
                    raise RuntimeError("Proposer did not write pending_eval.json")
                pending_path.write_text(json.dumps({"candidates": candidates}, indent=2) + "\n")
            shutil.copy(pending_path, iter_dir / "pending_eval.json")
            candidates = json.loads(pending_path.read_text()).get("candidates", [])

        valid_names = []
        prompt_only_rejects = []
        for candidate in candidates:
            if PROMPT_ONLY and "name" not in candidate:
                # The proposer occasionally labels the slot with a synonym.
                # Recover instead of aborting the round; the default mode keeps
                # its strict behaviour.
                for alias in ("slot", "harness", "id"):
                    if isinstance(candidate.get(alias), str):
                        candidate["name"] = candidate[alias]
                        print(f"{_ts()} pending_eval: mapped '{alias}' -> 'name' ({candidate['name']})")
                        break
            name = candidate["name"]
            if PROMPT_ONLY:
                # A constraint violation must not kill the whole MH run: skip
                # the candidate and record why, so the next proposer session
                # sees it in the archive.
                try:
                    validate_candidate(run_dir, name)
                except Exception as exc:
                    print(f"{_ts()} prompt-only reject {name}: {exc}")
                    prompt_only_rejects.append(
                        {
                            "iteration": iteration,
                            "harness": name,
                            "status": "rejected_prompt_only",
                            "reason": str(exc)[:300],
                            "hypothesis": candidate.get("hypothesis", ""),
                            "axis": candidate.get("axis", "?"),
                        }
                    )
                    continue
            else:
                validate_candidate(run_dir, name)
            valid_names.append(name)
        if not valid_names:
            if prompt_only_rejects:
                append_evolution_summary(run_dir, prompt_only_rejects)
            raise RuntimeError("No valid candidates to evaluate.")

        run_sweep(config, build_harness_list(run_dir, valid_names), logs_dir, force=args.force)
        frontier = print_frontier(logs_dir, config)
        rows = candidate_summary_rows(iteration, candidates, logs_dir, frontier)
        early_stop = find_early_stop_candidate(rows, valid_names, early_stop_success_rate)
        accepted = early_stop["harness"] if early_stop else pick_accepted(frontier, run_dir)
        set_accepted_harness(run_dir, accepted)
        append_evolution_summary(run_dir, rows)
        if PROMPT_ONLY and prompt_only_rejects:
            append_evolution_summary(run_dir, prompt_only_rejects)
        comparison = {
            "iteration": iteration,
            "accepted_harness": accepted,
            "candidates": candidates,
            "summary": rows,
            "frontier": frontier,
            "early_stop": early_stop,
        }
        (iter_dir / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
        print(f"{_ts()} accepted={accepted}")
        if early_stop:
            print(
                f"{_ts()} early_stop: proposed harness {accepted} reached "
                f"{float(early_stop['avg_success_rate']) * 100:.1f}% success"
            )
            break

        if es_min_iters > 0:
            _es_top = (
                float(((frontier or {}).get("_best") or {}).get("avg_success_rate", 0.0))
                if (frontier or {}).get("_best")
                else None
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="chess-puzzle-autoharness")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--iterations", type=int, default=1)
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
    parser.add_argument("--start-iteration", type=int, default=1)
    parser.add_argument("--proposals-per-iter", type=int, default=None)
    parser.add_argument("--proposer-model", default=None)
    parser.add_argument("--proposer-effort", default=None)
    parser.add_argument("--propose-timeout", type=int, default=None)
    parser.add_argument("--early-stop-success-rate", type=float, default=None)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--use-api-key", action="store_true")
    parser.add_argument(
        "--prompt-only",
        action="store_true",
        help="Restrict the search space to SYSTEM_PROMPT / USER_PROMPT: use the "
        "prompt-only proposer skill and reject any candidate whose non-prompt "
        "AST differs from h0 (default: off; existing behavior)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global PROMPT_ONLY
    PROMPT_ONLY = bool(getattr(args, "prompt_only", False))
    if PROMPT_ONLY:
        print(f"{_ts()} prompt-only mode: skill={PROMPT_ONLY_SKILL_DIR}")
    run_evolve(args)


if __name__ == "__main__":
    main()
