"""Benchmark/frontier utilities for chess-puzzle-v0 AutoHarness runs."""

from __future__ import annotations

import json
import re
import traceback
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from environments.chess_puzzle.env import evaluate_harness

DEFAULT_SEED = 42


def load_config(config_path: Path) -> dict[str, Any]:
    return json.loads(Path(config_path).read_text())


def get_model_short_name(model_id: str) -> str:
    return model_id.split("/")[-1].lower()


def run_dir(base: Path, profile: str, harness: str, model_short: str, seed: int) -> Path:
    leaf = model_short if seed == DEFAULT_SEED else f"{model_short}_seed{seed}"
    return base / profile / harness / leaf


def parse_run_path(base: Path, filepath: Path) -> dict[str, Any] | None:
    try:
        rel = filepath.parent.relative_to(base)
        if len(rel.parts) != 3:
            return None
        profile, harness, model_leaf = rel.parts
    except (ValueError, IndexError):
        return None
    match = re.match(r"^(.+)_seed(\d+)$", model_leaf)
    if match:
        model_short = match.group(1)
        seed = int(match.group(2))
    else:
        model_short = model_leaf
        seed = DEFAULT_SEED
    return {"profile": profile, "harness": harness, "model_short": model_short, "seed": seed}


def _models_from_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    if config.get("models"):
        return [dict(model) for model in config["models"]]
    llm_cfg = config.get("eval", {}).get("llm") or {}
    return [dict(llm_cfg)] if llm_cfg else [{"provider": "mock", "model": "mock"}]


def run_sweep(
    config: dict[str, Any],
    harnesses: list[tuple[str, Path]],
    logs_dir: Path,
    *,
    force: bool = False,
) -> list[tuple[str, bool]]:
    profiles = config["task_profiles"]
    eval_cfg = config.get("eval", {})
    models = _models_from_config(config)
    seeds = config.get("seeds", [int(eval_cfg.get("seed", DEFAULT_SEED))])
    rows: list[tuple[str, bool]] = []
    for profile_name, profile_cfg in profiles.items():
        dataset_path = profile_cfg["dataset_path"]
        limit = int(profile_cfg.get("limit", eval_cfg.get("limit", 256)))
        for model_cfg in models:
            model_short = get_model_short_name(str(model_cfg.get("model", "mock")))
            for seed in seeds:
                for harness_name, harness_path in harnesses:
                    out_dir = run_dir(logs_dir, profile_name, harness_name, model_short, int(seed))
                    desc = f"{profile_name}/{harness_name}/{model_short}_seed{seed}"
                    val_file = out_dir / "val.json"
                    if val_file.exists() and not force:
                        rows.append((desc, True))
                        continue
                    print(f"  [bench] {desc}", flush=True)
                    try:
                        raw_policy_max = eval_cfg.get("policy_max_tokens")
                        policy_max_tokens = None
                        if raw_policy_max not in {None, "", "null"}:
                            policy_max_tokens = int(raw_policy_max)
                        evaluate_harness(
                            harness_path=harness_path,
                            dataset_path=dataset_path,
                            llm_config=model_cfg,
                            output_dir=out_dir,
                            limit=limit,
                            seed=int(seed),
                            assistant_token_budget=int(eval_cfg.get("assistant_token_budget", 8129)),
                            policy_max_tokens=policy_max_tokens,
                        )
                        rows.append((desc, True))
                    except Exception as exc:
                        out_dir.mkdir(parents=True, exist_ok=True)
                        (out_dir / "error.txt").write_text(
                            f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
                        )
                        rows.append((desc, False))
    return rows


def load_results(base_dir: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    results = {}
    for val in base_dir.rglob("val.json"):
        parsed = parse_run_path(base_dir, val)
        if not parsed:
            continue
        try:
            data = json.loads(val.read_text())
            data["_seed"] = parsed["seed"]
            results[(parsed["model_short"], parsed["profile"], parsed["harness"])] = data
        except json.JSONDecodeError:
            continue
    return results


def compute_pareto(points: list[tuple[str, float, float]]) -> list[tuple[str, float, float]]:
    sorted_points = sorted(points, key=lambda item: (-item[1], item[2]))
    pareto = []
    best_turns = float("inf")
    for name, score, turns in sorted_points:
        if turns <= best_turns:
            pareto.append((name, score, turns))
            best_turns = turns
    return pareto


def print_frontier(logs_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    results = load_results(logs_dir)
    if not results:
        print("No results found.")
        return {}

    by_harness: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"success": [], "turns": [], "reward": []}
    )
    for (_, _, harness), data in results.items():
        by_harness[harness]["success"].append(float(data.get("success_rate", 0.0)))
        by_harness[harness]["turns"].append(float(data.get("mean_turn_count", 0.0)))
        by_harness[harness]["reward"].append(float(data.get("avg_reward", 0.0)))

    points = [
        (harness, mean(stats["success"]), mean(stats["turns"]))
        for harness, stats in by_harness.items()
        if stats["success"]
    ]
    pareto = compute_pareto(points)
    pareto_names = {name for name, _, _ in pareto}
    frontier = {
        "_pareto": [
            {
                "harness": name,
                "avg_success_rate": round(score, 4),
                "avg_mean_turn_count": round(turns, 3),
            }
            for name, score, turns in pareto
        ]
    }
    if points:
        best = sorted(points, key=lambda item: (-item[1], item[2]))[0]
        frontier["_best"] = {
            "harness": best[0],
            "avg_success_rate": round(best[1], 4),
            "avg_mean_turn_count": round(best[2], 3),
        }
    for profile in config["task_profiles"]:
        entries = [
            {
                "harness": harness,
                "model_short": model_short,
                "success_rate": float(data.get("success_rate", 0.0)),
                "mean_turn_count": float(data.get("mean_turn_count", 0.0)),
            }
            for (model_short, p, harness), data in results.items()
            if p == profile
        ]
        if entries:
            best = max(entries, key=lambda item: (item["success_rate"], -item["mean_turn_count"]))
            frontier[profile] = {
                "best_harness": best["harness"],
                "model_short": best["model_short"],
                "success_rate": round(best["success_rate"], 4),
                "mean_turn_count": round(best["mean_turn_count"], 3),
            }

    path = logs_dir / "frontier_val.json"
    path.write_text(json.dumps(frontier, indent=2) + "\n")
    print("\nPARETO FRONTIER (solved rate up, policy calls down)")
    print(f"  {'harness':<28} {'score':>7} {'calls':>9}")
    for name, score, turns in sorted(points, key=lambda item: (-item[1], item[2])):
        mark = " *" if name in pareto_names else ""
        print(f"  {name + mark:<28} {score * 100:>6.1f}% {turns:>9.1f}")
    print(f"Wrote {path}")
    return frontier
