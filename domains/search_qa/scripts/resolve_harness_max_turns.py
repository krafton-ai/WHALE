#!/usr/bin/env python3
"""Resolve the effective MAX_TURNS for a Search-R1 harness file."""

from __future__ import annotations

import ast
import importlib.util
import os
import sys
from pathlib import Path


def _emit(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, int):
        return False
    print(value)
    return True


def _literal_int(node: ast.AST) -> int | None:
    try:
        value = ast.literal_eval(node)
    except Exception:
        return None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _class_max_turns(tree: ast.Module, class_name: str) -> int | None:
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for stmt in node.body:
            value_node = None
            if isinstance(stmt, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "MAX_TURNS" for t in stmt.targets
            ):
                value_node = stmt.value
            elif (
                isinstance(stmt, ast.AnnAssign)
                and isinstance(stmt.target, ast.Name)
                and stmt.target.id == "MAX_TURNS"
            ):
                value_node = stmt.value
            if value_node is None:
                continue
            value = _literal_int(value_node)
            if value is not None:
                return value
    return None


def _candidate_bases(tree: ast.Module) -> list[str]:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "CandidateEnv":
            names: list[str] = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    names.append(base.id)
                elif isinstance(base, ast.Attribute):
                    names.append(base.attr)
            return names
    return []


def _fallback_static_max_turns(path: Path, repo_root: Path) -> int | None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    value = _class_max_turns(tree, "CandidateEnv")
    if value is not None:
        return value

    alias_to_searchr1 = any(
        isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "CandidateEnv" for t in node.targets)
        and isinstance(node.value, ast.Name)
        and node.value.id == "SearchR1Env"
        for node in tree.body
    )
    if alias_to_searchr1:
        value = _class_max_turns(tree, "SearchR1Env")
        if value is not None:
            return value

    if "SearchR1Env" in _candidate_bases(tree):
        base_path = repo_root / "environments" / "search_r1" / "base_harness.py"
        base_tree = ast.parse(base_path.read_text(encoding="utf-8"), filename=str(base_path))
        return _class_max_turns(base_tree, "SearchR1Env")

    return None


def _import_max_turns(path: Path, repo_root: Path) -> int | None:
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(path.parent))
    # Meta-harness candidates often reuse earlier candidates with imports like
    # `from harnesses.h6.harness import ...`. Add the run directory that owns
    # `harnesses/` so MAX_TURNS resolution matches verl's meta_harness_hook.
    if path.parent.parent.name == "harnesses":
        sys.path.insert(0, str(path.parent.parent.parent))

    spec = importlib.util.spec_from_file_location("selected_searchr1_harness", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not build import spec")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    env_class = getattr(mod, "CandidateEnv", None)
    value = getattr(env_class, "MAX_TURNS", None)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: resolve_harness_max_turns.py HARNESS_PATH", file=sys.stderr)
        return 2

    path = Path(argv[1]).resolve()
    repo_root = Path(os.environ.get("PROJECT_DIR") or os.environ.get("REPO") or Path.cwd()).resolve()
    if not path.is_file():
        print(f"missing harness: {path}", file=sys.stderr)
        return 1

    try:
        if _emit(_import_max_turns(path, repo_root)):
            return 0
    except Exception as exc:
        import_error = exc
    else:
        import_error = RuntimeError("CandidateEnv.MAX_TURNS is not an integer")

    if _emit(_fallback_static_max_turns(path, repo_root)):
        return 0

    print(f"failed to resolve MAX_TURNS from {path}: {import_error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
