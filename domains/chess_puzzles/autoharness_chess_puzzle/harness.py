from __future__ import annotations

import ast
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

FORBIDDEN_IMPORTS = {
    "asyncio",
    "builtins",
    "chess",
    "importlib",
    "io",
    "os",
    "pathlib",
    "pickle",
    "shutil",
    "socket",
    "subprocess",
    "sys",
    "threading",
}
FORBIDDEN_CALLS = {"__import__", "compile", "eval", "exec", "globals", "locals", "open"}
ALLOWED_IMPORTS = {"__future__", "collections", "itertools", "math", "random", "re", "statistics"}


@dataclass
class HarnessModule:
    path: Path
    propose_action: Callable[[str], str]
    is_legal_action: Callable[[str, str], bool]
    source: str
    format_observation: Callable[..., str] | None = None
    parse_action: Callable[..., str] | None = None
    system_prompt: str | None = None
    user_prompt: str | None = None
    format_retry_budget: int | None = None
    illegal_move_retry_budget: int | None = None
    max_turns: int | None = None


class HarnessValidationError(ValueError):
    pass


class _SafetyVisitor(ast.NodeVisitor):
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".", 1)[0]
            if root in FORBIDDEN_IMPORTS or root not in ALLOWED_IMPORTS:
                raise HarnessValidationError(f"Forbidden import: {alias.name}")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        root = (node.module or "").split(".", 1)[0]
        if root in FORBIDDEN_IMPORTS or root not in ALLOWED_IMPORTS:
            raise HarnessValidationError(f"Forbidden import: {node.module}")

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            raise HarnessValidationError(f"Forbidden call: {node.func.id}")
        if isinstance(node.func, ast.Attribute) and node.func.attr.startswith("__"):
            raise HarnessValidationError(f"Forbidden dunder attribute call: {node.func.attr}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__"):
            raise HarnessValidationError(f"Forbidden dunder attribute: {node.attr}")
        self.generic_visit(node)


SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "chr": chr,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "Exception": Exception,
    "filter": filter,
    "float": float,
    "frozenset": frozenset,
    "int": int,
    "isinstance": isinstance,
    "iter": iter,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "ord": ord,
    "pow": pow,
    "print": print,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "ArithmeticError": ArithmeticError,
    "AttributeError": AttributeError,
    "IndexError": IndexError,
    "KeyError": KeyError,
    "LookupError": LookupError,
    "RuntimeError": RuntimeError,
    "StopIteration": StopIteration,
    "TypeError": TypeError,
    "ValueError": ValueError,
    "ZeroDivisionError": ZeroDivisionError,
}


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".", 1)[0]
    if root not in ALLOWED_IMPORTS:
        raise ImportError(f"Import is not allowed in generated harness: {name}")
    return __import__(name, globals, locals, fromlist, level)


SAFE_BUILTINS = dict(SAFE_BUILTINS, __import__=_safe_import)


def validate_source(source: str) -> None:
    tree = ast.parse(source)
    _SafetyVisitor().visit(tree)
    names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    missing = {"propose_action", "is_legal_action"} - names
    if missing:
        raise HarnessValidationError(f"Missing required function(s): {sorted(missing)}")


def _optional_int(globals_dict: dict[str, Any], key: str) -> int | None:
    raw = globals_dict.get(key)
    if raw is None:
        return None
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return None


def load_harness(path: str | Path) -> HarnessModule:
    p = Path(path)
    source = p.read_text()
    validate_source(source)
    globals_dict = {
        "__builtins__": dict(SAFE_BUILTINS),
        "math": math,
        "random": random,
        "re": re,
    }
    exec(compile(source, str(p), "exec"), globals_dict)
    propose = globals_dict.get("propose_action")
    legal = globals_dict.get("is_legal_action")
    if not callable(propose) or not callable(legal):
        raise HarnessValidationError("Required harness functions are not callable.")
    formatter: Any = globals_dict.get("format_observation")
    parser: Any = globals_dict.get("parse_action")
    system_prompt = globals_dict.get("SYSTEM_PROMPT")
    user_prompt = globals_dict.get("USER_PROMPT")
    return HarnessModule(
        path=p,
        propose_action=propose,
        is_legal_action=legal,
        source=source,
        format_observation=formatter if callable(formatter) else None,
        parse_action=parser if callable(parser) else None,
        system_prompt=str(system_prompt) if isinstance(system_prompt, str) else None,
        user_prompt=str(user_prompt) if isinstance(user_prompt, str) else None,
        format_retry_budget=_optional_int(globals_dict, "FORMAT_RETRY_BUDGET"),
        illegal_move_retry_budget=_optional_int(globals_dict, "ILLEGAL_MOVE_RETRY_BUDGET"),
        max_turns=_optional_int(globals_dict, "MAX_TURNS"),
    )
