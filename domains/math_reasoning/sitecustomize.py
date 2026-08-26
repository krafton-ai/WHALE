"""Runtime patches used only when ReTool training opts in.

Python imports ``sitecustomize`` during interpreter startup when this repository
is on ``PYTHONPATH``. Keep the patches behind an environment flag so normal
repo Python commands remain unaffected.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(_to_jsonable(k)): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]

    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return _to_jsonable(value.tolist())
        if isinstance(value, np.generic):
            return _to_jsonable(value.item())
    except Exception:
        pass

    try:
        import torch

        if isinstance(value, torch.Tensor):
            return _to_jsonable(value.detach().cpu().tolist())
    except Exception:
        pass

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _to_jsonable(item())
        except Exception:
            pass

    return str(value)


def _patch_verl_rollout_dump(ray_trainer: Any) -> None:
    if getattr(ray_trainer.RayPPOTrainer, "_retool_json_dump_patch", False):
        return

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for key, values in reward_extra_infos_dict.items():
            try:
                values_len = len(values)
            except TypeError:
                continue
            if values_len == n:
                base_data[key] = values

        lines = []
        for i in range(n):
            entry = {key: _to_jsonable(values[i]) for key, values in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    ray_trainer.RayPPOTrainer._dump_generations = _dump_generations
    ray_trainer.RayPPOTrainer._retool_json_dump_patch = True
    print("[sitecustomize] patched verl RayPPOTrainer rollout JSON dump", flush=True)


def _patch_verl_basetool_print(base_tool: Any) -> None:
    if getattr(base_tool.BaseTool, "_retool_quiet_patch", False):
        return

    def _quiet_basetool_init(self, config, tool_schema):
        self.config = config
        self.tool_schema = tool_schema or self.get_openai_tool_schema()
        assert self.tool_schema is not None, "Tool schema is not set!"
        self.name = self.tool_schema.function.name

    base_tool.BaseTool.__init__ = _quiet_basetool_init
    base_tool.BaseTool._retool_quiet_patch = True
    print("[sitecustomize] silenced verl BaseTool schema print", flush=True)


def _maybe_patch_loaded_modules() -> None:
    ray_trainer = sys.modules.get("verl.trainer.ppo.ray_trainer")
    if ray_trainer is not None and hasattr(ray_trainer, "RayPPOTrainer"):
        _patch_verl_rollout_dump(ray_trainer)

    base_tool = sys.modules.get("verl.tools.base_tool")
    if base_tool is not None and hasattr(base_tool, "BaseTool"):
        _patch_verl_basetool_print(base_tool)


if os.environ.get("RETOOL_PATCH_VERL_RUNTIME") == "1":
    import builtins

    _orig_import = builtins.__import__

    def _patched_import(name, globals=None, locals=None, fromlist=(), level=0):
        module = _orig_import(name, globals, locals, fromlist, level)
        if name.startswith("verl.trainer.ppo.ray_trainer") or name.startswith("verl.tools.base_tool"):
            _maybe_patch_loaded_modules()
        elif name in {"verl.trainer.ppo", "verl.tools"}:
            _maybe_patch_loaded_modules()
        return module

    builtins.__import__ = _patched_import
    _maybe_patch_loaded_modules()
