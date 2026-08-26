"""Pre-import patches for the ReTool verl trainer entrypoint.

This mirrors the Search-R1 H200 workaround for Ray worker prestart storms:
bare ``@ray.remote`` decorators get ``num_cpus=1`` unless the caller already
declares a resource budget. It also silences BaseTool's repeated tool-schema
JSON dump, which is extremely noisy during multi-turn rollout.
"""

import logging
import os
import sys

import ray

_RESOURCE_KEYS = {"num_cpus", "num_gpus", "resources", "accelerator_type", "memory"}
_orig_remote = ray.remote


def _patched_remote(*args, **kwargs):
    if args and not kwargs:
        return _orig_remote(num_cpus=1)(args[0])
    if any(key in kwargs for key in _RESOURCE_KEYS):
        return _orig_remote(*args, **kwargs)
    kwargs["num_cpus"] = 1
    return _orig_remote(*args, **kwargs)


ray.remote = _patched_remote
print("[retool_preamble] ray.remote patched to default num_cpus=1", file=sys.stderr, flush=True)

import verl.tools.base_tool as _base_tool  # noqa: E402


def _quiet_basetool_init(self, config, tool_schema):
    self.config = config
    self.tool_schema = tool_schema or self.get_openai_tool_schema()
    assert self.tool_schema is not None, "Tool schema is not set!"
    self.name = self.tool_schema.function.name


_base_tool.BaseTool.__init__ = _quiet_basetool_init
print("[retool_preamble] BaseTool tool-schema print silenced", file=sys.stderr, flush=True)

logging.getLogger("verl").setLevel(logging.WARNING)

_harness_path = os.environ.get("HARNESS_PATH", "").strip()
if _harness_path:
    try:
        from verl.utils.meta_harness_hook import _cache_prompt_strings, get_candidate_env_class

        _klass = get_candidate_env_class()
        if _klass is not None:
            _cache_prompt_strings(_klass)
            print(
                f"[retool_preamble] HARNESS_PATH loaded: {_harness_path} "
                f"({getattr(_klass, '__name__', 'CandidateEnv')})",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(
                f"[retool_preamble] WARNING: HARNESS_PATH did not load: {_harness_path}",
                file=sys.stderr,
                flush=True,
            )
    except Exception as exc:
        print(
            f"[retool_preamble] WARNING: failed to preload HARNESS_PATH={_harness_path}: {exc}",
            file=sys.stderr,
            flush=True,
        )

from verl.trainer.main_ppo import main as _main  # noqa: E402


if __name__ == "__main__":
    _main()
