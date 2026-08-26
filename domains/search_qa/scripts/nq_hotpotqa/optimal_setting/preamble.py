"""Pre-import patches for the verl trainer entrypoint.

Applied to address two issues observed in our run logs (advised by an external
agent reviewing the 24873 trace):

  Priority 2 — Ray Python-worker prestart storm.
    Ray's raylet warned "384 PYTHON worker processes have been started ...
    6x the maximum expected startup concurrency (64)" because most @ray.remote
    decorators in verl are bare and Ray treats them as num_cpus=0/fractional,
    so the scheduler over-spawns prestart workers. Forcing num_cpus=1 on every
    decorator that doesn't set it explicitly bounds the worker pool to the
    real CPU budget and stops the storm. (Decorators that already specify
    num_cpus or any of {num_gpus, resources, accelerator_type, memory} are
    left untouched so verl's intentional CPU/GPU partitions still hold.)

  Priority 3 — log-spam reduction.
    BaseTool.__init__ in verl/tools/base_tool.py prints the entire tool schema
    as pretty JSON for every constructed tool, which gets repeated 381+ times
    across rollout cluster spam. This patch silences that single print call
    while leaving the rest of stdout untouched.

These are import-time patches applied before any verl module is loaded, so
trainer/agent/reward decorators all see the patched ray.remote when their
modules import. After the patches we delegate to verl.trainer.main_ppo's
Hydra-decorated main() so the rest of the pipeline runs unchanged.
"""
import json
import logging
import sys

# ---- Priority 2: default num_cpus=1 on @ray.remote without explicit budget ----
import ray

_RESOURCE_KEYS = {"num_cpus", "num_gpus", "resources", "accelerator_type", "memory"}
_orig_remote = ray.remote


def _patched_remote(*args, **kwargs):
    # Two call shapes:
    #   @ray.remote                      → ray.remote(cls_or_fn)            (positional, no kwargs)
    #   @ray.remote(num_cpus=…)          → ray.remote(num_cpus=…)(cls_or_fn) (kwargs only)
    #
    # We only want to inject num_cpus=1 on shape #1 (no resource budget set).
    # Shape #2 already declares its own budget — leave it alone.
    if args and not kwargs:
        # Bare @ray.remote applied directly to a class/function. Wrap it with
        # num_cpus=1 unless the underlying class/function has its own ray
        # options elsewhere (rare). Safe default.
        return _orig_remote(num_cpus=1)(args[0])
    if any(k in kwargs for k in _RESOURCE_KEYS):
        # Caller specified some resource budget — respect it verbatim.
        return _orig_remote(*args, **kwargs)
    # Caller passed only non-resource options (e.g. concurrency_groups). Add
    # default num_cpus=1.
    kwargs["num_cpus"] = 1
    return _orig_remote(*args, **kwargs)


ray.remote = _patched_remote
print("[preamble] ray.remote patched to default num_cpus=1 on bare decorators",
      file=sys.stderr, flush=True)


# ---- Priority 3: silence the per-tool schema dump in verl/tools/base_tool.py --
# BaseTool.__init__ does `print(json.dumps(self.tool_schema.model_dump(...), indent=2))`
# once per tool instance. With multi-turn rollout we have many tool instances,
# each printing the same long JSON, which clutters logs without information.
import verl.tools.base_tool as _bt

_orig_basetool_init = _bt.BaseTool.__init__


def _quiet_basetool_init(self, config, tool_schema):
    # Faithful reproduction of upstream verl/tools/base_tool.py:36-41 minus the
    # `print(json.dumps(...))` line at the end.
    self.config = config
    self.tool_schema = tool_schema or self.get_openai_tool_schema()
    assert self.tool_schema is not None, "Tool schema is not set!"
    self.name = self.tool_schema.function.name
    # (skipped) print(json.dumps(self.tool_schema.model_dump(...), indent=2))


_bt.BaseTool.__init__ = _quiet_basetool_init
print("[preamble] BaseTool.__init__ tool-schema print silenced",
      file=sys.stderr, flush=True)


# Optionally bump verl logger level so DEBUG-ish messages are quieter; the
# trainer's argparse / Hydra still controls console verbosity but a lot of
# library-internal logs go through stdlib logging.
logging.getLogger("verl").setLevel(logging.WARNING)


# ---- Priority 4: meta-harness HARNESS_PATH bookkeeping ----------------------
# When HARNESS_PATH is set, verl/utils/dataset/rl_dataset.py and
# verl/experimental/agent_loop/tool_agent_loop.py call into
# verl/utils/meta_harness_hook.py to override SYSTEM_PROMPT / USER_PROMPT_TEMPLATE
# / env_response with the candidate's. We just log the active path here so the
# training log makes the harness wiring visible at job start.
import os as _os

_harness_path = _os.environ.get("HARNESS_PATH", "").strip()
if _harness_path:
    print(f"[preamble] HARNESS_PATH={_harness_path} (meta-harness candidate active)",
          file=sys.stderr, flush=True)
    try:
        from verl.utils.meta_harness_hook import (
            _cache_prompt_strings,
            get_candidate_env_class,
        )

        klass = get_candidate_env_class()
        if klass is None:
            print("[preamble] WARNING: HARNESS_PATH set but CandidateEnv not loadable — "
                  "training will proceed with default verl behavior",
                  file=sys.stderr, flush=True)
        else:
            print(f"[preamble] CandidateEnv = {klass.__module__}.{klass.__name__}",
                  file=sys.stderr, flush=True)
            # Stash SYSTEM_PROMPT / USER_PROMPT_TEMPLATE in env vars so child
            # processes (HF datasets filter pool, Ray AgentLoopWorker actors)
            # can apply the override WITHOUT re-importing the candidate (which
            # drags in the full verifiers tree from FSx — caused 30+ min hangs
            # in run 29044 under FSx contention).
            _cache_prompt_strings(klass)
    except Exception as exc:
        print(f"[preamble] WARNING: meta-harness hook init failed: {exc}",
              file=sys.stderr, flush=True)
else:
    print("[preamble] HARNESS_PATH unset (default verl rollout)",
          file=sys.stderr, flush=True)


# ---- Hand off to the real entrypoint ------------------------------------------
# verl.trainer.main_ppo defines a Hydra-decorated `main()` that re-parses sys.argv
# (we leave argv intact so user-supplied Hydra overrides flow through unchanged).
from verl.trainer.main_ppo import main as _main  # noqa: E402

if __name__ == "__main__":
    _main()
