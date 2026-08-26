"""Pre-import patches for the chess-puzzle disaggregated RSFT entrypoint."""

from __future__ import annotations

import faulthandler
import os
import sys

import ray

_traceback_s_raw = os.environ.get("CHESS_PUZZLE_RSFT_STARTUP_TRACEBACK_S", "0")
try:
    _traceback_s = int(_traceback_s_raw)
except ValueError:
    _traceback_s = 0
if _traceback_s > 0:
    faulthandler.dump_traceback_later(_traceback_s, repeat=True, file=sys.stderr)
    print(
        f"[chess_puzzle_preamble_disagg_rsft] startup traceback dump enabled every {_traceback_s}s",
        file=sys.stderr,
        flush=True,
    )

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
print("[chess_puzzle_preamble_disagg_rsft] ray.remote patched to default num_cpus=1", file=sys.stderr, flush=True)

import verl.experimental.agent_loop.chess_puzzle_agent_loop  # noqa: F401,E402

print("[chess_puzzle_preamble_disagg_rsft] chess_puzzle_agent loop registered", file=sys.stderr, flush=True)

if os.environ.get("CHESS_PUZZLE_HARNESS_PATH") or os.environ.get("HARNESS_PATH"):
    harness = os.environ.get("CHESS_PUZZLE_HARNESS_PATH") or os.environ.get("HARNESS_PATH")
    print(f"[chess_puzzle_preamble_disagg_rsft] harness={harness}", file=sys.stderr, flush=True)

from verl.trainer.main_textarena_disagg_rsft import main as _main  # noqa: E402


if __name__ == "__main__":
    _main()

