"""Optional process-wide debugging hooks for chess-puzzle Slurm runs.

This module is imported automatically by Python when the repository root is on
PYTHONPATH. It is inert unless CHESS_PUZZLE_RSFT_WORKER_TRACEBACK_S is set.
"""

from __future__ import annotations

import faulthandler
import os
import sys


def _positive_int(name: str) -> int:
    try:
        return int(os.environ.get(name, "0"))
    except ValueError:
        return 0


_traceback_s = _positive_int("CHESS_PUZZLE_RSFT_WORKER_TRACEBACK_S")
if _traceback_s > 0:
    repeat = os.environ.get("CHESS_PUZZLE_RSFT_WORKER_TRACEBACK_REPEAT", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    faulthandler.dump_traceback_later(_traceback_s, repeat=repeat, file=sys.stderr)
    print(
        f"[sitecustomize] faulthandler traceback dump enabled every {_traceback_s}s "
        f"repeat={repeat} pid={os.getpid()}",
        file=sys.stderr,
        flush=True,
    )
