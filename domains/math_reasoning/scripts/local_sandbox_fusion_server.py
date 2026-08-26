#!/usr/bin/env python3
"""Small local Sandbox Fusion-compatible HTTP server for ReTool.

The verl ReTool tool calls a Sandbox Fusion endpoint with POST /run_code and
expects a JSON response with compile_result and run_result. This server
implements the Python subset needed for ReTool math rollouts so Slurm jobs can
be self-contained when no cluster-wide Sandbox Fusion service is available.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def _finished_result(stdout: str = "", stderr: str = "", return_code: int = 0, execution_time: float = 0.0) -> dict:
    return {
        "status": "Finished",
        "stdout": stdout,
        "stderr": stderr,
        "return_code": return_code,
        "execution_time": execution_time,
    }


def _timeout_result(stdout: str = "", stderr: str = "", execution_time: float = 0.0) -> dict:
    return {
        "status": "TimeLimitExceeded",
        "stdout": stdout,
        "stderr": stderr,
        "return_code": -9,
        "execution_time": execution_time,
    }


def _set_limits(memory_limit_mb: int) -> None:
    # Best-effort limits for child process only. Avoid CPU RLIMIT here because
    # Python process startup can be noisy on loaded nodes; subprocess timeout is
    # the primary runtime guard.
    if memory_limit_mb > 0:
        limit = int(memory_limit_mb) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def run_python_code(payload: dict[str, Any], python_bin: str) -> dict[str, Any]:
    code = str(payload.get("code") or "")
    stdin = payload.get("stdin")
    if stdin is None:
        stdin = ""
    else:
        stdin = str(stdin)

    compile_timeout = int(payload.get("compile_timeout") or 10)
    run_timeout = int(payload.get("run_timeout") or compile_timeout)
    timeout = max(1, run_timeout)
    memory_limit_mb = int(payload.get("memory_limit_MB") or 1024)
    language = str(payload.get("language") or "python").lower()

    if language not in {"python", "python3"}:
        return {
            "status": "Failed",
            "compile_result": _finished_result(),
            "run_result": _finished_result(
                stderr=f"Unsupported language for local sandbox: {language}\n",
                return_code=2,
            ),
        }

    with tempfile.TemporaryDirectory(prefix="retool-sandbox-") as tmp:
        script_path = Path(tmp) / "main.py"
        script_path.write_text(code, encoding="utf-8")

        start = time.monotonic()
        try:
            proc = subprocess.run(
                [python_bin, "-I", str(script_path)],
                input=stdin,
                text=True,
                capture_output=True,
                timeout=timeout,
                cwd=tmp,
                env={
                    "PATH": os.environ.get("PATH", ""),
                    "PYTHONIOENCODING": "utf-8",
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                },
                preexec_fn=lambda: _set_limits(memory_limit_mb),
            )
            duration = time.monotonic() - start
            top_status = "Success" if proc.returncode == 0 else "Failed"
            return {
                "status": top_status,
                "compile_result": _finished_result(),
                "run_result": _finished_result(
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    return_code=proc.returncode,
                    execution_time=duration,
                ),
            }
        except subprocess.TimeoutExpired as exc:
            duration = time.monotonic() - start
            stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", "replace")
            stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", "replace")
            return {
                "status": "Failed",
                "compile_result": _finished_result(),
                "run_result": _timeout_result(stdout=stdout, stderr=stderr, execution_time=duration),
            }
        except Exception as exc:
            duration = time.monotonic() - start
            return {
                "status": "Failed",
                "compile_result": _finished_result(),
                "run_result": _finished_result(
                    stderr=f"LocalSandboxError: {type(exc).__name__}: {exc}\n",
                    return_code=1,
                    execution_time=duration,
                ),
            }


class SandboxHandler(BaseHTTPRequestHandler):
    server_version = "RetoolLocalSandbox/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        if getattr(self.server, "quiet", False):
            return
        super().log_message(fmt, *args)

    def _send_json(self, status: int, body: dict[str, Any]) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path in {"/health", "/healthz"}:
            self._send_json(200, {"status": "ok"})
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/run_code":
            self._send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8") if raw else "{}")
        except Exception as exc:
            self._send_json(400, {"status": "Failed", "error": f"bad request: {exc}"})
            return

        future = self.server.executor.submit(run_python_code, payload, self.server.python_bin)
        self._send_json(200, future.result())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max-workers", type=int, default=128)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), SandboxHandler)
    server.executor = ThreadPoolExecutor(max_workers=args.max_workers)
    server.python_bin = args.python_bin
    server.quiet = args.quiet
    print(
        f"[local_sandbox] serving http://{args.host}:{args.port}/run_code "
        f"workers={args.max_workers} python={args.python_bin}",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.executor.shutdown(wait=False, cancel_futures=True)
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
