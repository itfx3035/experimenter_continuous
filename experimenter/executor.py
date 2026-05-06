"""
SubprocessExecutor — runs LLM-generated Python code in a child process.

Why subprocess instead of exec()
---------------------------------
- Real OS-level timeout: the child process is killed (SIGKILL) when it exceeds
  the time limit, not just "signalled".
- Process isolation: the child has its own memory space; it cannot corrupt the
  parent's state.
- No exec() with full builtins: the parent never calls eval/exec on untrusted
  code, so there is no risk of the generated code accessing parent-process
  internals directly.

The execution model
-------------------
The caller provides two Python source strings:

  implementation — the candidate code (function definitions, class, etc.)
  harness        — appended after the implementation; when run as __main__,
                   calls the implementation and prints a single JSON object
                   to stdout with at least {"passed": <bool>, "details": <str>}.

Both are concatenated into a single temp file and run with `sys.executable`.
The temp file is removed regardless of outcome.

Note on security
----------------
The child process still has the same OS-level permissions as the parent.  For
production use, consider running it in a container or with a reduced-privilege
OS account.  This is a significant improvement over exec() but is not a full
sandbox.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Optional

# Regex that matches an unindented `if __name__ == "__main__":` guard.
# We replace it with `if False:` so that the implementation's self-test /
# benchmark block does not run when the harness is in control.
_MAIN_GUARD_RE = re.compile(
    r'^(if\s+__name__\s*==\s*[\'"]__main__[\'"]\s*:)',
    re.MULTILINE,
)


@dataclass
class ExecutionResult:
    success: bool
    stdout: str
    stderr: str
    elapsed_seconds: float
    timed_out: bool = False

    @property
    def combined_output(self) -> str:
        parts = []
        if self.stdout.strip():
            parts.append(self.stdout.strip())
        if self.stderr.strip():
            parts.append(f"[stderr]\n{self.stderr.strip()}")
        return "\n".join(parts)

    def parsed_json(self) -> Optional[dict]:
        """Try to parse the last JSON object from stdout."""
        for line in reversed(self.stdout.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    pass
        return None


class SubprocessExecutor:
    def __init__(self, default_timeout: int = 30, tmp_dir: Optional[str] = None) -> None:
        self.default_timeout = default_timeout
        self._tmp_dir = tmp_dir

    def run(
        self,
        implementation: str,
        harness: str,
        timeout: Optional[int] = None,
    ) -> ExecutionResult:
        """
        Run  implementation + harness  as a Python script in a child process.

        Returns ExecutionResult.  The child process is hard-killed if it
        exceeds the timeout; execution_result.timed_out will be True in that
        case.
        """
        timeout = timeout if timeout is not None else self.default_timeout
        # Suppress the implementation's __main__ block so the harness owns
        # execution and controls what is printed to stdout.
        safe_impl = _MAIN_GUARD_RE.sub(r"if False:  # __main__ suppressed by harness", implementation)
        script = f"{safe_impl}\n\n{harness}"

        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                prefix="experimeter_",
                dir=self._tmp_dir,
                delete=False,
                encoding="utf-8",
            ) as fh:
                fh.write(script)
                tmp_path = fh.name

            t0 = time.monotonic()
            try:
                proc = subprocess.run(
                    [sys.executable, tmp_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                elapsed = time.monotonic() - t0
                return ExecutionResult(
                    success=proc.returncode == 0,
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    elapsed_seconds=elapsed,
                    timed_out=False,
                )
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - t0
                return ExecutionResult(
                    success=False,
                    stdout="",
                    stderr=f"Execution timed out after {timeout}s",
                    elapsed_seconds=elapsed,
                    timed_out=True,
                )
        except Exception as exc:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=str(exc),
                elapsed_seconds=0.0,
            )
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
