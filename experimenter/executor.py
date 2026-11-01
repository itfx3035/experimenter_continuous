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

Both are concatenated into a single temp file and run with `sys.executable -I`
(isolated mode: no user site-packages, no PYTHONPATH, no PYTHONSTARTUP, no
.pth execution), with a deliberately minimal environment.  The temp file is
removed regardless of outcome.

Note on security
----------------
The child process still has the same OS-level permissions as the parent.  For
production use, consider running it in a container or with a reduced-privilege
OS account.  This is a significant improvement over exec() but is not a full
sandbox.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

from .config import ExecutorConfig
from .parsing import parse_json_output, strip_main_guard

try:  # POSIX only; used for best-effort resource caps
    import resource
except ImportError:  # pragma: no cover - Windows
    resource = None  # type: ignore[assignment]


def _child_env(scrub: bool = True) -> dict[str, str]:
    """
    Minimal environment for the child.

    The generated code is untrusted: inheriting the full environment lets
    PYTHONSTARTUP, PYTHONPATH or a user .pth file execute arbitrary code before
    the harness ever runs.  Keeping the environment small makes runs reproducible
    as well as safer.  Pass scrub=False to deliberately inherit everything.
    """
    if not scrub:
        return dict(os.environ)
    env = {"PATH": os.environ.get("PATH", ""), "LANG": "C.UTF-8"}
    for var in ("SYSTEMROOT", "TEMP", "TMP"):
        if var in os.environ:
            env[var] = os.environ[var]
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONHASHSEED"] = "0"
    return env


def _limits_preexec(memory_mb: int | None, cpu_seconds: int | None):
    """Return a preexec_fn applying RLIMIT_AS / RLIMIT_CPU, or None."""
    if resource is None or (memory_mb is None and cpu_seconds is None):
        return None

    def _apply() -> None:  # runs in the child, before exec
        if memory_mb:
            lim = int(memory_mb) * 1024 * 1024
            try:
                resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
            except (ValueError, OSError, AttributeError):
                pass
        if cpu_seconds:
            try:
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
            except (ValueError, OSError, AttributeError):
                pass

    return _apply


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

    def parsed_json(self) -> dict | None:
        """
        Return the last JSON object from stdout (also handles pretty-printed
        multi-line JSON).  Returns None for non-object JSON — the callers use
        .get("score") directly, so a list here would crash them.
        """
        return parse_json_output(self.stdout)


class SubprocessExecutor:
    def __init__(
        self,
        default_timeout: int = 30,
        tmp_dir: str | None = None,
        memory_mb: int | None = None,
        cpu_seconds: int | None = None,
        isolated: bool = True,
        scrub_env: bool = True,
        config: ExecutorConfig | None = None,
    ) -> None:
        if config is not None:
            default_timeout = config.timeout
            memory_mb = config.memory_mb
            cpu_seconds = config.cpu_seconds
            isolated = config.isolated
            scrub_env = config.scrub_env
        self.default_timeout = default_timeout
        self._tmp_dir = tmp_dir
        self.memory_mb = memory_mb
        self.cpu_seconds = cpu_seconds
        self.isolated = isolated
        self.scrub_env = scrub_env

    def _argv(self, script_path: str) -> list[str]:
        args = [sys.executable]
        if self.isolated:
            args.append("-I")
        args.append(script_path)
        return args

    def run(
        self,
        implementation: str,
        harness: str,
        timeout: int | None = None,
    ) -> ExecutionResult:
        """
        Run  implementation + harness  as a Python script in a child process.

        Returns ExecutionResult.  The child process is hard-killed if it
        exceeds the timeout; execution_result.timed_out will be True in that
        case.  Output already produced before the timeout is kept — losing it
        makes long benchmarks look like total failures.
        """
        timeout = timeout if timeout is not None else self.default_timeout
        # Suppress the implementation's __main__ block so the harness owns
        # execution and controls what is printed to stdout.
        _, safe_impl = strip_main_guard(implementation)
        script = f"{safe_impl}\n\n{harness}"

        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                prefix="experimenter_",
                dir=self._tmp_dir,
                delete=False,
                encoding="utf-8",
            ) as fh:
                fh.write(script)
                tmp_path = fh.name

            t0 = time.monotonic()
            proc = subprocess.Popen(
                self._argv(tmp_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                env=_child_env(self.scrub_env),
                text=True,
                errors="replace",
                preexec_fn=_limits_preexec(self.memory_mb, self.cpu_seconds),
            )
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
                timed_out = False
            except subprocess.TimeoutExpired:
                proc.kill()
                # Drain whatever the child already printed before being killed.
                try:
                    stdout, stderr = proc.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    stdout, stderr = "", ""
                timed_out = True
                stderr = (stderr or "") + f"\nExecution timed out after {timeout}s"

            elapsed = time.monotonic() - t0
            return ExecutionResult(
                success=proc.returncode == 0 and not timed_out,
                stdout=stdout or "",
                stderr=stderr or "",
                elapsed_seconds=elapsed,
                timed_out=timed_out,
            )
        except Exception as exc:  # noqa: BLE001
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


__all__ = ["ExecutionResult", "SubprocessExecutor"]
