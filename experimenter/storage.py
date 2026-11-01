"""
File-based persistence layer.

Design decisions
----------------
- All writes go through _atomic_write (temp-file + os.replace) so a crash
  mid-write cannot corrupt existing files; the temp file is unlinked if the
  write itself fails.
- memory.jsonl is append-only during normal operation.  After compression,
  rewrite_iterations() replaces it with only the kept iterations, keeping the
  file compact.
- Everything evicted from memory.jsonl is appended to archive.jsonl first, so a
  compacted (or drop-oldest-trimmed) run is still fully inspectable on disk.
- load_all_iterations() reads only lines that are currently in the file,
  so loading after compression will never re-inflate the in-memory list.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .iteration import CompressedMemory, ExperimentState, Iteration


def _process_alive(pid: int) -> bool:
    """Best-effort liveness check; assume alive when the check is unavailable."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True   # exists, owned by another user
    except (OSError, AttributeError):  # Windows: os.kill(pid, 0) is not supported
        return True


class Storage:
    def __init__(self, storage_dir: str, experiment_name: str) -> None:
        # Sanitise the name: "../../etc" must not escape storage_dir.
        safe = os.path.basename(str(experiment_name).strip().strip(os.sep)) or "experiment"
        self.name = safe
        self.base = Path(storage_dir) / safe
        self.artifacts_dir = self.base / "artifacts"
        self.code_dir = self.base / "code"
        self.tmp_dir = self.code_dir / "tmp"
        self._ensure_dirs()

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _ensure_dirs(self) -> None:
        for d in (self.base, self.artifacts_dir, self.code_dir, self.tmp_dir):
            d.mkdir(parents=True, exist_ok=True)

    def _atomic_write(self, path: Path, content: str) -> None:
        """Write content to a sibling temp file, then atomically rename it."""
        tmp: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=path.parent, delete=False, suffix=".tmp", encoding="utf-8"
            ) as fh:
                fh.write(content)
                tmp = fh.name
            os.replace(tmp, path)  # atomic on POSIX; best-effort on Windows
            tmp = None
        finally:
            # Do not leave a stray temp file behind if the write/rename failed.
            if tmp:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    def exists(self) -> bool:
        return (self.base / "state.json").exists()

    # ------------------------------------------------------------------ #
    #  Lock (two processes writing the same experiment would corrupt it)
    # ------------------------------------------------------------------ #

    def lock_path(self) -> Path:
        return self.base / "lock"

    def acquire_lock(self) -> bool:
        """
        Claim the experiment directory for this process.

        Returns False when a *live* process already holds the lock.  A lock left
        behind by a crashed process is stale (its pid is dead) and is taken over.
        """
        path = self.lock_path()
        pid = os.getpid()
        if path.exists():
            holder = self._read_lock_pid(path)
            if holder is not None and holder != pid and _process_alive(holder):
                return False
        self._atomic_write(path, f"{pid}\n")
        return True

    def release_lock(self) -> None:
        """Release the lock only if we still own it (never delete another run's lock)."""
        path = self.lock_path()
        try:
            if path.exists() and self._read_lock_pid(path) == os.getpid():
                os.unlink(path)
        except OSError:
            pass

    @staticmethod
    def _read_lock_pid(path: Path) -> int | None:
        try:
            return int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return None

    def held_by_other(self) -> int | None:
        """Pid of a live lock holder, or None if the lock is free/stale/ours."""
        pid = self._read_lock_pid(self.lock_path())
        if pid is None or pid == os.getpid():
            return None
        return pid if _process_alive(pid) else None

    # ------------------------------------------------------------------ #
    #  Experiment state
    # ------------------------------------------------------------------ #

    def save_state(self, state: ExperimentState) -> None:
        self._atomic_write(self.base / "state.json", json.dumps(state.to_dict(), indent=2))

    def load_state(self) -> ExperimentState | None:
        path = self.base / "state.json"
        if not path.exists():
            return None
        try:
            return ExperimentState.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, KeyError) as exc:
            raise ValueError(f"Corrupt state.json in {self.base}: {exc}") from exc

    # ------------------------------------------------------------------ #
    #  Iterations  (append-only JSONL, rewritten after compression)
    # ------------------------------------------------------------------ #

    def append_iteration(self, iteration: Iteration) -> None:
        with open(self.base / "memory.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(iteration.to_dict()) + "\n")

    def load_all_iterations(self) -> list[Iteration]:
        """Read every line currently in memory.jsonl.  Returns [] if the file is absent."""
        path = self.base / "memory.jsonl"
        if not path.exists():
            return []
        result: list[Iteration] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    result.append(Iteration.from_dict(json.loads(line)))
                except (json.JSONDecodeError, KeyError):
                    pass  # skip corrupt lines rather than crashing
        return result

    def rewrite_iterations(self, iterations: list[Iteration]) -> None:
        """
        Atomically replace memory.jsonl with exactly the given iterations.
        Called after compression to keep the file compact.
        """
        content = "".join(json.dumps(it.to_dict()) + "\n" for it in iterations)
        self._atomic_write(self.base / "memory.jsonl", content)

    def archive_iterations(self, iterations: list[Iteration]) -> None:
        """
        Append iterations that are being evicted from memory.jsonl to
        archive.jsonl.  Without this, compression and the drop-oldest fallback
        would destroy that history on disk.
        """
        if not iterations:
            return
        path = self.base / "archive.jsonl"
        with open(path, "a", encoding="utf-8") as fh:
            for it in iterations:
                fh.write(json.dumps(it.to_dict()) + "\n")

    # ------------------------------------------------------------------ #
    #  Compressed memory
    # ------------------------------------------------------------------ #

    def save_compressed(self, entries: list[CompressedMemory]) -> None:
        self._atomic_write(
            self.base / "compressed.json",
            json.dumps([e.to_dict() for e in entries], indent=2),
        )

    def load_compressed(self) -> list[CompressedMemory]:
        path = self.base / "compressed.json"
        if not path.exists():
            return []
        try:
            return [CompressedMemory.from_dict(d) for d in json.loads(path.read_text(encoding="utf-8"))]
        except (json.JSONDecodeError, KeyError):
            return []

    # ------------------------------------------------------------------ #
    #  Artifacts (best result, etc.)
    # ------------------------------------------------------------------ #

    def save_artifact(self, name: str, content: str) -> None:
        self._atomic_write(self.artifacts_dir / name, content)

    def load_artifact(self, name: str) -> str | None:
        path = self.artifacts_dir / name
        return path.read_text(encoding="utf-8") if path.exists() else None

    # ------------------------------------------------------------------ #
    #  Code snapshots
    # ------------------------------------------------------------------ #

    def save_code(self, iteration_number: int, code: str, ext: str = "py") -> None:
        self._atomic_write(self.code_dir / f"iteration_{iteration_number}.{ext}", code)

    def save_execution_result(self, iteration_number: int, result: dict) -> None:
        self._atomic_write(
            self.code_dir / f"iteration_{iteration_number}_result.json",
            json.dumps(result, indent=2),
        )
