"""
File-based persistence layer.

Design decisions
----------------
- All writes go through _atomic_write (temp-file + os.replace) so a crash
  mid-write cannot corrupt existing files.
- memory.jsonl is append-only during normal operation.  After compression,
  rewrite_iterations() replaces it with only the kept iterations, keeping the
  file compact.
- load_all_iterations() reads only lines that are currently in the file,
  so loading after compression will never re-inflate the in-memory list.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import List, Optional

from .iteration import CompressedMemory, ExperimentState, Iteration


class Storage:
    def __init__(self, storage_dir: str, experiment_name: str) -> None:
        self.base = Path(storage_dir) / experiment_name
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
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, delete=False, suffix=".tmp", encoding="utf-8"
        ) as fh:
            fh.write(content)
            tmp = fh.name
        os.replace(tmp, path)  # atomic on POSIX; best-effort on Windows

    def exists(self) -> bool:
        return (self.base / "state.json").exists()

    # ------------------------------------------------------------------ #
    #  Experiment state
    # ------------------------------------------------------------------ #

    def save_state(self, state: ExperimentState) -> None:
        self._atomic_write(self.base / "state.json", json.dumps(state.to_dict(), indent=2))

    def load_state(self) -> Optional[ExperimentState]:
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

    def load_all_iterations(self) -> List[Iteration]:
        """Read every line currently in memory.jsonl.  Returns [] if the file is absent."""
        path = self.base / "memory.jsonl"
        if not path.exists():
            return []
        result: List[Iteration] = []
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

    def rewrite_iterations(self, iterations: List[Iteration]) -> None:
        """
        Atomically replace memory.jsonl with exactly the given iterations.
        Called after compression to keep the file compact.
        """
        content = "".join(json.dumps(it.to_dict()) + "\n" for it in iterations)
        self._atomic_write(self.base / "memory.jsonl", content)

    # ------------------------------------------------------------------ #
    #  Compressed memory
    # ------------------------------------------------------------------ #

    def save_compressed(self, entries: List[CompressedMemory]) -> None:
        self._atomic_write(
            self.base / "compressed.json",
            json.dumps([e.to_dict() for e in entries], indent=2),
        )

    def load_compressed(self) -> List[CompressedMemory]:
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

    def load_artifact(self, name: str) -> Optional[str]:
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
