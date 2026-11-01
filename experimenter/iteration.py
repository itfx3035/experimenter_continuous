"""
Data model: Iteration, CompressedMemory, ExperimentState.

All objects are serialisable to plain dicts so they can be persisted as JSON.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from .config import TOKEN_CHAR_FACTOR


def _as_int(value, default: int = 0) -> int:
    """Coerce a persisted value to int without raising (JSON can hold strings)."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_str(value, default: str = "") -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return default
    return str(value)


def _as_str_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_as_str(v) for v in value]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ------------------------------------------------------------------ #
#  Iteration
# ------------------------------------------------------------------ #

@dataclass
class Iteration:
    """One complete turn of the improvement loop."""

    number: int
    timestamp: str = field(default_factory=_now)

    # Set during STRATEGIC REVIEW
    analysis: str = ""
    proposal: str = ""

    # Set during IMPLEMENT
    implementation: str = ""

    # Set during EXECUTE (only when criteria.requires_execution is True)
    execution_output: str = ""   # combined stdout + stderr, truncated
    execution_success: bool = True
    execution_elapsed: float = 0.0

    # Set during EVALUATE
    evaluation_score: float = 0.0   # strictly 0.0 – 1.0
    evaluation_reasoning: str = ""

    # ---------------------------------------------------------------- #

    def token_estimate(self) -> int:
        """Rough token count (characters / TOKEN_CHAR_FACTOR) for context-budget tracking."""
        return (
            len(self.analysis)
            + len(self.proposal)
            + len(self.implementation)
            + len(self.execution_output)
            + len(self.evaluation_reasoning)
        ) // TOKEN_CHAR_FACTOR

    def to_dict(self) -> dict:
        return {
            "number": self.number,
            "timestamp": self.timestamp,
            "analysis": self.analysis,
            "proposal": self.proposal,
            "implementation": self.implementation,
            "execution_output": self.execution_output,
            "execution_success": self.execution_success,
            "execution_elapsed": self.execution_elapsed,
            "evaluation_score": self.evaluation_score,
            "evaluation_reasoning": self.evaluation_reasoning,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Iteration:
        """Tolerant: a corrupt field falls back to a default instead of raising,
        so one bad line in memory.jsonl cannot abort a resume."""
        return cls(
            number=_as_int(d.get("number"), 0),
            timestamp=_as_str(d.get("timestamp", "")),
            analysis=_as_str(d.get("analysis", "")),
            proposal=_as_str(d.get("proposal", "")),
            implementation=_as_str(d.get("implementation", "")),
            execution_output=_as_str(d.get("execution_output", "")),
            execution_success=bool(d.get("execution_success", True)),
            execution_elapsed=_as_float(d.get("execution_elapsed", 0.0)),
            evaluation_score=_as_float(d.get("evaluation_score", 0.0)),
            evaluation_reasoning=_as_str(d.get("evaluation_reasoning", "")),
        )


# ------------------------------------------------------------------ #
#  CompressedMemory
# ------------------------------------------------------------------ #

@dataclass
class CompressedMemory:
    """LLM-generated summary for a range of iterations that have been evicted."""

    iteration_range: tuple[int, int]   # (first, last) inclusive
    summary: str
    key_findings: list[str]
    best_score: float
    timestamp: str = field(default_factory=_now)

    def to_dict(self) -> dict:
        return {
            "iteration_range": list(self.iteration_range),
            "summary": self.summary,
            "key_findings": self.key_findings,
            "best_score": self.best_score,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CompressedMemory:
        """Tolerant: string/number mix-ups in persisted JSON do not raise."""
        raw_range = d.get("iteration_range") or [0, 0]
        if not isinstance(raw_range, (list, tuple)) or len(raw_range) < 2:
            raw_range = [0, 0]
        return cls(
            iteration_range=(_as_int(raw_range[0]), _as_int(raw_range[1])),
            summary=_as_str(d.get("summary", "")),
            key_findings=_as_str_list(d.get("key_findings", [])),
            best_score=_as_float(d.get("best_score", 0.0)),
            timestamp=_as_str(d.get("timestamp", "")),
        )

    def token_estimate(self) -> int:
        """
        Rough token count for the rendered summary.

        Uses the real rendered size instead of a flat constant, because every
        compressed block is included verbatim in the prompt and the list of
        blocks grows over a long run.
        """
        return len(self.format_for_llm()) // TOKEN_CHAR_FACTOR

    def format_for_llm(self) -> str:
        findings = "\n".join(f"  - {f}" for f in self.key_findings)
        return (
            f"[Compressed iterations {self.iteration_range[0]}–{self.iteration_range[1]}, "
            f"best score: {self.best_score:.2f}]\n"
            f"Summary: {self.summary}\n"
            f"Key findings:\n{findings}"
        )


# ------------------------------------------------------------------ #
#  ExperimentState
# ------------------------------------------------------------------ #

@dataclass
class ExperimentState:
    """Persisted metadata for a single experiment run."""

    name: str
    prompt: str
    current_iteration: int = 0
    max_iterations: int = 50
    start_time: float = 0.0
    last_update: float = 0.0
    best_iteration: int = 0
    best_score: float = 0.0
    # "created" | "running" | "paused" | "completed" | "stopped" | "time_exhausted"
    status: str = "created"
    total_elapsed_seconds: float = 0.0
    # Consecutive iterations without a best-score improvement.  Persisted so a
    # resumed run keeps telling the LLM "the current strategy is not working".
    stagnation_count: int = 0
    # Serialised Criteria dict; None until criteria are generated.
    criteria: dict | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "prompt": self.prompt,
            "current_iteration": self.current_iteration,
            "max_iterations": self.max_iterations,
            "start_time": self.start_time,
            "last_update": self.last_update,
            "best_iteration": self.best_iteration,
            "best_score": self.best_score,
            "status": self.status,
            "total_elapsed_seconds": self.total_elapsed_seconds,
            "stagnation_count": self.stagnation_count,
            "criteria": self.criteria,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ExperimentState:
        """Tolerant: falls back to defaults for missing/oddly-typed fields."""
        return cls(
            name=_as_str(d.get("name", "")),
            prompt=_as_str(d.get("prompt", "")),
            current_iteration=_as_int(d.get("current_iteration", 0)),
            max_iterations=_as_int(d.get("max_iterations", 50), 50),
            start_time=_as_float(d.get("start_time", 0.0)),
            last_update=_as_float(d.get("last_update", 0.0)),
            best_iteration=_as_int(d.get("best_iteration", 0)),
            best_score=_as_float(d.get("best_score", 0.0)),
            status=_as_str(d.get("status", "created"), "created"),
            total_elapsed_seconds=_as_float(d.get("total_elapsed_seconds", 0.0)),
            stagnation_count=_as_int(d.get("stagnation_count", 0)),
            criteria=d.get("criteria"),
        )
