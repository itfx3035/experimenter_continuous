"""
Data model: Iteration, CompressedMemory, ExperimentState.

All objects are serialisable to plain dicts so they can be persisted as JSON.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Tuple


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
        """Rough token count (characters / 4) for context-budget tracking."""
        return (
            len(self.analysis)
            + len(self.proposal)
            + len(self.implementation)
            + len(self.execution_output)
            + len(self.evaluation_reasoning)
        ) // 4

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
    def from_dict(cls, d: dict) -> "Iteration":
        return cls(
            number=d["number"],
            timestamp=d.get("timestamp", ""),
            analysis=d.get("analysis", ""),
            proposal=d.get("proposal", ""),
            implementation=d.get("implementation", ""),
            execution_output=d.get("execution_output", ""),
            execution_success=d.get("execution_success", True),
            execution_elapsed=d.get("execution_elapsed", 0.0),
            evaluation_score=d.get("evaluation_score", 0.0),
            evaluation_reasoning=d.get("evaluation_reasoning", ""),
        )


# ------------------------------------------------------------------ #
#  CompressedMemory
# ------------------------------------------------------------------ #

@dataclass
class CompressedMemory:
    """LLM-generated summary for a range of iterations that have been evicted."""

    iteration_range: Tuple[int, int]   # (first, last) inclusive
    summary: str
    key_findings: List[str]
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
    def from_dict(cls, d: dict) -> "CompressedMemory":
        return cls(
            iteration_range=tuple(d["iteration_range"]),  # type: ignore[arg-type]
            summary=d["summary"],
            key_findings=d["key_findings"],
            best_score=float(d["best_score"]),
            timestamp=d.get("timestamp", ""),
        )

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
    # "created" | "running" | "paused" | "completed" | "stopped"
    status: str = "created"
    total_elapsed_seconds: float = 0.0
    # Serialised Criteria dict; None until criteria are generated.
    criteria: Optional[dict] = None

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
            "criteria": self.criteria,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentState":
        return cls(
            name=d["name"],
            prompt=d["prompt"],
            current_iteration=d.get("current_iteration", 0),
            max_iterations=d.get("max_iterations", 50),
            start_time=d.get("start_time", 0.0),
            last_update=d.get("last_update", 0.0),
            best_iteration=d.get("best_iteration", 0),
            best_score=d.get("best_score", 0.0),
            status=d.get("status", "created"),
            total_elapsed_seconds=d.get("total_elapsed_seconds", 0.0),
            criteria=d.get("criteria"),
        )
