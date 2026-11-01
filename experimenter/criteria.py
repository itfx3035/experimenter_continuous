"""
Criteria — generated once by the LLM at experiment start.

The LLM reads the user's single prompt and produces:
  - A plain-language description of what the task requires.
  - A list of concrete, measurable success indicators.
  - An evaluation rubric (how to map results to a 0.0–1.0 score).

Whether code execution is needed is specified explicitly by the user via the
`requires_execution` argument on `Experiment`, not inferred by the LLM.

The execution harness (Python source that is appended to each candidate
implementation and prints {"passed": bool, "details": str} to stdout) is
generated lazily before the first execution and refreshed whenever execution
fails unexpectedly.
"""

from dataclasses import dataclass


@dataclass
class Criteria:
    task_description: str
    success_indicators: list[str]
    evaluation_rubric: str
    requires_execution: bool
    # Populated lazily; empty string means "not yet generated".
    execution_harness: str = ""
    # For execution tasks where the harness reports a `speedup` field:
    # the speedup at which score = 1.0 on a log scale (e.g. 100.0 means
    # 100× speedup earns a perfect score).  0.0 means unset — fall back
    # to LLM scoring.
    speedup_target: float = 0.0

    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        return {
            "task_description": self.task_description,
            "success_indicators": self.success_indicators,
            "evaluation_rubric": self.evaluation_rubric,
            "requires_execution": self.requires_execution,
            "execution_harness": self.execution_harness,
            "speedup_target": self.speedup_target,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Criteria":
        """Tolerant: state.json written by an older version may miss keys or
        store numbers as strings, and a resume should not crash on that."""
        indicators = d.get("success_indicators") or []
        if not isinstance(indicators, list):
            indicators = [str(indicators)]
        try:
            speedup = float(d.get("speedup_target", 0.0))
        except (TypeError, ValueError):
            speedup = 0.0
        return cls(
            task_description=str(d.get("task_description", "")),
            success_indicators=[str(s) for s in indicators],
            evaluation_rubric=str(d.get("evaluation_rubric", "")),
            requires_execution=bool(d.get("requires_execution", False)),
            execution_harness=str(d.get("execution_harness", "")),
            speedup_target=speedup,
        )

    def format_for_llm(self) -> str:
        indicators = "\n".join(f"  {i}. {s}" for i, s in enumerate(self.success_indicators, 1))
        rubric_line = f"Evaluation rubric: {self.evaluation_rubric}"
        if self.requires_execution and self.speedup_target > 1.0:
            rubric_line += f"\nScoring scale: {self.speedup_target:.0f}× speedup = score 1.0 (log scale)"
        return (
            f"Task: {self.task_description}\n\n"
            f"Success indicators:\n{indicators}\n\n"
            f"{rubric_line}"
        )
