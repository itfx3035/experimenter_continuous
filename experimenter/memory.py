"""
MemoryManager — in-memory iteration store with LLM-based compression.

Design decisions
----------------
- Context is built with rich content (analysis, proposal, implementation snippet,
  execution output, evaluation reasoning) so the LLM has real signal to work with.
- The "best" iteration bucket and "recent" iterations bucket can overlap; we
  deduplicate by iteration number so the same iteration is never shown twice.
- Scores are in [0.0, 1.0]; thresholds that reference scores use the same range.
- After compression, rewrite_iterations() replaces memory.jsonl so the file
  stays compact and loading after resume will not re-inflate the list.
- The MemoryManager is the single source of truth for the in-memory list; the
  Storage is its persistence backend.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from .config import CONTEXT_TOKEN_BUDGET, COMPRESSION_TRIGGER, CONTEXT_SAFETY_MARGIN
from .criteria import Criteria
from .iteration import CompressedMemory, Iteration
from .llm import LLMClient
from .storage import Storage

logger = logging.getLogger(__name__)

# How many recent iterations to show verbatim in context.
_RECENT_N = 8
# How many top-scoring iterations (outside the recent window) to show.
_BEST_N = 4
# Maximum characters for an implementation snippet in the non-full-code path.
_IMPL_SNIPPET = 500
# Maximum characters for execution output in context.
_EXEC_SNIPPET = 400


class MemoryManager:
    def __init__(self, storage: Storage, llm: LLMClient) -> None:
        self._storage = storage
        self._llm = llm
        self.iterations: List[Iteration] = []
        self.compressed: List[CompressedMemory] = []

    # ------------------------------------------------------------------ #
    #  Load / add
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        """Restore state from storage (called on resume)."""
        self.iterations = self._storage.load_all_iterations()
        self.compressed = self._storage.load_compressed()

    def add(self, iteration: Iteration) -> None:
        """Append a completed iteration to the in-memory list and persist it."""
        self.iterations.append(iteration)
        self._storage.append_iteration(iteration)

    # ------------------------------------------------------------------ #
    #  Queries
    # ------------------------------------------------------------------ #

    def best(self) -> Optional[Iteration]:
        """Return the best iteration by score, preferring successful executions."""
        if not self.iterations:
            return None
        # For execution tasks, prefer iterations where the harness ran cleanly.
        # Fall back to all iterations only if nothing succeeded (e.g. very early run).
        successful = [it for it in self.iterations if it.execution_success]
        candidates = successful if successful else self.iterations
        return max(candidates, key=lambda it: it.evaluation_score)

    def past_proposals(self) -> List[str]:
        """
        Return a deduplicated list of all proposals from the in-memory iterations
        plus the summary key_findings from compressed blocks.  Used to build an
        explicit "already tried" reminder for the strategic-review prompt.
        """
        seen: set = set()
        result: List[str] = []
        # Oldest first: compressed summaries first, then in-memory
        for cm in self.compressed:
            for finding in cm.key_findings:
                key = finding.strip()
                if key and key not in seen:
                    seen.add(key)
                    result.append(f"[compressed] {key}")
        for it in self.iterations:
            key = it.proposal.strip()
            if key and key not in seen:
                seen.add(key)
                result.append(f"[iter {it.number}, score={it.evaluation_score:.2f}] {key}")
        return result

    def estimate_tokens(self) -> int:
        """Rough token budget consumed by the current in-memory list."""
        total = sum(it.token_estimate() for it in self.iterations)
        total += len(self.compressed) * 200  # compressed entries are tiny
        return total

    def should_compress(self) -> bool:
        return self.estimate_tokens() > CONTEXT_TOKEN_BUDGET * COMPRESSION_TRIGGER

    def is_over_budget(self) -> bool:
        """Check if current token estimate exceeds the hard budget."""
        return self.estimate_tokens() > CONTEXT_TOKEN_BUDGET

    def ensure_within_budget(self, criteria: Criteria) -> None:
        """
        Repeatedly compress until we're under the hard budget.
        
        If compression alone is insufficient, drop oldest iterations one at a time.
        This is a safety net to prevent context overflow errors from the LLM.
        """
        max_compress_attempts = 10
        attempts = 0
        
        while self.is_over_budget() and attempts < max_compress_attempts:
            logger.info(
                "Context over budget (%d tokens vs %d budget) — compressing (attempt %d/%d)",
                self.estimate_tokens(), CONTEXT_TOKEN_BUDGET, attempts + 1, max_compress_attempts
            )
            self.compress(criteria)
            attempts += 1
        
        if self.is_over_budget():
            logger.warning(
                "Context still over budget after %d compression attempts; "
                "dropping oldest iterations", attempts
            )
            # Drop oldest iterations one at a time until under budget
            while self.is_over_budget() and self.iterations:
                oldest = self.iterations.pop(0)
                logger.info("Dropped iteration %d to reduce context size", oldest.number)
                self._storage.rewrite_iterations(self.iterations)
            
            if self.is_over_budget():
                # Last resort: log warning but continue (LLM may still work with slightly over budget)
                logger.warning(
                    "Context still over budget (%d tokens) after all mitigation; "
                    "proceeding but LLM call may fail",
                    self.estimate_tokens()
                )

    # ------------------------------------------------------------------ #
    #  Compression
    # ------------------------------------------------------------------ #

    def compress(self, criteria: Criteria) -> None:
        """
        Compress the oldest half of the in-memory iterations into a
        CompressedMemory entry, then rewrite memory.jsonl to contain only
        the kept iterations.
        """
        if len(self.iterations) < 4:
            return  # not enough history to compress usefully

        n_compress = max(2, len(self.iterations) // 2)
        to_compress = self.iterations[:n_compress]
        keep = self.iterations[n_compress:]

        iterations_text = "\n\n".join(
            (
                f"=== Iteration {it.number} (score={it.evaluation_score:.2f}) ===\n"
                f"Proposal: {it.proposal}\n"
                f"Implementation snippet: {it.implementation[:300]}\n"
                f"Execution output: {it.execution_output[:200]}\n"
                f"Evaluation: {it.evaluation_reasoning}"
            )
            for it in to_compress
        )
        iteration_range = (to_compress[0].number, to_compress[-1].number)

        try:
            compressed = self._llm.compress(iterations_text, criteria, iteration_range)
        except Exception as exc:  # noqa: BLE001
            # Compression is best-effort: keep all iterations rather than crashing.
            logger.warning("Compression LLM call failed, skipping: %s", exc)
            return

        self.compressed.append(compressed)
        self.iterations = keep

        # Atomically rewrite the JSONL so loading after resume stays compact.
        self._storage.rewrite_iterations(keep)
        self._storage.save_compressed(self.compressed)
        logger.info(
            "Compressed iterations %d–%d, keeping %d",
            iteration_range[0],
            iteration_range[1],
            len(keep),
        )

    # ------------------------------------------------------------------ #
    #  Context building
    # ------------------------------------------------------------------ #

    def build_context(self, include_full_code: bool = False) -> str:
        """
        Build a context string to pass to the LLM for strategic review and
        implementation.  Includes compressed summaries, top-performing
        iterations, and the most recent iterations.
        """
        parts: List[str] = []

        # 1. Compressed summaries (oldest history)
        if self.compressed:
            parts.append("=== COMPRESSED HISTORY ===")
            for cm in self.compressed:
                parts.append(cm.format_for_llm())

        if not self.iterations:
            return "\n\n".join(parts) if parts else "No history yet — this is the first iteration."

        # 2. Top-performing iterations (excluding recent window)
        recent_nums = {it.number for it in self.iterations[-_RECENT_N:]}
        best_iters = sorted(
            [it for it in self.iterations if it.number not in recent_nums],
            key=lambda it: it.evaluation_score,
            reverse=True,
        )[:_BEST_N]

        if best_iters:
            parts.append("=== TOP PERFORMING ITERATIONS ===")
            for it in sorted(best_iters, key=lambda x: x.number):
                parts.append(self._format_iteration(it, include_full_code))

        # 3. Recent iterations
        recent = self.iterations[-_RECENT_N:]
        best_set = {it.number for it in best_iters}
        parts.append("=== RECENT ITERATIONS ===")
        for it in recent:
            label = " [also top-scoring]" if it.number in best_set else ""
            parts.append(self._format_iteration(it, include_full_code, label=label))

        return "\n\n".join(parts)

    def _format_iteration(
        self, it: Iteration, include_full_code: bool, label: str = ""
    ) -> str:
        lines = [
            f"--- Iteration {it.number}{label} (score={it.evaluation_score:.2f}) ---",
            f"Analysis:  {it.analysis}",
            f"Proposal:  {it.proposal}",
        ]
        if include_full_code:
            lines.append(f"Implementation:\n```python\n{it.implementation}\n```")
        else:
            snippet = it.implementation[:_IMPL_SNIPPET]
            suffix = "…" if len(it.implementation) > _IMPL_SNIPPET else ""
            lines.append(f"Implementation snippet: {snippet}{suffix}")

        if it.execution_output:
            out = it.execution_output[:_EXEC_SNIPPET]
            suffix = "…" if len(it.execution_output) > _EXEC_SNIPPET else ""
            status = "OK" if it.execution_success else "FAILED"
            lines.append(f"Execution [{status}]: {out}{suffix}")

        lines.append(f"Evaluation: {it.evaluation_reasoning}")
        return "\n".join(lines)
