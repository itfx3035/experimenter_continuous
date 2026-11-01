"""
MemoryManager — in-memory iteration store with LLM-based compression.

Design decisions
----------------
- Context is built with rich content (analysis, proposal, implementation snippet,
  execution output, evaluation reasoning) so the LLM has real signal to work with.
- The "best" iteration bucket and "recent" iterations bucket can overlap; we
  deduplicate by iteration number so the same iteration is never shown twice.
- Scores are in [0.0, 1.0]; thresholds that reference scores use the same range.
- The token budget comes from the OllamaConfig (num_ctx * safety margin) rather
  than a hard-coded constant, so changing num_ctx actually changes when we
  compress.  Compressed blocks are counted at their real rendered size, and the
  budget reserves room for the model's reply — a context that "just fits" still
  fails at call time if nothing is left for num_predict.
- Window sizes (recent / top-scoring / compressed blocks / already-tried list)
  come from MemoryConfig instead of module constants, so long runs stop growing.
- The tracked best iteration is never evicted by compression or by the
  drop-oldest fallback, so "best" always means the global best, not just the
  best of whatever happens to still be in memory.
- Compressed history is capped: once max_compressed_blocks is reached, the
  oldest blocks are merged into one (rolling summary) instead of appending
  forever.
- After compression, rewrite_iterations() replaces memory.jsonl so the file
  stays compact; evicted iterations are archived to archive.jsonl first so the
  drop-oldest fallback cannot silently destroy history.
- The MemoryManager is the single source of truth for the in-memory list; the
  Storage is its persistence backend.
"""

from __future__ import annotations

import logging

from .config import CONTEXT_TOKEN_BUDGET, MemoryConfig
from .criteria import Criteria
from .iteration import CompressedMemory, Iteration
from .llm import LLMClient
from .storage import Storage

logger = logging.getLogger(__name__)

# Maximum characters for an implementation snippet in the non-full-code path.
_IMPL_SNIPPET = 500
# Maximum characters for execution output in context.
_EXEC_SNIPPET = 400


class MemoryManager:
    def __init__(
        self,
        storage: Storage,
        llm: LLMClient | None = None,
        memory_config: MemoryConfig | None = None,
        token_budget: int | None = None,
    ) -> None:
        self._storage = storage
        self._llm = llm
        self.config = memory_config or MemoryConfig()
        # Raw budget (num_ctx * safety margin) when supplied by the caller,
        # otherwise the fallback constant.
        self.token_budget = int(token_budget or CONTEXT_TOKEN_BUDGET)
        self.iterations: list[Iteration] = []
        self.compressed: list[CompressedMemory] = []
        # Number/score of the highest-scoring iteration seen so far (global best).
        self._best_number: int | None = None
        self._best_score: float = -1.0

    # ------------------------------------------------------------------ #
    #  Load / add
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        """Restore state from storage (called on resume)."""
        self.iterations = self._storage.load_all_iterations()
        self.compressed = self._storage.load_compressed()
        self._best_number = None
        self._best_score = -1.0
        for it in self.iterations:
            self._track_best(it)

    def add(self, iteration: Iteration) -> None:
        """Append a completed iteration to the in-memory list and persist it."""
        self.iterations.append(iteration)
        self._storage.append_iteration(iteration)
        self._track_best(iteration)

    def _track_best(self, iteration: Iteration) -> None:
        """Update the tracked global best (ties keep the earliest iteration)."""
        if self._best_number is None or iteration.evaluation_score > self._best_score:
            self._best_number = iteration.number
            self._best_score = iteration.evaluation_score

    # ------------------------------------------------------------------ #
    #  Queries
    # ------------------------------------------------------------------ #

    def best(self) -> Iteration | None:
        """Return the tracked global best; fall back to the best still in memory."""
        if not self.iterations:
            return None
        tracked = next((it for it in self.iterations if it.number == self._best_number), None)
        if tracked is not None:
            return tracked
        # Tracked best was evicted by an older compression run (legacy data):
        # prefer cleanly-executed iterations, as before.
        successful = [it for it in self.iterations if it.execution_success]
        candidates = successful if successful else self.iterations
        return max(candidates, key=lambda it: it.evaluation_score)

    def past_proposals(self) -> list[str]:
        """
        Return a deduplicated list of proposals from the in-memory iterations
        plus the summary key_findings from compressed blocks, capped at
        `max_tried_items` (most recent first — those are the ones the model is
        most likely to repeat).

        Without a cap this list grows every iteration and eventually dominates
        the prompt, which is exactly the growth compression is supposed to prevent.
        """
        seen: set = set()
        result: list[str] = []

        # Oldest first: compressed summaries first, then in-memory.
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

        cap = self.config.max_tried_items
        if len(result) > cap:
            dropped = len(result) - cap
            result = result[dropped:]
            result.insert(0, f"[{dropped} older entr{'y' if dropped == 1 else 'ies'} truncated]")
        return result

    # ------------------------------------------------------------------ #
    #  Budget
    # ------------------------------------------------------------------ #

    def budget(self) -> int:
        """Budget available for the rendered context (response room excluded)."""
        return self.config.effective_budget(self.token_budget)

    def estimate_tokens(self) -> int:
        """Rough token budget consumed by the current in-memory list."""
        total = sum(it.token_estimate() for it in self.iterations)
        # Compressed blocks are rendered verbatim in every prompt, so their real
        # size matters — a flat per-entry constant badly undercounts them.
        total += sum(cm.token_estimate() for cm in self.compressed)
        return total

    def should_compress(self) -> bool:
        return self.estimate_tokens() > self.budget() * self.config.compression_trigger

    def is_over_budget(self) -> bool:
        """Check if current token estimate exceeds the effective budget."""
        return self.estimate_tokens() > self.budget()

    def ensure_within_budget(self, criteria: Criteria) -> None:
        """
        Repeatedly compress until we're under the effective budget.

        If compression alone is insufficient, drop oldest iterations one at a time.
        This is a safety net to prevent context overflow errors from the LLM.
        """
        max_compress_attempts = 10
        attempts = 0

        while self.is_over_budget() and attempts < max_compress_attempts:
            logger.info(
                "Context over budget (%d tokens vs %d budget) — compressing (attempt %d/%d)",
                self.estimate_tokens(), self.budget(), attempts + 1, max_compress_attempts,
            )
            before = len(self.compressed)
            self.compress(criteria)
            attempts += 1
            # Compression made no progress (too few iterations, or the LLM call
            # failed): merging blocks is the only remaining lever.
            if len(self.compressed) == before and len(self.iterations) < 4:
                break

        if self.is_over_budget():
            logger.warning(
                "Context still over budget after %d compression attempts; "
                "dropping oldest iterations", attempts
            )
            # Drop oldest iterations one at a time until under budget.  The
            # tracked best is never dropped: losing it would make the next
            # context and the reported best_result regress below best_score.
            while self.is_over_budget():
                droppable = [it for it in self.iterations if it.number != self._best_number]
                if not droppable:
                    break
                oldest = droppable[0]
                logger.info("Dropped iteration %d to reduce context size", oldest.number)
                self.iterations.remove(oldest)
                # Archive the evicted iteration so dropping is recoverable.
                self._storage.archive_iterations([oldest])
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
        the kept iterations.  The tracked best iteration is always kept.

        The compressed-block list is capped: when it exceeds
        `max_compressed_blocks` the oldest blocks are merged into one.
        """
        if len(self.iterations) < 4:
            return  # not enough history to compress usefully

        n_compress = max(2, len(self.iterations) // 2)
        first_half = self.iterations[:n_compress]
        # Never evict the global best — it must stay visible in the context.
        to_compress = [it for it in first_half if it.number != self._best_number]
        if not to_compress:
            return
        protected = [it for it in first_half if it.number == self._best_number]
        keep = protected + self.iterations[n_compress:]

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

        compressed.key_findings = compressed.key_findings[: self.config.max_findings_per_block]
        self.compressed.append(compressed)
        self.iterations = keep

        # Cap the compressed history (rolling merge).
        if len(self.compressed) > self.config.max_compressed_blocks:
            self._merge_blocks(criteria)

        # Atomically rewrite the JSONL so loading after resume stays compact.
        self._storage.archive_iterations(to_compress)
        self._storage.rewrite_iterations(keep)
        self._storage.save_compressed(self.compressed)
        logger.info(
            "Compressed iterations %d–%d, keeping %d (%d summary blocks)",
            iteration_range[0],
            iteration_range[1],
            len(keep),
            len(self.compressed),
        )

    def _merge_blocks(self, criteria: Criteria) -> None:
        """Fold the oldest summary blocks into a single block (rolling summary)."""
        cap = self.config.max_compressed_blocks
        group = self.compressed[: max(2, len(self.compressed) - cap + 1)]
        if len(group) < 2:
            return

        merged: CompressedMemory | None = None
        if self._llm is not None:
            try:
                merged = self._llm.merge_summaries(group, criteria)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Merge LLM call failed, merging deterministically: %s", exc)

        if merged is None:
            merged = self._merge_deterministic(group)

        merged.key_findings = merged.key_findings[: self.config.max_findings_per_block]
        self.compressed = [merged] + self.compressed[len(group):]
        logger.info(
            "Merged %d summary blocks into one covering iterations %d–%d",
            len(group), merged.iteration_range[0], merged.iteration_range[1],
        )

    def _merge_deterministic(self, group: list[CompressedMemory]) -> CompressedMemory:
        """No-LLM fallback merge: union ranges, keep the best score, dedupe findings."""
        first = min(b.iteration_range[0] for b in group)
        last = max(b.iteration_range[1] for b in group)
        findings: list[str] = []
        seen: set = set()
        # Newest findings matter more than the oldest ones.
        for block in reversed(group):
            for f in block.key_findings:
                key = f.strip()
                if key and key not in seen:
                    seen.add(key)
                    findings.append(key)
        summary = " ".join(b.summary.strip() for b in group if b.summary.strip())
        return CompressedMemory(
            iteration_range=(first, last),
            summary=summary[:1500],
            key_findings=findings[: self.config.max_findings_per_block],
            best_score=max((b.best_score for b in group), default=0.0),
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
        parts: list[str] = []

        # 1. Compressed summaries (oldest history) — capped, oldest first.
        if self.compressed:
            blocks = self.compressed[: self.config.max_compressed_blocks]
            parts.append("=== COMPRESSED HISTORY ===")
            for cm in blocks:
                parts.append(cm.format_for_llm())
            if len(self.compressed) > len(blocks):
                parts.append(f"[{len(self.compressed) - len(blocks)} older summaries omitted]")

        if not self.iterations:
            return "\n\n".join(parts) if parts else "No history yet — this is the first iteration."

        # 2. Top-performing iterations (excluding recent window)
        recent_window = self.config.recent_window
        recent = self.iterations[-recent_window:]
        recent_nums = {it.number for it in recent}
        best_iters = sorted(
            [it for it in self.iterations if it.number not in recent_nums],
            key=lambda it: it.evaluation_score,
            reverse=True,
        )[: self.config.best_window]

        if best_iters:
            parts.append("=== TOP PERFORMING ITERATIONS ===")
            for it in sorted(best_iters, key=lambda x: x.number):
                parts.append(self._format_iteration(it, include_full_code))

        # 3. Recent iterations
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
