import json
import tempfile
import unittest

from experimenter.config import CONTEXT_TOKEN_BUDGET, MemoryConfig, OllamaConfig
from experimenter.criteria import Criteria
from experimenter.iteration import CompressedMemory, Iteration
from experimenter.memory import MemoryManager
from experimenter.storage import Storage

from . import stubs  # noqa: F401  (langchain stubs, so memory.py can import)

CRITERIA = Criteria(
    task_description="make it faster",
    success_indicators=["speed"],
    evaluation_rubric="log scale",
    requires_execution=True,
)


class FakeLLM:
    """Deterministic stand-in: compression/merging are the only calls MemoryManager makes."""

    def __init__(self, merge_via_llm=True):
        self.merge_via_llm = merge_via_llm
        self.merge_calls = 0

    def compress(self, text, criteria, iteration_range):
        return CompressedMemory(
            iteration_range=iteration_range,
            summary="summary",
            key_findings=["finding"],
            best_score=0.5,
        )

    def merge_summaries(self, blocks, criteria):
        self.merge_calls += 1
        if not self.merge_via_llm:
            raise RuntimeError("merge failed")
        first = min(b.iteration_range[0] for b in blocks)
        last = max(b.iteration_range[1] for b in blocks)
        return CompressedMemory(
            iteration_range=(first, last),
            summary="merged",
            key_findings=[f for b in blocks for f in b.key_findings],
            best_score=max(b.best_score for b in blocks),
        )


def _iteration(n, score, size=100):
    return Iteration(
        number=n,
        proposal="p",
        analysis="a",
        implementation="x" * size,
        evaluation_score=score,
        evaluation_reasoning="r",
    )


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.storage = Storage(self.tmp.name, "exp")
        self.mm = MemoryManager(self.storage, FakeLLM())

    def test_budget_follows_num_ctx(self):
        self.assertEqual(self.mm.token_budget, CONTEXT_TOKEN_BUDGET)
        small = MemoryManager(
            self.storage, FakeLLM(),
            token_budget=OllamaConfig(num_ctx=8192).get_context_budget(),
        )
        self.assertEqual(small.token_budget, 4915)
        # Room is reserved for the model's reply, so the usable budget is smaller.
        self.assertLess(small.budget(), small.token_budget)
        cfg = MemoryConfig(reserved_response_tokens=0, static_prompt_tokens=0)
        self.assertEqual(
            MemoryManager(self.storage, FakeLLM(), cfg, 4915).budget(), 4915
        )

    def test_compressed_blocks_are_counted_at_real_size(self):
        big = CompressedMemory(
            iteration_range=(1, 4), summary="y" * 4000, key_findings=["k"], best_score=0.4
        )
        self.mm.compressed.append(big)
        self.assertGreater(big.token_estimate(), 1000)

    def test_compression_keeps_the_best_iteration(self):
        # Best is inside the half that would otherwise be evicted.
        for n, score in [(1, 0.2), (2, 0.9), (3, 0.3), (4, 0.4), (5, 0.5), (6, 0.6)]:
            self.mm.add(_iteration(n, score))
        self.assertEqual(self.mm.best().number, 2)
        self.mm.compress(CRITERIA)
        self.assertEqual(self.mm.best().number, 2)
        self.assertIn(2, [it.number for it in self.mm.iterations])
        # Persisted file agrees with memory after compaction.
        self.assertIn(2, [it.number for it in self.storage.load_all_iterations()])

    def test_compressed_history_is_capped_by_rolling_merge(self):
        cfg = MemoryConfig(max_compressed_blocks=3)
        mm = MemoryManager(self.storage, FakeLLM(), cfg)
        for _ in range(5):
            for n, score in [(1, 0.2), (2, 0.3), (3, 0.4), (4, 0.5)]:
                mm.add(_iteration(n, score))
            mm.compress(CRITERIA)
            mm.iterations = mm.iterations[:0]  # keep it simple: blocks only
        self.assertLessEqual(len(mm.compressed), cfg.max_compressed_blocks)
        self.assertGreaterEqual(mm._llm.merge_calls, 2)

    def test_merge_falls_back_to_deterministic_when_llm_fails(self):
        cfg = MemoryConfig(max_compressed_blocks=2, max_findings_per_block=3)
        mm = MemoryManager(self.storage, FakeLLM(merge_via_llm=False), cfg)
        for _ in range(4):
            for n in (1, 2, 3, 4):
                mm.add(_iteration(n, 0.1))
            mm.compress(CRITERIA)
            mm.iterations = []
        self.assertLessEqual(len(mm.compressed), cfg.max_compressed_blocks)
        self.assertLessEqual(len(mm.compressed[0].key_findings), cfg.max_findings_per_block)

    def test_drop_oldest_never_drops_the_best_and_archives_it(self):
        cfg = MemoryConfig(reserved_response_tokens=0, static_prompt_tokens=0)
        tiny = MemoryManager(self.storage, FakeLLM(), cfg, token_budget=600)
        for n, score in [(1, 0.1), (2, 0.9), (3, 0.2)]:
            tiny.add(_iteration(n, score, size=1500))
        tiny.ensure_within_budget(CRITERIA)
        self.assertEqual(tiny.best().number, 2)
        self.assertIn(2, [it.number for it in tiny.iterations])
        self.assertIn(2, [it.number for it in self.storage.load_all_iterations()])
        # Evicted iterations survive on disk.
        archive = self.storage.base / "archive.jsonl"
        self.assertTrue(archive.exists())
        dropped = [json.loads(line)["number"] for line in archive.read_text().splitlines()]
        self.assertIn(1, dropped)

    def test_load_restores_tracked_best(self):
        for n, score in [(1, 0.9), (2, 0.4)]:
            self.mm.add(_iteration(n, score))
        fresh = MemoryManager(self.storage, FakeLLM())
        fresh.load()
        self.assertEqual(fresh.best().number, 1)

    def test_past_proposals_is_deduplicated(self):
        first = _iteration(1, 0.1)
        first.proposal = "same idea"
        second = _iteration(2, 0.2)
        second.proposal = "same idea"
        self.mm.add(first)
        self.mm.add(second)
        self.assertEqual(len(self.mm.past_proposals()), 1)

    def test_past_proposals_is_capped(self):
        # Without a cap the "already tried" list grows every iteration forever.
        cfg = MemoryConfig(max_tried_items=5)
        mm = MemoryManager(self.storage, FakeLLM(), cfg)
        mm.iterations = [_iteration(n, 0.1) for n in range(1, 15)]
        for i, it in enumerate(mm.iterations):
            it.proposal = f"idea {i}"   # distinct, so nothing is deduplicated
        tried = mm.past_proposals()
        self.assertEqual(len(tried), cfg.max_tried_items + 1)  # entries + marker
        self.assertTrue(any("truncated" in t for t in tried))

    def test_recent_window_still_shows_kept_best(self):
        for n, score in [(1, 0.9), (2, 0.3), (3, 0.4), (4, 0.5), (5, 0.6)]:
            self.mm.add(_iteration(n, score))
        self.mm.compress(CRITERIA)
        ctx = self.mm.build_context()
        self.assertIn("score=0.90", ctx)

    def test_build_context_windows_are_configurable_and_capped(self):
        cfg = MemoryConfig(recent_window=2, best_window=1, max_compressed_blocks=1)
        mm = MemoryManager(self.storage, FakeLLM(), cfg)
        mm.compressed = [
            CompressedMemory((1, 3), "s1", ["k"], 0.5),
            CompressedMemory((4, 6), "s2", ["k2"], 0.6),
        ]
        mm.iterations = [_iteration(n, n / 10.0) for n in range(1, 11)]
        ctx = mm.build_context()
        # 2 recent + 1 top-scoring, and only max_compressed_blocks summaries.
        self.assertEqual(ctx.count("--- Iteration"), cfg.recent_window + cfg.best_window)
        self.assertEqual(ctx.count("=== COMPRESSED HISTORY ==="), 1)
        self.assertIn("older summaries omitted", ctx)
        self.assertIn("s1", ctx)
        self.assertNotIn("s2", ctx)

    def test_build_context_does_not_duplicate_best_and_recent(self):
        for n in range(1, 12):
            self.mm.add(_iteration(n, n / 20.0))
        ctx = self.mm.build_context()
        numbers = [line.split("Iteration ")[1].split("(")[0].strip() for line in ctx.splitlines()
                   if line.startswith("--- Iteration")]
        self.assertEqual(len(numbers), len(set(numbers)))


if __name__ == "__main__":
    unittest.main()
