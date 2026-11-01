import json
import os
import tempfile
import unittest

from experimenter.criteria import Criteria
from experimenter.iteration import CompressedMemory, ExperimentState, Iteration
from experimenter.storage import Storage


def _iteration(n, score, impl="def f():\n    return 1"):
    return Iteration(number=n, evaluation_score=score, implementation=impl,
                     proposal=f"p{n}", analysis=f"a{n}")


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.storage = Storage(self.tmp.name, "exp1")

    def test_atomic_write_leaves_no_temp_files(self):
        self.storage.save_artifact("best_result.txt", "content")
        leftovers = [p for p in os.listdir(self.storage.artifacts_dir) if p.endswith(".tmp")]
        self.assertEqual(leftovers, [])
        self.assertEqual(self.storage.load_artifact("best_result.txt"), "content")

    def test_append_then_load(self):
        self.storage.append_iteration(_iteration(1, 0.3))
        self.storage.append_iteration(_iteration(2, 0.6))
        self.assertEqual([it.number for it in self.storage.load_all_iterations()], [1, 2])

    def test_rewrite_after_compaction(self):
        self.storage.append_iteration(_iteration(1, 0.3))
        self.storage.append_iteration(_iteration(2, 0.6))
        self.storage.rewrite_iterations([_iteration(2, 0.6)])
        self.assertEqual([it.number for it in self.storage.load_all_iterations()], [2])

    def test_corrupt_line_is_skipped(self):
        path = self.storage.base / "memory.jsonl"
        path.write_text('{"number": 1}\nnot json\n{"number": 2}\n', encoding="utf-8")
        self.assertEqual([it.number for it in self.storage.load_all_iterations()], [1, 2])

    def test_missing_state_is_none_but_corrupt_state_raises(self):
        """"No experiment yet" and "the file is damaged" are different situations;
        conflating them makes resume() report the wrong error."""
        self.assertIsNone(self.storage.load_state())
        (self.storage.base / "state.json").write_text("{ not json", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.storage.load_state()

    def test_evicted_iterations_are_archived(self):
        it = _iteration(1, 0.2)
        self.storage.archive_iterations([it])
        path = self.storage.base / "archive.jsonl"
        self.assertTrue(path.exists())
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["number"], 1)

    def test_experiment_name_cannot_escape_storage_dir(self):
        hostile = Storage(self.tmp.name, "../escape")
        self.assertEqual(hostile.name, "escape")
        self.assertTrue(os.path.realpath(hostile.base).startswith(os.path.realpath(self.tmp.name)))

    # ---- lock -------------------------------------------------------- #

    def test_lock_is_acquired_and_released(self):
        self.assertTrue(self.storage.acquire_lock())
        self.assertEqual(int(self.storage.lock_path().read_text().strip()), os.getpid())
        self.storage.release_lock()
        self.assertFalse(self.storage.lock_path().exists())

    def test_lock_held_by_a_live_process_is_not_stolen(self):
        # Our parent process is a guaranteed-alive foreign pid.
        self.storage.lock_path().write_text(str(os.getppid()), encoding="utf-8")
        self.assertFalse(self.storage.acquire_lock())
        self.assertEqual(self.storage.held_by_other(), os.getppid())
        # release_lock must not delete another process's lock.
        self.storage.release_lock()
        self.assertEqual(int(self.storage.lock_path().read_text().strip()), os.getppid())

    def test_stale_lock_from_a_dead_pid_is_taken_over(self):
        self.storage.lock_path().write_text("999999", encoding="utf-8")
        self.assertIsNone(self.storage.held_by_other())
        self.assertTrue(self.storage.acquire_lock())
        self.assertEqual(int(self.storage.lock_path().read_text().strip()), os.getpid())

    # ---- tolerant loading ------------------------------------------- #

    def test_iteration_from_dict_tolerates_bad_types(self):
        d = {"number": "3", "evaluation_score": "0.5", "key_findings": None}
        it = Iteration.from_dict(d)
        self.assertEqual(it.number, 3)
        self.assertEqual(it.evaluation_score, 0.5)
        self.assertEqual(it.proposal, "")

    def test_state_from_dict_tolerates_missing_fields(self):
        state = ExperimentState.from_dict({"name": "x", "prompt": "y", "stagnation_count": "4"})
        self.assertEqual(state.stagnation_count, 4)
        self.assertEqual(state.status, "created")

    def test_compressed_from_dict_tolerates_bad_range(self):
        cm = CompressedMemory.from_dict({
            "iteration_range": ["1", 5], "summary": "s", "key_findings": "not a list",
            "best_score": "0.4",
        })
        self.assertEqual(cm.iteration_range, (1, 5))
        self.assertEqual(cm.key_findings, [])
        self.assertEqual(cm.best_score, 0.4)

    def test_criteria_from_dict_tolerates_bad_types(self):
        c = Criteria.from_dict({
            "task_description": "t", "success_indicators": "one item",
            "evaluation_rubric": "r", "requires_execution": True,
            "speedup_target": "abc",
        })
        self.assertEqual(c.success_indicators, ["one item"])
        self.assertEqual(c.speedup_target, 0.0)


if __name__ == "__main__":
    unittest.main()
