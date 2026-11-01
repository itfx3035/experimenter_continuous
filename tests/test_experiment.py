import os
import signal
import tempfile
import threading
import unittest

from experimenter.config import ExecutorConfig, MemoryConfig, OllamaConfig
from experimenter.criteria import Criteria
from experimenter.experiment import Experiment
from experimenter.iteration import CompressedMemory

from . import stubs  # noqa: F401


class FakeLLM:
    """No-network stand-in for LLMClient."""

    def __init__(self, implement_behaviour="ok"):
        self.implement_behaviour = implement_behaviour
        self.calls = 0

    def generate_criteria(self, prompt, requires_execution):
        return Criteria(
            task_description="task",
            success_indicators=["ok"],
            evaluation_rubric="rubric",
            requires_execution=requires_execution,
        )

    def generate_initial(self, criteria):
        return "def f():\n    return 1\n"

    def generate_execution_harness(self, implementation, criteria):
        return 'import json\nprint(json.dumps({"passed": True, "score": 0.5}))'

    def strategic_review(self, context, criteria, tried, stagnation_count=0):
        return ("analysis", "proposal", [])

    def implement(self, proposal, current, context, criteria, avoid=None):
        self.calls += 1
        if self.implement_behaviour == "syntax_error":
            return "def f(:\n    pass"
        if self.implement_behaviour == "interrupt" and self.calls >= 2:
            raise KeyboardInterrupt
        return "def f():\n    return 2\n"

    def evaluate(self, result, criteria, previous_best_score, previous_best_summary):
        return (0.6, "good")

    def compress(self, text, criteria, iteration_range):
        return CompressedMemory(iteration_range=iteration_range, summary="s",
                                key_findings=["k"], best_score=0.6)


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.orig_sigint = signal.getsignal(signal.SIGINT)
        self.orig_sigterm = signal.getsignal(signal.SIGTERM)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(lambda: signal.signal(signal.SIGINT, self.orig_sigint))
        self.addCleanup(lambda: signal.signal(signal.SIGTERM, self.orig_sigterm))

    def _experiment(self, llm, name="exp", max_iterations=2, requires_execution=False):
        exp = Experiment(
            name=name,
            prompt="do better",
            requires_execution=requires_execution,
            max_iterations=max_iterations,
            storage_dir=self.tmp.name,
            verbose=False,
            ollama_config=OllamaConfig(),
        )
        exp._llm = llm
        return exp

    def test_failed_iteration_is_recorded_in_memory(self):
        """A skipped iteration must still land in memory.jsonl, otherwise
        past_proposals() never sees it and the LLM can loop on the same idea."""
        exp = self._experiment(FakeLLM("syntax_error"), requires_execution=True)
        result = exp.run()
        self.assertEqual(result["total_iterations"], 2)
        recorded = exp._memory.iterations
        self.assertEqual(len(recorded), 2)
        self.assertTrue(all(it.evaluation_score == 0.0 for it in recorded))
        self.assertIn("failed after all retries", recorded[0].evaluation_reasoning)
        self.assertEqual(len(exp._memory.past_proposals()), 1)
        # Nothing scored, so nothing was stored as best.
        self.assertEqual(result["best_iteration"], 0)

    def test_run_initial_seeds_the_baseline(self):
        exp = self._experiment(FakeLLM(), max_iterations=1)
        exp.stop()          # stop before the loop so the seed is not overwritten
        result = exp.run(initial="def f():\n    return 0\n")
        self.assertEqual(result["status"], "stopped")
        self.assertEqual(exp._current, "def f():\n    return 0")
        self.assertEqual(exp._storage.load_artifact("best_result.txt"),
                         "def f():\n    return 0")

    def test_invalid_initial_falls_back_to_llm_baseline(self):
        exp = self._experiment(FakeLLM(), max_iterations=1, requires_execution=True)
        exp.stop()
        exp.run(initial="def f(:\n    pass")
        self.assertEqual(exp._current, "def f():\n    return 1\n")

    def test_interrupt_saves_state_before_propagating(self):
        exp = self._experiment(FakeLLM("interrupt"), max_iterations=3)
        with self.assertRaises(KeyboardInterrupt):
            exp.run()
        saved = exp._storage.load_state()
        self.assertEqual(saved.status, "stopped")

    def test_second_signal_raises_first_sets_stop(self):
        exp = self._experiment(FakeLLM(), max_iterations=1)
        exp._handle_signal(signal.SIGINT, None)
        self.assertTrue(exp._stop_requested)
        with self.assertRaises(KeyboardInterrupt):
            exp._handle_signal(signal.SIGINT, None)

    def test_construction_off_main_thread_does_not_crash(self):
        errors = []

        def _build():
            try:
                Experiment(
                    name="threaded",
                    prompt="do better",
                    requires_execution=False,
                    storage_dir=self.tmp.name,
                    verbose=False,
                    ollama_config=OllamaConfig(),
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        t = threading.Thread(target=_build)
        t.start()
        t.join(10)
        self.assertEqual(errors, [])

    def test_memory_and_executor_configs_are_wired_through(self):
        mem = MemoryConfig(recent_window=4, max_compressed_blocks=2)
        exe = ExecutorConfig(memory_mb=128, isolated=False)
        exp = Experiment(
            name="wired", prompt="do better", requires_execution=True,
            storage_dir=self.tmp.name, verbose=False, ollama_config=OllamaConfig(),
            memory_config=mem, executor_config=exe,
        )
        # LLM and memory must share one config, otherwise the fit decision and the
        # compression decision use different budgets.
        self.assertIs(exp._memory.config, mem)
        self.assertIs(exp._llm._mem_cfg, mem)
        self.assertEqual(exp._executor.memory_mb, 128)
        self.assertFalse(exp._executor.isolated)
        self.assertEqual(exp._memory.budget(), mem.effective_budget(exp._config.context_budget()))

    # ---- lock -------------------------------------------------------- #

    def test_lock_prevents_two_processes_on_the_same_experiment(self):
        exp = self._experiment(FakeLLM(), name="locked", max_iterations=1)
        exp.stop()
        # A foreign, definitely-alive pid (our parent) holds the lock.
        exp._storage.lock_path().write_text(str(os.getppid()), encoding="utf-8")
        with self.assertRaises(RuntimeError):
            exp.run()
        # release_lock must not delete another process's lock.
        self.assertEqual(int(exp._storage.lock_path().read_text().strip()), os.getppid())

    def test_stale_lock_is_taken_over_and_lock_released_after_run(self):
        exp = self._experiment(FakeLLM(), name="stale", max_iterations=1)
        exp._storage.lock_path().write_text("999999", encoding="utf-8")
        exp.run()
        self.assertFalse(exp._storage.lock_path().exists())

    # ---- stagnation ------------------------------------------------- #

    def test_stagnation_count_is_persisted_across_resume(self):
        exp = self._experiment(FakeLLM(), name="stag", max_iterations=2)
        exp.run()   # iteration 1 improves from 0.0, iteration 2 does not
        saved = exp._storage.load_state()
        self.assertEqual(saved.stagnation_count, 1)

        resumed = self._experiment(FakeLLM(), name="stag", max_iterations=3)
        resumed.stop()
        resumed.run()
        self.assertEqual(resumed._state.stagnation_count, 1)

    def test_resume_without_saved_criteria_uses_the_user_flag(self):
        """The LLM cannot infer requires_execution — it must come from the config."""
        exp = self._experiment(FakeLLM(), name="nocriteria", max_iterations=1,
                               requires_execution=True)
        exp.stop()
        exp.run()
        state = exp._storage.load_state()
        state.criteria = None
        exp._storage.save_state(state)

        resumed = self._experiment(FakeLLM(), name="nocriteria", max_iterations=1,
                                   requires_execution=True)
        resumed.stop()
        resumed.run()
        self.assertIsNotNone(resumed._criteria)
        self.assertTrue(resumed._criteria.requires_execution)


if __name__ == "__main__":
    unittest.main()
