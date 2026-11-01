"""
Experiment — top-level orchestrator.

Public API
----------
    exp = Experiment(
        name="fast_fibonacci",
        prompt="Optimize fibonacci(n) for speed. n can be up to 35.",
        requires_execution=True,
        ollama_config=OllamaConfig(model="llama3.1"),
    )
    result = exp.run(initial="def fibonacci(n): ...")

    # Resume later (same name + storage_dir detects existing state)
    exp2 = Experiment(name="fast_fibonacci", prompt="...", requires_execution=True)
    result = exp2.run()   # detects existing state and resumes automatically

    exp2.stop()    # graceful stop after the current iteration
    exp2.get_status()

Loop steps per iteration
------------------------
  1. STRATEGIC REVIEW  — LLM analyses history and proposes next action.
  2. IMPLEMENT         — LLM writes the new implementation.
  3. SYNTAX CHECK      — ast.parse() for code tasks (retried up to 3×).
  4. EXECUTE           — subprocess runs implementation + harness (code tasks).
  5. EVALUATE          — LLM scores result 0.0–1.0 against criteria.
  6. STORE             — everything persisted atomically.
  7. COMPRESS          — if context budget is exceeded, old history is summarised.

Notes
-----
- A lock file (storage/lock) prevents two processes from writing the same
  experiment directory; a stale lock from a dead pid is taken over.
- Signal handlers are installed only from the main thread; the first Ctrl-C
  means "save and stop", the second aborts immediately.
- Stagnation count is persisted in state.json, so a resumed run keeps the
  "this strategy has not improved for N iterations" warning.
"""

from __future__ import annotations

import ast
import logging
import signal
import time

from .config import ExecutorConfig, ExperimentConfig, MemoryConfig, OllamaConfig
from .criteria import Criteria
from .executor import ExecutionResult, SubprocessExecutor
from .iteration import ExperimentState, Iteration
from .llm import LLMClient
from .log_setup import setup_logging
from .memory import MemoryManager
from .storage import Storage

logger = logging.getLogger(__name__)



class Experiment:
    def __init__(
        self,
        name: str,
        prompt: str,
        requires_execution: bool,
        max_iterations: int = 50,
        max_time_seconds: float = 7200.0,
        storage_dir: str = "./experiments",
        verbose: bool = True,
        ollama_config: OllamaConfig | None = None,
        memory_config: MemoryConfig | None = None,
        executor_config: ExecutorConfig | None = None,
        use_lock: bool = True,
    ) -> None:
        # Use JSON configuration if no explicit config is provided
        if ollama_config is None:
            ollama_config = OllamaConfig.from_json()

        self._config = ExperimentConfig(
            name=name,
            prompt=prompt,
            requires_execution=requires_execution,
            max_iterations=max_iterations,
            max_time_seconds=max_time_seconds,
            storage_dir=storage_dir,
            verbose=verbose,
            ollama_config=ollama_config,
            memory=memory_config or MemoryConfig(),
            executor=executor_config or ExecutorConfig(),
        )
        self._storage = Storage(storage_dir, name)
        self._use_lock = use_lock
        setup_logging(self._storage.base)
        # LLMClient and MemoryManager share the same MemoryConfig so the fit and
        # the compression decision use the same effective budget.
        self._llm = LLMClient(self._config.ollama_config, self._config.memory)
        self._executor = SubprocessExecutor(
            tmp_dir=str(self._storage.tmp_dir), config=self._config.executor
        )
        # Budget derived from the model's actual context window, not a constant.
        self._memory = MemoryManager(
            self._storage,
            self._llm,
            memory_config=self._config.memory,
            token_budget=self._config.context_budget(),
        )

        self._state: ExperimentState | None = None
        self._criteria: Criteria | None = None
        # Current best implementation/content text (seeded from initial or restored).
        self._current: str = ""

        self._stop_requested = False
        self._pause_requested = False
        self._signal_count = 0

        # signal.signal() only works from the main thread.  Guard so the library
        # can also be used inside threads / async apps instead of crashing.
        try:
            signal.signal(signal.SIGINT, self._handle_signal)
            signal.signal(signal.SIGTERM, self._handle_signal)
        except ValueError:
            logger.warning(
                "Signal handlers not installed (Experiment created outside the "
                "main thread); use exp.stop() to request a graceful stop."
            )

    # ------------------------------------------------------------------ #
    #  Signal handling
    # ------------------------------------------------------------------ #

    def _handle_signal(self, _signum: int, _frame: object) -> None:
        self._signal_count += 1
        if self._signal_count >= 2:
            # Second interrupt: abort immediately (state is still saved by
            # _run_loop's finally block).
            self._log("Second interrupt — aborting immediately.")
            raise KeyboardInterrupt
        self._log("Interrupt received — will stop after this iteration.")
        self._stop_requested = True

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def run(self, initial: str = "") -> dict:
        """
        Start a new experiment, or resume an existing one automatically.

        `initial` seeds the baseline for a brand-new experiment.  If it is empty
        (or the experiment already exists), the LLM generates the baseline from
        the prompt.
        """
        if self._storage.exists():
            self._log("Existing experiment found — resuming.")
            return self._with_lock(self._resume)
        return self._with_lock(lambda: self._start(initial))

    def resume(self) -> dict:
        """
        Resume a previously saved experiment.

        Raises RuntimeError if no saved state exists, and ValueError if
        state.json exists but is corrupt (Storage.load_state).
        """
        if not self._storage.exists():
            raise RuntimeError(f"No saved experiment '{self._config.name}' found.")
        return self._with_lock(self._resume)

    def _with_lock(self, fn) -> dict:
        """Guard the run so two processes cannot write the same experiment."""
        if not self._use_lock:
            return fn()
        if not self._storage.acquire_lock():
            holder = self._storage.held_by_other()
            raise RuntimeError(
                f"Experiment '{self._config.name}' is already running in process "
                f"{holder}; two runs sharing {self._storage.base} would corrupt it. "
                "Pass use_lock=False to override."
            )
        try:
            return fn()
        finally:
            self._storage.release_lock()

    def stop(self) -> None:
        """Request a graceful stop after the current iteration."""
        self._stop_requested = True

    def pause(self) -> None:
        """Request a pause after the current iteration."""
        self._pause_requested = True

    def get_status(self) -> dict:
        if self._state is None:
            return {"status": "not_started"}
        return {
            "status": self._state.status,
            "current_iteration": self._state.current_iteration,
            "best_score": self._state.best_score,
            "elapsed_seconds": self._state.total_elapsed_seconds,
        }

    # ------------------------------------------------------------------ #
    #  Internals — startup / resume
    # ------------------------------------------------------------------ #

    def _start(self, initial: str = "") -> dict:
        self._log("Generating success criteria from your prompt…")
        self._criteria = self._llm.generate_criteria(
            self._config.prompt, self._config.requires_execution
        )
        self._log(f"Task understood: {self._criteria.task_description}")
        self._log(f"Requires execution: {self._criteria.requires_execution}")

        self._log("Generating initial baseline…")
        if initial.strip():
            self._current = initial.strip()
            if self._criteria.requires_execution:
                try:
                    ast.parse(self._current)
                except SyntaxError as exc:
                    self._log(
                        f"  Seed `initial` is not valid Python ({exc}) — "
                        "generating a baseline instead."
                    )
                    self._current = self._llm.generate_initial(self._criteria)
        else:
            self._current = self._llm.generate_initial(self._criteria)
        self._storage.save_artifact("best_result.txt", self._current)

        if self._criteria.requires_execution:
            self._criteria.execution_harness = self._generate_harness_with_smoketest(
                self._current
            )

        self._state = ExperimentState(
            name=self._config.name,
            prompt=self._config.prompt,
            max_iterations=self._config.max_iterations,
            start_time=time.time(),
            last_update=time.time(),
            criteria=self._criteria.to_dict(),
        )
        self._save_state()
        return self._run_loop()

    def _resume(self) -> dict:
        state = self._storage.load_state()
        if state is None:
            raise RuntimeError("state.json is missing or corrupt — cannot resume.")
        self._state = state

        if self._state.criteria:
            self._criteria = Criteria.from_dict(self._state.criteria)
        else:
            self._log("No saved criteria — regenerating from prompt…")
            # requires_execution is a user choice, not an LLM guess — it must be
            # passed here or this call raises TypeError.
            self._criteria = self._llm.generate_criteria(
                self._config.prompt, self._config.requires_execution
            )
            self._state.criteria = self._criteria.to_dict()

        self._memory.load()
        # Prefer the persisted artifact: after compression the global best may no
        # longer be in memory.jsonl, so best() could report a worse iteration
        # than state.best_score.
        best = self._memory.best()
        self._current = self._storage.load_artifact("best_result.txt") or (
            best.implementation if best else ""
        )
        if not self._current:
            self._log("No prior implementation found — regenerating initial baseline…")
            self._current = self._llm.generate_initial(self._criteria)

        self._log(
            f"Resumed at iteration {self._state.current_iteration}, "
            f"best score {self._state.best_score:.2f}"
        )
        return self._run_loop()

    # ------------------------------------------------------------------ #
    #  Main loop
    # ------------------------------------------------------------------ #

    def _run_loop(self) -> dict:
        assert self._state is not None  # always set by _start/_resume

        self._state.status = "running"
        self._save_state()
        loop_start = time.time()
        # Persisted across resumes: a resumed run should still know it has been
        # plateauing, otherwise the stagnation warning resets to zero.
        stagnation_count = self._state.stagnation_count

        try:
            while True:
                if self._stop_requested:
                    self._state.status = "stopped"
                    break
                if self._pause_requested:
                    self._state.status = "paused"
                    self._pause_requested = False
                    break
                if self._state.current_iteration >= self._config.max_iterations:
                    self._state.status = "completed"
                    break
                elapsed = self._state.total_elapsed_seconds + (time.time() - loop_start)
                if elapsed >= self._config.max_time_seconds:
                    # A time-limited run is not the same as hitting the target.
                    self._state.status = "time_exhausted"
                    break

                prev_best = self._state.best_score
                self._state.current_iteration += 1
                n = self._state.current_iteration
                self._log(f"--- Iteration {n} / {self._config.max_iterations} ---")

                try:
                    self._run_iteration(n, stagnation_count)
                except KeyboardInterrupt:
                    raise
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Unhandled error in iteration %d", n)
                    self._log(f"Iteration {n} failed with error: {exc}  (continuing)")

                if self._state.best_score > prev_best + self._config.stagnation_threshold:
                    stagnation_count = 0
                else:
                    stagnation_count += 1
                self._state.stagnation_count = stagnation_count
        except KeyboardInterrupt:
            # Second Ctrl-C: save what we have, then let the caller see the interrupt.
            self._state.status = "stopped"
            self._log("Interrupted — saving state before exiting.")
            raise
        finally:
            self._state.total_elapsed_seconds += time.time() - loop_start
            self._save_state()

        self._log(
            f"Experiment {self._state.status}. "
            f"Best score: {self._state.best_score:.2f} "
            f"(iteration {self._state.best_iteration})"
        )

        best = self._memory.best()
        return {
            "status": self._state.status,
            "best_score": self._state.best_score,
            "best_iteration": self._state.best_iteration,
            "total_iterations": self._state.current_iteration,
            "best_result": best.implementation if best else self._current,
        }

    # ------------------------------------------------------------------ #
    #  Single iteration
    # ------------------------------------------------------------------ #

    def _run_iteration(self, iteration_num: int, stagnation_count: int = 0) -> None:
        assert self._criteria is not None

        iteration = Iteration(number=iteration_num)

        # --- 1. STRATEGIC REVIEW ---
        context = self._memory.build_context(include_full_code=False)
        tried = self._memory.past_proposals()
        analysis, proposal, avoid = self._llm.strategic_review(
            context, self._criteria, tried, stagnation_count=stagnation_count
        )
        iteration.analysis = analysis
        iteration.proposal = proposal
        self._log(f"  Proposal: {proposal[:120]}")

        # --- 2. IMPLEMENT (with syntax retry for code tasks) ---
        impl = self._implement_with_retry(proposal, context, avoid=avoid)
        if impl is None:
            # Record the failure: without an Iteration the proposal never enters
            # past_proposals(), so the LLM can loop on the same failing idea.
            msg = "  Implementation failed after all retries — recording and skipping."
            self._log(msg)
            logger.warning(msg)
            iteration.evaluation_score = 0.0
            iteration.evaluation_reasoning = (
                "Implementation failed after all retries (LLM error or syntax error)."
            )
            self._memory.add(iteration)
            self._save_state()
            return
        iteration.implementation = impl

        # --- 3. EXECUTE ---
        if self._criteria.requires_execution:
            exec_result = self._execute(impl)
            iteration.execution_output = exec_result.combined_output[:2000]
            iteration.execution_success = exec_result.success
            iteration.execution_elapsed = exec_result.elapsed_seconds

            result_for_eval = (
                f"Execution {'succeeded' if exec_result.success else 'FAILED'}.\n"
                f"Elapsed: {exec_result.elapsed_seconds:.3f}s\n"
                f"Output:\n{iteration.execution_output}\n\n"
                f"Implementation:\n```python\n{impl}\n```"
            )
        else:
            result_for_eval = f"Implementation:\n```python\n{impl}\n```"

        # --- 4. EVALUATE ---
        if self._criteria.requires_execution:
            parsed = exec_result.parsed_json()
            if parsed is None:
                # Harness crashed with no JSON output
                score = 0.0
                reasoning = "Harness produced no JSON output (Python crash or timeout)."
            elif "score" in parsed:
                # Harness computed the score deterministically from the rubric.
                # Honour the harness's own `correct` field as a safeguard: if
                # the implementation failed correctness checks the score must
                # be 0.0 regardless of what the scoring formula produced.
                if parsed.get("correct") is False:
                    score = 0.0
                    reasoning = "Correctness checks failed. " + str(parsed.get("details", ""))[:350]
                    logger.warning("Harness reported correct=False — overriding score to 0.0")
                else:
                    try:
                        score = max(0.0, min(1.0, float(parsed["score"])))
                    except (TypeError, ValueError):
                        score = 0.0
                    reasoning = str(parsed.get("details", f"Harness score: {score:.2f}"))[:400]
            else:
                # Harness has no score field — fall back to LLM evaluator
                logger.warning("Harness JSON has no 'score' field — falling back to LLM evaluator")
                score, reasoning = self._llm.evaluate(
                    result_for_eval, self._criteria,
                    *self._prev_best_for_eval(),
                )
        else:
            score, reasoning = self._llm.evaluate(
                result_for_eval, self._criteria,
                *self._prev_best_for_eval(),
            )

        iteration.evaluation_score = score
        iteration.evaluation_reasoning = reasoning
        self._log(f"  Score: {score:.2f} — {reasoning[:120]}")

        # --- 5. STORE ---
        self._memory.add(iteration)

        # Always save the iteration content so every run is inspectable on disk.
        ext = "py" if self._criteria.requires_execution else "txt"
        self._storage.save_code(iteration_num, impl, ext=ext)

        if self._criteria.requires_execution:
            self._storage.save_execution_result(
                iteration_num,
                {
                    "success": exec_result.success,
                    "elapsed_seconds": exec_result.elapsed_seconds,
                    "timed_out": exec_result.timed_out,
                    "output": iteration.execution_output,
                },
            )

        # Track global best
        if score > self._state.best_score:
            self._state.best_score = score
            self._state.best_iteration = iteration_num
            self._current = impl
            self._storage.save_artifact("best_result.txt", impl)
            self._log(f"  ★ New best! Score {score:.2f}")

        self._save_state()

        # --- 6. COMPRESS ---
        if self._memory.should_compress():
            self._log("  Memory budget reached — compressing history…")
            self._memory.compress(self._criteria)
        
        # Ensure we're within the hard budget ceiling (Fix #1)
        if self._memory.is_over_budget():
            self._log("  Context exceeds hard budget — enforcing ceiling…")
            self._memory.ensure_within_budget(self._criteria)

    # ------------------------------------------------------------------ #
    #  Implementation helpers
    # ------------------------------------------------------------------ #

    def _implement_with_retry(
        self, proposal: str, context: str, max_retries: int = 3,
        avoid: list[str] | None = None,
    ) -> str | None:
        """
        Ask the LLM to implement the proposal.  For code tasks, verify syntax
        and retry with the error message if it fails.  Transient LLM errors
        (timeouts, empty responses) are also retried up to max_retries times.
        """
        assert self._criteria is not None
        current_proposal = proposal

        for attempt in range(max_retries):
            try:
                impl = self._llm.implement(
                    current_proposal, self._current, context, self._criteria,
                    avoid=avoid,
                )
                if self._criteria.requires_execution:
                    ast.parse(impl)  # raises SyntaxError if invalid
                return impl
            except SyntaxError as exc:
                self._log(f"  Syntax error (attempt {attempt + 1}): {exc}")
                current_proposal = (
                    f"{proposal}\n\n"
                    f"[Previous attempt had a Python syntax error: {exc}. "
                    "Fix the syntax and try again.]"
                )
            except Exception as exc:  # noqa: BLE001 — transient LLM error, retry
                if attempt == max_retries - 1:
                    logger.error(
                        "implement() failed after %d attempts: %s", max_retries, exc, exc_info=True
                    )
                    break
                logger.warning("implement() error (attempt %d/%d): %s", attempt + 1, max_retries, exc)
                time.sleep(2 ** attempt)

        return None

    # Smoke-test uses a longer timeout than the per-iteration default because
    # the harness is run against the naive baseline, which may be very slow
    # (e.g. recursive fibonacci(35) takes ~5s per call).
    _HARNESS_SMOKETEST_TIMEOUT = 120
    _HARNESS_MAX_ATTEMPTS = 3

    def _generate_harness_with_smoketest(self, implementation: str) -> str:
        """
        Generate an execution harness and immediately smoke-test it by running
        it against `implementation`.  Retries up to _HARNESS_MAX_ATTEMPTS times
        if the harness crashes (produces no parseable JSON output).

        Returns the last generated harness even if all attempts fail — the
        per-iteration crash-regen path in _execute() is the final safety net.
        """
        assert self._criteria is not None
        harness = ""
        for attempt in range(self._HARNESS_MAX_ATTEMPTS):
            self._log(
                f"  Generating harness (attempt {attempt + 1}/{self._HARNESS_MAX_ATTEMPTS})…"
            )
            harness = self._llm.generate_execution_harness(implementation, self._criteria)
            result = self._executor.run(
                implementation, harness, timeout=self._HARNESS_SMOKETEST_TIMEOUT
            )
            parsed = result.parsed_json()
            if parsed is not None and "score" in parsed:
                try:
                    score_val = float(parsed["score"])
                    # Correctness failure → score must be 0.0 even if speedup is high
                    if parsed.get("correct") is False:
                        score_val = 0.0
                    assert 0.0 <= score_val <= 1.0
                    self._log(f"  Harness smoke-test passed (score={score_val:.2f}).")
                    return harness
                except (TypeError, ValueError, AssertionError):
                    fail_reason = f"score field invalid: {parsed.get('score')!r}"
            elif parsed is not None:
                fail_reason = "JSON output has no 'score' field"
            else:
                fail_reason = "no JSON in output"
            remaining = self._HARNESS_MAX_ATTEMPTS - attempt - 1
            logger.warning(
                "Harness smoke-test attempt %d/%d failed (%s). Output: %s",
                attempt + 1, self._HARNESS_MAX_ATTEMPTS, fail_reason,
                result.combined_output[:400],
            )
            self._log(
                f"  Harness smoke-test failed ({fail_reason}) — "
                f"{'retrying' if remaining else 'giving up, will retry during iterations'}…"
            )
        return harness

    def _execute(self, implementation: str) -> ExecutionResult:
        """
        Run the implementation through the execution harness.
        If no harness exists yet, generate one.
        If execution fails (non-timeout), regenerate the harness once and retry.
        """
        assert self._criteria is not None

        # Lazily generate the harness on first use (fallback: resume with no saved harness).
        if not self._criteria.execution_harness:
            self._criteria.execution_harness = self._generate_harness_with_smoketest(
                implementation
            )
            if self._state is not None:
                self._state.criteria = self._criteria.to_dict()

        result = self._executor.run(implementation, self._criteria.execution_harness)

        # Regenerate the harness only when it has crashed at the Python level —
        # i.e. the process exited non-zero AND produced no parseable JSON.
        # If the harness ran successfully and returned {"passed": false} that is
        # a legitimate result (e.g. speedup < 1.0) and we must NOT discard it.
        if (
            not result.success
            and not result.timed_out
            and result.parsed_json() is None
        ):
            self._log("  Harness crashed (no JSON output) — refreshing harness and retrying…")
            self._criteria.execution_harness = self._llm.generate_execution_harness(
                implementation, self._criteria
            )
            if self._state is not None:
                self._state.criteria = self._criteria.to_dict()
            result = self._executor.run(implementation, self._criteria.execution_harness)

        return result

    # ------------------------------------------------------------------ #
    #  Persistence helper
    # ------------------------------------------------------------------ #

    def _prev_best_for_eval(self) -> tuple:
        """Return (prev_score, prev_summary) for the LLM evaluate fallback path."""
        assert self._criteria is not None
        best = self._memory.best()
        if best is None:
            return 0.0, "No previous result yet."
        if self._criteria.requires_execution:
            summary = (
                f"Execution {'succeeded' if best.execution_success else 'FAILED'}.\n"
                f"Elapsed: {best.execution_elapsed:.3f}s\n"
                f"Output:\n{best.execution_output}\n\n"
                f"Implementation:\n```python\n{best.implementation}\n```\n\n"
                f"Evaluator reasoning: {best.evaluation_reasoning}"
            )
        else:
            summary = (
                f"Implementation:\n```python\n{best.implementation}\n```\n\n"
                f"Evaluator reasoning: {best.evaluation_reasoning}"
            )
        return best.evaluation_score, summary

    def _save_state(self) -> None:
        if self._state is None:
            return
        if self._criteria:
            self._state.criteria = self._criteria.to_dict()
        self._storage.save_state(self._state)

    def _log(self, msg: str) -> None:
        if self._config.verbose:
            print(f"[Experimenter] {msg}")
        # logger is experimenter.experiment → via the package logger to experiment.log
        logger.info(msg)
