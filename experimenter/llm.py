"""
LLMClient — all communication with the Ollama model.

Design decisions
----------------
- One clean _invoke() method with exponential-backoff retry and a hard timeout.
  The timeout uses a daemon thread so a hung LLM call cannot prevent process
  exit (fixed from original: daemon=True).
- A single 0.0–1.0 evaluation scale throughout (original had two competing
  scales that caused confusion).
- JSON parsing is tolerant: parsing.extract_json() finds the first {...} block in
  the LLM output regardless of surrounding prose or markdown fences (it lives in
  parsing.py so it can be unit-tested without langchain installed).
- Code extraction strips ``` fences but otherwise returns the text verbatim so
  the caller can perform its own syntax check before using it.
- _fit_context() returns the truncated message that is actually sent.  The old
  _check_context_length() mutated a local parameter, so its truncation was a
  no-op and oversized prompts were sent to the model unchanged.
- No module-level TimeoutError redefinition; we use the Python built-in.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from .config import TOKEN_CHAR_FACTOR, MemoryConfig, OllamaConfig
from .criteria import Criteria
from .iteration import CompressedMemory
from .parsing import extract_code, extract_json, fit_to_budget

logger = logging.getLogger(__name__)

_SYSTEM_RESEARCHER = (
    "You are an expert researcher running an iterative improvement experiment. "
    "You are methodical, creative, and always build on prior findings. "
    "Follow instructions exactly; never add unrequested explanations."
)


# ------------------------------------------------------------------ #
#  Parsing helpers
# ------------------------------------------------------------------ #

def _extract_json(text: str) -> dict:
    """Alias for parsing.extract_json (kept for existing call sites)."""
    return extract_json(text)


def _extract_code(text: str) -> str:
    """Alias for parsing.extract_code (kept for existing call sites)."""
    return extract_code(text)


# ------------------------------------------------------------------ #
#  Response validators
# ------------------------------------------------------------------ #

def _validate_json_keys(*required_keys: str) -> Callable[[str], str | None]:
    """Return a validator that checks the response is valid JSON with the given keys."""
    def _validate(text: str) -> str | None:
        try:
            data = _extract_json(text)
        except ValueError as exc:
            return f"Response must be a valid JSON object: {exc}"
        missing = [k for k in required_keys if k not in data]
        if missing:
            return f"JSON is missing required keys: {missing}"
        return None
    return _validate


def _validate_non_empty_code(text: str) -> str | None:
    """Return an error string if the response has no extractable content.

    Deliberately lenient: `_extract_code` falls back to the raw text when the
    model omits fences, so only an empty response is rejected here — real
    syntax checking happens in Experiment._implement_with_retry.
    """
    if not _extract_code(text):
        return "Response must contain non-empty code inside a ```python ... ``` fence"
    return None


def _validate_evaluate_response(text: str) -> str | None:
    """Validate that the evaluate response has a score in [0.0, 1.0]."""
    try:
        data = _extract_json(text)
    except ValueError as exc:
        return f"Response must be a valid JSON object: {exc}"
    if "score" not in data:
        return "JSON must contain a 'score' key"
    try:
        s = float(data["score"])
        if not (0.0 <= s <= 1.0):
            return f"'score' must be between 0.0 and 1.0, got {s}"
    except (TypeError, ValueError):
        return f"'score' must be a float, got {data['score']!r}"
    return None


# ------------------------------------------------------------------ #
#  LLMClient
# ------------------------------------------------------------------ #

class LLMClient:
    def __init__(
        self, config: OllamaConfig, memory_config: MemoryConfig | None = None
    ) -> None:
        self._config = config
        # Same knobs MemoryManager uses, so the fit here and the compression
        # decision there agree on what "fits in the window" means.
        self._mem_cfg = memory_config or MemoryConfig()
        self._llm: ChatOllama | None = None

    @property
    def llm(self) -> ChatOllama:
        if self._llm is None:
            self._llm = ChatOllama(
                base_url=self._config.base_url,
                model=self._config.model,
                temperature=self._config.temperature,
                num_ctx=self._config.num_ctx,
            )
        return self._llm

    # ---------------------------------------------------------------- #
    #  Core invocation
    # ---------------------------------------------------------------- #

    def _invoke(
        self,
        system: str,
        user: str,
        max_retries: int = 3,
        label: str = "",
        validate: Callable[[str], str | None] | None = None,
    ) -> str:
        """
        Call the LLM with exponential-backoff retry and optional content validation.

        Parameters
        ----------
        validate:
            Optional callable that receives the raw response text and returns
            either None (response is valid) or an error string describing why
            it is invalid.  On failure the LLM is re-prompted with a correction
            hint appended to the user message.
        """
        # Pre-call validation: check the context fits the model's window AND use
        # the (possibly truncated) message that will actually be sent.
        user = self._fit_context(system, user, label)

        label_str = label or "llm"
        last_error: Exception = RuntimeError("no attempts made")
        current_user = user

        _TRUNC = 2000

        def _trunc(text: str) -> str:
            if len(text) <= _TRUNC:
                return text
            return text[:_TRUNC] + f"\n[… {len(text) - _TRUNC} chars truncated]"

        def _banner(attempt: int, suffix: str = "") -> str:
            mid = f" {label_str} · attempt {attempt + 1}/{max_retries} "
            if suffix:
                mid += f"· {suffix} "
            return "═" * 6 + mid + "═" * 6

        for attempt in range(max_retries):
            logger.debug("%s", _banner(attempt))
            if attempt == 0:
                logger.debug("SYSTEM:\n%s", _trunc(system))
                logger.debug("USER:\n%s", _trunc(current_user))
            else:
                logger.debug("SYSTEM: [unchanged]")
                logger.debug("USER (with correction hint):\n%s", _trunc(current_user))

            messages = [SystemMessage(content=system), HumanMessage(content=current_user)]
            try:
                raw = self._call_with_timeout(messages)
                if not (raw and raw.strip()):
                    logger.warning("%s  Empty LLM response", _banner(attempt, "EMPTY"))
                    last_error = RuntimeError("empty response")
                else:
                    response = raw.strip()
                    if validate is not None:
                        error = validate(response)
                        if error:
                            logger.warning("%s  Validation failed: %s", _banner(attempt, "INVALID"), error)
                            logger.debug("INVALID RESPONSE (%d chars):\n%s", len(response), _trunc(response))
                            current_user = (
                                f"{user}\n\n"
                                f"[Your previous response was invalid: {error}.\n"
                                f"Your previous response was:\n{response}\n"
                                "Fix the issues above and respond again.]"
                            )
                            last_error = ValueError(f"validation failed: {error}")
                        else:
                            logger.debug("RESPONSE (%d chars):\n%s", len(response), _trunc(response))
                            return response
                    else:
                        logger.debug("RESPONSE (%d chars):\n%s", len(response), _trunc(response))
                        return response
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.error("%s  Call failed: %s", _banner(attempt, "ERROR"), exc, exc_info=True)
            if attempt < max_retries - 1:
                time.sleep(2**attempt)  # 1s, 2s, 4s …

        raise RuntimeError(f"LLM failed after {max_retries} attempts: {last_error}") from last_error

    def _fit_context(self, system: str, user: str, label: str) -> str:
        """
        Return the user message, truncated if the combined system+user context
        does not fit inside the model's context window (safety margin applied).

        Raises ValueError only when the system prompt alone is already too large.
        """
        cfg = self._mem_cfg
        budget_tokens = cfg.effective_budget(self._config.get_context_budget())
        system_tokens = max(1, len(system) // TOKEN_CHAR_FACTOR)
        available = budget_tokens - system_tokens

        if available <= 0:
            raise ValueError(
                f"System prompt alone ({system_tokens} tokens) exceeds the usable "
                f"context budget ({budget_tokens} tokens, num_ctx={self._config.num_ctx})."
            )

        estimated = len(user) // TOKEN_CHAR_FACTOR
        if estimated <= available:
            return user

        logger.error(
            "Context too large (~%d tokens, ~%d for the system prompt, budget %d tokens "
            "with %d reserved for the response) for %s call",
            estimated, system_tokens, budget_tokens, cfg.reserved_response_tokens, label,
        )
        logger.warning("Truncating user message to ~%d tokens", available)
        return fit_to_budget(user, available)

    def _call_with_timeout(self, messages: list, timeout: int | None = None) -> str:
        timeout = timeout or self._config.timeout
        result: list = []
        error: list = []

        def _run() -> None:
            try:
                response = self.llm.invoke(messages)
                result.append(response.content)
            except Exception as exc:  # noqa: BLE001
                error.append(exc)

        # daemon=True: if this thread is still running when the main thread
        # exits, it will not prevent the process from terminating.
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout)

        if t.is_alive():
            raise TimeoutError(f"LLM call exceeded {timeout}s")
        if error:
            raise error[0]
        if not result:
            raise RuntimeError("LLM thread completed without producing a result")
        return result[0]

    def merge_summaries(
        self, blocks: list[CompressedMemory], criteria: Criteria
    ) -> CompressedMemory:
        """
        Merge several compressed blocks into one.

        Without this the compressed-history section grows one block per
        compression forever — the "compact" context keeps expanding and the
        oldest insights get pushed out of the window anyway.
        """
        block_text = "\n\n".join(b.format_for_llm() for b in blocks)
        first = min(b.iteration_range[0] for b in blocks)
        last = max(b.iteration_range[1] for b in blocks)
        user = (
            f"CRITERIA:\n{criteria.format_for_llm()}\n\n"
            f"SUMMARIES TO MERGE (iterations #{first}–#{last}):\n{block_text}\n\n"
            "Respond with JSON:\n"
            "  summary      (str)          — one merged summary of the whole range\n"
            "  key_findings (list of str)  — the most important insights across all blocks\n"
            "  best_score   (float)        — highest score achieved in this range\n"
        )
        raw = self._invoke(
            _SYSTEM_RESEARCHER, user, label="merge_summaries",
            validate=_validate_json_keys("summary", "key_findings", "best_score"),
        )
        data = _extract_json(raw)
        findings = data.get("key_findings", [])
        if not isinstance(findings, list):
            findings = []
        return CompressedMemory(
            iteration_range=(first, last),
            summary=data.get("summary", ""),
            key_findings=findings,
            best_score=max((b.best_score for b in blocks), default=float(data.get("best_score", 0.0))),
        )

    # ---------------------------------------------------------------- #
    #  High-level operations
    # ---------------------------------------------------------------- #

    def generate_initial(self, criteria: Criteria) -> str:
        """
        Generate a baseline implementation or content to seed the experiment.

        For code tasks this is a simple, obviously correct but unoptimised
        implementation.  For non-code tasks it is a first rough draft.
        The result will be improved iteratively — it only needs to be a
        reasonable starting point, not a good one.
        """
        if criteria.requires_execution:
            system = (
                "You write simple, readable Python code. "
                "Given a task description, write a naive but correct baseline implementation. "
                "Do not optimise — correctness matters more than speed. "
                "Return ONLY the code inside a single ```python ... ``` fence."
            )
            user = (
                f"Task: {criteria.task_description}\n\n"
                f"Success indicators:\n"
                + "\n".join(f"  - {s}" for s in criteria.success_indicators)
                + "\n\nWrite the baseline implementation:"
            )
            raw = self._invoke(
                system, user, label="generate_initial", validate=_validate_non_empty_code
            )
            return _extract_code(raw)
        else:
            system = (
                "You are a knowledgeable writer. "
                "Given a task description, produce a first rough draft. "
                "It does not need to be polished — it will be improved iteratively. "
                "Return only the content, no meta-commentary."
            )
            user = (
                f"Task: {criteria.task_description}\n\n"
                "Write a first draft:"
            )
            return self._invoke(system, user, label="generate_initial")

    def generate_criteria(self, prompt: str, requires_execution: bool) -> Criteria:
        """
        Analyse the user's prompt and produce structured success criteria.

        The harness is NOT generated here (no implementation exists yet).
        It is generated lazily in generate_execution_harness().
        """
        speedup_field = (
            "  speedup_target     (float) — realistic maximum speedup achievable vs a naive "
            "Python baseline for this task (e.g. 50.0 for pure-Python algorithmic optimisation, "
            "500.0 for tasks using numpy/C extensions, 10.0 for I/O-bound work). "
            "Score 1.0 is awarded at this speedup on a log scale.\n"
            if requires_execution else
            "  speedup_target     (float) — set to 0.0 (not an execution task).\n"
        )
        system = (
            "You are an expert in experimental design. "
            "Given a user's goal, analyse what the task requires and produce a structured plan.\n\n"
            "Respond with a single JSON object with exactly these keys:\n"
            "  task_description   (str)  — what the task actually requires, in your own words\n"
            "  success_indicators (list of str) — 3-6 concrete, measurable outcomes\n"
            "  evaluation_rubric  (str)  — how to map results to a score from 0.0 (failure) "
            "to 1.0 (perfect), with 4-5 distinct milestones.\n"
            + speedup_field +
            "\nRubric guidelines:\n"
            "- For performance/speed tasks: express milestones as speedup multipliers over the "
            "initial baseline (e.g. 1.5×, 3×, 10×, 50×). Use a logarithmic scale so that each "
            "milestone requires meaningfully more effort than the previous one. "
            "Do NOT use small percentage bands (e.g. 10%, 20%) — these collapse into the same score.\n"
            "- For quality/correctness tasks: define bands based on completeness and correctness, "
            "with each band requiring a qualitatively better result than the previous.\n"
            "- Each milestone should be clearly distinguishable from adjacent ones.\n"
        )
        raw = self._invoke(
            system, f"User goal:\n{prompt}", label="generate_criteria",
            validate=_validate_json_keys("task_description", "success_indicators", "evaluation_rubric"),
        )
        data = _extract_json(raw)
        return Criteria(
            task_description=data.get("task_description", prompt),
            success_indicators=data.get("success_indicators", []),
            evaluation_rubric=data.get("evaluation_rubric", ""),
            requires_execution=requires_execution,
            execution_harness="",  # generated lazily
            speedup_target=float(data.get("speedup_target", 0.0)),
        )

    def generate_execution_harness(self, implementation: str, criteria: Criteria) -> str:
        """
        Write a Python harness that tests the given implementation.

        The harness is appended after the implementation source and run as a
        script.  It must print a single JSON object to stdout including a
        deterministic `score` field computed from the evaluation rubric.
        """
        system = (
            "You write Python test harnesses for iterative optimization experiments. "
            "You will receive an implementation and evaluation criteria including a rubric. "
            "Write Python code that, when appended to the implementation and run as a script, "
            "prints exactly one JSON line to stdout with these required fields:\n"
            '  {"passed": <bool>, "correct": <bool>, "score": <float 0.0-1.0>, "details": <str>}\n\n'
            "CORRECTNESS GATES SCORING (highest priority rule):\n"
            "- Run all correctness/unit checks FIRST, before any timing.\n"
            "- If ANY correctness check fails: set correct=False, score=0.0, passed=False.\n"
            "- Only compute speed/quality metrics and a non-zero score when correct=True.\n\n"
            "SCORING (when correct=True):\n"
            "- Read the evaluation rubric carefully and embed its thresholds as constants.\n"
            "- Compute `score` deterministically from measured metrics. Do NOT call any LLM.\n"
            "- Example for an accuracy/quality task "
            "(rubric: accuracy 0.7→0.3, 0.85→0.6, 0.95→0.85, 0.99→1.0):\n"
            "    import numpy as np\n"
            "    score = float(np.interp(accuracy, [0.5, 0.7, 0.85, 0.95, 0.99], "
            "[0.0, 0.3, 0.6, 0.85, 1.0]))\n"
            "- Example for a performance/speedup task (log scale, target=50x):\n"
            "    import math\n"
            "    score = min(1.0, max(0.0, math.log(max(speedup, 1e-9)) / math.log(50)))\n\n"
            "Hard requirements:\n"
            "- Do NOT redefine any function or class already defined by the implementation.\n"
            "- The implementation's `if __name__ == '__main__':` block has been suppressed — "
            "your harness is the only code that runs at the top level.\n"
            "- Wrap everything in a try/except; on any exception set "
            "passed=False, score=0.0, details=full traceback.\n"
            "- Do NOT use importlib or __file__ to reload the implementation — "
            "its functions are already in the global scope.\n"
            "- Output ONLY the harness code — no fences, no explanations.\n\n"
            "Performance measurement (when criteria mention speed/performance):\n"
            "- Define a self-contained naive reference function with a DIFFERENT name "
            "(e.g. `_reference_impl`).\n"
            "- For correctness checks use a FAST helper — do NOT call the slow naive "
            "reference for all test inputs.\n"
            "- Time both functions with `timeit.Timer(lambda: ...).autorange()`.\n"
            "  IMPORTANT: autorange() returns (number_of_loops, total_time). Example:\n"
            "    import timeit\n"
            "    n_ref, t_ref = timeit.Timer(lambda: _reference_impl(35)).autorange()\n"
            "    n_opt, t_opt = timeit.Timer(lambda: fibonacci(35)).autorange()\n"
            "    speedup = (t_ref / n_ref) / (t_opt / n_opt)\n"
            "  If the reference function is slow (e.g. naive recursive), use a smaller\n"
            "  n (e.g. 30 instead of 35) or a fixed number of repeats\n"
            "  (e.g. timeit.timeit(lambda: _reference_impl(30), number=3) / 3)\n"
            "  to avoid timing out. The optimized version can still be timed at n=35.\n"
            "- Also include `\"speedup\": <float>, \"optimized_us\": <float>, "
            "\"reference_us\": <float>` in the JSON (times in microseconds).\n"
        )
        user = (
            f"Criteria:\n{criteria.format_for_llm()}\n\n"
            f"Implementation:\n```python\n{implementation}\n```\n\n"
            "Write the harness:"
        )
        return self._invoke(
            system, user, label="generate_harness", validate=_validate_non_empty_code
        ).strip()

    def strategic_review(
        self,
        context: str,
        criteria: Criteria,
        tried_approaches: list[str],
        stagnation_count: int = 0,
    ) -> tuple[str, str, list[str]]:
        """
        Review experiment history and propose the next concrete improvement.
        Returns (analysis, proposal, avoid).
        """
        tried_section = ""
        if tried_approaches:
            tried_lines = "\n".join(f"  - {t}" for t in tried_approaches)
            tried_section = (
                f"APPROACHES ALREADY TRIED (you MUST NOT repeat or paraphrase any of these):\n"
                f"{tried_lines}\n\n"
            )
        stagnation_section = ""
        if stagnation_count >= 3:
            stagnation_section = (
                f"⚠ STAGNATION WARNING: The score has NOT improved for {stagnation_count} "
                f"consecutive iterations. The current strategy is not working.\n"
                f"You MUST propose something fundamentally different — a different algorithm, "
                f"data structure, or approach that has NOT been tried before. "
                f"Incremental tweaks to existing approaches will not break the plateau.\n\n"
            )
        user = (
            f"EXPERIMENT GOAL:\n{criteria.format_for_llm()}\n\n"
            f"{stagnation_section}"
            f"{tried_section}"
            f"HISTORY:\n{context}\n\n"
            "Based on the history, respond with a JSON object:\n"
            "  analysis (str) — patterns observed, what worked, what failed\n"
            "  proposal (str) — a NOVEL, concrete improvement not listed in APPROACHES ALREADY TRIED\n"
            "  avoid    (list of str) — specific approaches not to repeat in future iterations\n"
        )
        raw = self._invoke(
            _SYSTEM_RESEARCHER, user, label="strategic_review",
            validate=_validate_json_keys("analysis", "proposal"),
        )
        data = _extract_json(raw)
        avoid = data.get("avoid", [])
        if not isinstance(avoid, list):
            avoid = []
        return data.get("analysis", ""), data.get("proposal", ""), avoid

    def implement(
        self,
        proposal: str,
        current: str,
        context: str,
        criteria: Criteria,
        avoid: list[str] | None = None,
    ) -> str:
        """
        Generate a new implementation based on the proposal.
        Returns raw source code (no fences).
        """
        avoid_section = ""
        if avoid:
            avoid_lines = "\n".join(f"  - {a}" for a in avoid)
            avoid_section = (
                f"DO NOT USE any of these approaches (already tried):\n"
                f"{avoid_lines}\n\n"
            )
        system = (
            "You are an expert developer. Implement the given proposal exactly. "
            "Return ONLY the implementation inside a single ```python ... ``` fence. "
            "No text outside the fence."
        )
        user = (
            f"GOAL:\n{criteria.format_for_llm()}\n\n"
            f"PROPOSAL:\n{proposal}\n\n"
            f"{avoid_section}"
            f"CURRENT IMPLEMENTATION:\n```python\n{current}\n```\n\n"
            f"RECENT CONTEXT:\n{context}\n\n"
            f"YOUR TASK: Implement this proposal exactly: {proposal}\n\n"
            "Write the new implementation:"
        )
        raw = self._invoke(
            system, user, label="implement", validate=_validate_non_empty_code
        )
        return _extract_code(raw)

    def evaluate(
        self,
        result: str,
        criteria: Criteria,
        previous_best_score: float,
        previous_best_summary: str,
    ) -> tuple[float, str]:
        """
        Score a result against the criteria.
        Returns (score: float 0.0–1.0, reasoning: str).
        """
        system = (
            "You are a strict, consistent evaluator. "
            "Score the result from 0.0 to 1.0 following the rubric exactly. "
            "The same result must always receive the same score. "
            "Apply the rubric independently — derive the score entirely from the rubric "
            "and the result shown. Do NOT anchor on or adjust relative to the previous best score. "
            "The result includes both execution output AND the implementation source code — "
            "you MUST evaluate all rubric dimensions including code quality. "
            "When the execution output contains a `speedup` field, you MUST use that value "
            "directly to determine the score from the rubric — do not ignore it or estimate."
        )
        user = (
            f"CRITERIA:\n{criteria.format_for_llm()}\n\n"
            f"PREVIOUS BEST (for comparison only — do not anchor your score on it):\n{previous_best_summary}\n\n"
            f"RESULT TO EVALUATE:\n{result}\n\n"
            'Respond with JSON: {"score": <float 0.0-1.0>, "reasoning": <str>}'
        )
        raw = self._invoke(
            system, user, label="evaluate", validate=_validate_evaluate_response
        )
        data = _extract_json(raw)
        score = max(0.0, min(1.0, float(data.get("score", 0.0))))
        return score, data.get("reasoning", "")

    def compress(
        self,
        iterations_text: str,
        criteria: Criteria,
        iteration_range: tuple[int, int],
    ) -> CompressedMemory:
        """Summarise a batch of iterations that are being evicted from context."""
        user = (
            f"CRITERIA:\n{criteria.format_for_llm()}\n\n"
            f"ITERATIONS TO COMPRESS (#{iteration_range[0]}–#{iteration_range[1]}):\n"
            f"{iterations_text}\n\n"
            "Respond with JSON:\n"
            "  summary      (str)          — concise summary of what was tried and learned\n"
            "  key_findings (list of str)  — 3–7 bullet points of the most important insights\n"
            "  best_score   (float)        — highest score achieved in this range\n"
        )
        raw = self._invoke(
            _SYSTEM_RESEARCHER, user, label="compress",
            validate=_validate_json_keys("summary", "key_findings", "best_score"),
        )
        data = _extract_json(raw)
        return CompressedMemory(
            iteration_range=iteration_range,
            summary=data.get("summary", ""),
            key_findings=data.get("key_findings", []),
            best_score=float(data.get("best_score", 0.0)),
        )
