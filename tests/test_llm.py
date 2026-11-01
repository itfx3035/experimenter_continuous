import unittest

from experimenter.config import TOKEN_CHAR_FACTOR, MemoryConfig, OllamaConfig
from experimenter.criteria import Criteria
from experimenter.iteration import CompressedMemory
from experimenter.llm import LLMClient
from experimenter.parsing import extract_json, fit_to_budget

from . import stubs  # noqa: F401


class FitContextTests(unittest.TestCase):
    def test_short_prompt_is_returned_unchanged(self):
        llm = LLMClient(OllamaConfig(num_ctx=32768))
        self.assertEqual(llm._fit_context("sys", "user", "test"), "user")

    def test_oversized_prompt_is_actually_truncated(self):
        """Regression test: the old implementation mutated a local variable, so
        the truncation never reached the message that was really sent."""
        llm = LLMClient(OllamaConfig(num_ctx=1000))
        user = "x" * 5000
        fitted = llm._fit_context("system prompt", user, "implement")
        self.assertLess(len(fitted), len(user))
        self.assertIn("truncated", fitted)

    def test_response_room_is_reserved_before_truncating(self):
        """A context that 'just fits' still fails if nothing is left for num_predict."""
        cfg = MemoryConfig(reserved_response_tokens=1024, static_prompt_tokens=600)
        llm = LLMClient(OllamaConfig(num_ctx=4096), cfg)
        fitted = llm._fit_context("sys", "x" * 20000, "implement")
        budget = cfg.effective_budget(llm._config.get_context_budget())
        self.assertEqual(budget, int(4096 * 0.6) - 1024 - 600)
        content = fitted.split("\n[context")[0]      # the marker itself is not context
        self.assertLessEqual(len(content) // TOKEN_CHAR_FACTOR, budget - 1)

    def test_truncation_prefers_a_line_boundary(self):
        cfg = MemoryConfig(reserved_response_tokens=0, static_prompt_tokens=0)
        llm = LLMClient(OllamaConfig(num_ctx=300), cfg)   # floored to MIN_EFFECTIVE_BUDGET
        user = "line\n" * 400
        fitted = llm._fit_context("sys", user, "implement")
        self.assertIn("truncated", fitted)
        self.assertLess(len(fitted), len(user))
        body = fitted.split("\n[context")[0]
        self.assertEqual(body.splitlines(), ["line"] * len(body.splitlines()))
        self.assertEqual(fit_to_budget("abc", 10), "abc")

    def test_system_prompt_alone_too_large_raises(self):
        llm = LLMClient(OllamaConfig(num_ctx=100))
        with self.assertRaises(ValueError):
            llm._fit_context("y" * 2000, "user", "implement")

    def test_llm_and_memory_share_the_same_config(self):
        cfg = MemoryConfig()
        llm = LLMClient(OllamaConfig(num_ctx=8192), cfg)
        self.assertIs(llm._mem_cfg, cfg)
        self.assertLess(llm._mem_cfg.effective_budget(8192 * 0.6), 8192 * 0.6)


class ValidatorTests(unittest.TestCase):
    def test_json_validator_reports_missing_keys(self):
        from experimenter.llm import _validate_json_keys
        v = _validate_json_keys("analysis", "proposal")
        self.assertIsNone(v('{"analysis": "a", "proposal": "p"}'))
        self.assertIn("proposal", v('{"analysis": "a"}'))

    def test_evaluate_validator_rejects_out_of_range_score(self):
        from experimenter.llm import _validate_evaluate_response
        self.assertIsNone(_validate_evaluate_response('{"score": 0.5, "reasoning": "x"}'))
        self.assertIn("between", _validate_evaluate_response('{"score": 1.4}'))

    def test_code_validator(self):
        from experimenter.llm import _validate_non_empty_code
        # The validator is deliberately lenient: _extract_code falls back to the
        # raw text when the model omits fences, so only an empty response fails
        # (syntax is checked later, in _implement_with_retry).
        self.assertIsNone(_validate_non_empty_code("```python\nx = 1\n```"))
        self.assertIsNone(_validate_non_empty_code("x = 1"))
        self.assertIn("non-empty", _validate_non_empty_code("   "))

    def test_extract_json_still_available_through_parsing(self):
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})


class MergeSummaryTests(unittest.TestCase):
    class FakeLLM(LLMClient):
        def __init__(self, response):
            super().__init__(OllamaConfig())
            self.response = response

        def _invoke(self, system, user, max_retries=3, label="", validate=None):
            return self.response

    def test_merge_covers_the_whole_range_and_keeps_best_score(self):
        blocks = [
            CompressedMemory((1, 4), "s1", ["a"], 0.2),
            CompressedMemory((5, 9), "s2", ["b"], 0.7),
        ]
        criteria = Criteria("t", ["i"], "r", True)
        merged = self.FakeLLM('{"summary": "m", "key_findings": ["a", "b"], "best_score": 0.7}').merge_summaries(blocks, criteria)
        self.assertEqual(merged.iteration_range, (1, 9))
        self.assertEqual(merged.best_score, 0.7)
        self.assertEqual(merged.key_findings, ["a", "b"])

    def test_merge_handles_a_non_list_findings_field(self):
        blocks = [CompressedMemory((1, 2), "s", ["a"], 0.1)]
        merged = self.FakeLLM('{"summary": "m", "key_findings": "oops", "best_score": 0.3}').merge_summaries(
            blocks, Criteria("t", ["i"], "r", True)
        )
        self.assertEqual(merged.key_findings, [])


if __name__ == "__main__":
    unittest.main()
