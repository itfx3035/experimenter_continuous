import os
import pathlib
import tempfile
import unittest

import experimenter
from experimenter.config import (
    CONTEXT_TOKEN_BUDGET,
    ExperimentConfig,
    MemoryConfig,
    OllamaConfig,
    load_ollama_config_from_json,
)


class MemoryConfigTests(unittest.TestCase):
    def test_effective_budget_reserves_response_room(self):
        cfg = MemoryConfig(reserved_response_tokens=1024, static_prompt_tokens=600)
        self.assertEqual(cfg.effective_budget(6000), 4376)

    def test_effective_budget_has_a_floor(self):
        # Below the floor compression is pointless — the prompt would be cut to nothing.
        cfg = MemoryConfig()
        self.assertEqual(cfg.effective_budget(100), 512)

    def test_validation(self):
        for kwargs in (
            {"recent_window": 0},
            {"best_window": 0},
            {"max_compressed_blocks": 0},
            {"compression_trigger": 0},
            {"compression_trigger": 1.5},
            {"reserved_response_tokens": -1},
        ):
            with self.assertRaises(ValueError):
                MemoryConfig(**kwargs)

    def test_defaults_are_bounded_growth(self):
        cfg = MemoryConfig()
        self.assertEqual(cfg.max_compressed_blocks, 10)
        self.assertEqual(cfg.max_tried_items, 20)
        self.assertLessEqual(cfg.max_findings_per_block, 10)


class OllamaConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_defaults(self):
        cfg = OllamaConfig()
        self.assertEqual(cfg.timeout, 600)
        self.assertEqual(cfg.num_ctx, 32768)
        self.assertEqual(cfg.get_context_budget(), 19660)

    def test_validation(self):
        with self.assertRaises(ValueError):
            OllamaConfig(num_ctx=0)
        with self.assertRaises(ValueError):
            OllamaConfig(timeout=0)
        with self.assertRaises(ValueError):
            OllamaConfig(temperature=3)

    def test_from_json_coerces_types(self):
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        tmp.write('{"model": "m", "temperature": \"0.2\", \"timeout\": \"90\", \"num_ctx\": \"4096\"}')
        tmp.close()
        cfg = OllamaConfig.from_json(tmp.name)
        self.assertEqual(cfg.model, "m")
        self.assertEqual(cfg.temperature, 0.2)
        self.assertEqual(cfg.timeout, 90)
        self.assertEqual(cfg.num_ctx, 4096)

    def test_missing_file_falls_back_to_defaults(self):
        self.assertEqual(load_ollama_config_from_json("/nonexistent.json")["model"], "llama3.1")

    def test_non_object_json_falls_back_to_defaults(self):
        path = pathlib.Path(self.tmp.name) / "bad.json"
        path.write_text('[1, 2, 3]', encoding="utf-8")
        self.assertEqual(load_ollama_config_from_json(str(path))["model"], "llama3.1")

    def test_lookup_order_is_cwd_before_package(self):
        cwd = os.getcwd()
        os.chdir(self.tmp.name)
        try:
            # Nothing in the working directory → package default.
            self.assertEqual(load_ollama_config_from_json()["model"], "llama3.1")
            (pathlib.Path(self.tmp.name) / "ollama_config.json").write_text(
                '{"model": "from-cwd"}', encoding="utf-8"
            )
            self.assertEqual(load_ollama_config_from_json()["model"], "from-cwd")
            # An explicit path still wins over both.
            other = pathlib.Path(self.tmp.name) / "explicit.json"
            other.write_text('{"model": "explicit"}', encoding="utf-8")
            self.assertEqual(load_ollama_config_from_json(str(other))["model"], "explicit")
        finally:
            os.chdir(cwd)

    def test_example_template_is_shipped_and_parses(self):
        template = pathlib.Path(experimenter.__file__).parent / "ollama_config.example.json"
        self.assertTrue(template.exists())
        cfg = load_ollama_config_from_json(str(template))
        self.assertEqual(cfg["model"], "llama3.1")
        self.assertEqual(OllamaConfig.from_json(str(template)).num_ctx, 32768)


class ExperimentConfigTests(unittest.TestCase):
    def test_context_budget_follows_ollama_config(self):
        cfg = ExperimentConfig(name="x", prompt="p", requires_execution=True,
                               ollama_config=OllamaConfig(num_ctx=8192))
        self.assertEqual(cfg.context_budget(), 4915)

    def test_validation(self):
        with self.assertRaises(ValueError):
            ExperimentConfig(name="x", prompt="p", requires_execution=True, max_iterations=0)
        with self.assertRaises(ValueError):
            ExperimentConfig(name="x", prompt="p", requires_execution=True, max_time_seconds=0)
        with self.assertRaises(ValueError):
            ExperimentConfig(name="x", prompt="p", requires_execution=True,
                             stagnation_threshold=-0.01)

    def test_fallback_constant_is_only_a_fallback(self):
        self.assertEqual(CONTEXT_TOKEN_BUDGET, 20_000)


if __name__ == "__main__":
    unittest.main()
