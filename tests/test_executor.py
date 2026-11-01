import os
import tempfile
import unittest

from experimenter.config import ExecutorConfig
from experimenter.executor import SubprocessExecutor, _child_env
from experimenter.parsing import has_main_guard, strip_main_guard


class MainGuardTests(unittest.TestCase):
    def test_top_level_guard_is_removed(self):
        src = 'def f():\n    return 1\nif __name__ == "__main__":\n    print("runs")\n'
        removed, cleaned = strip_main_guard(src)
        self.assertTrue(removed)
        self.assertNotIn("__main__", cleaned)
        self.assertIn("def f():", cleaned)

    def test_indented_guard_is_detected(self):
        src = 'if True:\n    if __name__ == "__main__":\n        print("x")\n'
        self.assertTrue(has_main_guard(src))
        # ...but it is not stripped: removing it would leave `if True:` with an
        # empty body, which is a SyntaxError.
        self.assertEqual(strip_main_guard(src), (False, src))

    def test_string_literal_is_not_a_guard(self):
        # The old regex matched this and silently rewrote a line of code.
        src = 's = \'if __name__ == "__main__": print(1)\'\nprint(s)'
        self.assertFalse(has_main_guard(src))
        self.assertEqual(strip_main_guard(src), (False, src))

    def test_harness_keeps_control(self):
        # Implementation guard removed => the harness is the only __main__ block.
        impl = 'def f():\n    return 2\nif __name__ == "__main__":\n    print("self-test")\n'
        harness = 'import json\nprint(json.dumps({"score": 0.5, "passed": f() == 2}))'
        exec_ = SubprocessExecutor(default_timeout=10)
        result = exec_.run(impl, harness)
        self.assertEqual(result.parsed_json(), {"score": 0.5, "passed": True})
        self.assertNotIn("self-test", result.stdout)


class EnvironmentTests(unittest.TestCase):
    def test_environment_is_minimal(self):
        env = _child_env()
        self.assertNotIn("PYTHONSTARTUP", env)
        self.assertNotIn("PYTHONPATH", env)
        self.assertEqual(env.get("PYTHONHASHSEED"), "0")
        self.assertEqual(sorted(env), sorted(["PATH", "LANG", "PYTHONIOENCODING",
                                              "PYTHONUNBUFFERED", "PYTHONHASHSEED"]))

    def test_startup_file_and_path_are_ignored(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        # Would be executed automatically if PYTHONSTARTUP were inherited.
        startup = os.path.join(tmp.name, "startup.py")
        with open(startup, "w") as fh:
            fh.write("print('startup ran')\n")
        env = dict(_child_env(), PYTHONSTARTUP=startup, PYTHONPATH=tmp.name)
        import subprocess
        import sys
        proc = subprocess.run(
            [sys.executable, "-I", "-c", "import sys; print('ok')"],
            capture_output=True, text=True, env=env,
        )
        self.assertEqual(proc.stdout.strip(), "ok")
        self.assertNotIn("startup ran", proc.stdout)

    def test_isolated_flag_is_used(self):
        exec_ = SubprocessExecutor(tmp_dir=tempfile.gettempdir())
        argv = exec_._argv("script.py")
        self.assertIn("-I", argv)
        exec_.isolated = False
        self.assertNotIn("-I", exec_._argv("script.py"))


class ExecutorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.exec_ = SubprocessExecutor(default_timeout=10, tmp_dir=self.tmp.name)

    def test_harness_result_is_parsed(self):
        impl = "def f(n):\n    return n * 2\n"
        harness = 'import json\nprint(json.dumps({"passed": f(2) == 4, "score": 1.0, "details": "ok"}))'
        result = self.exec_.run(impl, harness)
        self.assertTrue(result.success)
        parsed = result.parsed_json()
        self.assertEqual(parsed["score"], 1.0)

    def test_pretty_printed_json_is_parsed(self):
        harness = 'import json\nprint(json.dumps({"score": 0.42}, indent=2))'
        result = self.exec_.run("x = 1", harness)
        self.assertEqual(result.parsed_json(), {"score": 0.42})

    def test_non_dict_json_returns_none(self):
        harness = 'import json\nprint(json.dumps([1, 2, 3]))'
        result = self.exec_.run("x = 1", harness)
        self.assertIsNone(result.parsed_json())

    def test_timeout_kills_child(self):
        result = self.exec_.run("x = 1", "import time\ntime.sleep(5)\n", timeout=1)
        self.assertTrue(result.timed_out)
        self.assertFalse(result.success)

    def test_partial_output_survives_timeout(self):
        harness = ('import json, sys, time\n'
                   'print(json.dumps({"score": 0.1, "details": "partial"}))\n'
                   'sys.stdout.flush()\n'
                   'time.sleep(5)\n')
        result = self.exec_.run("x = 1", harness, timeout=1)
        self.assertTrue(result.timed_out)
        self.assertEqual(result.parsed_json(), {"score": 0.1, "details": "partial"})

    def test_memory_limit_is_enforced(self):
        exec_ = SubprocessExecutor(default_timeout=10, tmp_dir=self.tmp.name, memory_mb=64)
        harness = "import sys\nbytearray(512 * 1024 * 1024)\nprint('allocated')"
        result = exec_.run("x = 1", harness)
        self.assertFalse(result.success)
        self.assertNotIn("allocated", result.stdout)

    def test_non_utf8_output_does_not_crash_the_run(self):
        harness = 'import json, sys\nsys.stdout.buffer.write(b"\\xff\\xfe")\nprint(json.dumps({"score": 0.5}))'
        result = self.exec_.run("x = 1", harness)
        self.assertTrue(result.success)
        self.assertEqual(result.parsed_json(), {"score": 0.5})

    def test_stdin_is_closed_so_input_cannot_hang(self):
        harness = 'import json\ntry:\n    input()\nexcept Exception:\n    pass\nprint(json.dumps({"score": 0.3}))'
        result = self.exec_.run("x = 1", harness)
        self.assertEqual(result.parsed_json(), {"score": 0.3})

    def test_temp_file_is_cleaned_up(self):
        self.exec_.run("x = 1", 'import json\nprint(json.dumps({"score": 0.0}))')
        self.assertEqual([p for p in os.listdir(self.tmp.name) if p.endswith(".py")], [])

    def test_scrub_env_can_be_disabled_deliberately(self):
        os.environ["EXPERIMENTER_TEST_SECRET"] = "secret"
        self.addCleanup(os.environ.pop, "EXPERIMENTER_TEST_SECRET", None)
        self.assertIsNone(_child_env(True).get("EXPERIMENTER_TEST_SECRET"))
        self.assertEqual(_child_env(False).get("EXPERIMENTER_TEST_SECRET"), "secret")
        self.assertEqual(_child_env(SubprocessExecutor(scrub_env=False).scrub_env).get(
            "EXPERIMENTER_TEST_SECRET"), "secret")

    def test_executor_config_flows_through(self):
        cfg = ExecutorConfig(timeout=45, memory_mb=256, cpu_seconds=30, isolated=False)
        ex = SubprocessExecutor(config=cfg)
        self.assertEqual(ex.default_timeout, 45)
        self.assertEqual(ex.memory_mb, 256)
        self.assertEqual(ex.cpu_seconds, 30)
        self.assertFalse(ex.isolated)
        for kwargs in ({"timeout": 0}, {"memory_mb": 0}, {"cpu_seconds": 0}):
            with self.assertRaises(ValueError):
                ExecutorConfig(**kwargs)

    def test_run_uses_config_timeout(self):
        ex = SubprocessExecutor(config=ExecutorConfig(timeout=1), tmp_dir=self.tmp.name)
        result = ex.run("x = 1", 'import time\ntime.sleep(3)\nprint("late")')
        self.assertTrue(result.timed_out)
        self.assertFalse(result.success)


if __name__ == "__main__":
    unittest.main()
