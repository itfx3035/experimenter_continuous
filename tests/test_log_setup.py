import logging
import tempfile
import unittest
from pathlib import Path

from experimenter.log_setup import setup_logging

from . import stubs  # noqa: F401


class LogSetupTests(unittest.TestCase):
    """Regression test: handlers used to be attached to `experimeter.*`, which
    never matched `logging.getLogger(__name__)`, so the log files stayed empty."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.experiment_log = logging.getLogger("experimenter.experiment")
        self.package_log = logging.getLogger("experimenter")
        self.llm_log = logging.getLogger("experimenter.llm")
        for lg in (self.package_log, self.experiment_log, self.llm_log):
            for h in lg.handlers:
                h.close()
            lg.handlers.clear()

    def tearDown(self):
        for lg in (self.package_log, self.experiment_log, self.llm_log):
            for h in lg.handlers:
                h.close()
            lg.handlers.clear()

    def test_package_logger_receives_module_messages(self):
        setup_logging(self.base)
        self.experiment_log.info("hello from experiment")
        logging.getLogger("experimenter.memory").info("hello from memory")
        self.llm_log.info("hello from llm")
        for h in self.package_log.handlers:
            h.flush()
        for h in self.llm_log.handlers:
            h.flush()
        content = (self.base / "experiment.log").read_text()
        self.assertIn("hello from experiment", content)
        self.assertIn("hello from memory", content)
        self.assertNotIn("hello from llm", content)  # llm.log is separate
        llm_content = (self.base / "llm.log").read_text()
        self.assertIn("hello from llm", llm_content)

    def test_handlers_are_not_duplicated(self):
        setup_logging(self.base)
        setup_logging(self.base)
        self.assertEqual(len(self.package_log.handlers), 1)
        self.assertEqual(len(self.llm_log.handlers), 1)

    def test_creates_missing_directory(self):
        setup_logging(self.base / "nested")
        self.assertTrue((self.base / "nested" / "experiment.log").exists())


if __name__ == "__main__":
    unittest.main()
