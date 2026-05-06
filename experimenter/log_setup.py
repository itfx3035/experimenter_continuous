"""
Log setup — attaches file handlers to the two experiment-scoped loggers.

Called once from Experiment.__init__ with the experiment's storage directory.
Safe to call again (handlers are not duplicated).

  experimeter.experiment  →  {base}/experiment.log   (brief step-by-step)
  experimeter.llm         →  {base}/llm.log          (full prompts + responses)
"""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(base_dir: Path) -> None:
    _attach(
        name="experimeter.experiment",
        path=base_dir / "experiment.log",
        fmt="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    _attach(
        name="experimeter.llm",
        path=base_dir / "llm.log",
        fmt="%(asctime)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _attach(name: str, path: Path, fmt: str, datefmt: str) -> None:
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    # Don't bubble up to the root logger — keeps console clean unless the
    # caller explicitly configures a root handler.
    log.propagate = False

    resolved = str(path.resolve())
    for h in log.handlers:
        if isinstance(h, logging.FileHandler) and h.baseFilename == resolved:
            return  # already attached

    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    log.addHandler(handler)
