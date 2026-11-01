"""
Log setup — attaches file handlers to the experiment-scoped loggers.

Called once from Experiment.__init__ with the experiment's storage directory.
Safe to call again (handlers are not duplicated).

  experimenter (package logger)        →  {base}/experiment.log  (step-by-step)
  experimenter.llm                     →  {base}/llm.log          (prompts/responses)

Names must match `logging.getLogger(__name__)` in the modules, i.e. the real
package name (`experimenter`), not the README title.  Handlers are attached to
the package logger so every submodule (experiment, memory, storage) is covered;
`experimenter.llm` is handled separately (propagate=False) so its full prompts
do not also land in experiment.log.
"""

from __future__ import annotations

import logging
from pathlib import Path

_PKG = "experimenter"


def setup_logging(base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)

    # Package logger: catches experimenter, experimenter.experiment,
    # experimenter.memory, experimenter.storage, ...
    pkg = logging.getLogger(_PKG)
    pkg.setLevel(logging.DEBUG)
    pkg.propagate = False
    _attach(
        pkg,
        base_dir / "experiment.log",
        fmt="%(asctime)s  %(name)-24s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # LLM logger stays separate — full prompts/responses would flood the
    # step-by-step log.
    llm = logging.getLogger(f"{_PKG}.llm")
    llm.propagate = False  # do not duplicate into experiment.log
    _attach(
        llm,
        base_dir / "llm.log",
        fmt="%(asctime)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _attach(log: logging.Logger, path: Path, fmt: str, datefmt: str) -> None:
    log.setLevel(logging.DEBUG)
    resolved = str(path.resolve())
    for h in log.handlers:
        if isinstance(h, logging.FileHandler) and h.baseFilename == resolved:
            return  # already attached

    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    log.addHandler(handler)
