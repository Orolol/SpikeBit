"""Minimal logging helpers used across Phase 0/1 scripts.

We deliberately keep this tiny: a unified stdlib logger with a nicer
formatter and a `log_metrics` helper that writes one JSON line per call
for offline analysis. Phase 3+ will plug W&B on top.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s :: %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


class JsonlMetricsWriter:
    """Append one JSON line per call. Lightweight stand-in for W&B."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", buffering=1)  # line-buffered
        self._t0 = time.time()

    def log(self, step: int, **metrics: Any) -> None:
        record = {"step": step, "wall_s": time.time() - self._t0, **metrics}
        self._fh.write(json.dumps(record) + "\n")

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> JsonlMetricsWriter:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
