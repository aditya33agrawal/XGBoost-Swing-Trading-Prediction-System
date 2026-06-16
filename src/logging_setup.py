"""Centralised logging configuration.

Call ``setup_logging()`` once at process start (CLI entry point or the first
Colab cell).  Everything else in the codebase just does::

    import logging
    logger = logging.getLogger(__name__)
    logger.info("...")

and the messages are formatted consistently with timestamps so progress is
visible in a Colab cell's live output.
"""
from __future__ import annotations

import logging
import os
import sys
import time


class _ElapsedFormatter(logging.Formatter):
    """Formatter that prefixes each line with wall-clock time + elapsed seconds."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._start = time.time()

    def format(self, record: logging.LogRecord) -> str:
        record.elapsed = f"{time.time() - self._start:7.1f}s"
        return super().format(record)


def setup_logging(level: str | int = "INFO") -> logging.Logger:
    """Configure the root logger.  Idempotent — safe to call multiple times.

    Level can be overridden with the ``LOG_LEVEL`` env var (handy in Colab where
    you can set ``os.environ['LOG_LEVEL'] = 'DEBUG'`` before running).
    """
    level = os.environ.get("LOG_LEVEL", level)
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    # Remove handlers a previous call (or Colab itself) may have installed so we
    # don't get duplicated lines.
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        _ElapsedFormatter(
            fmt="%(asctime)s | +%(elapsed)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.addHandler(handler)
    root.setLevel(level)

    # Quiet down noisy third-party loggers.
    for noisy in ("yfinance", "urllib3", "matplotlib", "numexpr"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logging.getLogger("pipeline")
