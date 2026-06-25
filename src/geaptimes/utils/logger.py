"""Minimal logging setup shared across geapTimes."""

import logging

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def get_logger(name: str = "geaptimes", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger, attaching a stream handler exactly once."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger
