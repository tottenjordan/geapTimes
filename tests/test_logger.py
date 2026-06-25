"""Tests for the shared logger helper."""

import logging

from geaptimes.utils.logger import get_logger


def test_get_logger_attaches_handler_once() -> None:
    logger = get_logger("geaptimes-test")
    assert logger.level == logging.INFO
    assert len(logger.handlers) == 1
    # idempotent: calling again does not add another handler
    again = get_logger("geaptimes-test")
    assert again is logger
    assert len(again.handlers) == 1
