"""Unit tests for vnfake.utils.logging."""

from __future__ import annotations

import logging

from vnfake.utils.logging import get_logger


def test_get_logger_returns_namespaced_logger() -> None:
    logger = get_logger("vnfake.test")
    assert isinstance(logger, logging.Logger)
    assert logger.name.startswith("vnfake")


def test_get_logger_prepends_namespace() -> None:
    logger = get_logger("mymodule")
    assert logger.name == "vnfake.mymodule"
