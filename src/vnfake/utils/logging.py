"""Structured logging for vnfake.

All application code MUST obtain loggers via :func:`get_logger`. Direct
``logging.getLogger`` and ``print`` are disallowed in ``src/``.
"""

from __future__ import annotations

import logging
import os
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"
_NAMESPACE = "vnfake"
_CONFIGURED = False


def _resolve_level() -> int:
    raw = os.getenv("VNFAKE_LOG_LEVEL", "INFO").upper()
    level = logging.getLevelName(raw)
    return level if isinstance(level, int) else logging.INFO


def _configure_namespace() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    logger = logging.getLogger(_NAMESPACE)
    logger.setLevel(_resolve_level())
    logger.propagate = False
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger with the shared configuration.

    Args:
        name: Logger name, typically ``__name__``.

    Returns:
        A logger living under the ``vnfake`` namespace.
    """
    _configure_namespace()
    if name == _NAMESPACE or name.startswith(f"{_NAMESPACE}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_NAMESPACE}.{name}")
