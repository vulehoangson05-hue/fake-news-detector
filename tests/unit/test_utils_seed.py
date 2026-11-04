"""Unit tests for vnfake.utils.seed."""

from __future__ import annotations

import random

from vnfake.utils.seed import set_seed


def test_set_seed_is_deterministic() -> None:
    set_seed(123)
    a = [random.random() for _ in range(5)]
    set_seed(123)
    b = [random.random() for _ in range(5)]
    assert a == b
