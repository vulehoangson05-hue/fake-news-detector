"""Unit tests for vnfake.data.segmentation (no model/Java loaded)."""

from __future__ import annotations

import pytest

from vnfake.data.segmentation import (
    SegmenterProtocol,
    UndertheseaSegmenter,
    get_segmenter,
)


class _FakeSegmenter:
    """Deterministic mock: joins tokens with ``_``."""

    def segment(self, text: str) -> str:
        return "_".join(text.split()) if text.strip() else ""

    def segment_batch(self, texts: list[str]) -> list[str]:
        return [self.segment(t) for t in texts]


def test_get_segmenter_underthesea() -> None:
    seg = get_segmenter("underthesea")
    assert isinstance(seg, UndertheseaSegmenter)


def test_get_segmenter_cached() -> None:
    assert get_segmenter("underthesea") is get_segmenter("underthesea")


def test_get_segmenter_vncorenlp_requires_dir() -> None:
    with pytest.raises(ValueError):
        get_segmenter("vncorenlp")


def test_get_segmenter_unknown_backend() -> None:
    with pytest.raises(ValueError):
        get_segmenter("unknown")  # type: ignore[arg-type]


def test_fake_segmenter_satisfies_protocol() -> None:
    fake = _FakeSegmenter()
    assert isinstance(fake, SegmenterProtocol)
    assert fake.segment("hello world") == "hello_world"
    assert fake.segment_batch(["a b", "c d"]) == ["a_b", "c_d"]


def test_underthesea_empty_input() -> None:
    # Should not trigger model load (early return before lazy import).
    seg = UndertheseaSegmenter()
    assert seg.segment("") == ""
