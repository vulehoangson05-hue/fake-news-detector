"""Unit tests for vnfake.data.chunking."""

from __future__ import annotations

import dataclasses
import uuid

import pytest

from vnfake.data.chunking import (
    Chunk,
    chunk_record,
    make_chunk_id,
    sliding_windows,
    split_sentences,
)
from vnfake.schemas.kb import KBRecord


class _FakeSegmenter:
    def segment(self, text: str) -> str:
        return text.replace(" ", "_")

    def segment_batch(self, texts: list[str]) -> list[str]:
        return [self.segment(t) for t in texts]


def _kb(text: str, rec_id: str = "doc-1") -> KBRecord:
    return KBRecord(
        id=rec_id,
        title="Tiêu đề",
        text=text,
        label="real",
        url=f"https://example.com/{rec_id}",
        crawl_date="2024-01-01",
    )


def test_split_sentences_basic() -> None:
    out = split_sentences("Tôi yêu Việt Nam. Đất nước đẹp!")
    assert out == ["Tôi yêu Việt Nam.", "Đất nước đẹp!"]


def test_split_sentences_newlines() -> None:
    out = split_sentences("Câu một.\n\nCâu hai.")
    assert out == ["Câu một.", "Câu hai."]


def test_split_sentences_single() -> None:
    assert split_sentences("Chỉ một câu") == ["Chỉ một câu"]


def test_split_sentences_empty() -> None:
    assert split_sentences("") == []
    assert split_sentences("   \n\n") == []


def test_split_sentences_question_and_ellipsis() -> None:
    out = split_sentences("Thật sao? Đúng vậy… Tôi đi.")
    assert out == ["Thật sao?", "Đúng vậy…", "Tôi đi."]


def test_sliding_windows_basic() -> None:
    out = sliding_windows(list("abcde"), window=3, overlap=1)
    assert out == [["a", "b", "c"], ["c", "d", "e"]]


def test_sliding_windows_no_extra_chunk() -> None:
    out = sliding_windows(list("abc"), window=3, overlap=1)
    assert out == [["a", "b", "c"]]


def test_sliding_windows_short_input() -> None:
    assert sliding_windows(["a"], window=3, overlap=1) == [["a"]]


def test_sliding_windows_empty() -> None:
    assert sliding_windows([], window=3, overlap=1) == []


def test_sliding_windows_validation() -> None:
    with pytest.raises(ValueError):
        sliding_windows(["a"], window=0, overlap=0)
    with pytest.raises(ValueError):
        sliding_windows(["a"], window=2, overlap=2)


def test_make_chunk_id_deterministic() -> None:
    a = make_chunk_id("doc-1", 0)
    b = make_chunk_id("doc-1", 0)
    assert a == b
    assert uuid.UUID(a).version == 5


def test_make_chunk_id_varies_by_idx() -> None:
    assert make_chunk_id("doc-1", 0) != make_chunk_id("doc-1", 1)


def test_chunk_record_produces_chunks() -> None:
    record = _kb("Câu một. Câu hai. Câu ba. Câu bốn.")
    chunks = chunk_record(
        record,
        source="dantri",
        segmenter=_FakeSegmenter(),
        window=2,
        overlap=0,
    )
    assert len(chunks) == 2
    assert chunks[0].text == "Câu một. Câu hai."
    assert chunks[1].text == "Câu ba. Câu bốn."
    assert chunks[0].chunk_idx == 0
    assert chunks[1].chunk_idx == 1
    assert chunks[0].source == "dantri"
    assert chunks[0].text_segmented == "Câu_một._Câu_hai."


def test_chunk_record_empty_text() -> None:
    record = _kb("   ")
    assert chunk_record(record, source="x", segmenter=None, window=2, overlap=1) == []


def test_chunk_record_no_segmenter_leaves_empty_segmented() -> None:
    record = _kb("Một câu.")
    chunks = chunk_record(record, source="x", segmenter=None, window=3, overlap=1)
    assert chunks[0].text_segmented == ""


def test_chunk_to_dict_keys() -> None:
    record = _kb("A. B.")
    chunks = chunk_record(record, source="vnexpress", segmenter=None, window=2, overlap=0)
    d = chunks[0].to_dict()
    assert set(d.keys()) == {
        "chunk_id",
        "doc_id",
        "chunk_idx",
        "title",
        "text",
        "text_segmented",
        "url",
        "crawl_date",
        "source",
    }


def test_chunk_dataclass_frozen() -> None:
    c = Chunk(
        chunk_id="id",
        doc_id="d",
        chunk_idx=0,
        title="t",
        text="x",
        text_segmented="",
        url="u",
        crawl_date="2024",
        source="s",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.text = "y"  # type: ignore[misc]
