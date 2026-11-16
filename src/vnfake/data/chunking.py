"""Sentence-aware sliding-window chunking with deterministic IDs.

Produces chunks matching the schema in ARCHITECTURE §4.2. Chunk IDs are
UUID5 of ``(doc_id, chunk_idx)`` so re-running on the same corpus yields the
same IDs (idempotent upsert).
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Literal

from vnfake.data.segmentation import SegmenterProtocol
from vnfake.schemas.kb import KBRecord
from vnfake.utils.logging import get_logger

logger = get_logger(__name__)

_CHUNK_NS = uuid.NAMESPACE_DNS
# Split on terminal punctuation followed by whitespace. Newlines are handled first.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+", re.UNICODE)
_PARAGRAPH_SPLIT_RE = re.compile(r"\n+")


@dataclass(frozen=True, slots=True)
class Chunk:
    """One KB chunk (see ARCHITECTURE §4.2)."""

    chunk_id: str
    doc_id: str
    chunk_idx: int
    title: str
    text: str
    text_segmented: str
    url: str
    crawl_date: str
    source: str
    published_date: str | None = None
    ingest_source: Literal["static", "rss"] = "static"

    def to_dict(self) -> dict[str, object]:
        """Return a JSONL-serialisable dict.

        Omits ``published_date`` and ``ingest_source`` when at their default values
        (``None`` and ``"static"``) to preserve exact byte identity of the static KB.
        """
        d: dict[str, object] = {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "chunk_idx": self.chunk_idx,
            "title": self.title,
            "text": self.text,
            "text_segmented": self.text_segmented,
            "url": self.url,
            "crawl_date": self.crawl_date,
            "source": self.source,
        }
        if self.published_date is not None:
            d["published_date"] = self.published_date
        if self.ingest_source != "static":
            d["ingest_source"] = self.ingest_source
        return d


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on ``. ! ? …`` + whitespace and on newlines.

    Preserves terminal punctuation. Empty fragments are dropped.

    Known limitation: abbreviations like ``"TP. HCM"`` are split incorrectly.
    Upgrade to ``underthesea.sent_tokenize`` in Phase 2+ if this matters.

    Args:
        text: NFC-normalized input text.

    Returns:
        Non-empty sentences in original order.
    """
    if not text or not text.strip():
        return []
    sentences: list[str] = []
    for paragraph in _PARAGRAPH_SPLIT_RE.split(text):
        for fragment in _SENT_SPLIT_RE.split(paragraph):
            frag = fragment.strip()
            if frag:
                sentences.append(frag)
    return sentences


def sliding_windows(items: list[str], window: int, overlap: int) -> list[list[str]]:
    """Group ``items`` into windows of length ``window`` with ``overlap`` shared.

    Args:
        items: Ordered items to group.
        window: Number of items per window.
        overlap: Items shared between adjacent windows.

    Raises:
        ValueError: If ``window < 1`` or ``not 0 <= overlap < window``.

    Returns:
        List of windows; each window is a non-empty slice of ``items``.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if not 0 <= overlap < window:
        raise ValueError(f"overlap must satisfy 0 <= overlap < window; got {overlap} < {window}")
    if not items:
        return []
    step = window - overlap
    windows: list[list[str]] = []
    i = 0
    while i < len(items):
        windows.append(items[i : i + window])
        if i + window >= len(items):
            break
        i += step
    return windows


def make_chunk_id(doc_id: str, chunk_idx: int) -> str:
    """Deterministic chunk UUID5 from ``(doc_id, chunk_idx)``."""
    return str(uuid.uuid5(_CHUNK_NS, f"vnfake:kb:{doc_id}:{chunk_idx}"))


def chunk_record(
    record: KBRecord,
    *,
    source: str,
    segmenter: SegmenterProtocol | None,
    window: int,
    overlap: int,
) -> list[Chunk]:
    """Split a single KB record into chunks.

    Args:
        record: Validated, cleaned KB record.
        source: ``"dantri"`` or ``"vnexpress"``.
        segmenter: Word segmenter for ``text_segmented`` (``None`` → empty string).
        window: Sentences per chunk.
        overlap: Overlapping sentences between adjacent chunks.

    Returns:
        Ordered chunks (empty if the record has no sentences).
    """
    sentences = split_sentences(record.text)
    if not sentences:
        return []
    windows = sliding_windows(sentences, window, overlap)
    chunks: list[Chunk] = []
    for idx, window_sents in enumerate(windows):
        raw = " ".join(window_sents).strip()
        if not raw:
            continue
        seg = segmenter.segment(raw) if segmenter is not None else ""
        chunks.append(
            Chunk(
                chunk_id=make_chunk_id(record.id, idx),
                doc_id=record.id,
                chunk_idx=idx,
                title=record.title,
                text=raw,
                text_segmented=seg,
                url=record.url,
                crawl_date=record.crawl_date,
                source=source,
                published_date=record.published_date,
                ingest_source=record.ingest_source,
            )
        )
    return chunks
