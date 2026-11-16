"""Unit tests for Phase 2.5 Task 4.1: KBRecord & Chunk schema backward compatibility."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from vnfake.data.chunking import chunk_record
from vnfake.retrieval.index_builder import compute_index_version, iter_chunk_jsonl
from vnfake.retrieval.vector_store import _hit_to_scored
from vnfake.schemas.kb import KBRecord


def test_static_kb_record_reads_date_and_dumps_crawl_date() -> None:
    """Static JSONL records only have ``date``; dumping must emit ``crawl_date``."""
    raw = {
        "id": "dt_001",
        "title": "Tiêu đề bài báo tĩnh",
        "text": "Nội dung bài báo tĩnh từ Dân Trí.",
        "label": "real",
        "url": "https://dantri.com.vn/bai-1.htm",
        "date": "2024-05-01",
    }
    rec = KBRecord.model_validate(raw)
    assert rec.crawl_date == "2024-05-01"
    assert rec.published_date is None
    assert rec.ingest_source == "static"

    dumped = rec.model_dump()
    assert "crawl_date" in dumped
    assert "date" not in dumped
    assert dumped["crawl_date"] == "2024-05-01"
    assert dumped["published_date"] is None
    assert dumped["ingest_source"] == "static"


def test_rss_kb_record_and_chunk_fields() -> None:
    """RSS records carry ``published_date`` and ``ingest_source='rss'`` into Chunk payload."""
    rec = KBRecord.model_validate(
        {
            "id": "rss_001",
            "title": "Tin thời sự mới",
            "text": "Câu thứ nhất của bài báo. Câu thứ hai của bài báo.",
            "label": "real",
            "url": "https://vnexpress.net/tin-moi.html",
            "crawl_date": "2026-11-08T10:05:00Z",
            "published_date": "2026-11-08T09:30:00Z",
            "ingest_source": "rss",
        }
    )
    chunks = chunk_record(rec, source="vnexpress", segmenter=None, window=3, overlap=1)
    assert len(chunks) == 1
    payload = chunks[0].to_dict()
    assert payload["crawl_date"] == "2026-11-08T10:05:00Z"
    assert payload["published_date"] == "2026-11-08T09:30:00Z"
    assert payload["ingest_source"] == "rss"


def test_legacy_payload_without_new_fields_reads_cleanly(tmp_path: Path) -> None:
    """Existing Qdrant points and JSONL chunks without ``published_date``/``ingest_source`` must work."""
    legacy_payload = {
        "chunk_id": "cid_1",
        "doc_id": "doc_1",
        "chunk_idx": 0,
        "title": "Tiêu đề cũ",
        "text": "Văn bản cũ",
        "text_segmented": "Văn_bản cũ",
        "url": "https://vnexpress.net/cu.html",
        "crawl_date": "2024-01-01",
        "source": "vnexpress",
    }
    fake_hit = SimpleNamespace(id="cid_1", payload=legacy_payload, score=0.75)
    scored = _hit_to_scored(fake_hit)
    assert scored.published_date is None
    assert scored.ingest_source == "static"
    assert scored.crawl_date == "2024-01-01"

    jsonl_path = tmp_path / "legacy_chunks.jsonl"
    jsonl_path.write_text(json.dumps(legacy_payload, ensure_ascii=False) + "\n", encoding="utf-8")
    loaded = list(iter_chunk_jsonl(jsonl_path))
    assert len(loaded) == 1
    assert loaded[0].published_date is None
    assert loaded[0].ingest_source == "static"


def test_existing_kb_chunks_index_version_unchanged() -> None:
    """Confirm static snapshot index_version remains 8b525e25ad54 (Brief #2 §3.1)."""
    chunks_path = Path("data/processed/kb_chunks.jsonl")
    if chunks_path.exists():
        assert compute_index_version(chunks_path) == "8b525e25ad54"
