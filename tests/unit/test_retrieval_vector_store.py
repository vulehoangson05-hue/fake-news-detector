"""Unit tests for vnfake.retrieval.vector_store using QdrantClient(':memory:')."""

from __future__ import annotations

import pytest
from qdrant_client import QdrantClient

from vnfake.data.chunking import Chunk, make_chunk_id
from vnfake.retrieval.vector_store import QdrantStore, ScoredChunk

_DIM = 4


def _chunk(i: int) -> Chunk:
    return Chunk(
        chunk_id=make_chunk_id(f"doc-{i}", 0),
        doc_id=f"doc-{i}",
        chunk_idx=0,
        title=f"Title {i}",
        text=f"Text body {i}",
        text_segmented=f"Text_body_{i}",
        url=f"https://example.com/{i}",
        crawl_date="2024-01-01",
        source="dantri" if i % 2 == 0 else "vnexpress",
    )


def _store() -> QdrantStore:
    return QdrantStore(QdrantClient(":memory:"))


def test_recreate_collection_new() -> None:
    store = _store()
    store.recreate_collection("c1", dim=_DIM)
    assert "c1" in {c.name for c in store._client.get_collections().collections}


def test_recreate_collection_existing_no_force_raises() -> None:
    store = _store()
    store.recreate_collection("c1", dim=_DIM)
    with pytest.raises(ValueError, match="force=True"):
        store.recreate_collection("c1", dim=_DIM)


def test_recreate_collection_existing_force_ok() -> None:
    store = _store()
    store.recreate_collection("c1", dim=_DIM)
    store.recreate_collection("c1", dim=_DIM, force=True)
    assert "c1" in {c.name for c in store._client.get_collections().collections}


def test_upsert_and_search_roundtrip() -> None:
    store = _store()
    store.recreate_collection("c1", dim=_DIM)
    chunks = [_chunk(i) for i in range(3)]
    vectors = [[1.0, 0.0, 0.0, 0.0] for _ in chunks]
    store.upsert_chunks("c1", chunks, vectors)

    hits = store.search("c1", [1.0, 0.0, 0.0, 0.0], top_k=5)
    assert len(hits) == 3
    assert all(isinstance(h, ScoredChunk) for h in hits)
    assert all(h.url.startswith("https://") for h in hits)
    assert {h.source for h in hits} == {"dantri", "vnexpress"}
    # Cosine similarity of identical vectors is ~1.
    assert all(h.score > 0.99 for h in hits)


def test_upsert_length_mismatch_raises() -> None:
    store = _store()
    store.recreate_collection("c1", dim=_DIM)
    with pytest.raises(ValueError, match="len"):
        store.upsert_chunks("c1", [_chunk(0)], [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])


def test_upsert_empty_noop() -> None:
    store = _store()
    store.recreate_collection("c1", dim=_DIM)
    store.upsert_chunks("c1", [], [])  # must not raise


def test_search_empty_collection_returns_empty() -> None:
    store = _store()
    store.recreate_collection("c1", dim=_DIM)
    assert store.search("c1", [1.0, 0.0, 0.0, 0.0], top_k=5) == []


def test_search_rejects_bad_top_k() -> None:
    store = _store()
    store.recreate_collection("c1", dim=_DIM)
    with pytest.raises(ValueError, match="top_k"):
        store.search("c1", [1.0, 0.0, 0.0, 0.0], top_k=0)


def test_search_preserves_payload_fields() -> None:
    store = _store()
    store.recreate_collection("c1", dim=_DIM)
    chunk = _chunk(7)
    store.upsert_chunks("c1", [chunk], [[1.0, 0.0, 0.0, 0.0]])
    hits = store.search("c1", [1.0, 0.0, 0.0, 0.0], top_k=1)
    assert len(hits) == 1
    h = hits[0]
    assert h.chunk_id == chunk.chunk_id
    assert h.doc_id == chunk.doc_id
    assert h.chunk_idx == chunk.chunk_idx
    assert h.title == chunk.title
    assert h.text == chunk.text
    assert h.text_segmented == chunk.text_segmented
    assert h.url == chunk.url
    assert h.crawl_date == chunk.crawl_date
    assert h.source == chunk.source
