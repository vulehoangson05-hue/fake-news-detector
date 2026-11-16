"""Unit tests for vnfake.retrieval.vector_store (in-memory Qdrant)."""

from __future__ import annotations

import pytest
from qdrant_client import QdrantClient

from vnfake.data.chunking import Chunk, make_chunk_id
from vnfake.retrieval.embedder import SparseVectorData
from vnfake.retrieval.vector_store import QdrantStore, ScoredChunk

_DIM = 4


def _chunk(i: int) -> Chunk:
    return Chunk(
        chunk_id=make_chunk_id(f"doc-{i}", 0),
        doc_id=f"doc-{i}",
        chunk_idx=0,
        title=f"Title {i}",
        text=f"Text body {i}",
        text_segmented="",
        url=f"https://example.com/{i}",
        crawl_date="2024-01-01",
        source="dantri" if i % 2 == 0 else "vnexpress",
    )


def _sparse(indices: list[int], values: list[float]) -> SparseVectorData:
    return SparseVectorData(indices=indices, values=values)


def _store() -> QdrantStore:
    return QdrantStore(QdrantClient(":memory:"))


def test_recreate_collection_creates_named_vectors() -> None:
    store = _store()
    store.recreate_collection("c1", dim=_DIM)
    info = store._client.get_collection("c1")
    assert "dense" in info.config.params.vectors  # type: ignore[operator]
    assert "sparse" in info.config.params.sparse_vectors  # type: ignore[operator]


def test_recreate_collection_existing_no_force_raises() -> None:
    store = _store()
    store.recreate_collection("c1", dim=_DIM)
    with pytest.raises(ValueError, match="force=True"):
        store.recreate_collection("c1", dim=_DIM)


def test_recreate_collection_existing_force_ok() -> None:
    store = _store()
    store.recreate_collection("c1", dim=_DIM)
    store.recreate_collection("c1", dim=_DIM, force=True)


def test_upsert_and_dense_search_roundtrip() -> None:
    store = _store()
    store.recreate_collection("c1", dim=_DIM)
    chunks = [_chunk(i) for i in range(3)]
    dense = [[1.0, 0.0, 0.0, 0.0] for _ in chunks]
    sparse = [_sparse([1, 2], [1.0, 0.5]) for _ in chunks]
    store.upsert_chunks("c1", chunks, dense, sparse)

    hits = store.search_dense("c1", [1.0, 0.0, 0.0, 0.0], top_k=5)
    assert len(hits) == 3
    assert all(isinstance(h, ScoredChunk) for h in hits)
    assert all(h.score > 0.99 for h in hits)


def test_sparse_search_returns_matching_chunks() -> None:
    store = _store()
    store.recreate_collection("c1", dim=_DIM)
    chunks = [_chunk(0), _chunk(1)]
    dense = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    sparse = [_sparse([10, 20], [1.0, 1.0]), _sparse([30], [1.0])]
    store.upsert_chunks("c1", chunks, dense, sparse)

    hits = store.search_sparse("c1", _sparse([10], [1.0]), top_k=2)
    assert len(hits) >= 1
    assert hits[0].chunk_id == chunks[0].chunk_id


def test_hybrid_search_fuses_dense_and_sparse() -> None:
    store = _store()
    store.recreate_collection("c1", dim=_DIM)
    chunks = [_chunk(i) for i in range(4)]
    dense = [
        [0.8, 0.6, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    sparse = [
        _sparse([1], [1.0]),
        _sparse([7], [1.0]),
        _sparse([7], [1.0]),
        _sparse([9], [1.0]),
    ]
    store.upsert_chunks("c1", chunks, dense, sparse)

    hits = store.search_hybrid(
        "c1",
        dense_query=[1.0, 0.0, 0.0, 0.0],
        sparse_query=_sparse([7], [1.0]),
        top_k=3,
        dense_prefetch_k=3,
        sparse_prefetch_k=3,
    )
    assert len(hits) == 3
    assert hits[0].chunk_id == chunks[1].chunk_id


def test_upsert_length_mismatch_raises() -> None:
    store = _store()
    store.recreate_collection("c1", dim=_DIM)
    with pytest.raises(ValueError, match="Length mismatch"):
        store.upsert_chunks("c1", [_chunk(0)], [[1.0, 0.0, 0.0, 0.0]], [])


def test_search_rejects_bad_top_k() -> None:
    store = _store()
    store.recreate_collection("c1", dim=_DIM)
    with pytest.raises(ValueError, match="top_k"):
        store.search_dense("c1", [1.0, 0.0, 0.0, 0.0], top_k=0)


def test_search_preserves_payload() -> None:
    store = _store()
    store.recreate_collection("c1", dim=_DIM)
    chunk = _chunk(7)
    store.upsert_chunks("c1", [chunk], [[1.0, 0.0, 0.0, 0.0]], [_sparse([1], [1.0])])
    h = store.search_dense("c1", [1.0, 0.0, 0.0, 0.0], top_k=1)[0]
    assert h.chunk_id == chunk.chunk_id
    assert h.url == chunk.url
    assert h.source == chunk.source
