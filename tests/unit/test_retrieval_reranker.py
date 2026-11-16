"""Unit tests for vnfake.retrieval.reranker (no real model load)."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from vnfake.retrieval.reranker import BgeReranker, RerankerProtocol
from vnfake.retrieval.vector_store import ScoredChunk


class _DummyReranker:
    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        return list(reversed(candidates))[:top_k]


def _chunk(i: int) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=f"c{i}",
        doc_id=f"d{i}",
        chunk_idx=0,
        title=f"T{i}",
        text=f"text {i}",
        text_segmented="",
        url=f"https://x/{i}",
        crawl_date="2024-01-01",
        source="dantri",
        score=0.5,
    )


def test_dummy_satisfies_protocol() -> None:
    assert isinstance(_DummyReranker(), RerankerProtocol)


def test_dummy_reranker_reorders_and_truncates() -> None:
    out = _DummyReranker().rerank("q", [_chunk(0), _chunk(1), _chunk(2)], top_k=2)
    assert [c.chunk_id for c in out] == ["c2", "c1"]


def test_bge_reranker_lazy_init() -> None:
    rer = BgeReranker("BAAI/bge-reranker-v2-m3")
    assert rer._model is None


def test_bge_reranker_empty_returns_empty() -> None:
    rer = BgeReranker("BAAI/bge-reranker-v2-m3")
    assert rer.rerank("q", [], top_k=5) == []
    assert rer._model is None


def test_bge_reranker_rejects_bad_top_k() -> None:
    rer = BgeReranker("BAAI/bge-reranker-v2-m3")
    with pytest.raises(ValueError, match="top_k"):
        rer.rerank("q", [_chunk(0)], top_k=0)
