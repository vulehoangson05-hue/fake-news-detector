"""Unit tests for vnfake.retrieval.embedder (no real model load)."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from vnfake.retrieval.embedder import (
    BgeM3Embedder,
    EmbedderProtocol,
    SparseVectorData,
)


class _DummyEmbedder:
    def __init__(self, dim: int = 4) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed_dense(self, texts: Sequence[str], batch_size: int) -> list[list[float]]:
        return [[float(i + 1)] * self._dim for i in range(len(texts))]

    def embed_sparse(self, texts: Sequence[str], batch_size: int) -> list[SparseVectorData]:
        return [SparseVectorData(indices=[i + 1], values=[0.5]) for i in range(len(texts))]


def test_dummy_satisfies_protocol() -> None:
    assert isinstance(_DummyEmbedder(), EmbedderProtocol)


def test_bge_m3_lazy_init() -> None:
    emb = BgeM3Embedder("BAAI/bge-m3", dim=1024)
    assert emb._model is None
    assert emb.dim == 1024


def test_bge_m3_empty_dense() -> None:
    emb = BgeM3Embedder("BAAI/bge-m3", dim=1024)
    assert emb.embed_dense([], batch_size=8) == []
    assert emb._model is None


def test_bge_m3_empty_sparse() -> None:
    emb = BgeM3Embedder("BAAI/bge-m3", dim=1024)
    assert emb.embed_sparse([], batch_size=8) == []
    assert emb._model is None


def test_bge_m3_rejects_bad_batch_size() -> None:
    emb = BgeM3Embedder("BAAI/bge-m3", dim=1024)
    with pytest.raises(ValueError, match="batch_size"):
        emb.embed_sparse(["a"], batch_size=0)
