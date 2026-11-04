"""Unit tests for vnfake.retrieval.embedder (no real model load)."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from vnfake.retrieval.embedder import BgeM3Embedder, EmbedderProtocol


class _DummyEmbedder:
    def __init__(self, dim: int = 4) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed_dense(self, texts: Sequence[str], batch_size: int) -> list[list[float]]:
        return [[float(i + 1)] * self._dim for i in range(len(texts))]


def test_dummy_embedder_satisfies_protocol() -> None:
    assert isinstance(_DummyEmbedder(), EmbedderProtocol)


def test_bge_m3_embedder_lazy_init() -> None:
    emb = BgeM3Embedder("BAAI/bge-m3", dim=1024)
    assert emb._model is None  # private, but we're testing the lazy contract
    assert emb.dim == 1024


def test_bge_m3_embedder_empty_returns_empty() -> None:
    emb = BgeM3Embedder("BAAI/bge-m3", dim=1024)
    assert emb.embed_dense([], batch_size=8) == []
    assert emb._model is None  # empty input must not trigger load


def test_bge_m3_embedder_rejects_bad_batch_size() -> None:
    emb = BgeM3Embedder("BAAI/bge-m3", dim=1024)
    with pytest.raises(ValueError, match="batch_size"):
        emb.embed_dense(["a"], batch_size=0)


def test_dummy_embedder_returns_correct_shape() -> None:
    emb = _DummyEmbedder(dim=8)
    out = emb.embed_dense(["a", "b", "c"], batch_size=2)
    assert len(out) == 3
    assert all(len(row) == 8 for row in out)
