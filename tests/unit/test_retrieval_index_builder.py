"""Unit tests for vnfake.retrieval.index_builder (in-memory + dummy embedder)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from vnfake.config import (
    AppConfig,
    EmbeddingConfig,
    PathsConfig,
    RetrievalConfig,
    SparseEmbeddingConfig,
)
from vnfake.data.chunking import make_chunk_id
from vnfake.retrieval.embedder import SparseVectorData
from vnfake.retrieval.index_builder import (
    build_index,
    compute_index_version,
    iter_chunk_batches,
    iter_chunk_jsonl,
)
from vnfake.retrieval.vector_store import QdrantStore

_DIM = 4


class _DummyEmbedder:
    def __init__(self, dim: int = _DIM) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed_dense(self, texts: Sequence[str], batch_size: int) -> list[list[float]]:
        return [[float(i + 1)] * self._dim for i in range(len(texts))]

    def embed_sparse(self, texts: Sequence[str], batch_size: int) -> list[SparseVectorData]:
        return [SparseVectorData(indices=[i + 1], values=[1.0]) for i in range(len(texts))]


def _chunk_dict(i: int) -> dict[str, object]:
    return {
        "chunk_id": make_chunk_id(f"doc-{i}", 0),
        "doc_id": f"doc-{i}",
        "chunk_idx": 0,
        "title": f"Title {i}",
        "text": f"Text {i}",
        "text_segmented": "",
        "url": f"https://example.com/{i}",
        "crawl_date": "2024-01-01",
        "source": "dantri",
    }


def _write_chunks(path: Path, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(_chunk_dict(i), ensure_ascii=False) for i in range(n)) + "\n",
        encoding="utf-8",
    )


def _cfg(tmp: Path) -> AppConfig:
    return AppConfig(
        seed=42,
        label_map={0: "real", 1: "fake", 2: "unverifiable"},
        paths=PathsConfig(
            kb_dantri=tmp / "kb" / "dantri.jsonl",
            kb_vnexpress=tmp / "kb" / "vnexpress.jsonl",
            vifactcheck_dir=tmp / "vifactcheck",
            interim_dir=tmp / "interim",
            processed_dir=tmp / "processed",
            artifacts_dir=tmp / "artifacts",
        ),
        retrieval=RetrievalConfig(
            embedding=EmbeddingConfig(model_name="dummy", dim=_DIM, batch_size=2, use_fp16=False),
            sparse=SparseEmbeddingConfig(weight_threshold=0.0, batch_size=2),
            qdrant_collection_prefix="kb_chunks",
        ),
    )


def test_compute_index_version_deterministic(tmp_path: Path) -> None:
    chunks = tmp_path / "c.jsonl"
    _write_chunks(chunks, 3)
    v1 = compute_index_version(chunks)
    v2 = compute_index_version(chunks)
    assert v1 == v2
    assert len(v1) == 12


def test_compute_index_version_changes_with_content(tmp_path: Path) -> None:
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    _write_chunks(a, 3)
    _write_chunks(b, 4)
    assert compute_index_version(a) != compute_index_version(b)


def test_iter_chunk_jsonl_skips_malformed(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    path.write_text(
        json.dumps(_chunk_dict(0)) + "\n{bad\n" + json.dumps(_chunk_dict(1)) + "\n",
        encoding="utf-8",
    )
    chunks = list(iter_chunk_jsonl(path))
    assert len(chunks) == 2


def test_iter_chunk_jsonl_missing_field_skipped(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    bad = _chunk_dict(0)
    del bad["url"]
    path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    assert list(iter_chunk_jsonl(path)) == []


def test_iter_chunk_batches_respects_size(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    _write_chunks(path, 5)
    assert [len(b) for b in iter_chunk_batches(path, batch_size=2)] == [2, 2, 1]


def test_build_index_end_to_end(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    chunks = cfg.paths.processed_dir / "kb_chunks.jsonl"
    _write_chunks(chunks, 5)
    store = QdrantStore(QdrantClient(":memory:"))

    stats = build_index(
        cfg,
        chunks_path=chunks,
        embedder=_DummyEmbedder(),
        store=store,
        show_progress=False,
    )
    assert stats["chunks"] == 5
    assert stats["skipped"] is False
    assert (cfg.paths.artifacts_dir / "index_manifest.json").exists()

    hits = store.search_dense(str(stats["collection"]), [1.0] * _DIM, top_k=10)
    assert len(hits) == 5


def test_build_index_skips_when_up_to_date(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    chunks = cfg.paths.processed_dir / "kb_chunks.jsonl"
    _write_chunks(chunks, 2)
    store = QdrantStore(QdrantClient(":memory:"))
    build_index(
        cfg, chunks_path=chunks, embedder=_DummyEmbedder(), store=store, show_progress=False
    )
    stats = build_index(
        cfg, chunks_path=chunks, embedder=_DummyEmbedder(), store=store, show_progress=False
    )
    assert stats["skipped"] is True
    assert stats["chunks"] == 2


def test_build_index_force_rebuilds(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    chunks = cfg.paths.processed_dir / "kb_chunks.jsonl"
    _write_chunks(chunks, 2)
    store = QdrantStore(QdrantClient(":memory:"))
    build_index(
        cfg, chunks_path=chunks, embedder=_DummyEmbedder(), store=store, show_progress=False
    )
    stats = build_index(
        cfg,
        chunks_path=chunks,
        embedder=_DummyEmbedder(),
        store=store,
        force=True,
        show_progress=False,
    )
    assert stats["skipped"] is False


def test_manifest_contains_vector_names(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    chunks = cfg.paths.processed_dir / "kb_chunks.jsonl"
    _write_chunks(chunks, 2)
    store = QdrantStore(QdrantClient(":memory:"))
    build_index(
        cfg, chunks_path=chunks, embedder=_DummyEmbedder(), store=store, show_progress=False
    )
    manifest = json.loads(
        (cfg.paths.artifacts_dir / "index_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["dense_vector_name"] == "dense"
    assert manifest["sparse_vector_name"] == "sparse"


def test_build_index_missing_file_raises(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    store = QdrantStore(QdrantClient(":memory:"))
    with pytest.raises(FileNotFoundError):
        build_index(
            cfg,
            chunks_path=tmp_path / "nope.jsonl",
            embedder=_DummyEmbedder(),
            store=store,
            show_progress=False,
        )
