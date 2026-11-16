"""Unit tests for vnfake.evaluation.retrieval_eval (in-memory Qdrant + dummy models)."""

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
from vnfake.data.chunking import Chunk, make_chunk_id
from vnfake.evaluation.retrieval_eval import (
    EvalQuery,
    compute_ranking_metrics,
    evaluate_retrieval,
    extract_word_ngrams,
    format_markdown_report,
    hit_rate_at_k,
    iter_eval_queries,
    make_proxy_matcher,
    match_explicit_metadata,
    mrr_at_k,
    ndcg_at_k,
    ngram_containment,
    resolve_collection_name,
)
from vnfake.retrieval.embedder import SparseVectorData
from vnfake.retrieval.vector_store import QdrantStore, ScoredChunk

_DIM = 4


class _DummyEmbedder:
    @property
    def dim(self) -> int:
        return _DIM

    def embed_dense(self, texts: Sequence[str], batch_size: int) -> list[list[float]]:
        return [[1.0] * _DIM for _ in texts]

    def embed_sparse(self, texts: Sequence[str], batch_size: int) -> list[SparseVectorData]:
        return [SparseVectorData(indices=[1, 2], values=[1.0, 0.5]) for _ in texts]


class _DummyReranker:
    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        return list(candidates[:top_k])


def _scored_chunk(
    chunk_id: str = "cid-1",
    doc_id: str = "doc-1",
    text: str = "Ngân hàng Nhà nước bán vàng miếng bình ổn thị trường",
    source: str = "vnexpress",
    score: float = 0.9,
) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        chunk_idx=0,
        title="Tin kinh tế",
        text=text,
        text_segmented="",
        url="https://vnexpress.net/tin-1",
        crawl_date="2024-01-01",
        source=source,
        score=score,
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
            retrieve_k=5,
            rerank_k=2,
            embedding=EmbeddingConfig(model_name="dummy", dim=_DIM, batch_size=2, use_fp16=False),
            sparse=SparseEmbeddingConfig(weight_threshold=0.0, batch_size=2),
            qdrant_collection_prefix="kb_chunks",
        ),
    )


def test_extract_word_ngrams_and_containment() -> None:
    ngrams = extract_word_ngrams("Ngân hàng Nhà nước", 2)
    assert "ngân hàng" in ngrams
    assert "nhà nước" in ngrams
    score = ngram_containment(
        "Ngân hàng Nhà nước bán vàng",
        "Hôm nay Ngân hàng Nhà nước bán vàng miếng SJC",
        2,
    )
    assert score == pytest.approx(1.0)
    assert ngram_containment("", "bất kỳ", 2) == 0.0


def test_ranking_metrics_values() -> None:
    rels = [False, True, False, True]
    assert hit_rate_at_k(rels, 1) == 0.0
    assert hit_rate_at_k(rels, 2) == 1.0
    assert mrr_at_k(rels, 4) == pytest.approx(0.5)
    assert 0.0 < ndcg_at_k(rels, 4) < 1.0

    agg = compute_ranking_metrics([rels, [True, False]], [1, 2])
    assert agg["hit_rate@1"] == pytest.approx(0.5)
    assert agg["hit_rate@2"] == pytest.approx(1.0)
    assert agg["mrr@2"] == pytest.approx(0.75)


def test_explicit_and_proxy_matcher() -> None:
    chunk = _scored_chunk(chunk_id="target-chunk", source="dantri")
    q_explicit = EvalQuery(
        statement="Không liên quan",
        evidence="",
        label=0,
        gold_chunk_ids=frozenset({"target-chunk"}),
    )
    assert match_explicit_metadata(q_explicit, chunk) is True

    matcher = make_proxy_matcher(ngram_n=2, threshold=0.3)
    q_lexical = EvalQuery(
        statement="Ngân hàng Nhà nước bán vàng miếng",
        evidence="",
        label=0,
    )
    assert matcher(q_lexical, chunk) is True


def test_iter_eval_queries_and_evaluate_end_to_end(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.paths.processed_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.artifacts_dir.mkdir(parents=True, exist_ok=True)

    collection = "kb_chunks_test"
    (cfg.paths.artifacts_dir / "index_manifest.json").write_text(
        json.dumps({"collection": collection}),
        encoding="utf-8",
    )
    assert resolve_collection_name(cfg) == collection

    dev_path = cfg.paths.processed_dir / "dev_clean.jsonl"
    samples = [
        {
            "Statement": "Ngân hàng Nhà nước bán vàng miếng bình ổn thị trường",
            "Context": "",
            "labels": 0,
            "Evidence": "Ngân hàng Nhà nước bán vàng miếng",
        },
        {
            "Statement": "Thông tin hoàn toàn khác biệt không có trong kho tri thức",
            "Context": "",
            "labels": 1,
            "Evidence": "Không khớp",
        },
    ]
    dev_path.write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in samples) + "\n",
        encoding="utf-8",
    )
    loaded = list(iter_eval_queries(dev_path))
    assert len(loaded) == 2

    store = QdrantStore(QdrantClient(":memory:"))
    store.recreate_collection(collection, dim=_DIM, force=True)
    chunk = Chunk(
        chunk_id=make_chunk_id("doc-1", 0),
        doc_id="doc-1",
        chunk_idx=0,
        title="Kinh tế",
        text="Ngân hàng Nhà nước bán vàng miếng bình ổn thị trường qua 4 ngân hàng",
        text_segmented="",
        url="https://vnexpress.net/1",
        crawl_date="2024-01-01",
        source="vnexpress",
    )
    store.upsert_chunks(
        collection,
        [chunk],
        [[1.0] * _DIM],
        [SparseVectorData(indices=[1, 2], values=[1.0, 0.5])],
    )

    report = evaluate_retrieval(
        cfg,
        store=store,
        embedder=_DummyEmbedder(),
        reranker=_DummyReranker(),
    )
    assert report.num_queries == 2
    assert report.proxy_rerank_metrics["hit_rate@2"] == pytest.approx(0.5)
    md = format_markdown_report(report)
    assert "Báo cáo Đánh giá Retrieval & Reranking" in md
    assert collection in md
