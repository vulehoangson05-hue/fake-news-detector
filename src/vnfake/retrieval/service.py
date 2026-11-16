"""Phase 2.5 — Unified Retrieval Service (Rewrite -> Hybrid Search -> Rerank).

Implements ARCHITECTURE.md §5.3.1 & Brief #2 §2.2.5:
- ``RetrievalService.search(claim)`` orchestrates optional LLM query rewrite,
  Qdrant hybrid search (either ``rewritten_only`` or ``original+rewritten`` fused
  with RRF), and cross-encoder reranking.
- **Hard invariant**: Reranker (and downstream classifier) ALWAYS receives the
  normalized **original** claim, never the rewritten keyword string.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

from vnfake.config import AppConfig
from vnfake.data.cleaning import normalize_text
from vnfake.retrieval.embedder import EmbedderProtocol
from vnfake.retrieval.hybrid import RankedItem, rrf_fuse
from vnfake.retrieval.query_rewriter import (
    NoOpQueryRewriter,
    QueryRewriter,
    RewriteResult,
    RewriteStatus,
)
from vnfake.retrieval.reranker import RerankerProtocol
from vnfake.retrieval.vector_store import QdrantStore, ScoredChunk
from vnfake.utils.logging import get_logger

logger = get_logger(__name__)

_MS_PER_SECOND = 1000.0


@dataclass(frozen=True, slots=True)
class RetrievalSearchResult:
    """Result of ``RetrievalService.search(claim)``."""

    claim: str
    evidences: list[ScoredChunk]
    candidates: list[ScoredChunk]
    rewrite_status: RewriteStatus
    rewrite_result: RewriteResult
    retrieve_ms: float
    rerank_ms: float
    total_ms: float


def fuse_candidate_lists(
    lists: Sequence[Sequence[ScoredChunk]],
    *,
    rrf_k: int = 60,
    top_k: int = 20,
) -> list[ScoredChunk]:
    """Fuse multiple ``ScoredChunk`` candidate lists via RRF and preserve chunk metadata."""
    chunk_by_id: dict[str, ScoredChunk] = {}
    ranked_lists: list[list[RankedItem]] = []
    for cand_list in lists:
        ranked: list[RankedItem] = []
        for c in cand_list:
            if c.chunk_id not in chunk_by_id:
                chunk_by_id[c.chunk_id] = c
            ranked.append(RankedItem(doc_id=c.chunk_id, score=c.score))
        ranked_lists.append(ranked)

    fused_items = rrf_fuse(ranked_lists, k=rrf_k, top_k=top_k)
    return [
        replace(chunk_by_id[item.doc_id], score=item.score)
        for item in fused_items
        if item.doc_id in chunk_by_id
    ]


class RetrievalService:
    """End-to-end retrieval + reranking service with optional LLM query rewrite."""

    def __init__(
        self,
        cfg: AppConfig,
        *,
        store: QdrantStore,
        embedder: EmbedderProtocol,
        reranker: RerankerProtocol,
        collection: str,
        rewriter: QueryRewriter | None = None,
    ) -> None:
        self._cfg = cfg
        self._store = store
        self._embedder = embedder
        self._reranker = reranker
        self._collection = collection
        self._rewriter: QueryRewriter = rewriter or NoOpQueryRewriter()

    def _hybrid_search_single(self, query_text: str, top_k: int) -> list[ScoredChunk]:
        dense_vec = self._embedder.embed_dense(
            [query_text], batch_size=self._cfg.retrieval.embedding.batch_size
        )[0]
        sparse_vec = self._embedder.embed_sparse(
            [query_text], batch_size=self._cfg.retrieval.sparse.batch_size
        )[0]
        return self._store.search_hybrid(
            self._collection,
            dense_query=dense_vec,
            sparse_query=sparse_vec,
            top_k=top_k,
            dense_prefetch_k=self._cfg.retrieval.hybrid.dense_prefetch_k,
            sparse_prefetch_k=self._cfg.retrieval.hybrid.sparse_prefetch_k,
        )

    def search(
        self,
        claim: str,
        *,
        top_k: int | None = None,
        retrieve_k: int | None = None,
        fusion: Literal["rewritten_only", "original+rewritten"] | None = None,
    ) -> RetrievalSearchResult:
        """Execute ``claim -> (rewrite) -> hybrid search -> rerank(original_claim)``."""
        t_start = time.perf_counter()
        claim_nfc = normalize_text(claim)
        effective_retrieve_k = (
            retrieve_k if retrieve_k is not None else self._cfg.retrieval.retrieve_k
        )
        effective_top_k = top_k if top_k is not None else self._cfg.retrieval.rerank_k
        effective_fusion = fusion if fusion is not None else self._cfg.query_rewrite.fusion

        if self._cfg.query_rewrite.enabled:
            rewrite_res = self._rewriter.rewrite(claim_nfc)
        else:
            rewrite_res = RewriteResult(
                text=claim_nfc,
                status="disabled",
                latency_ms=0.0,
                raw_output=None,
            )

        t_ret = time.perf_counter()
        rewrite_succeeded = rewrite_res.status in ("ok", "cache_hit")
        if (
            rewrite_succeeded
            and effective_fusion == "original+rewritten"
            and rewrite_res.text != claim_nfc
        ):
            orig_hits = self._hybrid_search_single(claim_nfc, effective_retrieve_k)
            rew_hits = self._hybrid_search_single(rewrite_res.text, effective_retrieve_k)
            candidates = fuse_candidate_lists(
                [orig_hits, rew_hits],
                rrf_k=self._cfg.retrieval.hybrid.rrf_k,
                top_k=effective_retrieve_k,
            )
        else:
            query_for_retrieve = rewrite_res.text if rewrite_succeeded else claim_nfc
            candidates = self._hybrid_search_single(query_for_retrieve, effective_retrieve_k)
        retrieve_ms = (time.perf_counter() - t_ret) * _MS_PER_SECOND

        # Hard invariant: Reranker ALWAYS receives the original normalized claim.
        t_rerank = time.perf_counter()
        evidences = self._reranker.rerank(claim_nfc, candidates, top_k=effective_top_k)
        rerank_ms = (time.perf_counter() - t_rerank) * _MS_PER_SECOND
        total_ms = (time.perf_counter() - t_start) * _MS_PER_SECOND

        return RetrievalSearchResult(
            claim=claim_nfc,
            evidences=evidences,
            candidates=candidates,
            rewrite_status=rewrite_res.status,
            rewrite_result=rewrite_res,
            retrieve_ms=retrieve_ms,
            rerank_ms=rerank_ms,
            total_ms=total_ms,
        )
