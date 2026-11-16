"""Phase 4 (Task 4.3) — End-to-end Live Verification Pipeline (`LiveVerifier`).

Implements Brief #5 §4 & §12.5:
1. Rule-based segmented query builder ``build_keyword_query_v2`` (Q1v2) with optional
   Gemini Flash query rewrite (disabled by default via ``cfg.query_rewrite.enabled``).
2. Live web retrieval via ``TavilyWebSearcher`` (whitelist ``include_domains``,
   disk credit guard, disk response cache, parallel polite fetching of up to 3 pages).
3. Unified Phase 3 BM25 sentence selection over combined hits (top-3 sentences,
   ``<= 150`` words), retaining ``{text, url, domain}`` provenance per sentence.
4. Cross-encoder relevance scoring (``bge-reranker-v2-m3``) and Relevance Gate (``tau_rel``).
5. 3-seed temperature-calibrated PhoBERT ensemble (``EnsembleVerifier``) + Confidence Gate
   (``tau_conf``), setting ``show_probs=False`` whenever blocked by a gate.
6. Graceful fallback chain with request deadline (``live_deadline_s``):
   Tavily live -> Disk Cache -> Local Qdrant KB (lazy ``bge-m3`` only on fallback) -> ``unverifiable``.

Important VRAM constraint (Brief #5 §4.7):
``BAAI/bge-m3`` MUST NOT be imported or loaded during normal ``live`` or ``cache`` execution.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from typing import Literal, Protocol, cast

import numpy as np
import torch
from pydantic import BaseModel, Field

from vnfake.config import AppConfig, SearchConfig, load_config, load_search_config
from vnfake.data.cleaning import normalize_text
from vnfake.data.segmentation import SegmenterProtocol, get_segmenter
from vnfake.pipeline.decision import DecisionGate, load_vc_ensemble
from vnfake.retrieval.evidence_selector import (
    SourcedEvidenceResult,
    select_evidence_with_sources,
)
from vnfake.retrieval.query_rewriter import NoOpQueryRewriter, QueryRewriter
from vnfake.retrieval.reranker import BgeReranker
from vnfake.retrieval.web_search import (
    TavilyWebSearcher,
    WebSearchHit,
    WebSearchResult,
    build_keyword_query_v2,
    build_query_q0,
    build_query_q1,
    extract_domain,
    select_evidence_from_web_hits,
)
from vnfake.utils.logging import get_logger

logger = get_logger(__name__)

_MS_PER_SECOND = 1000.0
VerdictLiteral = Literal["real", "fake", "unverifiable"]
ModeLiteral = Literal["live", "cache", "fallback"]


class EvidenceItem(BaseModel):
    """Single selected evidence sentence with source attribution."""

    text: str
    url: str = ""
    domain: str = ""


class LatencyBreakdownMs(BaseModel):
    """Per-stage wall-clock latency breakdown in milliseconds."""

    query_ms: float = 0.0
    search_ms: float = 0.0
    fetch_ms: float = 0.0
    select_ms: float = 0.0
    rerank_ms: float = 0.0
    verify_ms: float = 0.0
    total_ms: float = 0.0


class LiveVerifyResult(BaseModel):
    """Structured end-to-end verification response (Brief #5 §4.5 & §12.5)."""

    claim: str
    query: str
    verdict: VerdictLiteral
    probs: dict[str, float] = Field(
        default_factory=lambda: {"real": 0.0, "fake": 0.0, "unverifiable": 1.0}
    )
    show_probs: bool = False
    relevance: float = 0.0
    confidence: float = 0.0
    evidence: list[EvidenceItem] = Field(default_factory=list)
    latency_ms: LatencyBreakdownMs = Field(default_factory=LatencyBreakdownMs)
    mode: ModeLiteral
    reason: str


class RerankScorerProtocol(Protocol):
    """Minimal scoring protocol so unit tests can inject a mock reranker."""

    def score_pair(self, query: str, text: str) -> float: ...


class VerifierProtocol(Protocol):
    """Minimal verifier protocol so unit tests can inject a mock ensemble."""

    def predict_single(
        self,
        statement: str,
        evidence: str,
        *,
        presegmented: bool = False,
    ) -> np.ndarray: ...


class LiveVerifier:
    """Orchestrates live Vietnamese claim verification with fallback & gating."""

    def __init__(
        self,
        cfg: AppConfig | None = None,
        search_cfg: SearchConfig | None = None,
        *,
        device: str | None = None,
        use_fp16: bool | None = None,
        demo_mode: bool | None = None,
        total_deadline_s: float | None = None,
        short_circuit_low_relevance: bool = True,
        low_vram_sequential: bool = False,
        searcher: TavilyWebSearcher | None = None,
        reranker: RerankScorerProtocol | None = None,
        ensemble: VerifierProtocol | None = None,
        segmenter: SegmenterProtocol | None = None,
        query_rewriter: QueryRewriter | None = None,
        kb_fallback_fn: Callable[[str], Sequence[EvidenceItem]] | None = None,
        kb_collection: str | None = None,
    ) -> None:
        self.cfg = cfg or load_config()
        self.search_cfg = search_cfg or load_search_config()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = self.cfg.decision.use_fp16 if use_fp16 is None else bool(use_fp16)

        if demo_mode is None:
            env_demo = os.getenv("DEMO_MODE", "").strip().lower() in {"1", "true", "yes", "on"}
            self.demo_mode = bool(self.cfg.decision.demo_mode or env_demo)
        else:
            self.demo_mode = bool(demo_mode)

        self.total_deadline_s = (
            float(self.cfg.decision.live_deadline_s)
            if total_deadline_s is None
            else float(total_deadline_s)
        )
        self.short_circuit_low_relevance = short_circuit_low_relevance
        self.low_vram_sequential = low_vram_sequential

        self._segmenter = segmenter or get_segmenter(
            self.cfg.segmentation.backend,
            str(self.cfg.segmentation.vncorenlp_dir),
        )
        self._query_rewriter: QueryRewriter = query_rewriter or NoOpQueryRewriter()
        self._searcher = searcher or TavilyWebSearcher(self.search_cfg)
        self._reranker: RerankScorerProtocol | None = reranker
        self._ensemble: VerifierProtocol | None = ensemble
        self._kb_fallback_fn = kb_fallback_fn
        self._kb_collection = kb_collection
        self._gate = DecisionGate(
            tau_rel=self.cfg.decision.tau_rel,
            tau_conf=self.cfg.decision.tau_conf,
            class_names=self.cfg.label_map,
        )

    def _get_reranker(self) -> RerankScorerProtocol:
        if self._reranker is None:
            rer_cfg = self.cfg.retrieval.reranker
            effective_fp16 = self.use_fp16 and self.device.startswith("cuda")
            self._reranker = BgeReranker(
                rer_cfg.model_name,
                batch_size=rer_cfg.batch_size,
                use_fp16=effective_fp16,
                max_length=rer_cfg.max_length,
                device=self.device,
            )
        return self._reranker

    def _get_ensemble(self) -> VerifierProtocol:
        if self._ensemble is None:
            self._ensemble = load_vc_ensemble(
                self.cfg,
                device=self.device,
                use_fp16=self.use_fp16,
            )
        return self._ensemble

    def warmup(self) -> None:
        """Eagerly load reranker and 3-seed verifier onto ``self.device``."""
        reranker = self._get_reranker()
        # Force lazy model load on concrete BgeReranker
        load_fn = getattr(reranker, "_load", None)
        if callable(load_fn):
            load_fn()
        ens = self._get_ensemble()
        ens_load = getattr(ens, "_ensure_loaded", None)
        if callable(ens_load):
            ens_load()

    def unload(self) -> None:
        """Release loaded reranker & ensemble weights from CPU/GPU memory."""
        for obj in (self._reranker, self._ensemble):
            if obj is not None:
                unload_fn = getattr(obj, "unload", None)
                if callable(unload_fn):
                    unload_fn()
        self._reranker = None
        self._ensemble = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _lookup_cached_web(self, query_str: str, claim_nfc: str = "") -> WebSearchResult | None:
        candidate_queries = [query_str]
        if claim_nfc:
            for alt_q in (build_query_q1(claim_nfc), build_query_q0(claim_nfc)):
                if alt_q and alt_q not in candidate_queries:
                    candidate_queries.append(alt_q)

        alt_searcher: TavilyWebSearcher | None = None
        if isinstance(self._searcher, TavilyWebSearcher) and self.search_cfg.max_results != 10:
            alt_cfg = self.search_cfg.model_copy(update={"max_results": 10})
            alt_searcher = TavilyWebSearcher(alt_cfg)

        for cand_q in candidate_queries:
            cached = self._searcher.search_cached_only(cand_q, variant="Q1v2")
            if cached is not None:
                return cached
            if alt_searcher is not None:
                alt_cached = alt_searcher.search_cached_only(
                    cand_q,
                    variant="Q1v2",
                    max_results=self.search_cfg.max_results,
                )
                if alt_cached is not None:
                    return alt_cached
        return None

    def _query_local_kb(self, claim_nfc: str) -> list[EvidenceItem]:
        """Lazy local Qdrant KB lookup invoked ONLY when live search & cache fail."""
        if self._kb_fallback_fn is not None:
            return list(self._kb_fallback_fn(claim_nfc))

        # Lazy imports ensure BAAI/bge-m3 is NEVER imported/loaded in live or cache paths.
        from vnfake.retrieval.embedder import BgeM3Embedder
        from vnfake.retrieval.vector_store import QdrantStore

        store = QdrantStore.from_env(
            dense_name=self.cfg.retrieval.dense_vector_name,
            sparse_name=self.cfg.retrieval.sparse_vector_name,
        )
        collection = self._kb_collection or os.getenv("QDRANT_COLLECTION", "").strip()
        if not collection:
            prefix = self.cfg.retrieval.qdrant_collection_prefix
            available = [c.name for c in store._client.get_collections().collections]
            matching = sorted(name for name in available if name.startswith(prefix))
            if not matching:
                raise RuntimeError(
                    f"No Qdrant collection matching prefix {prefix!r} found (available={available})."
                )
            collection = matching[-1]
        elif not store._client.collection_exists(collection):
            raise RuntimeError(f"Qdrant collection {collection!r} does not exist.")

        emb_cfg = self.cfg.retrieval.embedding
        embedder = BgeM3Embedder(
            emb_cfg.model_name,
            emb_cfg.dim,
            use_fp16=self.use_fp16,
            device=self.device,
            weight_threshold=self.cfg.retrieval.sparse.weight_threshold,
        )
        try:
            dense_q = embedder.embed_dense([claim_nfc], batch_size=1)[0]
            sparse_q = embedder.embed_sparse([claim_nfc], batch_size=1)[0]
            chunks = store.search_hybrid(
                collection,
                dense_query=dense_q,
                sparse_query=sparse_q,
                top_k=self.cfg.retrieval.rerank_k,
                dense_prefetch_k=self.cfg.retrieval.hybrid.dense_prefetch_k,
                sparse_prefetch_k=self.cfg.retrieval.hybrid.sparse_prefetch_k,
            )
        finally:
            embedder.unload()

        return [
            EvidenceItem(
                text=c.text,
                url=c.url,
                domain=extract_domain(c.url) or c.source,
            )
            for c in chunks
            if c.text.strip()
        ]

    def _build_fallback_unverifiable(
        self,
        *,
        claim_nfc: str,
        query_str: str,
        reason: str,
        mode: ModeLiteral,
        latency: LatencyBreakdownMs,
        t_start: float,
        evidence: list[EvidenceItem] | None = None,
        relevance: float = 0.0,
    ) -> LiveVerifyResult:
        latency.total_ms = round((time.perf_counter() - t_start) * _MS_PER_SECOND, 2)
        return LiveVerifyResult(
            claim=claim_nfc,
            query=query_str,
            verdict="unverifiable",
            probs={"real": 0.0, "fake": 0.0, "unverifiable": 1.0},
            show_probs=False,
            relevance=round(float(relevance), 4),
            confidence=0.0,
            evidence=evidence or [],
            latency_ms=latency,
            mode=mode,
            reason=reason,
        )

    def _resolve_query(self, claim_nfc: str, *, use_llm_rewrite: bool | None) -> str:
        rule_query = build_keyword_query_v2(claim_nfc, segmenter=self._segmenter)
        rewrite_enabled = (
            self.cfg.query_rewrite.enabled if use_llm_rewrite is None else bool(use_llm_rewrite)
        )
        if not rewrite_enabled:
            return rule_query

        rewriter = self._query_rewriter
        if isinstance(rewriter, NoOpQueryRewriter):
            # Keep Gemini optional and lazy-imported
            try:
                from vnfake.retrieval.query_rewriter import GeminiQueryRewriter

                qr_cfg = self.cfg.query_rewrite.model_copy(update={"enabled": True})
                rewriter = GeminiQueryRewriter(qr_cfg, online=True)
            except Exception as exc:
                logger.warning("Optional Gemini rewriter init failed; falling back to Q1v2: %s", exc)
                return rule_query

        try:
            res = rewriter.rewrite(claim_nfc)
            if res.status in ("ok", "cache_hit") and res.text.strip():
                return res.text.strip()
        except Exception as exc:
            logger.warning("Optional Gemini rewriter failed; falling back to Q1v2: %s", exc)
        return rule_query

    def verify(
        self,
        claim: str,
        *,
        use_llm_rewrite: bool | None = None,
    ) -> LiveVerifyResult:
        """Execute full claim verification workflow with deadline & fallback guarantees."""
        t_start = time.perf_counter()
        latency = LatencyBreakdownMs()
        claim_nfc = normalize_text(claim)

        def _remaining_s() -> float:
            return self.total_deadline_s - (time.perf_counter() - t_start)

        if not claim_nfc:
            return self._build_fallback_unverifiable(
                claim_nfc="",
                query_str="",
                reason="nhận định đầu vào rỗng",
                mode="fallback",
                latency=latency,
                t_start=t_start,
            )

        # 1. Query builder (Q1v2 rule-based + optional LLM rewrite)
        t_q0 = time.perf_counter()
        query_str = self._resolve_query(claim_nfc, use_llm_rewrite=use_llm_rewrite)
        latency.query_ms = round((time.perf_counter() - t_q0) * _MS_PER_SECOND, 2)

        # 2. Retrieve candidate articles/texts: Live -> Cache -> Local KB
        mode: ModeLiteral = "live"
        hits: list[WebSearchHit] = []
        kb_items: list[EvidenceItem] = []
        web_failure_reason: str | None = None

        t_search0 = time.perf_counter()
        if self.demo_mode:
            cached = self._lookup_cached_web(query_str, claim_nfc=claim_nfc)
            if cached is not None and cached.hits:
                mode = "cache"
                hits = cached.hits
                latency.fetch_ms = cached.fetch_ms
                latency.search_ms = round(
                    max((time.perf_counter() - t_search0) * _MS_PER_SECOND - cached.fetch_ms, 0.0),
                    2,
                )
        if not hits:
            if _remaining_s() <= 0.0:
                return self._build_fallback_unverifiable(
                    claim_nfc=claim_nfc,
                    query_str=query_str,
                    reason="vượt quá giới hạn thời gian xử lý (deadline)",
                    mode="fallback",
                    latency=latency,
                    t_start=t_start,
                )
            try:
                search_res = self._searcher.search(query_str, variant="Q1v2")
                mode = "cache" if search_res.from_cache else "live"
                hits = search_res.hits
                latency.fetch_ms = search_res.fetch_ms
                elapsed_ms = (time.perf_counter() - t_search0) * _MS_PER_SECOND
                latency.search_ms = round(max(elapsed_ms - search_res.fetch_ms, 0.0), 2)
            except Exception as exc:
                web_failure_reason = str(exc)
                logger.warning("Tavily live search unavailable (%s); trying cache/KB fallback", exc)
                cached = self._lookup_cached_web(query_str, claim_nfc=claim_nfc)
                if cached is not None and cached.hits:
                    mode = "cache"
                    hits = cached.hits
                    latency.fetch_ms = cached.fetch_ms
                    elapsed_ms = (time.perf_counter() - t_search0) * _MS_PER_SECOND
                    latency.search_ms = round(max(elapsed_ms - cached.fetch_ms, 0.0), 2)
                else:
                    mode = "fallback"
                    elapsed_ms = (time.perf_counter() - t_search0) * _MS_PER_SECOND
                    latency.search_ms = round(elapsed_ms, 2)

        # Fallback to Local KB if web search & cache yielded nothing
        if not hits:
            mode = "fallback"
            if _remaining_s() <= 0.5:
                reason = "hết thời gian chờ trước khi truy vấn kho tri thức cục bộ"
                if web_failure_reason:
                    reason = f"{reason} ({web_failure_reason})"
                return self._build_fallback_unverifiable(
                    claim_nfc=claim_nfc,
                    query_str=query_str,
                    reason=reason,
                    mode=mode,
                    latency=latency,
                    t_start=t_start,
                )
            t_kb0 = time.perf_counter()
            try:
                kb_items = self._query_local_kb(claim_nfc)
            except Exception as kb_exc:
                logger.warning("Local KB fallback failed: %s", kb_exc)
                kb_items = []
            latency.search_ms = round(
                latency.search_ms + (time.perf_counter() - t_kb0) * _MS_PER_SECOND, 2
            )

            if not kb_items:
                reason = "không tìm thấy bằng chứng từ tìm kiếm web và kho cục bộ"
                if web_failure_reason:
                    reason = f"{reason} ({web_failure_reason})"
                return self._build_fallback_unverifiable(
                    claim_nfc=claim_nfc,
                    query_str=query_str,
                    reason=reason,
                    mode=mode,
                    latency=latency,
                    t_start=t_start,
                )

        # 3. Evidence Selection (Unified Phase 3 BM25 selector, top-3 sentences <= 150 words)
        t_sel0 = time.perf_counter()
        sourced: SourcedEvidenceResult
        if hits:
            sourced = select_evidence_from_web_hits(
                claim_nfc, hits, top_k=3, max_tokens=150
            )
        else:
            sourced = select_evidence_with_sources(
                claim_nfc,
                [(it.text, it.url, it.domain) for it in kb_items],
                top_k=3,
                max_tokens=150,
            )
        latency.select_ms = round((time.perf_counter() - t_sel0) * _MS_PER_SECOND, 2)

        evidence_items = [
            EvidenceItem(text=it.text, url=it.url, domain=it.domain) for it in sourced.items
        ]
        ev_text = sourced.text.strip()
        if not ev_text:
            return self._build_fallback_unverifiable(
                claim_nfc=claim_nfc,
                query_str=query_str,
                reason="không trích xuất được câu bằng chứng phù hợp từ tài liệu tìm thấy",
                mode=mode,
                latency=latency,
                t_start=t_start,
                evidence=[],
            )

        # 4. Relevance Scoring + Gate
        if _remaining_s() <= 0.0:
            return self._build_fallback_unverifiable(
                claim_nfc=claim_nfc,
                query_str=query_str,
                reason="vượt quá giới hạn thời gian xử lý trước bước xếp hạng liên quan",
                mode="fallback",
                latency=latency,
                t_start=t_start,
                evidence=evidence_items,
            )

        t_rer0 = time.perf_counter()
        reranker = self._get_reranker()
        relevance = float(reranker.score_pair(claim_nfc, ev_text))
        latency.rerank_ms = round((time.perf_counter() - t_rer0) * _MS_PER_SECOND, 2)
        if self.low_vram_sequential:
            unload_fn = getattr(reranker, "unload", None)
            if callable(unload_fn):
                unload_fn()
                self._reranker = None

        if self.short_circuit_low_relevance and relevance < self._gate.tau_rel:
            dec = self._gate.decide(relevance, [0.0, 0.0, 1.0])
            latency.total_ms = round((time.perf_counter() - t_start) * _MS_PER_SECOND, 2)
            return LiveVerifyResult(
                claim=claim_nfc,
                query=query_str,
                verdict=cast(VerdictLiteral, dec.verdict),
                probs=dec.probs,
                show_probs=dec.show_probs,
                relevance=dec.relevance_score,
                confidence=0.0,
                evidence=evidence_items,
                latency_ms=latency,
                mode=mode,
                reason=dec.reason,
            )

        # 5. Verifier 3-Seed Ensemble + Confidence Gate
        if _remaining_s() <= 0.0:
            return self._build_fallback_unverifiable(
                claim_nfc=claim_nfc,
                query_str=query_str,
                reason="vượt quá giới hạn thời gian xử lý trước bước suy luận mô hình",
                mode="fallback",
                latency=latency,
                t_start=t_start,
                evidence=evidence_items,
                relevance=relevance,
            )

        t_ver0 = time.perf_counter()
        ensemble = self._get_ensemble()
        probs_arr = ensemble.predict_single(claim_nfc, ev_text, presegmented=False)
        dec = self._gate.decide(relevance, probs_arr)
        latency.verify_ms = round((time.perf_counter() - t_ver0) * _MS_PER_SECOND, 2)
        if self.low_vram_sequential:
            unload_ens = getattr(ensemble, "unload", None)
            if callable(unload_ens):
                unload_ens()
                self._ensemble = None

        latency.total_ms = round((time.perf_counter() - t_start) * _MS_PER_SECOND, 2)
        return LiveVerifyResult(
            claim=claim_nfc,
            query=query_str,
            verdict=cast(VerdictLiteral, dec.verdict),
            probs=dec.probs,
            show_probs=dec.show_probs,
            relevance=dec.relevance_score,
            confidence=dec.confidence,
            evidence=evidence_items,
            latency_ms=latency,
            mode=mode,
            reason=dec.reason,
        )
