"""Cross-encoder reranker for BAAI/bge-reranker-v2-m3.

Lazy load (copilot-instructions §5). Scores query-passage pairs directly;
used after hybrid retrieval to narrow ``retrieve_k`` → ``rerank_k``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import torch

from vnfake.retrieval.vector_store import ScoredChunk
from vnfake.utils.logging import get_logger

logger = get_logger(__name__)


@runtime_checkable
class RerankerProtocol(Protocol):
    """Structural interface so tests can inject a dummy."""

    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]: ...


class BgeReranker:
    """BAAI/bge-reranker-v2-m3 cross-encoder wrapper."""

    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int = 16,
        use_fp16: bool = True,
        max_length: int = 512,
        device: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._use_fp16 = use_fp16
        self._max_length = max_length
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: object | None = None

    def _load(self) -> object:
        if self._model is None:
            from FlagEmbedding import FlagReranker

            effective_fp16 = self._use_fp16 and self._device == "cuda"
            logger.info(
                "Loading reranker: %s on %s (fp16=%s)",
                self._model_name,
                self._device,
                effective_fp16,
            )
            self._model = FlagReranker(
                self._model_name,
                use_fp16=effective_fp16,
                devices=self._device,
            )
        return self._model

    def score_pairs(self, pairs: Sequence[Sequence[str] | tuple[str, str]]) -> list[float]:
        """Score arbitrary ``(query, text)`` pairs and return normalized relevance floats."""
        if not pairs:
            return []
        model = self._load()
        pair_lists = [[str(q), str(t)] for q, t in pairs]
        raw_scores = model.compute_score(  # type: ignore[attr-defined]
            pair_lists,
            batch_size=self._batch_size,
            max_length=self._max_length,
            normalize=True,
        )
        if isinstance(raw_scores, (int, float)):
            return [float(raw_scores)]
        return [float(s) for s in raw_scores]

    def score_pair(self, query: str, text: str) -> float:
        """Score a single ``(query, text)`` pair."""
        scores = self.score_pairs([(query, text)])
        return scores[0] if scores else 0.0

    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        """Return top-``top_k`` candidates sorted by reranker score.

        ``score`` on returned chunks is the reranker score (first-stage score
        is discarded so downstream layers see a single comparable value).
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if not candidates:
            return []

        scores = self.score_pairs([[query, c.text] for c in candidates])

        if len(scores) != len(candidates):
            raise RuntimeError(
                f"reranker returned {len(scores)} scores for {len(candidates)} pairs"
            )

        reranked = [
            ScoredChunk(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                chunk_idx=c.chunk_idx,
                title=c.title,
                text=c.text,
                text_segmented=c.text_segmented,
                url=c.url,
                crawl_date=c.crawl_date,
                source=c.source,
                score=s,
                published_date=c.published_date,
                ingest_source=c.ingest_source,
            )
            for c, s in zip(candidates, scores, strict=True)
        ]
        reranked.sort(key=lambda c: c.score, reverse=True)
        return reranked[:top_k]
