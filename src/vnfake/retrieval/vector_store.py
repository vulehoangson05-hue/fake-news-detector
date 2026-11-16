"""Qdrant vector store wrapper (named dense + sparse vectors, hybrid search).

All Qdrant clients MUST be created here (copilot-instructions §2). Uses the
Universal Query API (``client.query_points``) — ``client.search`` was removed
in qdrant-client >= 1.12.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from vnfake.data.chunking import Chunk
from vnfake.retrieval.embedder import SparseVectorData
from vnfake.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """One search hit, flattened for direct consumption by the API layer."""

    chunk_id: str
    doc_id: str
    chunk_idx: int
    title: str
    text: str
    text_segmented: str
    url: str
    crawl_date: str
    source: str
    score: float
    published_date: str | None = None
    ingest_source: str = "static"


class QdrantStore:
    """Thin wrapper around :class:`QdrantClient` for named-vector collections."""

    def __init__(
        self,
        client: QdrantClient,
        *,
        dense_name: str = "dense",
        sparse_name: str = "sparse",
    ) -> None:
        self._client = client
        self._dense_name = dense_name
        self._sparse_name = sparse_name

    @classmethod
    def from_env(
        cls,
        url: str | None = None,
        api_key: str | None = None,
        *,
        dense_name: str = "dense",
        sparse_name: str = "sparse",
    ) -> QdrantStore:
        """Build a store from ``QDRANT_URL`` / ``QDRANT_API_KEY`` env vars."""
        import os

        resolved_url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        resolved_key = api_key or os.getenv("QDRANT_API_KEY") or None
        logger.info("Connecting to Qdrant at %s", resolved_url)
        return cls(
            QdrantClient(url=resolved_url, api_key=resolved_key),
            dense_name=dense_name,
            sparse_name=sparse_name,
        )

    def recreate_collection(self, name: str, dim: int, *, force: bool = False) -> None:
        """Create ``name`` with named dense + sparse vectors."""
        existing = {c.name for c in self._client.get_collections().collections}
        if name in existing:
            if not force:
                raise ValueError(
                    f"Collection {name!r} already exists; pass force=True to recreate."
                )
            self._client.delete_collection(name)
            logger.info("Deleted existing collection %s", name)
        self._client.create_collection(
            collection_name=name,
            vectors_config={
                self._dense_name: models.VectorParams(size=dim, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={self._sparse_name: models.SparseVectorParams()},
        )
        logger.info("Created collection %s (dense dim=%d, sparse enabled)", name, dim)

    def upsert_chunks(
        self,
        name: str,
        chunks: Sequence[Chunk],
        dense_vectors: Sequence[Sequence[float]],
        sparse_vectors: Sequence[SparseVectorData],
        *,
        batch_size: int = 128,
    ) -> None:
        """Idempotent upsert of chunks + named dense/sparse vectors."""
        n = len(chunks)
        if len(dense_vectors) != n or len(sparse_vectors) != n:
            raise ValueError(
                f"Length mismatch: chunks={n}, dense={len(dense_vectors)}, "
                f"sparse={len(sparse_vectors)}"
            )
        if n == 0:
            logger.warning("upsert_chunks called with empty input")
            return

        points = [
            models.PointStruct(
                id=c.chunk_id,
                vector={
                    self._dense_name: list(dv),
                    self._sparse_name: models.SparseVector(indices=sv.indices, values=sv.values),
                },
                payload=c.to_dict(),
            )
            for c, dv, sv in zip(chunks, dense_vectors, sparse_vectors, strict=True)
        ]
        for start in range(0, len(points), batch_size):
            self._client.upsert(
                collection_name=name,
                points=points[start : start + batch_size],
            )
        logger.info("Upserted %d chunks into %s", len(points), name)

    def search_dense(
        self,
        name: str,
        query_vector: Sequence[float],
        top_k: int,
        *,
        with_payload: bool = True,
    ) -> list[ScoredChunk]:
        """Dense-only search."""
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        response = self._client.query_points(
            collection_name=name,
            query=list(query_vector),
            using=self._dense_name,
            limit=top_k,
            with_payload=with_payload,
        )
        return [_hit_to_scored(p) for p in response.points if p is not None]

    def search_sparse(
        self,
        name: str,
        query_sparse: SparseVectorData,
        top_k: int,
        *,
        with_payload: bool = True,
    ) -> list[ScoredChunk]:
        """Sparse-only search."""
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        response = self._client.query_points(
            collection_name=name,
            query=models.SparseVector(indices=query_sparse.indices, values=query_sparse.values),
            using=self._sparse_name,
            limit=top_k,
            with_payload=with_payload,
        )
        return [_hit_to_scored(p) for p in response.points if p is not None]

    def search_hybrid(
        self,
        name: str,
        dense_query: Sequence[float],
        sparse_query: SparseVectorData,
        top_k: int,
        *,
        dense_prefetch_k: int,
        sparse_prefetch_k: int,
        with_payload: bool = True,
    ) -> list[ScoredChunk]:
        """Hybrid search with server-side RRF fusion."""
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        response = self._client.query_points(
            collection_name=name,
            prefetch=[
                models.Prefetch(
                    query=list(dense_query),
                    using=self._dense_name,
                    limit=dense_prefetch_k,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_query.indices, values=sparse_query.values
                    ),
                    using=self._sparse_name,
                    limit=sparse_prefetch_k,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=with_payload,
        )
        return [_hit_to_scored(p) for p in response.points if p is not None]


def _hit_to_scored(hit: object) -> ScoredChunk:
    """Convert a Qdrant ``ScoredPoint`` to our flattened dataclass."""
    payload = getattr(hit, "payload", None) or {}
    raw_pub = payload.get("published_date")
    pub_date = str(raw_pub) if raw_pub is not None else None
    raw_ingest = payload.get("ingest_source")
    ingest_source = str(raw_ingest) if raw_ingest is not None else "static"
    return ScoredChunk(
        chunk_id=str(getattr(hit, "id", "")),
        doc_id=str(payload.get("doc_id", "")),
        chunk_idx=int(payload.get("chunk_idx", 0)),
        title=str(payload.get("title", "")),
        text=str(payload.get("text", "")),
        text_segmented=str(payload.get("text_segmented", "")),
        url=str(payload.get("url", "")),
        crawl_date=str(payload.get("crawl_date", "")),
        source=str(payload.get("source", "")),
        score=float(getattr(hit, "score", 0.0)),
        published_date=pub_date,
        ingest_source=ingest_source,
    )
