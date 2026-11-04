"""Qdrant vector store wrapper.

All Qdrant clients MUST be created here (copilot-instructions §2). The store
owns collection lifecycle and payload mapping; it does not know about models
or config beyond what callers pass in.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from vnfake.data.chunking import Chunk
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


class QdrantStore:
    """Thin wrapper around :class:`QdrantClient`.

    Args:
        client: A ``QdrantClient`` instance. Callers should obtain one via
            :meth:`from_env` in production, or ``QdrantClient(":memory:")``
            in tests.
    """

    def __init__(self, client: QdrantClient) -> None:
        self._client = client

    @classmethod
    def from_env(cls, url: str | None = None, api_key: str | None = None) -> QdrantStore:
        """Build a store from ``QDRANT_URL`` / ``QDRANT_API_KEY`` env vars.

        Defaults to ``http://localhost:6333`` (docker-compose service).
        """
        import os

        resolved_url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        resolved_key = api_key or os.getenv("QDRANT_API_KEY") or None
        logger.info("Connecting to Qdrant at %s", resolved_url)
        return cls(QdrantClient(url=resolved_url, api_key=resolved_key))

    def recreate_collection(self, name: str, dim: int, *, force: bool = False) -> None:
        """Create ``name`` with a single dense vector of size ``dim`` (cosine).

        Args:
            name: Collection name (typically ``kb_chunks_<version>``).
            dim: Vector dimensionality (must match the embedder).
            force: If True and the collection exists, delete then recreate.

        Raises:
            ValueError: If the collection exists and ``force`` is False.
        """
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
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        logger.info("Created collection %s (dim=%d, cosine)", name, dim)

    def upsert_chunks(
        self,
        name: str,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        *,
        batch_size: int = 128,
    ) -> None:
        """Idempotent upsert of chunks + their dense vectors.

        Point ID = ``chunk.chunk_id`` (UUID5), so re-running overwrites rather
        than duplicates.

        Args:
            name: Target collection.
            chunks: Chunks in order; ``chunks[i]`` pairs with ``vectors[i]``.
            vectors: Dense vectors, each of length equal to the collection dim.
            batch_size: Number of points per ``upsert`` call.

        Raises:
            ValueError: If ``len(chunks) != len(vectors)``.
        """
        if len(chunks) != len(vectors):
            raise ValueError(f"len(chunks)={len(chunks)} != len(vectors)={len(vectors)}")
        if not chunks:
            logger.warning("upsert_chunks called with empty input")
            return

        points = [
            PointStruct(id=c.chunk_id, vector=list(v), payload=c.to_dict())
            for c, v in zip(chunks, vectors, strict=True)
        ]
        for start in range(0, len(points), batch_size):
            self._client.upsert(
                collection_name=name,
                points=points[start : start + batch_size],
            )
        logger.info("Upserted %d chunks into %s", len(points), name)

    def search(
        self,
        name: str,
        query_vector: Sequence[float],
        top_k: int,
        *,
        with_payload: bool = True,
    ) -> list[ScoredChunk]:
        """Return the top-``k`` nearest chunks for a query vector.

        Args:
            name: Collection name.
            query_vector: Dense query vector (length = collection dim).
            top_k: Number of hits to return.
            with_payload: Whether to fetch payload (required for evidence URL).

        Returns:
            Ordered hits (highest score first).
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        response = self._client.query_points(
            collection_name=name,
            query=list(query_vector),
            limit=top_k,
            with_payload=with_payload,
        )
        # ``response.points`` is ``list[ScoredPoint]`` on qdrant-client>=1.10.
        # Filter out any ``None`` defensively (some embedded builds return them).
        return [_hit_to_scored(p) for p in response.points if p is not None]


def _hit_to_scored(hit: object) -> ScoredChunk:
    """Convert a Qdrant ``ScoredPoint`` to our flattened dataclass."""
    payload = getattr(hit, "payload", None) or {}
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
    )
