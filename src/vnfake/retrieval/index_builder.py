"""KB index builder (manifest + resume + dense/sparse)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from vnfake.config import AppConfig
from vnfake.data.chunking import Chunk
from vnfake.retrieval.embedder import EmbedderProtocol
from vnfake.retrieval.vector_store import QdrantStore
from vnfake.utils.logging import get_logger

logger = get_logger(__name__)

_REQUIRED_FIELDS: tuple[str, ...] = (
    "chunk_id",
    "doc_id",
    "chunk_idx",
    "title",
    "text",
    "url",
    "crawl_date",
    "source",
)
_MANIFEST_NAME = "index_manifest.json"


def compute_index_version(chunks_path: Path, *, length: int = 12) -> str:
    """Return a short SHA1 over the chunks file content (deterministic)."""
    h = hashlib.sha1()
    with chunks_path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:length]


def iter_chunk_jsonl(path: Path) -> Iterator[Chunk]:
    """Stream chunks from JSONL, skipping malformed lines with a warning."""
    skipped = total = 0
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                skipped += 1
                logger.warning("Skip malformed chunk at line %d: %s", lineno, exc)
                continue
            missing = [k for k in _REQUIRED_FIELDS if k not in obj]
            if missing:
                skipped += 1
                logger.warning("Skip chunk at line %d: missing %s", lineno, missing)
                continue
            raw_pub = obj.get("published_date")
            pub_date = str(raw_pub) if raw_pub is not None else None
            raw_ingest = obj.get("ingest_source")
            ingest_src: Literal["static", "rss"] = "rss" if raw_ingest == "rss" else "static"
            yield Chunk(
                chunk_id=str(obj["chunk_id"]),
                doc_id=str(obj["doc_id"]),
                chunk_idx=int(obj["chunk_idx"]),
                title=str(obj["title"]),
                text=str(obj["text"]),
                text_segmented=str(obj.get("text_segmented", "")),
                url=str(obj["url"]),
                crawl_date=str(obj["crawl_date"]),
                source=str(obj["source"]),
                published_date=pub_date,
                ingest_source=ingest_src,
            )
    if skipped:
        logger.warning("iter_chunk_jsonl: skipped %d/%d lines", skipped, total)


def iter_chunk_batches(chunks_path: Path, batch_size: int) -> Iterator[list[Chunk]]:
    """Yield lists of at most ``batch_size`` chunks."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    batch: list[Chunk] = []
    for chunk in iter_chunk_jsonl(chunks_path):
        batch.append(chunk)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _text_for_embedding(chunk: Chunk) -> str:
    """String embedded for ``chunk`` (title + raw text)."""
    return f"{chunk.title}\n{chunk.text}" if chunk.title else chunk.text


def _wrap_progress(iterator: Iterable[list[Chunk]], enabled: bool) -> Iterable[list[Chunk]]:
    """Wrap ``iterator`` in a tqdm progress bar if enabled."""
    if not enabled:
        return iterator
    try:
        from tqdm import tqdm
    except ImportError:
        logger.warning("tqdm not installed; progress bar disabled")
        return iterator
    return cast(Iterable[list[Chunk]], tqdm(iterator, desc="Embedding", unit="batch"))


def _manifest_path(cfg: AppConfig) -> Path:
    return cfg.paths.artifacts_dir / _MANIFEST_NAME


def _write_manifest(cfg: AppConfig, manifest: dict[str, object]) -> None:
    p = _manifest_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote manifest %s", p)


def _collection_point_count(store: QdrantStore, collection: str) -> int:
    """Return the number of points in ``collection``; 0 if missing/unreachable."""
    try:
        return int(store._client.count(collection_name=collection, exact=True).count)
    except Exception as exc:
        logger.debug("count() failed for %s: %s", collection, exc)
        return 0


def build_index(
    cfg: AppConfig,
    *,
    chunks_path: Path,
    embedder: EmbedderProtocol,
    store: QdrantStore,
    index_version: str | None = None,
    force: bool = False,
    show_progress: bool = True,
) -> dict[str, object]:
    """Build (or resume) the KB index with named dense + sparse vectors.

    Behaviour:
        * ``force=True``                  → delete + recreate, full re-embed.
        * Collection missing              → create + full embed.
        * Exists + count == expected      → skip (writes manifest with skipped=True).
        * Exists + count != expected      → re-upsert (idempotent resume).

    Returns:
        Stats dict (``collection``, ``index_version``, ``chunks``,
        ``embedding_dim``, ``skipped``).
    """
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    version = index_version or compute_index_version(chunks_path)
    collection = f"{cfg.retrieval.qdrant_collection_prefix}_{version}"
    batch_size = cfg.retrieval.embedding.batch_size
    sparse_batch_size = cfg.retrieval.sparse.batch_size
    existing = {c.name for c in store._client.get_collections().collections}
    expected_total = sum(1 for _ in iter_chunk_jsonl(chunks_path))

    if collection in existing and not force:
        current = _collection_point_count(store, collection)
        if current == expected_total and expected_total > 0:
            logger.info(
                "Index already up to date: %s (%d chunks); skipping.",
                collection,
                current,
            )
            _write_manifest(
                cfg,
                {
                    "collection": collection,
                    "index_version": version,
                    "chunks": current,
                    "embedding_dim": embedder.dim,
                    "chunks_file": str(chunks_path),
                    "updated_at": datetime.now(UTC).isoformat(),
                    "skipped": True,
                },
            )
            return {
                "collection": collection,
                "index_version": version,
                "chunks": current,
                "embedding_dim": embedder.dim,
                "skipped": True,
            }
        logger.warning(
            "Collection %s has %d/%d points; resuming (idempotent upsert).",
            collection,
            current,
            expected_total,
        )
    else:
        store.recreate_collection(collection, dim=embedder.dim, force=force)

    total = 0
    for batch in _wrap_progress(iter_chunk_batches(chunks_path, batch_size), show_progress):
        texts = [_text_for_embedding(c) for c in batch]
        dense = embedder.embed_dense(texts, batch_size=batch_size)
        sparse = embedder.embed_sparse(texts, batch_size=sparse_batch_size)
        store.upsert_chunks(collection, batch, dense, sparse)
        total += len(batch)

    stats: dict[str, object] = {
        "collection": collection,
        "index_version": version,
        "chunks": total,
        "embedding_dim": embedder.dim,
        "skipped": False,
    }
    _write_manifest(
        cfg,
        {
            **stats,
            "chunks_file": str(chunks_path),
            "updated_at": datetime.now(UTC).isoformat(),
            "dense_vector_name": cfg.retrieval.dense_vector_name,
            "sparse_vector_name": cfg.retrieval.sparse_vector_name,
        },
    )
    logger.info("Index built: %s", stats)
    return stats
