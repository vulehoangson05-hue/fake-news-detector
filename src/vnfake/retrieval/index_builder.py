"""KB index builder (consumed by ``scripts/03_index_kb.py``).

Logic lives here per copilot-instructions §2 ("``scripts/`` chỉ là CLI mỏng gọi
``src/``").
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import cast

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


def compute_index_version(chunks_path: Path, *, length: int = 12) -> str:
    """Return a short SHA1 over the chunks file content (deterministic)."""
    h = hashlib.sha1()
    with chunks_path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:length]


def iter_chunk_jsonl(path: Path) -> Iterator[Chunk]:
    """Stream chunks from JSONL, skipping malformed lines with a warning.

    Args:
        path: JSONL produced by ``scripts/02_prepare_data.py``.

    Yields:
        Validated :class:`Chunk` objects in file order.
    """
    skipped = 0
    total = 0
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
                logger.warning("Skip chunk at line %d: missing fields %s", lineno, missing)
                continue
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
    """Return the string embedded for ``chunk`` (per ARCHITECTURE §5.1 step 5).

    bge-m3 is a multilingual model — it handles Vietnamese well without word
    segmentation, so we embed the raw text (not ``text_segmented``).
    """
    return f"{chunk.title}\n{chunk.text}" if chunk.title else chunk.text


def _wrap_progress(iterator: Iterable[list[Chunk]], enabled: bool) -> Iterable[list[Chunk]]:
    """Wrap ``iterator`` in a tqdm progress bar if enabled and available."""
    if not enabled:
        return iterator
    try:
        from tqdm import tqdm  # local import — optional dependency
    except ImportError:
        logger.warning("tqdm not installed; progress bar disabled")
        return iterator
    return cast(Iterable[list[Chunk]], tqdm(iterator, desc="Embedding", unit="batch"))


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
    """Recreate the KB collection and upsert all chunks.

    Args:
        cfg: Full app config (uses ``retrieval.embedding`` and
            ``retrieval.qdrant_collection_prefix``).
        chunks_path: Processed chunks JSONL.
        embedder: Dense embedder (``BgeM3Embedder`` in prod, dummy in tests).
        store: Target :class:`QdrantStore`.
        index_version: Override for the version suffix (otherwise content SHA1).
        force: Delete + recreate if collection exists.
        show_progress: Show a tqdm progress bar.

    Returns:
        Stats dict (collection name, index_version, chunks, embedding_dim).
    """
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    version = index_version or compute_index_version(chunks_path)
    collection = f"{cfg.retrieval.qdrant_collection_prefix}_{version}"
    batch_size = cfg.retrieval.embedding.batch_size

    store.recreate_collection(collection, dim=embedder.dim, force=force)

    total = 0
    for batch in _wrap_progress(iter_chunk_batches(chunks_path, batch_size), show_progress):
        vectors = embedder.embed_dense(
            [_text_for_embedding(c) for c in batch], batch_size=batch_size
        )
        store.upsert_chunks(collection, batch, vectors)
        total += len(batch)

    stats: dict[str, object] = {
        "collection": collection,
        "index_version": version,
        "chunks": total,
        "embedding_dim": embedder.dim,
    }
    logger.info("Index built: %s", stats)
    return stats
