"""Phase 2.5 Task 6 — On-demand RSS Ingestion & KB Updater (Pipeline D).

Implements Brief #2 §5 and Brief #3 §D:
- Fetches enabled RSS feeds from ``configs/rss.yaml`` using ``HttpxPoliteFetcher``.
- Enforces three-tier deduplication via ``canonicalize_url``:
    1. Across feeds within current run.
    2. Against static KB (articles already in ``data/processed/kb_chunks.jsonl`` are skipped).
    3. Against dynamic KB manifest (articles in ``rss_manifest.json`` are skipped).
- Extracts article text via ``trafilatura``, cleans with :func:`clean_rss_article_text`.
- Breaks into chunks with :func:`chunk_record` (UUID5 deterministic chunk_id).
- Embeds chunks using dense + sparse embedder and upserts into Qdrant.
- Strictly adheres to Rule D8: Upserting to main collection ``kb_chunks_8b525e25ad54``
  is gated until PM human labeling is completed. Default is a test collection.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import feedparser

from vnfake.config import AppConfig
from vnfake.data.chunking import Chunk, chunk_record
from vnfake.data.rss_audit import (
    HttpFetcherProtocol,
    HttpxPoliteFetcher,
    RobotsChecker,
    RssConfig,
    canonicalize_url,
    parse_pubdate_iso8601_utc,
)
from vnfake.data.rss_cleaner import clean_rss_article_text
from vnfake.retrieval.embedder import EmbedderProtocol
from vnfake.retrieval.vector_store import QdrantStore
from vnfake.schemas.kb import KBRecord
from vnfake.utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_MANIFEST_PATH = Path("artifacts/rss_manifest.json")
_DEFAULT_STATIC_KB_PATH = Path("data/processed/kb_chunks.jsonl")


def _default_extractor(html: str, url: str) -> str:
    import trafilatura

    extracted = trafilatura.extract(
        html,
        url=url,
        favor_recall=True,
        include_comments=False,
        include_tables=False,
        output_format="txt",
    )
    return extracted or ""


@dataclass
class RssManifestData:
    """Tracking schema stored in ``artifacts/rss_manifest.json``."""

    rss_last_updated: str = ""
    dynamic_chunks: int = 0
    total_articles_ingested: int = 0
    ingested_urls: set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rss_last_updated": self.rss_last_updated,
            "dynamic_chunks": self.dynamic_chunks,
            "total_articles_ingested": self.total_articles_ingested,
            "ingested_urls": sorted(self.ingested_urls),
        }

    @classmethod
    def from_file(cls, path: Path) -> RssManifestData:
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                rss_last_updated=data.get("rss_last_updated", ""),
                dynamic_chunks=int(data.get("dynamic_chunks", 0)),
                total_articles_ingested=int(data.get("total_articles_ingested", 0)),
                ingested_urls=set(data.get("ingested_urls", [])),
            )
        except Exception as exc:
            logger.warning("Could not read rss_manifest.json (%s), starting fresh.", exc)
            return cls()


def load_static_kb_urls(static_kb_path: Path = _DEFAULT_STATIC_KB_PATH) -> set[str]:
    """Extract canonical URLs of articles in static KB to prevent duplicate dynamic ingestion."""
    if not static_kb_path.exists():
        return set()
    urls: set[str] = set()
    with static_kb_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
                raw_url = chunk.get("url")
                if raw_url:
                    c_url = canonicalize_url(raw_url)
                    if c_url:
                        urls.add(c_url)
            except Exception:
                continue
    return urls


@dataclass(frozen=True, slots=True)
class IngestionStats:
    """Summary of one RSS update run."""

    feeds_checked: int
    articles_discovered: int
    articles_skipped_cross_feed: int
    articles_skipped_static_kb: int
    articles_skipped_previously_ingested: int
    articles_fetched: int
    articles_cleaned: int
    chunks_created: int
    chunks_upserted: int
    dry_run: bool


class RssUpdater:
    """Orchestrates end-to-end on-demand RSS ingestion (Pipeline D)."""

    def __init__(
        self,
        cfg: AppConfig,
        rss_cfg: RssConfig,
        *,
        store: QdrantStore,
        embedder: EmbedderProtocol,
        fetcher: HttpFetcherProtocol | None = None,
        extractor_fn: Callable[[str, str], str] | None = None,
        manifest_path: Path = _DEFAULT_MANIFEST_PATH,
        static_kb_path: Path = _DEFAULT_STATIC_KB_PATH,
    ) -> None:
        self._cfg = cfg
        self._rss_cfg = rss_cfg
        self._store = store
        self._embedder = embedder
        self._manifest_path = manifest_path
        self._static_kb_path = static_kb_path

        self._fetcher: HttpFetcherProtocol = fetcher or HttpxPoliteFetcher(
            user_agent=rss_cfg.fetch.user_agent,
            timeout_s=rss_cfg.fetch.request_timeout_s,
            min_delay_s=rss_cfg.fetch.min_delay_between_requests_s,
        )
        self._extractor_fn = extractor_fn or _default_extractor
        self._manifest = RssManifestData.from_file(manifest_path)
        self._static_urls = load_static_kb_urls(static_kb_path)
        self._robots = RobotsChecker(
            self._fetcher,
            user_agent=rss_cfg.fetch.user_agent,
            respect=rss_cfg.fetch.respect_robots_txt,
        )

    def update(
        self,
        collection_name: str,
        *,
        dry_run: bool = False,
        max_articles_to_ingest: int | None = None,
    ) -> IngestionStats:
        """Run on-demand RSS feed fetching, deduping, chunking, and upserting."""
        seen_in_run: set[str] = set()

        total_discovered = 0
        skipped_cross_feed = 0
        skipped_static = 0
        skipped_prev = 0
        articles_fetched = 0
        articles_cleaned = 0
        total_chunks = 0
        total_upserted = 0

        enabled_feeds = [f for f in self._rss_cfg.feeds if f.enabled]
        logger.info(
            "Starting RSS update on %d enabled feeds -> collection=%s (dry_run=%s)",
            len(enabled_feeds),
            collection_name,
            dry_run,
        )

        new_chunks_to_upsert: list[Chunk] = []
        newly_ingested_urls: set[str] = set()

        for feed in enabled_feeds:
            try:
                resp = self._fetcher.get(feed.url)
            except Exception as exc:
                logger.warning("Failed to fetch feed %s (%s): %s", feed.id, feed.url, exc)
                continue

            parsed = feedparser.parse(resp.text)
            entries = getattr(parsed, "entries", [])
            for entry in entries:
                raw_url = getattr(entry, "link", None)
                if not raw_url:
                    continue
                c_url = canonicalize_url(raw_url)
                if not c_url:
                    continue

                total_discovered += 1

                # 1. Dedup cross-feed within run
                if c_url in seen_in_run:
                    skipped_cross_feed += 1
                    continue
                seen_in_run.add(c_url)

                # 2. Dedup against static KB
                if c_url in self._static_urls:
                    skipped_static += 1
                    continue

                # 3. Dedup against previously ingested dynamic articles
                if c_url in self._manifest.ingested_urls:
                    skipped_prev += 1
                    continue

                if max_articles_to_ingest and articles_fetched >= max_articles_to_ingest:
                    break

                # Check robots.txt
                allowed, _ = self._robots.is_allowed(c_url)
                if not allowed:
                    logger.info("Blocked by robots.txt: %s", c_url)
                    continue

                # Fetch article HTML
                articles_fetched += 1
                try:
                    art_resp = self._fetcher.get(c_url)
                except Exception as exc:
                    logger.warning("Failed to fetch article %s: %s", c_url, exc)
                    continue

                if art_resp.status_code != 200 or not art_resp.text:
                    continue

                raw_title = getattr(entry, "title", "")
                raw_pubdate = getattr(entry, "published", None) or getattr(entry, "pubDate", None)
                pubdate_iso = parse_pubdate_iso8601_utc(raw_pubdate)

                extracted_raw = self._extractor_fn(art_resp.text, c_url)
                cleaned_text = clean_rss_article_text(
                    extracted_raw,
                    title=raw_title,
                    min_chars=100,
                )
                if not cleaned_text:
                    continue

                articles_cleaned += 1
                doc_id = hashlib.sha1(c_url.encode("utf-8")).hexdigest()[:12]
                now_iso = datetime.now(UTC).isoformat()

                kb_rec = KBRecord(
                    id=doc_id,
                    title=raw_title,
                    text=cleaned_text,
                    label="real",
                    url=c_url,
                    crawl_date=now_iso,
                    published_date=pubdate_iso,
                    ingest_source="rss",
                )

                chunks = chunk_record(
                    kb_rec,
                    source=feed.source,
                    segmenter=None,
                    window=self._cfg.preprocessing.chunk.window,
                    overlap=self._cfg.preprocessing.chunk.overlap,
                )
                if chunks:
                    new_chunks_to_upsert.extend(chunks)
                    newly_ingested_urls.add(c_url)
                    total_chunks += len(chunks)

            if max_articles_to_ingest and articles_fetched >= max_articles_to_ingest:
                break

        if not dry_run and new_chunks_to_upsert:
            # Embed new chunks
            chunk_texts = [c.text for c in new_chunks_to_upsert]
            dense_vectors = self._embedder.embed_dense(
                chunk_texts, batch_size=self._cfg.retrieval.embedding.batch_size
            )
            sparse_vectors = self._embedder.embed_sparse(
                chunk_texts, batch_size=self._cfg.retrieval.sparse.batch_size
            )

            existing_collections = {
                c.name for c in self._store._client.get_collections().collections
            }
            if collection_name not in existing_collections:
                self._store.recreate_collection(
                    collection_name,
                    dim=self._cfg.retrieval.embedding.dim,
                    force=False,
                )

            self._store.upsert_chunks(
                collection_name,
                new_chunks_to_upsert,
                dense_vectors,
                sparse_vectors,
            )
            total_upserted = len(new_chunks_to_upsert)

            # Update manifest
            self._manifest.rss_last_updated = datetime.now(UTC).isoformat()
            self._manifest.dynamic_chunks += total_upserted
            self._manifest.total_articles_ingested += len(newly_ingested_urls)
            self._manifest.ingested_urls.update(newly_ingested_urls)
            self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
            self._manifest_path.write_text(
                json.dumps(self._manifest.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info("Updated RSS manifest at %s", self._manifest_path)

        stats = IngestionStats(
            feeds_checked=len(enabled_feeds),
            articles_discovered=total_discovered,
            articles_skipped_cross_feed=skipped_cross_feed,
            articles_skipped_static_kb=skipped_static,
            articles_skipped_previously_ingested=skipped_prev,
            articles_fetched=articles_fetched,
            articles_cleaned=articles_cleaned,
            chunks_created=total_chunks,
            chunks_upserted=total_upserted,
            dry_run=dry_run,
        )
        logger.info("RSS Update finished: %s", stats)
        return stats
