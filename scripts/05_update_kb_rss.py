"""Phase 2.5 Task 6 — CLI tool for On-Demand RSS Knowledge Base Updater (Pipeline D)."""

from __future__ import annotations

import argparse
from pathlib import Path

from vnfake.config import load_config
from vnfake.data.rss_audit import load_rss_config
from vnfake.data.rss_updater import RssUpdater
from vnfake.retrieval.embedder import BgeM3Embedder
from vnfake.retrieval.vector_store import QdrantStore
from vnfake.utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_TEST_COLLECTION = "test_kb_rss_temporary"
_PROTECTED_MAIN_COLLECTION = "kb_chunks_8b525e25ad54"


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for on-demand RSS updater."""
    parser = argparse.ArgumentParser(
        description="Phase 2.5 Task 6 — On-Demand RSS KB Ingestion (Pipeline D)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base.yaml"),
        help="Path to base config (default: configs/base.yaml)",
    )
    parser.add_argument(
        "--rss-config",
        type=Path,
        default=Path("configs/rss.yaml"),
        help="Path to rss.yaml (default: configs/rss.yaml)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=_DEFAULT_TEST_COLLECTION,
        help=f"Target Qdrant collection name (default: {_DEFAULT_TEST_COLLECTION})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and clean articles without writing to Qdrant or manifest",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=None,
        help="Maximum articles to fetch in this run (optional rate guard)",
    )
    parser.add_argument(
        "--force-production",
        action="store_true",
        help="Explicitly allow writing to protected main collection kb_chunks_8b525e25ad54 (Rule D8)",
    )
    args = parser.parse_args(argv)

    # Enforce Rule D8 (Freezing corpus)
    if args.collection == _PROTECTED_MAIN_COLLECTION and not args.force_production:
        logger.error(
            "RULE D8 VIOLATION: Upserting to %s is strictly blocked until C1 human labeling is complete! "
            "Use a test collection (e.g. %s) or pass --force-production if authorized by PM.",
            _PROTECTED_MAIN_COLLECTION,
            _DEFAULT_TEST_COLLECTION,
        )
        return 1

    cfg = load_config(args.config)
    rss_cfg = load_rss_config(args.rss_config)

    store = QdrantStore.from_env(
        dense_name=cfg.retrieval.dense_vector_name,
        sparse_name=cfg.retrieval.sparse_vector_name,
    )

    embedder = BgeM3Embedder(
        model_name=cfg.retrieval.embedding.model_name,
        dim=cfg.retrieval.embedding.dim,
        use_fp16=cfg.retrieval.embedding.use_fp16,
        weight_threshold=cfg.retrieval.sparse.weight_threshold,
    )

    updater = RssUpdater(
        cfg,
        rss_cfg,
        store=store,
        embedder=embedder,
    )

    stats = updater.update(
        args.collection,
        dry_run=args.dry_run,
        max_articles_to_ingest=args.max_articles,
    )

    print("\n--- RSS Ingestion Summary ---")
    print(f"Feeds Checked:                 {stats.feeds_checked}")
    print(f"Articles Discovered:           {stats.articles_discovered}")
    print(f"Skipped (Cross-feed Duplicate): {stats.articles_skipped_cross_feed}")
    print(f"Skipped (Static KB Match):     {stats.articles_skipped_static_kb}")
    print(f"Skipped (Previously Ingested): {stats.articles_skipped_previously_ingested}")
    print(f"Articles Fetched:              {stats.articles_fetched}")
    print(f"Articles Cleaned:              {stats.articles_cleaned}")
    print(f"Chunks Upserted:               {stats.chunks_upserted}")
    print(f"Dry Run Mode:                  {stats.dry_run}")
    print("-----------------------------\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
