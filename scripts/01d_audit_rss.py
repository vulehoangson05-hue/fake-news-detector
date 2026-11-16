"""Phase 2.5 Task 4.2 — Thin CLI wrapper to run RSS audit on the 13 configured feeds."""

from __future__ import annotations

import argparse
from pathlib import Path

from vnfake.config import load_config
from vnfake.data.rss_audit import (
    HttpxPoliteFetcher,
    load_rss_config,
    load_static_kb_urls,
    run_rss_audit,
    save_rss_audit_artifacts,
)
from vnfake.utils.logging import get_logger

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 2.5 Task 4.2 — Audit 13 RSS feeds (robots.txt, overlap, full-text extraction)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to base YAML config (default: configs/base.yaml)",
    )
    parser.add_argument(
        "--rss-config",
        type=Path,
        default=Path("configs/rss.yaml"),
        help="Path to RSS YAML config (default: configs/rss.yaml)",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=20,
        help="Max articles to fetch per feed for full-text audit (default: 20)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/audit_rss"),
        help="Output directory for raw audit JSON artifacts (default: artifacts/audit_rss)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    rss_cfg = load_rss_config(args.rss_config)
    static_urls = load_static_kb_urls(cfg)

    fetcher = HttpxPoliteFetcher(
        user_agent=rss_cfg.fetch.user_agent,
        timeout_s=rss_cfg.fetch.request_timeout_s,
        min_delay_s=rss_cfg.fetch.min_delay_between_requests_s,
    )
    try:
        report = run_rss_audit(
            rss_cfg,
            static_kb_urls=static_urls,
            fetcher=fetcher,
            max_articles_per_feed=args.max_articles,
            min_chars=cfg.preprocessing.min_chars,
        )
    finally:
        fetcher.close()

    summary_path, samples_path = save_rss_audit_artifacts(report, args.out_dir)
    logger.info(
        "RSS Audit completed | feeds=%d | raw_items=%d | unique_urls=%d | dantri_tin_moi_overlap=%d/%d (%.1f%%) | summary=%s | samples=%s",
        report.total_feeds,
        report.total_raw_items,
        report.total_unique_urls_across_feeds,
        report.dantri_tin_moi_nhat_overlap_with_other_dantri,
        report.dantri_tin_moi_nhat_total_urls,
        report.dantri_tin_moi_nhat_overlap_rate * 100.0,
        summary_path,
        samples_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
