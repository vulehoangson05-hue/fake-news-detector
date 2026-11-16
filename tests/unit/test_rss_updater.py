"""Unit tests for vnfake.data.rss_updater (mock HTTP, no network calls)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from vnfake.config import load_config
from vnfake.data.rss_audit import HttpResponseData, load_rss_config
from vnfake.data.rss_updater import RssUpdater

_FEED_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Dân Trí - Giáo dục</title>
    <item>
      <title>Học sinh đạt giải quốc tế về khoa học kỹ thuật</title>
      <link>https://dantri.com.vn/giao-duc/hoc-sinh-dat-giai.htm</link>
      <pubDate>Sun, 08 Nov 2026 10:00:00 +0700</pubDate>
    </item>
  </channel>
</rss>
"""

_ARTICLE_HTML = """<!DOCTYPE html>
<html>
<head><title>Học sinh đạt giải quốc tế về khoa học kỹ thuật</title></head>
<body>
<article>
Học sinh đạt giải quốc tế về khoa học kỹ thuật
Hà NộiNhóm học sinh trường THPT Chuyên đã xuất sắc giành huy chương vàng tại cuộc thi sáng tạo quốc tế.
Đề tài nghiên cứu ứng dụng trí tuệ nhân tạo trong chẩn đoán sớm bệnh lý được ban giám khảo đánh giá rất cao.
Các em đã nỗ lực làm việc trong suốt sáu tháng qua để hoàn thiện sản phẩm mẫu và kiểm nghiệm thực tế.
</article>
</body>
</html>
"""


class _MockFetcher:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get(self, url: str) -> HttpResponseData:
        self.calls.append(url)
        if "robots.txt" in url:
            return HttpResponseData(
                status_code=200,
                content_type="text/plain",
                text="User-agent: *\nAllow: /\n",
                url=url,
            )
        if "rss" in url or "feed" in url or url.endswith(".rss"):
            return HttpResponseData(
                status_code=200,
                content_type="application/rss+xml",
                text=_FEED_XML,
                url=url,
            )
        return HttpResponseData(
            status_code=200,
            content_type="text/html",
            text=_ARTICLE_HTML,
            url=url,
        )


def _mock_extractor(html: str, url: str) -> str:
    return (
        "Học sinh đạt giải quốc tế về khoa học kỹ thuật\n"
        "Hà NộiNhóm học sinh trường THPT Chuyên đã xuất sắc giành huy chương vàng tại cuộc thi sáng tạo quốc tế. "
        "Đề tài nghiên cứu ứng dụng trí tuệ nhân tạo trong chẩn đoán sớm bệnh lý được ban giám khảo đánh giá rất cao. "
        "Các em đã nỗ lực làm việc trong suốt sáu tháng qua để hoàn thiện sản phẩm mẫu và kiểm nghiệm thực tế."
    )


def test_rss_updater_idempotent_two_runs(tmp_path: Path) -> None:
    manifest_path = tmp_path / "rss_manifest.json"
    static_kb_path = tmp_path / "static_kb.jsonl"
    static_kb_path.write_text("", encoding="utf-8")

    # Minimal single feed config
    rss_yaml_content = """
fetch:
  user_agent: "VNFakeTestBot/1.0"
  request_timeout_s: 5.0
  min_delay_between_requests_s: 0.0
  respect_robots_txt: false
  use_conditional_get: false
ingest:
  label: "real"
  ingest_source: "rss"
feeds:
  - id: "test_feed"
    source: "dantri"
    category: "giao-duc"
    url: "https://dantri.com.vn/giao-duc.rss"
    enabled: true
"""
    rss_cfg_file = tmp_path / "rss.yaml"
    rss_cfg_file.write_text(rss_yaml_content, encoding="utf-8")
    rss_cfg = load_rss_config(rss_cfg_file)

    mock_store = MagicMock()
    mock_embedder = MagicMock()
    mock_embedder.embed_dense.return_value = [[0.1] * 1024]
    mock_embedder.embed_sparse.return_value = [MagicMock(indices=[1, 2], values=[0.5, 0.8])]

    cfg = load_config(Path("configs/base.yaml"))
    fetcher = _MockFetcher()

    updater = RssUpdater(
        cfg,
        rss_cfg,
        store=mock_store,
        embedder=mock_embedder,
        fetcher=fetcher,
        extractor_fn=_mock_extractor,
        manifest_path=manifest_path,
        static_kb_path=static_kb_path,
    )

    # Run 1: Ingests 1 new article
    stats1 = updater.update("test_collection")
    assert stats1.articles_cleaned == 1
    assert stats1.chunks_upserted >= 1
    assert mock_store.upsert_chunks.call_count == 1
    assert manifest_path.exists()

    # Run 2: Same feed -> 0 duplicates ingested!
    stats2 = updater.update("test_collection")
    assert stats2.articles_skipped_previously_ingested == 1
    assert stats2.articles_cleaned == 0
    assert stats2.chunks_upserted == 0
    # Store upsert was NOT called a second time
    assert mock_store.upsert_chunks.call_count == 1
