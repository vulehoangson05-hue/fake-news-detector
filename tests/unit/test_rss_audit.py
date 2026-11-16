"""Unit tests for Phase 2.5 Task 4.2: RSS audit logic (mocked HTTP, no network)."""

from __future__ import annotations

from pathlib import Path

from vnfake.data.rss_audit import (
    HttpResponseData,
    canonicalize_url,
    load_rss_config,
    parse_pubdate_iso8601_utc,
    run_rss_audit,
)

_SAMPLE_RSS_FEED_1 = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Dân Trí - Sức khỏe</title>
    <item>
      <title>Bài báo số 1 về sức khỏe cộng đồng</title>
      <link>https://dantri.com.vn/suc-khoe/bai-1.htm</link>
      <pubDate>Sun, 08 Nov 2026 08:30:00 +0700</pubDate>
    </item>
    <item>
      <title>Bài báo số 2 bị trùng trong cùng feed</title>
      <link>https://dantri.com.vn/suc-khoe/bai-1.htm/</link>
      <pubDate>Sun, 08 Nov 2026 08:30:00 +0700</pubDate>
    </item>
    <item>
      <title>Bài báo số 3 trùng với KB tĩnh</title>
      <link>https://dantri.com.vn/suc-khoe/bai-tinh.htm</link>
      <pubDate>Sun, 08 Nov 2026 09:00:00 +0700</pubDate>
    </item>
  </channel>
</rss>
"""

_SAMPLE_RSS_TIN_MOI = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Dân Trí - Tin mới nhất</title>
    <item>
      <title>Bài báo số 1 về sức khỏe cộng đồng</title>
      <link>https://dantri.com.vn/suc-khoe/bai-1.htm</link>
      <pubDate>Sun, 08 Nov 2026 08:30:00 +0700</pubDate>
    </item>
    <item>
      <title>Bài báo bị chặn bởi robots.txt</title>
      <link>https://dantri.com.vn/private/bai-chan.htm</link>
      <pubDate>Sun, 08 Nov 2026 10:00:00 +0700</pubDate>
    </item>
  </channel>
</rss>
"""


class _FakeFetcher:
    def __init__(self, responses: dict[str, HttpResponseData]) -> None:
        self._responses = responses
        self.requested_urls: list[str] = []

    def get(self, url: str) -> HttpResponseData:
        self.requested_urls.append(url)
        if url in self._responses:
            return self._responses[url]
        return HttpResponseData(status_code=404, content_type="text/plain", text="", url=url)


def test_load_rss_config_has_13_feeds() -> None:
    cfg = load_rss_config(Path("configs/rss.yaml"))
    assert len(cfg.feeds) == 13
    assert sum(1 for f in cfg.feeds if f.source == "vnexpress") == 8
    assert sum(1 for f in cfg.feeds if f.source == "dantri") == 5
    # ASCII header check
    cfg.fetch.user_agent.encode("ascii")


def test_parse_pubdate_and_canonicalize_url() -> None:
    iso = parse_pubdate_iso8601_utc("Sun, 08 Nov 2026 08:30:00 +0700")
    assert iso == "2026-11-08T01:30:00Z"
    assert parse_pubdate_iso8601_utc("invalid-date") is None

    u1 = canonicalize_url("https://DanTri.com.vn/suc-khoe/bai-1.htm/")
    u2 = canonicalize_url("https://dantri.com.vn/suc-khoe/bai-1.htm?utm_source=rss#top")
    u3 = canonicalize_url("http://www.dantri.com.vn/suc-khoe/bai-1.htm/amp/")
    u4 = canonicalize_url("https://amp.dantri.com.vn/suc-khoe/bai-1.htm.amp")
    assert u1 == u2 == u3 == u4 == "https://dantri.com.vn/suc-khoe/bai-1.htm"


def test_run_rss_audit_respects_robots_and_computes_overlap() -> None:
    rss_cfg = load_rss_config(Path("configs/rss.yaml"))
    # Keep only 2 feeds for the unit test
    rss_cfg = rss_cfg.model_copy(
        update={
            "feeds": [
                f for f in rss_cfg.feeds if f.id in ("dantri_suc_khoe", "dantri_tin_moi_nhat")
            ]
        }
    )

    robots_txt = "User-agent: *\nDisallow: /private/\n"
    long_article_text = (
        "Bộ Y tế vừa ban hành hướng dẫn mới về phòng chống dịch bệnh mùa đông xuân trên toàn quốc. "
        * 4
    )
    responses = {
        "https://dantri.com.vn/robots.txt": HttpResponseData(
            200, "text/plain", robots_txt, "https://dantri.com.vn/robots.txt"
        ),
        "https://dantri.com.vn/rss/suc-khoe.rss": HttpResponseData(
            200, "application/rss+xml", _SAMPLE_RSS_FEED_1, "https://dantri.com.vn/rss/suc-khoe.rss"
        ),
        "https://dantri.com.vn/rss/tin-moi-nhat.rss": HttpResponseData(
            200,
            "application/rss+xml",
            _SAMPLE_RSS_TIN_MOI,
            "https://dantri.com.vn/rss/tin-moi-nhat.rss",
        ),
        "https://dantri.com.vn/suc-khoe/bai-1.htm": HttpResponseData(
            200,
            "text/html",
            f"<html><body><p>{long_article_text}</p></body></html>",
            "https://dantri.com.vn/suc-khoe/bai-1.htm",
        ),
        "https://dantri.com.vn/suc-khoe/bai-tinh.htm": HttpResponseData(
            200,
            "text/html",
            f"<html><body><p>{long_article_text}</p></body></html>",
            "https://dantri.com.vn/suc-khoe/bai-tinh.htm",
        ),
    }
    fetcher = _FakeFetcher(responses)
    static_urls = {"https://dantri.com.vn/suc-khoe/bai-tinh.htm"}

    report = run_rss_audit(
        rss_cfg,
        static_kb_urls=static_urls,
        fetcher=fetcher,
        extractor_fn=lambda html, _: long_article_text if "body" in html else "",
        max_articles_per_feed=5,
        min_chars=100,
    )

    assert report.total_feeds == 2
    assert report.dantri_tin_moi_nhat_total_urls == 2
    assert report.dantri_tin_moi_nhat_overlap_with_other_dantri == 1
    assert report.dantri_tin_moi_nhat_overlap_rate == 0.5

    # Verify disallowed article URL was NEVER requested
    assert "https://dantri.com.vn/private/bai-chan.htm" not in fetcher.requested_urls

    by_id = {f.feed_id: f for f in report.feeds}
    sk = by_id["dantri_suc_khoe"]
    assert sk.num_items == 3
    assert sk.within_feed_dup_count == 1
    assert sk.cross_feed_dup_count == 1
    assert sk.static_kb_dup_count == 1
    assert sk.cleaned_survive_count == 2

    tm = by_id["dantri_tin_moi_nhat"]
    assert tm.articles_blocked_robots == 1
    assert tm.articles_attempted == 1
