"""Phase 2.5 Task 4.2 — RSS Feed & Full-Text Extraction Audit (Pipeline D prep).

Implements Brief #2 §3.2 and DATA_AUDIT.md §2b:
- Loads ``configs/rss.yaml`` with Pydantic validation.
- Respects ``user_agent``, ``request_timeout_s``, ``min_delay_between_requests_s``,
  and ``robots.txt`` (via ``urllib.robotparser.RobotFileParser``).
- Never fetches article pages that are disallowed by ``robots.txt`` (marks ``blocked``).
- Audits HTTP status, content-type, ``feedparser`` items, valid ``pubDate`` (ISO-8601 UTC)
  and ``link`` rates, within-feed / cross-feed / static-KB URL overlap (including
  ``dantri_tin_moi_nhat`` overlap with other Dân Trí feeds).
- Fetches up to ``max_articles_per_feed`` (default 20) articles per feed, extracts full
  text via ``trafilatura``, measures extraction success rate, char length mean/p95,
  empty/noise rate, and residual boilerplate markers, and saves extracted text samples
  (never raw HTML) to ``artifacts/audit_rss/``.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Literal, Protocol
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx
import yaml
from pydantic import BaseModel, Field

from vnfake.config import AppConfig
from vnfake.data.cleaning import is_too_short, normalize_text, strip_boilerplate
from vnfake.utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_RSS_CONFIG_PATH = Path("configs/rss.yaml")
_DEFAULT_AUDIT_OUT_DIR = Path("artifacts/audit_rss")
_DEFAULT_MAX_ARTICLES_PER_FEED = 20
_DEFAULT_MIN_CHARS = 100
_PERCENTILE_P95 = 95.0
_PERCENTILE_DIVISOR = 100.0

_NOISE_URL_PATTERNS: tuple[str, ...] = (
    "/video",
    "/anh/",
    "/photo",
    "/infographic",
    "/podcast",
    "/interactive",
    "/emagazine",
)
_RESIDUAL_BOILERPLATE_MARKERS: tuple[str, ...] = (
    "xem thêm:",
    "đọc thêm:",
    "tin liên quan",
    "bình luận của bạn",
    "ý kiến bạn đọc",
    "chia sẻ bài viết",
)


class RssFetchConfig(BaseModel):
    """HTTP / politeness settings in ``configs/rss.yaml: fetch``."""

    user_agent: str = Field(min_length=1)
    request_timeout_s: float = 15.0
    min_delay_between_requests_s: float = 1.0
    respect_robots_txt: bool = True
    use_conditional_get: bool = True


class RssIngestConfig(BaseModel):
    """Ingest metadata settings in ``configs/rss.yaml: ingest``."""

    max_items_per_feed: int | None = None
    label: Literal["real"] = "real"
    ingest_source: Literal["rss"] = "rss"


class RssFeedEntryConfig(BaseModel):
    """One RSS feed entry in ``configs/rss.yaml: feeds``."""

    id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    category: str = Field(min_length=1)
    url: str = Field(min_length=1)
    enabled: bool = True


class RssConfig(BaseModel):
    """Root schema for ``configs/rss.yaml``."""

    fetch: RssFetchConfig
    ingest: RssIngestConfig = RssIngestConfig()
    feeds: list[RssFeedEntryConfig]


@dataclass(frozen=True, slots=True)
class HttpResponseData:
    """Minimal HTTP response container for testable fetching."""

    status_code: int
    content_type: str
    text: str
    url: str


class HttpFetcherProtocol(Protocol):
    """Injectable HTTP fetcher so unit tests never access the network."""

    def get(self, url: str) -> HttpResponseData: ...


class HttpxPoliteFetcher:
    """Rate-limited ``httpx`` client enforcing per-domain minimum delay and ASCII User-Agent."""

    def __init__(
        self,
        *,
        user_agent: str,
        timeout_s: float,
        min_delay_s: float,
        clock_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        # Ensure ASCII-only header per configs/rss.yaml comment.
        user_agent.encode("ascii")
        self._user_agent = user_agent
        self._timeout_s = timeout_s
        self._min_delay_s = max(0.0, min_delay_s)
        self._clock = clock_fn
        self._sleep = sleep_fn
        self._last_request_by_domain: dict[str, float] = {}
        self._client = httpx.Client(
            headers={"User-Agent": self._user_agent},
            timeout=self._timeout_s,
            follow_redirects=True,
        )

    def close(self) -> None:
        self._client.close()

    def _wait_domain(self, url: str) -> None:
        domain = urlparse(url).netloc.lower()
        if not domain or self._min_delay_s <= 0:
            return
        now = self._clock()
        last = self._last_request_by_domain.get(domain)
        if last is not None:
            elapsed = now - last
            if elapsed < self._min_delay_s:
                self._sleep(self._min_delay_s - elapsed)
        self._last_request_by_domain[domain] = self._clock()

    def get(self, url: str) -> HttpResponseData:
        self._wait_domain(url)
        resp = self._client.get(url)
        ct = resp.headers.get("content-type", "")
        return HttpResponseData(
            status_code=resp.status_code,
            content_type=ct,
            text=resp.text,
            url=str(resp.url),
        )


def load_rss_config(path: Path = _DEFAULT_RSS_CONFIG_PATH) -> RssConfig:
    """Load and validate ``configs/rss.yaml``."""
    if not path.exists():
        raise FileNotFoundError(f"RSS config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return RssConfig.model_validate(raw)


def canonicalize_url(url: str) -> str:
    """Normalize article URL for deduplication (A6).

    Rules:
    1. Scheme must be ``http`` or ``https`` and is unified to ``https`` (lowercased).
    2. Host (netloc) is lowercased, default ports (``:80``, ``:443``) are stripped,
       and leading ``www.`` or ``amp.`` prefixes are removed.
    3. Query parameters (``?...``) and fragments (``#...``) are stripped completely.
    4. Path collapses duplicate slashes, removes trailing ``/``, strips trailing
       ``/amp`` or ``.amp`` suffixes (case-insensitive), and removes trailing ``/`` again.
    """
    raw = url.strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    scheme_raw = parsed.scheme.lower()
    if scheme_raw not in ("http", "https") or not parsed.netloc:
        return ""

    host = parsed.netloc.lower()
    if host.endswith(":80"):
        host = host[:-3]
    elif host.endswith(":443"):
        host = host[:-4]
    for prefix in ("www.", "amp."):
        if host.startswith(prefix):
            host = host[len(prefix) :]
    if not host:
        return ""

    path = re.sub(r"/{2,}", "/", parsed.path or "")
    path = re.sub(r"/+$", "", path)
    path = re.sub(r"(?:/amp|\.amp)$", "", path, flags=re.IGNORECASE)
    path = re.sub(r"/+$", "", path)
    return f"https://{host}{path}"


def parse_pubdate_iso8601_utc(raw_date: str | None) -> str | None:
    """Parse RFC-2822 / RSS ``pubDate`` string into ISO-8601 UTC string."""
    if not raw_date or not raw_date.strip():
        return None
    text = raw_date.strip()
    try:
        dt = parsedate_to_datetime(text)
        dt = dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)
        return dt.isoformat().replace("+00:00", "Z")
    except Exception:
        try:
            dt_iso = datetime.fromisoformat(text.replace("Z", "+00:00"))
            dt_iso = dt_iso.replace(tzinfo=UTC) if dt_iso.tzinfo is None else dt_iso.astimezone(UTC)
            return dt_iso.isoformat().replace("+00:00", "Z")
        except Exception:
            return None


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(float(v) for v in values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (pct / _PERCENTILE_DIVISOR) * (len(sorted_vals) - 1)
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return sorted_vals[lower]
    weight = pos - lower
    return sorted_vals[lower] * (1.0 - weight) + sorted_vals[upper] * weight


class RobotsChecker:
    """Caches ``robots.txt`` per domain and checks URL permission for ``user_agent``."""

    def __init__(
        self, fetcher: HttpFetcherProtocol, user_agent: str, *, respect: bool = True
    ) -> None:
        self._fetcher = fetcher
        self._user_agent = user_agent
        self._respect = respect
        self._parsers: dict[str, RobotFileParser | None] = {}

    def is_allowed(self, url: str) -> tuple[bool, str]:
        """Return ``(allowed, status_note)`` for ``url``."""
        if not self._respect:
            return True, "robots_check_disabled"
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base not in self._parsers:
            robots_url = f"{base}/robots.txt"
            try:
                resp = self._fetcher.get(robots_url)
                if resp.status_code == 200 and resp.text:
                    rp = RobotFileParser()
                    rp.parse(resp.text.splitlines())
                    self._parsers[base] = rp
                elif resp.status_code in (401, 403):
                    self._parsers[base] = None
                    return False, f"blocked_robots_http_{resp.status_code}"
                else:
                    # Standard RFC 9309: 404/missing robots.txt means unrestricted.
                    rp_empty = RobotFileParser()
                    rp_empty.parse([])
                    self._parsers[base] = rp_empty
            except Exception as exc:
                logger.warning("Failed to fetch %s: %s — treating as unrestricted", robots_url, exc)
                rp_fallback = RobotFileParser()
                rp_fallback.parse([])
                self._parsers[base] = rp_fallback

        rp_cached = self._parsers.get(base)
        if rp_cached is None:
            return False, "blocked_robots_disallow_all"
        allowed = rp_cached.can_fetch(self._user_agent, url)
        return (True, "allowed") if allowed else (False, "blocked_by_robots_txt")


def extract_article_text(html: str, url: str | None = None) -> str:
    """Extract main article text from HTML using ``trafilatura``."""
    if not html or not html.strip():
        return ""
    import trafilatura

    extracted = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=False,
        no_fallback=False,
    )
    if not extracted:
        return ""
    return normalize_text(extracted)


def detect_residual_boilerplate(text: str) -> list[str]:
    """Return any residual menu/comment/boilerplate markers found in ``text``."""
    lower = text.lower()
    return [m for m in _RESIDUAL_BOILERPLATE_MARKERS if m in lower]


def is_media_or_noise_url(url: str) -> bool:
    """Check whether URL points to a video/photo/infographic page."""
    lower = url.lower()
    return any(pat in lower for pat in _NOISE_URL_PATTERNS)


@dataclass(frozen=True, slots=True)
class ExtractedArticleSample:
    """Saved sample of an extracted article (stores clean text only, NEVER raw HTML)."""

    feed_id: str
    source: str
    category: str
    url: str
    title: str
    published_date: str | None
    raw_extracted_chars: int
    cleaned_chars: int
    survived_cleaning: bool
    is_media_url: bool
    residual_markers: list[str]
    cleaned_text_preview: str
    cleaned_text: str


@dataclass(frozen=True, slots=True)
class FeedAuditMetrics:
    """Per-feed audit metrics for ``docs/DATA_AUDIT.md`` §2b."""

    feed_id: str
    source: str
    category: str
    url: str
    enabled: bool
    robots_status: str
    http_status: int | None
    content_type: str
    rss_valid: bool
    num_items: int
    valid_pubdate_count: int
    valid_pubdate_rate: float
    valid_link_count: int
    valid_link_rate: float
    within_feed_dup_count: int
    within_feed_dup_rate: float
    cross_feed_dup_count: int
    cross_feed_dup_rate: float
    static_kb_dup_count: int
    static_kb_dup_rate: float
    articles_attempted: int
    articles_blocked_robots: int
    articles_http_ok: int
    fulltext_success_count: int
    fulltext_success_rate: float
    cleaned_survive_count: int
    cleaned_survive_rate: float
    empty_or_noise_count: int
    empty_or_noise_rate: float
    text_mean_chars: float
    text_p95_chars: float
    residual_boilerplate_count: int
    error: str | None = None


@dataclass(frozen=True, slots=True)
class RssAuditReport:
    """Full audit report across all configured RSS feeds."""

    timestamp_utc: str
    total_feeds: int
    total_raw_items: int
    total_unique_urls_across_feeds: int
    static_kb_total_urls: int
    dantri_tin_moi_nhat_total_urls: int
    dantri_tin_moi_nhat_overlap_with_other_dantri: int
    dantri_tin_moi_nhat_overlap_rate: float
    feeds: list[FeedAuditMetrics]
    manual_inspection_samples: list[ExtractedArticleSample]


def load_static_kb_urls(cfg: AppConfig) -> set[str]:
    """Load canonicalized URLs from the static KB JSONL files."""
    urls: set[str] = set()
    for path in (cfg.paths.kb_dantri, cfg.paths.kb_vnexpress):
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    u = canonicalize_url(str(obj.get("url", "")))
                    if u:
                        urls.add(u)
                except json.JSONDecodeError:
                    continue
    return urls


@dataclass(frozen=True, slots=True)
class _RawFeedFetchResult:
    feed_cfg: RssFeedEntryConfig
    robots_status: str
    http_status: int | None
    content_type: str
    rss_valid: bool
    entries: list[dict[str, object]]
    error: str | None


def run_rss_audit(
    rss_cfg: RssConfig,
    *,
    static_kb_urls: set[str],
    fetcher: HttpFetcherProtocol,
    extractor_fn: Callable[[str, str | None], str] = extract_article_text,
    max_articles_per_feed: int = _DEFAULT_MAX_ARTICLES_PER_FEED,
    min_chars: int = _DEFAULT_MIN_CHARS,
) -> RssAuditReport:
    """Execute the full RSS audit across all feeds in ``rss_cfg``."""
    import feedparser

    robots = RobotsChecker(
        fetcher,
        rss_cfg.fetch.user_agent,
        respect=rss_cfg.fetch.respect_robots_txt,
    )

    # Pass 1: Fetch all RSS feeds and collect item lists for cross-feed URL overlap.
    raw_feed_data: list[_RawFeedFetchResult] = []
    urls_by_feed: dict[str, list[str]] = {}
    url_feed_membership: dict[str, set[str]] = {}

    for feed_cfg in rss_cfg.feeds:
        allowed_feed, robots_note = robots.is_allowed(feed_cfg.url)
        if not allowed_feed:
            logger.warning("Feed %s blocked by robots.txt (%s)", feed_cfg.id, robots_note)
            raw_feed_data.append(
                _RawFeedFetchResult(
                    feed_cfg=feed_cfg,
                    robots_status=robots_note,
                    http_status=None,
                    content_type="",
                    rss_valid=False,
                    entries=[],
                    error=f"Blocked by robots.txt: {robots_note}",
                )
            )
            urls_by_feed[feed_cfg.id] = []
            continue

        try:
            resp = fetcher.get(feed_cfg.url)
        except Exception as exc:
            logger.warning("Failed to fetch RSS feed %s (%s): %s", feed_cfg.id, feed_cfg.url, exc)
            raw_feed_data.append(
                _RawFeedFetchResult(
                    feed_cfg=feed_cfg,
                    robots_status=robots_note,
                    http_status=None,
                    content_type="",
                    rss_valid=False,
                    entries=[],
                    error=str(exc),
                )
            )
            urls_by_feed[feed_cfg.id] = []
            continue

        parsed = feedparser.parse(resp.text)
        entries = [dict(e) for e in (getattr(parsed, "entries", []) or [])]
        rss_valid = resp.status_code == 200 and (
            len(entries) > 0 or not getattr(parsed, "bozo", False)
        )

        canon_urls: list[str] = []
        for entry in entries:
            raw_link = str(entry.get("link", "") or "").strip()
            cu = canonicalize_url(raw_link)
            if cu:
                canon_urls.append(cu)
                url_feed_membership.setdefault(cu, set()).add(feed_cfg.id)

        urls_by_feed[feed_cfg.id] = canon_urls
        raw_feed_data.append(
            _RawFeedFetchResult(
                feed_cfg=feed_cfg,
                robots_status=robots_note,
                http_status=resp.status_code,
                content_type=resp.content_type,
                rss_valid=rss_valid,
                entries=entries,
                error=None if resp.status_code == 200 else f"HTTP {resp.status_code}",
            )
        )

    # Compute dantri_tin_moi_nhat overlap with other Dân Trí feeds.
    tin_moi_urls = set(urls_by_feed.get("dantri_tin_moi_nhat", []))
    other_dantri_urls: set[str] = set()
    for f_cfg in rss_cfg.feeds:
        if f_cfg.source == "dantri" and f_cfg.id != "dantri_tin_moi_nhat":
            other_dantri_urls.update(urls_by_feed.get(f_cfg.id, []))
    tin_moi_overlap = len(tin_moi_urls & other_dantri_urls)
    tin_moi_overlap_rate = (tin_moi_overlap / len(tin_moi_urls)) if tin_moi_urls else 0.0

    # Pass 2: Per-feed item metrics + full-text extraction on up to max_articles_per_feed items.
    feed_metrics_list: list[FeedAuditMetrics] = []
    all_samples: list[ExtractedArticleSample] = []
    total_raw_items = 0

    for item_data in raw_feed_data:
        feed_cfg = item_data.feed_cfg
        entries = item_data.entries
        robots_status = item_data.robots_status
        http_status = item_data.http_status
        content_type = item_data.content_type
        rss_valid = item_data.rss_valid
        err_msg = item_data.error

        num_items = len(entries)
        total_raw_items += num_items

        valid_pubdate_count = 0
        valid_link_count = 0
        seen_in_feed: set[str] = set()
        within_feed_dup_count = 0
        unique_entries_for_fetch: list[tuple[str, str, str | None]] = []

        for entry in entries:
            raw_pub = str(entry.get("published", "") or entry.get("updated", "") or "").strip()
            pub_iso = parse_pubdate_iso8601_utc(raw_pub)
            if pub_iso is not None:
                valid_pubdate_count += 1

            raw_link = str(entry.get("link", "") or "").strip()
            cu = canonicalize_url(raw_link)
            if cu:
                valid_link_count += 1
                if cu in seen_in_feed:
                    within_feed_dup_count += 1
                else:
                    seen_in_feed.add(cu)
                    title = normalize_text(str(entry.get("title", "") or ""))
                    unique_entries_for_fetch.append((cu, title, pub_iso))

        unique_count = len(seen_in_feed)
        cross_feed_dup_count = sum(
            1 for u in seen_in_feed if len(url_feed_membership.get(u, set())) > 1
        )
        static_kb_dup_count = sum(1 for u in seen_in_feed if u in static_kb_urls)

        # Fetch up to max_articles_per_feed unique articles for full-text audit.
        to_fetch = unique_entries_for_fetch[: max(0, max_articles_per_feed)]
        articles_attempted = 0
        articles_blocked_robots = 0
        articles_http_ok = 0
        fulltext_success_count = 0
        cleaned_survive_count = 0
        empty_or_noise_count = 0
        residual_boilerplate_count = 0
        cleaned_lengths: list[float] = []

        for art_url, art_title, art_pub_iso in to_fetch:
            art_allowed, _ = robots.is_allowed(art_url)
            if not art_allowed:
                articles_blocked_robots += 1
                continue

            articles_attempted += 1
            try:
                art_resp = fetcher.get(art_url)
            except Exception as exc:
                logger.warning("Article fetch failed (%s): %s", art_url, exc)
                empty_or_noise_count += 1
                continue

            if art_resp.status_code != 200 or not art_resp.text:
                empty_or_noise_count += 1
                continue

            articles_http_ok += 1
            extracted_raw = extractor_fn(art_resp.text, art_url)
            raw_len = len(extracted_raw)
            cleaned = strip_boilerplate(normalize_text(extracted_raw)) if extracted_raw else ""
            cleaned_len = len(cleaned)
            survived = bool(cleaned) and not is_too_short(cleaned, min_chars)
            media_flag = is_media_or_noise_url(art_url)
            residuals = detect_residual_boilerplate(cleaned) if cleaned else []

            if raw_len > 0:
                fulltext_success_count += 1
            if survived:
                cleaned_survive_count += 1
                cleaned_lengths.append(float(cleaned_len))
            else:
                empty_or_noise_count += 1

            if residuals:
                residual_boilerplate_count += 1

            all_samples.append(
                ExtractedArticleSample(
                    feed_id=feed_cfg.id,
                    source=feed_cfg.source,
                    category=feed_cfg.category,
                    url=art_url,
                    title=art_title,
                    published_date=art_pub_iso,
                    raw_extracted_chars=raw_len,
                    cleaned_chars=cleaned_len,
                    survived_cleaning=survived,
                    is_media_url=media_flag,
                    residual_markers=residuals,
                    cleaned_text_preview=cleaned[:300] if cleaned else "",
                    cleaned_text=cleaned,
                )
            )

        denom_items = max(1, num_items)
        denom_unique = max(1, unique_count)
        denom_attempted = max(1, articles_attempted)

        feed_metrics_list.append(
            FeedAuditMetrics(
                feed_id=feed_cfg.id,
                source=feed_cfg.source,
                category=feed_cfg.category,
                url=feed_cfg.url,
                enabled=feed_cfg.enabled,
                robots_status=robots_status,
                http_status=http_status,
                content_type=content_type,
                rss_valid=rss_valid,
                num_items=num_items,
                valid_pubdate_count=valid_pubdate_count,
                valid_pubdate_rate=valid_pubdate_count / denom_items if num_items else 0.0,
                valid_link_count=valid_link_count,
                valid_link_rate=valid_link_count / denom_items if num_items else 0.0,
                within_feed_dup_count=within_feed_dup_count,
                within_feed_dup_rate=within_feed_dup_count / denom_items if num_items else 0.0,
                cross_feed_dup_count=cross_feed_dup_count,
                cross_feed_dup_rate=cross_feed_dup_count / denom_unique if unique_count else 0.0,
                static_kb_dup_count=static_kb_dup_count,
                static_kb_dup_rate=static_kb_dup_count / denom_unique if unique_count else 0.0,
                articles_attempted=articles_attempted,
                articles_blocked_robots=articles_blocked_robots,
                articles_http_ok=articles_http_ok,
                fulltext_success_count=fulltext_success_count,
                fulltext_success_rate=(
                    fulltext_success_count / denom_attempted if articles_attempted else 0.0
                ),
                cleaned_survive_count=cleaned_survive_count,
                cleaned_survive_rate=(
                    cleaned_survive_count / denom_attempted if articles_attempted else 0.0
                ),
                empty_or_noise_count=empty_or_noise_count,
                empty_or_noise_rate=(
                    empty_or_noise_count / denom_attempted if articles_attempted else 0.0
                ),
                text_mean_chars=(
                    sum(cleaned_lengths) / len(cleaned_lengths) if cleaned_lengths else 0.0
                ),
                text_p95_chars=_percentile(cleaned_lengths, _PERCENTILE_P95),
                residual_boilerplate_count=residual_boilerplate_count,
                error=err_msg,
            )
        )

    # Select up to 10 survived samples per source (vnexpress, dantri) for manual spot-check.
    spot_check_samples: list[ExtractedArticleSample] = []
    for src_name in ("vnexpress", "dantri"):
        src_samples = [s for s in all_samples if s.source == src_name and s.survived_cleaning]
        spot_check_samples.extend(src_samples[:10])

    return RssAuditReport(
        timestamp_utc=datetime.now(UTC).isoformat(),
        total_feeds=len(rss_cfg.feeds),
        total_raw_items=total_raw_items,
        total_unique_urls_across_feeds=len(url_feed_membership),
        static_kb_total_urls=len(static_kb_urls),
        dantri_tin_moi_nhat_total_urls=len(tin_moi_urls),
        dantri_tin_moi_nhat_overlap_with_other_dantri=tin_moi_overlap,
        dantri_tin_moi_nhat_overlap_rate=tin_moi_overlap_rate,
        feeds=feed_metrics_list,
        manual_inspection_samples=spot_check_samples,
    )


def save_rss_audit_artifacts(
    report: RssAuditReport, out_dir: Path = _DEFAULT_AUDIT_OUT_DIR
) -> tuple[Path, Path]:
    """Save audit summary and manual spot-check samples (cleaned text only, no raw HTML)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "rss_audit_summary.json"
    samples_path = out_dir / "manual_spotcheck_20_articles.json"

    summary_dict = asdict(report)
    # Keep summary JSON compact by storing manual samples in their dedicated file.
    samples_data = summary_dict.pop("manual_inspection_samples", [])

    summary_path.write_text(
        json.dumps(summary_dict, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    samples_path.write_text(
        json.dumps(samples_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Saved RSS audit summary to %s and samples to %s", summary_path, samples_path)
    return summary_path, samples_path
