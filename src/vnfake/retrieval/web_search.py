"""Tavily Web Search client, rule-based query builder, and spike evaluator (Brief #4 §3).

Enforces:
- ``TAVILY_API_KEY`` read strictly from environment / ``.env``; never logged.
- Disk caching of every raw Tavily API response under ``cfg.cache_dir``.
- Hard credit ceiling ``max_credits_spike`` (default 150) tracked on disk.
- Strict verification that every returned URL belongs to ``include_domains``.
- Polite page fetching (respecting ``robots.txt`` and ``rate_limit_s <= 1 req/s``)
  with ``trafilatura`` when Tavily ``raw_content`` is absent or shorter than 400 chars.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import random
import re
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx
import trafilatura
from dotenv import load_dotenv

from vnfake.config import SearchConfig
from vnfake.data.cleaning import normalize_text
from vnfake.data.segmentation import SegmenterProtocol, get_segmenter
from vnfake.retrieval.evidence_selector import (
    SourcedEvidenceResult,
    compute_3gram_containment,
    select_evidence_with_sources,
)
from vnfake.schemas.vifactcheck import ViFactCheckSample
from vnfake.utils.logging import get_logger

logger = get_logger(__name__)

_TAVILY_SEARCH_URL = "https://api.tavily.com/search"
_MAX_QUERY_CHARS = 400
_MIN_RAW_CONTENT_CHARS = 400
_MS_PER_SECOND = 1000.0

# High-frequency Vietnamese functional stopwords removed in Q1 rule-based query
_VI_STOPWORDS: frozenset[str] = frozenset(
    {
        "là",
        "và",
        "của",
        "có",
        "những",
        "các",
        "được",
        "trong",
        "cho",
        "với",
        "không",
        "đã",
        "này",
        "để",
        "khi",
        "về",
        "tại",
        "theo",
        "từ",
        "như",
        "sẽ",
        "một",
        "người",
        "ra",
        "trên",
        "đến",
        "vào",
        "còn",
        "nhưng",
        "vì",
        "do",
        "bị",
        "cũng",
        "rất",
        "nhiều",
        "hơn",
        "nếu",
        "mà",
        "thì",
        "rằng",
        "việc",
        "bằng",
        "qua",
        "lại",
        "đang",
        "sau",
        "trước",
        "nên",
        "hay",
        "hoặc",
        "cả",
        "đây",
        "đó",
        "vừa",
        "mới",
    }
)


@dataclass(frozen=True, slots=True)
class WebSearchHit:
    """One search result item returned by Tavily."""

    title: str
    url: str
    domain: str
    snippet: str
    full_text: str
    text_source: Literal["tavily_raw_content", "trafilatura_fetch", "tavily_snippet"]
    score: float


@dataclass(frozen=True, slots=True)
class WebSearchResult:
    """Structured result for one search query."""

    query: str
    variant: str
    hits: list[WebSearchHit]
    latency_ms: float
    from_cache: bool
    credits_used: int
    fetch_ms: float = 0.0


class TavilyTransportProtocol(Protocol):
    """Protocol for mocking Tavily HTTP calls in unit tests."""

    def search_post(self, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]: ...


class HttpxTavilyTransport:
    """Production ``httpx`` transport for Tavily REST API."""

    def search_post(self, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        with httpx.Client(timeout=timeout_s) as client:
            resp = client.post(_TAVILY_SEARCH_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, dict):
                raise ValueError("Tavily response is not a JSON object.")
            return data


def build_query_q0(statement: str, max_chars: int = _MAX_QUERY_CHARS) -> str:
    """Build Q0 query: NFC-normalized claim truncated cleanly to ``max_chars``."""
    clean = normalize_text(statement)
    if len(clean) <= max_chars:
        return clean
    cut = clean[:max_chars]
    last_space = cut.rfind(" ")
    return cut[:last_space].strip() if last_space > max_chars // 2 else cut.strip()


def build_query_q1(statement: str, max_chars: int = _MAX_QUERY_CHARS) -> str:
    """Build Q1 rule-based query: keep proper nouns, numbers, dates, and content words."""
    clean = normalize_text(statement)
    raw_tokens = re.findall(r"\d+[/-]\d+(?:[/-]\d+)?|\d+[.,]\d+|\w+", clean, flags=re.UNICODE)
    kept: list[str] = []
    seen_lower: set[str] = set()

    for tok in raw_tokens:
        low = tok.lower()
        is_cap_or_num = any(ch.isupper() or ch.isdigit() for ch in tok)
        if not is_cap_or_num and low in _VI_STOPWORDS:
            continue
        # Avoid consecutive duplicate content words while keeping numbers/proper nouns
        if not is_cap_or_num and low in seen_lower and len(kept) >= 8:
            continue
        seen_lower.add(low)
        kept.append(tok)

    result = " ".join(kept) if kept else clean
    return build_query_q0(result, max_chars=max_chars)


# Negation words MUST be preserved in Q1v2 (Brief #5 §11.6 & §12.5)
_VI_NEGATION_WORDS: frozenset[str] = frozenset(
    {
        "không",
        "chưa",
        "chẳng",
        "chả",
        "đừng",
        "chớ",
        "không_thể",
        "chưa_từng",
        "chưa_bao_giờ",
        "không_bao_giờ",
        "không_phải",
        "chẳng_hề",
        "không_hề",
    }
)

# Functional word-level stopwords for Q1v2 (strictly excluding negations and content nouns like "người", "việc")
_VI_STOPWORDS_V2: frozenset[str] = frozenset(
    {
        "là",
        "và",
        "của",
        "những",
        "các",
        "được",
        "trong",
        "cho",
        "với",
        "đã",
        "này",
        "để",
        "khi",
        "về",
        "tại",
        "theo",
        "từ",
        "như",
        "sẽ",
        "một",
        "trên",
        "đến",
        "vào",
        "còn",
        "nhưng",
        "vì",
        "do",
        "bị",
        "cũng",
        "rất",
        "nhiều",
        "hơn",
        "nếu",
        "mà",
        "thì",
        "rằng",
        "bằng",
        "qua",
        "lại",
        "đang",
        "sau",
        "trước",
        "nên",
        "hay",
        "hoặc",
        "cả",
        "đây",
        "đó",
        "vừa",
        "mới",
        # 2-syllable narrative / connective stopwords (Brief #5 §4.1 & Task 4.4)
        "cho_biết",
        "tỏ_ra",
        "về_việc",
        "đối_với",
        "liên_quan",
        "cùng_với",
        "thông_qua",
        "nhằm_mục_đích",
    }
)

_2SYLLABLE_STOPWORD_PAIRS: frozenset[tuple[str, str]] = frozenset(
    {
        ("cho", "biết"),
        ("tỏ", "ra"),
        ("về", "việc"),
        ("đối", "với"),
        ("liên", "quan"),
        ("cùng", "với"),
        ("thông", "qua"),
        ("nhằm", "mục_đích"),
        ("nhằm", "mục"),
    }
)


def build_keyword_query_v2(
    statement: str,
    *,
    segmenter: SegmenterProtocol | None = None,
    max_chars: int = _MAX_QUERY_CHARS,
) -> str:
    """Build Q1v2 keyword query: word-segment first, filter word-level stopwords, keep compounds & negations.

    Fixes Q1 bugs (Brief #5 §4.1, §11.6):
    - Segments Vietnamese words before filtering so compound nouns/verbs (``Chính_phủ``,
      ``cơ_sở_dữ_liệu``, ``làm_việc``) are never broken at syllable level.
    - Preserves negation words (``không``, ``chưa``, ``chẳng``, ``chả``, ``đừng``, ...).
    - Preserves common nouns such as ``người`` (e.g. ``người dùng``).
    """
    clean = normalize_text(statement)
    if not clean:
        return ""

    seg = segmenter or get_segmenter("underthesea")
    segmented = seg.segment(clean)

    raw_tokens = re.findall(r"\d+[/-]\d+(?:[/-]\d+)?|\d+[.,]\d+|\w+", segmented, flags=re.UNICODE)
    kept: list[str] = []
    seen_lower: set[str] = set()

    idx = 0
    while idx < len(raw_tokens):
        tok = raw_tokens[idx]
        low = tok.lower()
        is_neg = low in _VI_NEGATION_WORDS
        is_num = any(ch.isdigit() for ch in tok)
        is_cap = any(ch.isupper() for ch in tok)

        # Check consecutive 2-token narrative/connective stopwords (e.g. "cho biết", "về việc")
        if idx + 1 < len(raw_tokens):
            next_tok = raw_tokens[idx + 1]
            next_low = next_tok.lower()
            if (low, next_low) in _2SYLLABLE_STOPWORD_PAIRS or f"{low}_{next_low}" in _VI_STOPWORDS_V2:
                idx += 2
                continue

        # Remove functional stopwords:
        # 1. 2-syllable narrative/connective stopwords (e.g. "cho_biết", "đối_với", "tỏ_ra")
        # 2. Standalone single-word stopwords (e.g. "là", "và"), unless capitalized proper nouns
        if not is_neg and not is_num:
            norm_stop = low if "_" in low else low.replace(" ", "_")
            if norm_stop in _VI_STOPWORDS_V2 and ("_" in tok or idx == 0 or not is_cap):
                idx += 1
                continue

        # Light deduplication on repeated non-compound stopwords/generics only if query is long
        if not is_neg and not is_cap and not is_num and low in seen_lower and len(kept) >= 12:
            idx += 1
            continue

        seen_lower.add(low)
        kept.append(tok.replace("_", " "))
        idx += 1

    result = re.sub(r"\s+", " ", " ".join(kept)).strip()
    if not result:
        result = clean
    return build_query_q0(result, max_chars=max_chars)


build_query_q1v2 = build_keyword_query_v2


def extract_domain(url: str) -> str:
    """Extract normalized lowercase hostname (without leading ``www.``) from ``url``."""
    host = (urlparse(url).hostname or "").lower().strip()
    if host.startswith("www."):
        host = host[4:]
    return host


def is_allowed_domain(url: str, include_domains: list[str]) -> bool:
    """Return True iff ``url`` belongs to one of ``include_domains`` (or its subdomain)."""
    host = extract_domain(url)
    if not host:
        return False
    for allowed in include_domains:
        dom = allowed.lower().strip()
        if dom.startswith("www."):
            dom = dom[4:]
        if host == dom or host.endswith("." + dom):
            return True
    return False


def _compute_search_cache_key(
    query: str,
    cfg: SearchConfig,
) -> str:
    domains_sorted = ",".join(sorted(d.lower().strip() for d in cfg.include_domains))
    payload = (
        f"{normalize_text(query)}\n{cfg.provider}\n{cfg.search_depth}\n"
        f"{cfg.max_results}\n{cfg.topic}\n{domains_sorted}"
    ).encode()
    return hashlib.sha256(payload).hexdigest()


class TavilyCreditTracker:
    """Tracks Tavily API credits on disk and enforces ``max_credits_spike`` ceiling."""

    def __init__(self, cache_dir: Path, max_credits: int = 150) -> None:
        self._cache_dir = cache_dir
        self._max_credits = max_credits
        self._file = cache_dir / "credits_usage.json"

    def current_credits(self) -> int:
        if not self._file.exists():
            return 0
        try:
            data = json.loads(self._file.read_text(encoding="utf-8"))
            return int(data.get("credits_used", 0))
        except Exception:
            return 0

    def check_can_spend(self, cost: int = 1) -> None:
        used = self.current_credits()
        if used + cost > self._max_credits:
            raise RuntimeError(
                f"Tavily credit spike ceiling reached: used={used}, requested={cost}, "
                f"max_credits_spike={self._max_credits}. Halting per Brief #4 §5."
            )

    def record_spend(self, query: str, cost: int = 1) -> int:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        used = self.current_credits() + cost
        history: list[dict[str, object]] = []
        if self._file.exists():
            with contextlib.suppress(Exception):
                raw = json.loads(self._file.read_text(encoding="utf-8"))
                if isinstance(raw, dict) and isinstance(raw.get("calls"), list):
                    history = raw["calls"]
        history.append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "cost": cost,
                "query_preview": query[:80],
            }
        )
        payload = {
            "credits_used": used,
            "max_credits_spike": self._max_credits,
            "calls": history,
        }
        self._file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return used


class PoliteArticleFetcher:
    """Polite web page text extractor with ``robots.txt`` check, per-domain rate limit, and disk cache."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        user_agent: str,
        rate_limit_s: float = 1.0,
        timeout_s: float = 10.0,
    ) -> None:
        self._pages_dir = cache_dir / "pages"
        self._user_agent = user_agent
        self._rate_limit_s = max(1.0, rate_limit_s)
        self._timeout_s = timeout_s
        self._domain_last_fetch: dict[str, float] = {}
        self._lock = threading.Lock()
        self._robots_cache: dict[str, RobotFileParser | None] = {}

    def read_cached_text(self, url: str) -> str | None:
        """Read cached extracted page text from disk without making network requests."""
        url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()
        cache_path = self._pages_dir / f"{url_hash}.json"
        if cache_path.exists():
            with contextlib.suppress(Exception):
                obj = json.loads(cache_path.read_text(encoding="utf-8"))
                return str(obj.get("text", ""))
        return None

    def _is_allowed_by_robots(self, url: str) -> bool:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        with self._lock:
            cached = base in self._robots_cache
            rp_cached = self._robots_cache.get(base)

        if not cached:
            rp = RobotFileParser()
            robots_url = f"{base}/robots.txt"
            try:
                with httpx.Client(
                    timeout=5.0,
                    headers={"User-Agent": self._user_agent},
                    follow_redirects=True,
                ) as client:
                    resp = client.get(robots_url)
                    if resp.status_code == 200:
                        rp.parse(resp.text.splitlines())
                        rp_cached = rp
                    else:
                        rp_cached = None
            except Exception:
                rp_cached = None
            with self._lock:
                self._robots_cache[base] = rp_cached

        if rp_cached is None:
            return True
        return rp_cached.can_fetch(self._user_agent, url)

    def fetch_text(self, url: str) -> str:
        """Fetch and extract article text via ``trafilatura``, caching to disk."""
        cached_text = self.read_cached_text(url)
        if cached_text is not None:
            return cached_text

        if not self._is_allowed_by_robots(url):
            logger.warning("robots.txt disallows fetching URL: %s", url)
            return ""

        domain = extract_domain(url) or urlparse(url).netloc
        wait_s = 0.0
        with self._lock:
            last_ts = self._domain_last_fetch.get(domain, 0.0)
            now = time.monotonic()
            if last_ts > 0.0 and (now - last_ts) < self._rate_limit_s:
                wait_s = self._rate_limit_s - (now - last_ts)
                self._domain_last_fetch[domain] = last_ts + self._rate_limit_s
            else:
                self._domain_last_fetch[domain] = now

        if wait_s > 0.0:
            time.sleep(wait_s)

        extracted = ""
        try:
            with httpx.Client(
                timeout=self._timeout_s,
                headers={"User-Agent": self._user_agent},
                follow_redirects=True,
            ) as client:
                resp = client.get(url)
                if resp.status_code == 200 and resp.text:
                    extracted = normalize_text(trafilatura.extract(resp.text) or "")
        except Exception as exc:
            logger.warning("Polite fetch failed for %s: %s", url, exc)

        url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()
        cache_path = self._pages_dir / f"{url_hash}.json"
        with self._lock:
            self._pages_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps({"url": url, "text": extracted}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return extracted


class TavilyWebSearcher:
    """Tavily REST API client with disk cache, credit guard, and domain verification."""

    def __init__(
        self,
        cfg: SearchConfig,
        *,
        api_key: str | None = None,
        transport: TavilyTransportProtocol | None = None,
        page_fetcher: Callable[[str], str] | None = None,
    ) -> None:
        self._cfg = cfg
        self._api_key = api_key
        self._transport = transport or HttpxTavilyTransport()
        self._credit_tracker = TavilyCreditTracker(cfg.cache_dir, max_credits=cfg.max_credits_spike)
        self._default_fetcher = PoliteArticleFetcher(
            cfg.cache_dir,
            user_agent=cfg.user_agent,
            rate_limit_s=cfg.rate_limit_s,
            timeout_s=cfg.timeout_s,
        )
        self._custom_page_fetcher = page_fetcher
        self._fetch_page = page_fetcher or self._default_fetcher.fetch_text

    @property
    def credits_used(self) -> int:
        return self._credit_tracker.current_credits()

    def _resolve_api_key(self) -> str:
        load_dotenv(override=False)
        key = (self._api_key or os.getenv("TAVILY_API_KEY") or "").strip()
        if not key:
            raise RuntimeError("Missing TAVILY_API_KEY in environment / .env.")
        return key

    def _build_hits_from_raw(
        self,
        raw_data: dict[str, Any],
        *,
        max_results: int | None = None,
        allow_network_fetch: bool = True,
        max_parallel_fetch: int = 3,
    ) -> tuple[list[WebSearchHit], float]:
        results_list = raw_data.get("results", [])
        if not isinstance(results_list, list):
            results_list = []

        limit = max_results if max_results is not None else self._cfg.max_results
        parsed_items: list[tuple[str, str, str, str, float]] = []
        urls_to_fetch: list[str] = []

        for item in results_list[:limit]:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            if not is_allowed_domain(url, self._cfg.include_domains):
                raise ValueError(f"Returned URL {url!r} violates include_domains whitelist!")
            title = normalize_text(str(item.get("title") or ""))
            snippet = normalize_text(str(item.get("content") or ""))
            raw_content = normalize_text(str(item.get("raw_content") or ""))
            score = float(item.get("score") or 0.0)
            parsed_items.append((url, title, snippet, raw_content, score))
            if len(raw_content) < _MIN_RAW_CONTENT_CHARS:
                urls_to_fetch.append(url)

        fetch_t0 = time.perf_counter()
        fetched_by_url: dict[str, str] = {}
        if urls_to_fetch:
            if not allow_network_fetch and self._custom_page_fetcher is None:
                for u in urls_to_fetch:
                    fetched_by_url[u] = self._default_fetcher.read_cached_text(u) or ""
            elif len(urls_to_fetch) > 1 and max_parallel_fetch > 1:
                workers = min(max_parallel_fetch, len(urls_to_fetch))
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = {u: pool.submit(self._fetch_page, u) for u in urls_to_fetch}
                    for u, fut in futures.items():
                        try:
                            fetched_by_url[u] = fut.result()
                        except Exception as exc:
                            logger.warning("Parallel page fetch failed for %s: %s", u, exc)
                            fetched_by_url[u] = ""
            else:
                for u in urls_to_fetch:
                    fetched_by_url[u] = self._fetch_page(u)
        fetch_ms = (time.perf_counter() - fetch_t0) * _MS_PER_SECOND

        hits: list[WebSearchHit] = []
        for url, title, snippet, raw_content, score in parsed_items:
            if len(raw_content) >= _MIN_RAW_CONTENT_CHARS:
                full_text = f"{title}\n{snippet}\n{raw_content}".strip()
                text_source: Literal[
                    "tavily_raw_content", "trafilatura_fetch", "tavily_snippet"
                ] = "tavily_raw_content"
            else:
                fetched = fetched_by_url.get(url, "")
                if fetched:
                    full_text = f"{title}\n{snippet}\n{fetched}".strip()
                    text_source = "trafilatura_fetch"
                else:
                    full_text = f"{title}\n{snippet}\n{raw_content}".strip()
                    text_source = "tavily_snippet"

            hits.append(
                WebSearchHit(
                    title=title,
                    url=url,
                    domain=extract_domain(url),
                    snippet=snippet,
                    full_text=full_text,
                    text_source=text_source,
                    score=score,
                )
            )
        return hits, round(fetch_ms, 2)

    def search_cached_only(
        self,
        query: str,
        *,
        variant: str = "Q0",
        max_results: int | None = None,
    ) -> WebSearchResult | None:
        """Read Tavily result strictly from disk cache (0 credits, 0 network calls)."""
        clean_q = build_query_q0(query, max_chars=_MAX_QUERY_CHARS)
        cache_key = _compute_search_cache_key(clean_q, self._cfg)
        raw_cache_path = self._cfg.cache_dir / "raw" / f"{cache_key}.json"
        if not raw_cache_path.exists():
            return None

        t0 = time.perf_counter()
        raw_data = json.loads(raw_cache_path.read_text(encoding="utf-8"))
        latency_ms = float(raw_data.get("_cached_latency_ms", 0.0))
        hits, fetch_ms = self._build_hits_from_raw(
            raw_data,
            max_results=max_results,
            allow_network_fetch=False,
        )
        if latency_ms == 0.0:
            latency_ms = (time.perf_counter() - t0) * _MS_PER_SECOND
        return WebSearchResult(
            query=clean_q,
            variant=variant,
            hits=hits,
            latency_ms=round(latency_ms, 2),
            from_cache=True,
            credits_used=self.credits_used,
            fetch_ms=fetch_ms,
        )

    def search(
        self,
        query: str,
        *,
        variant: str = "Q0",
        max_results: int | None = None,
    ) -> WebSearchResult:
        """Search Tavily (or read from disk cache) and extract full text for top results."""
        clean_q = build_query_q0(query, max_chars=_MAX_QUERY_CHARS)
        cache_key = _compute_search_cache_key(clean_q, self._cfg)
        raw_cache_path = self._cfg.cache_dir / "raw" / f"{cache_key}.json"

        t0 = time.perf_counter()
        from_cache = False

        if raw_cache_path.exists():
            raw_data = json.loads(raw_cache_path.read_text(encoding="utf-8"))
            latency_ms = float(raw_data.get("_cached_latency_ms", 0.0))
            from_cache = True
        else:
            credit_cost = 1 if self._cfg.search_depth == "basic" else 2
            self._credit_tracker.check_can_spend(credit_cost)
            api_key = self._resolve_api_key()
            payload: dict[str, Any] = {
                "api_key": api_key,
                "query": clean_q,
                "search_depth": self._cfg.search_depth,
                "topic": self._cfg.topic,
                "max_results": self._cfg.max_results,
                "include_domains": list(self._cfg.include_domains),
                "include_raw_content": True,
            }
            call_t0 = time.perf_counter()
            raw_data = self._transport.search_post(payload, timeout_s=self._cfg.timeout_s)
            latency_ms = (time.perf_counter() - call_t0) * _MS_PER_SECOND
            self._credit_tracker.record_spend(clean_q, cost=credit_cost)

            raw_cache_path.parent.mkdir(parents=True, exist_ok=True)
            to_save = dict(raw_data)
            to_save["_cached_latency_ms"] = round(latency_ms, 2)
            to_save["_cached_at"] = datetime.now(UTC).isoformat()
            raw_cache_path.write_text(
                json.dumps(to_save, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        hits, fetch_ms = self._build_hits_from_raw(
            raw_data,
            max_results=max_results,
            allow_network_fetch=True,
            max_parallel_fetch=3,
        )

        if from_cache and latency_ms == 0.0:
            latency_ms = (time.perf_counter() - t0) * _MS_PER_SECOND

        return WebSearchResult(
            query=clean_q,
            variant=variant,
            hits=hits,
            latency_ms=round(latency_ms, 2),
            from_cache=from_cache,
            credits_used=self.credits_used,
            fetch_ms=fetch_ms,
        )


def select_evidence_from_web_hits(
    statement: str,
    hits: Sequence[WebSearchHit],
    *,
    top_k: int = 3,
    max_tokens: int = 150,
) -> SourcedEvidenceResult:
    """Run unified Phase 3 BM25 selector over combined sentences from top web search hits.

    Each sentence retains ``{text, url, domain}`` of its originating search result.
    """
    sourced_contexts = [
        (h.full_text, h.url, h.domain) for h in hits if h.full_text and h.full_text.strip()
    ]
    return select_evidence_with_sources(
        statement,
        sourced_contexts,
        top_k=top_k,
        max_tokens=max_tokens,
    )


def sample_stratified_dev_claims(
    samples: list[ViFactCheckSample],
    *,
    per_class: int = 10,
    seed: int = 42,
) -> list[tuple[str, ViFactCheckSample]]:
    """Sample ``per_class`` claims per label (30 total) deterministically with ``seed``."""
    by_label: dict[int, list[tuple[str, ViFactCheckSample]]] = {0: [], 1: [], 2: []}
    for idx, s in enumerate(samples):
        claim_id = f"tavily_dev_{idx:03d}"
        by_label[s.labels].append((claim_id, s))

    rng = random.Random(seed)
    chosen: list[tuple[str, ViFactCheckSample]] = []
    for lbl in (0, 1, 2):
        pool = list(by_label[lbl])
        rng.shuffle(pool)
        chosen.extend(pool[:per_class])

    chosen.sort(key=lambda x: x[0])
    return chosen


def _percentile(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    pos = (q / 100.0) * (len(s) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return s[lo]
    return s[lo] * (1.0 - (pos - lo)) + s[hi] * (pos - lo)


def run_tavily_spike(
    claims: list[tuple[str, ViFactCheckSample]],
    searcher: TavilyWebSearcher,
    label_map: dict[int, str],
) -> dict[str, Any]:
    """Run Q0 and Q1 queries across the 30 stratified dev claims and compute spike metrics."""
    variants = ("Q0", "Q1")
    all_latencies: list[float] = []
    variant_reports: dict[str, Any] = {}
    text_source_counts: dict[str, int] = {
        "tavily_raw_content": 0,
        "trafilatura_fetch": 0,
        "tavily_snippet": 0,
    }
    total_urls_verified = 0

    for var in variants:
        hits_05 = 0
        hits_03 = 0
        class_totals: dict[str, int] = {label_map[k]: 0 for k in sorted(label_map)}
        class_hits_05: dict[str, int] = {label_map[k]: 0 for k in sorted(label_map)}
        class_hits_03: dict[str, int] = {label_map[k]: 0 for k in sorted(label_map)}
        claim_details: list[dict[str, Any]] = []

        for claim_id, sample in claims:
            cls_name = label_map[sample.labels]
            class_totals[cls_name] += 1
            query_str = (
                build_query_q0(sample.statement)
                if var == "Q0"
                else build_query_q1(sample.statement)
            )
            res = searcher.search(query_str, variant=var)
            all_latencies.append(res.latency_ms)

            best_containment = 0.0
            best_url = ""
            best_title = ""
            best_source = ""
            for h in res.hits:
                total_urls_verified += 1
                text_source_counts[h.text_source] = text_source_counts.get(h.text_source, 0) + 1
                c = compute_3gram_containment(sample.evidence, h.full_text)
                if c > best_containment:
                    best_containment = c
                    best_url = h.url
                    best_title = h.title
                    best_source = h.text_source

            if not best_url and res.hits:
                best_url = res.hits[0].url
                best_title = res.hits[0].title
                best_source = res.hits[0].text_source

            if best_containment >= 0.5:
                hits_05 += 1
                class_hits_05[cls_name] += 1
            if best_containment >= 0.3:
                hits_03 += 1
                class_hits_03[cls_name] += 1

            claim_details.append(
                {
                    "claim_id": claim_id,
                    "label": cls_name,
                    "statement": sample.statement,
                    "gold_evidence": sample.evidence,
                    "query": res.query,
                    "num_results": len(res.hits),
                    "best_containment_3gram": round(best_containment, 4),
                    "hit_ge_05": best_containment >= 0.5,
                    "hit_ge_03": best_containment >= 0.3,
                    "best_url": best_url,
                    "best_title": best_title,
                    "best_text_source": best_source,
                    "latency_ms": res.latency_ms,
                    "top_urls": [h.url for h in res.hits],
                }
            )

        n = max(len(claims), 1)
        hit_examples = [d for d in claim_details if d["hit_ge_05"]][:5]
        if len(hit_examples) < 5:
            extra_hits = [
                d
                for d in sorted(
                    claim_details,
                    key=lambda x: -float(x["best_containment_3gram"]),
                )
                if d not in hit_examples and float(d["best_containment_3gram"]) > 0.0
            ]
            hit_examples.extend(extra_hits[: 5 - len(hit_examples)])

        miss_examples = [
            d
            for d in sorted(claim_details, key=lambda x: float(x["best_containment_3gram"]))
            if not d["hit_ge_05"]
        ][:5]

        variant_reports[var] = {
            "num_claims": len(claims),
            "hits_ge_05": hits_05,
            "pct_ge_05": round((hits_05 / n) * 100.0, 2),
            "hits_ge_03": hits_03,
            "pct_ge_03": round((hits_03 / n) * 100.0, 2),
            "per_class_ge_05": {
                k: f"{class_hits_05[k]}/{class_totals[k]} ({class_hits_05[k] / max(class_totals[k], 1) * 100:.1f}%)"
                for k in class_totals
            },
            "per_class_ge_03": {
                k: f"{class_hits_03[k]}/{class_totals[k]} ({class_hits_03[k] / max(class_totals[k], 1) * 100:.1f}%)"
                for k in class_totals
            },
            "hit_5_examples": hit_examples,
            "miss_5_examples": miss_examples,
            "claims": claim_details,
        }

    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "num_claims": len(claims),
        "credits_used": searcher.credits_used,
        "max_credits_spike": searcher._cfg.max_credits_spike,
        "latency_p50_ms": round(_percentile(all_latencies, 50.0), 2),
        "latency_p95_ms": round(_percentile(all_latencies, 95.0), 2),
        "total_urls_verified": total_urls_verified,
        "all_urls_in_whitelist": True,
        "text_source_counts": text_source_counts,
        "variants": variant_reports,
    }
