"""Phase 2.5 — LLM Query Rewrite module (Gemini Flash family only).

Implements ARCHITECTURE.md §5.3.1 and copilot-instructions.md §6.2-§6.3:
- ``QueryRewriter`` Protocol, ``NoOpQueryRewriter``, and ``GeminiQueryRewriter``.
- Lazy ``google-genai`` client creation (strictly isolated to this module).
- Startup health-check via ``models.get`` (fail-safe to NoOp unless ``required=True``).
- Disk cache keyed by ``sha256(claim_nfc + prompt_version + model)``; caches ONLY ``ok`` results.
- Client-side rate limiter (sliding 60-second RPM window) + bounded exponential backoff retry.
- Pure ``sanitize_keywords()`` enforcing single-line, no-markdown, no-chatter Vietnamese keywords.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import time
import unicodedata
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from vnfake.config import QueryRewriteConfig
from vnfake.data.cleaning import normalize_text
from vnfake.utils.logging import get_logger

logger = get_logger(__name__)

REQUIRED_PROMPT_RULE = (
    "Tuyệt đối CHỈ in ra từ khóa. KHÔNG chào hỏi, KHÔNG dạ vâng, KHÔNG giải thích."
)

RewriteStatus = Literal[
    "ok",
    "cache_hit",
    "disabled",
    "fallback:timeout",
    "fallback:rate_limit",
    "fallback:quota_exhausted",
    "fallback:api_error",
    "fallback:invalid_output",
]

_MS_PER_SECOND = 1000.0
_SECONDS_PER_MINUTE = 60.0
_BACKOFF_BASE_SECONDS = 1.0
_MAX_ACCEPTABLE_RAW_LINES = 3

_MARKDOWN_TOKENS_RE = re.compile(r"(\*\*|__|``|`|\*)")
_HEADER_OR_BULLET_RE = re.compile(r"^(?:#+\s*|[-*•+]\s+|\d+[.)]\s+)+")
_PREFIX_LABEL_RE = re.compile(
    r"^(?:từ\s*khóa|từ\s*khoá|keywords?|kết\s*quả|output|trả\s*lời)\s*[:\-\u2013\u2014]\s*",
    flags=re.IGNORECASE,
)
_SEPARATOR_RE = re.compile(r"[,;|/]+")
_MULTI_DIGIT_RE = re.compile(r"\d{2,}")
_SURROUNDING_QUOTES = "\"'\u201c\u201d\u2018\u2019\u00ab\u00bb"


@dataclass(frozen=True, slots=True)
class RewriteResult:
    """Output of a query rewrite attempt."""

    text: str
    status: RewriteStatus
    latency_ms: float
    raw_output: str | None = None


@runtime_checkable
class QueryRewriter(Protocol):
    """Protocol for query rewrite implementations."""

    def rewrite(self, claim: str) -> RewriteResult: ...


class LowLevelGeminiClient(Protocol):
    """Injectable low-level client interface so unit tests never touch the network."""

    def check_model(self, model_name: str) -> None: ...

    def generate(
        self,
        *,
        model_name: str,
        system_instruction: str,
        user_content: str,
        temperature: float,
        max_output_tokens: int,
        timeout_s: float,
    ) -> str: ...


class _GoogleGenAIClient:
    """Production wrapper around ``google.genai.Client`` (created only inside this module)."""

    def __init__(self, api_key: str) -> None:
        from google import genai

        self._client = genai.Client(api_key=api_key)

    def check_model(self, model_name: str) -> None:
        self._client.models.get(model=model_name)

    def generate(
        self,
        *,
        model_name: str,
        system_instruction: str,
        user_content: str,
        temperature: float,
        max_output_tokens: int,
        timeout_s: float,
    ) -> str:
        from google.genai import types

        timeout_ms = max(1, int(timeout_s * _MS_PER_SECOND))
        http_opts = types.HttpOptions(timeout=timeout_ms)
        cfg = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            http_options=http_opts,
        )
        resp = self._client.models.generate_content(
            model=model_name,
            contents=user_content,
            config=cfg,
        )
        return resp.text if resp.text is not None else ""


def _redact_api_key(message: str, api_key: str | None = None) -> str:
    """Redact API keys from error strings before logging."""
    out = message
    if api_key and len(api_key) >= 4:
        out = out.replace(api_key, "[REDACTED_API_KEY]")
    out = re.sub(r"([?&]key=)[^&\s\"']+", r"\1[REDACTED_API_KEY]", out, flags=re.IGNORECASE)
    out = re.sub(r"(AIza[0-9A-Za-z_-]{20,})", "[REDACTED_API_KEY]", out)
    return out


def load_prompt_template(path: Path) -> tuple[str, str, str]:
    """Load and validate ``configs/prompts/query_rewrite.txt``.

    Splits the prompt file into ``(system_instruction, user_template, prompt_sha256)``
    where ``system_instruction`` contains the strict rule and ``user_template`` contains
    the few-shot examples and ``<claim_data>{claim}</claim_data>`` delimiter.
    """
    if not path.exists():
        raise FileNotFoundError(f"Query rewrite prompt file not found: {path}")
    raw = path.read_text(encoding="utf-8")
    prompt_sha256 = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    lines = [ln for ln in raw.splitlines() if not ln.strip().startswith("#")]
    body = "\n".join(lines).strip()
    if REQUIRED_PROMPT_RULE not in body:
        raise ValueError(
            f"Prompt template {path} is missing required rule: {REQUIRED_PROMPT_RULE!r}"
        )
    if "<claim_data>" not in body or "{claim}" not in body:
        raise ValueError(
            f"Prompt template {path} must wrap {{claim}} inside <claim_data>...</claim_data>"
        )
    return REQUIRED_PROMPT_RULE, body, prompt_sha256


def sanitize_keywords(
    raw: str,
    *,
    source_claim: str | None = None,
    max_keywords: int = 15,
    max_chars: int = 256,
    forbidden_phrases: Sequence[str] = (),
    drop_terms: Sequence[str] = (),
) -> str | None:
    """Sanitize raw LLM output into a single-line space-separated keyword string.

    Pure function per Brief #2 §2.2.4, Brief #3 A5.3 & A9.2, and copilot-instructions §6.3:
    1. Unicode NFC normalization.
    2. Reject empty output, multi-line poems/lists (>= 4 lines), or any output
       containing ``forbidden_phrases`` anywhere.
    3. Take the first non-empty line; reject if longer than ``max_chars``.
    4. Strip markdown tokens, bullets, surrounding quotes, and prefixes like ``Từ khóa:``.
    5. Strip any colloquial/relative-time terms listed in ``drop_terms`` (case-insensitive).
    6. Replace punctuation separators (``,``, ``;``, ``|``) with spaces, collapse whitespace.
    7. If ``source_claim`` is provided, drop any token containing a multi-digit number
       (``\\d{2,}``) that does not appear in ``source_claim`` (A5.3 alien-number filter).
    8. Deduplicate keywords/tokens while preserving original first appearance order (A9.2).
    9. Truncate to ``max_keywords`` words and return ``None`` if invalid/empty.
    """
    if not raw:
        return None

    nfc = unicodedata.normalize("NFC", raw).strip()
    if not nfc:
        return None

    lower_full = nfc.lower()
    for phrase in forbidden_phrases:
        norm_phrase = unicodedata.normalize("NFC", phrase).lower()
        if norm_phrase and norm_phrase in lower_full:
            return None

    non_empty_lines = [ln.strip() for ln in nfc.splitlines() if ln.strip()]
    if not non_empty_lines or len(non_empty_lines) > _MAX_ACCEPTABLE_RAW_LINES:
        return None

    line = non_empty_lines[0]
    if len(line) > max_chars:
        return None

    # Strip markdown formatting, leading headers/bullets, quotes, and prefixes iteratively.
    line = _MARKDOWN_TOKENS_RE.sub("", line).strip()
    line = line.strip(_SURROUNDING_QUOTES).strip()
    line = _HEADER_OR_BULLET_RE.sub("", line).strip()
    line = line.strip(_SURROUNDING_QUOTES).strip()

    while True:
        updated = _PREFIX_LABEL_RE.sub("", line).strip()
        updated = updated.strip(_SURROUNDING_QUOTES).strip()
        updated = _HEADER_OR_BULLET_RE.sub("", updated).strip()
        if updated == line:
            break
        line = updated

    if not line or len(line) > max_chars:
        return None

    lower_line = line.lower()
    for phrase in forbidden_phrases:
        norm_phrase = unicodedata.normalize("NFC", phrase).lower()
        if norm_phrase and norm_phrase in lower_line:
            return None

    # Strip colloquial or relative-time terms configured in drop_terms
    for term in drop_terms:
        norm_term = unicodedata.normalize("NFC", term).strip()
        if norm_term:
            line = re.sub(
                rf"(?i)(?:\b|\s|^){re.escape(norm_term)}(?:\b|\s|$)",
                " ",
                line,
            )

    # Deduplicate keyword segments if separated by punctuation ([,;|\n])
    raw_segments = [s.strip() for s in re.split(r"[,;|]+", line) if s.strip()]
    seen_segments: set[str] = set()
    deduped_segments: list[str] = []
    for seg in raw_segments:
        norm_seg = unicodedata.normalize("NFC", seg).lower()
        if norm_seg not in seen_segments:
            seen_segments.add(norm_seg)
            deduped_segments.append(seg)
    line = " ".join(deduped_segments)

    line = _SEPARATOR_RE.sub(" ", line)
    words = [w.strip(_SURROUNDING_QUOTES + ".,:;!?()[]{}") for w in line.split()]
    words = [w for w in words if w]
    if not words:
        return None

    if source_claim is not None:
        claim_nfc = unicodedata.normalize("NFC", source_claim)
        allowed_numbers = set(_MULTI_DIGIT_RE.findall(claim_nfc))
        filtered_words: list[str] = []
        for word in words:
            word_numbers = _MULTI_DIGIT_RE.findall(word)
            if any(num not in allowed_numbers for num in word_numbers):
                continue
            filtered_words.append(word)
        words = filtered_words
        if not words:
            return None

    # Deduplicate consecutive identical word tokens (e.g. "tăng tăng" -> "tăng")
    deduped_words: list[str] = []
    for w in words:
        if not deduped_words or w.lower() != deduped_words[-1].lower():
            deduped_words.append(w)
    words = deduped_words

    if not words:
        return None

    if max_keywords > 0:
        words = words[:max_keywords]

    result = " ".join(words).strip()
    if not result or not any(ch.isalnum() for ch in result):
        return None
    return result


def compute_cache_key(
    claim_nfc: str,
    prompt_version: str,
    model: str,
    prompt_sha256: str = "",
) -> str:
    """Deterministic SHA-256 disk-cache key over ``(claim_nfc, prompt_version, prompt_sha256, model)``."""
    payload = f"{claim_nfc}\n{prompt_version}\n{prompt_sha256}\n{model}".encode()
    return hashlib.sha256(payload).hexdigest()


class DailyQuotaTracker:
    """Tracks real daily requests and enforces circuit-breaker on daily quota exhaustion (C0)."""

    def __init__(
        self,
        cache_dir: Path,
        budget: int = 80,
        timezone_str: str = "US/Pacific",
    ) -> None:
        self._cache_dir = cache_dir
        self._budget = max(1, budget)
        self._tz_str = timezone_str
        self._circuit_broken = False

    def _current_day(self) -> str:
        try:
            from zoneinfo import ZoneInfo

            return datetime.now(ZoneInfo(self._tz_str)).strftime("%Y-%m-%d")
        except Exception:
            return datetime.now(UTC).strftime("%Y-%m-%d")

    def _usage_file(self) -> Path:
        return self._cache_dir / f"usage_{self._current_day()}.json"

    def can_request(self) -> bool:
        if self._circuit_broken:
            return False
        p = self._usage_file()
        if not p.exists():
            return True
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return int(data.get("request_count", 0)) < self._budget
        except Exception:
            return True

    def record_request(self, note: str = "") -> int:
        day = self._current_day()
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        p = self._usage_file()
        data: dict[str, object] = {
            "date": day,
            "timezone": self._tz_str,
            "budget": self._budget,
            "request_count": 0,
            "calls": [],
        }
        if p.exists():
            with contextlib.suppress(Exception):
                raw = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    data = raw
        raw_count = data.get("request_count", 0)
        count = int(raw_count) if isinstance(raw_count, int | float | str) else 0
        count += 1
        data["request_count"] = count
        calls = data.get("calls")
        if isinstance(calls, list):
            calls.append({"timestamp": datetime.now(UTC).isoformat(), "note": note})
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return count

    def trip_circuit_breaker(self) -> None:
        self._circuit_broken = True


class SlidingWindowRateLimiter:
    """Client-side RPM rate limiter over a 60-second sliding window."""

    def __init__(
        self,
        rpm_limit: int,
        *,
        clock_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._rpm_limit = max(1, rpm_limit)
        self._clock = clock_fn
        self._sleep = sleep_fn
        self._timestamps: deque[float] = deque()

    def acquire(self) -> float:
        """Wait if necessary to stay within ``rpm_limit``; returns seconds slept."""
        now = self._clock()
        while self._timestamps and (now - self._timestamps[0]) >= _SECONDS_PER_MINUTE:
            self._timestamps.popleft()

        slept = 0.0
        if len(self._timestamps) >= self._rpm_limit:
            wait_s = _SECONDS_PER_MINUTE - (now - self._timestamps[0])
            if wait_s > 0:
                self._sleep(wait_s)
                slept = wait_s
                now = self._clock()
                while self._timestamps and (now - self._timestamps[0]) >= _SECONDS_PER_MINUTE:
                    self._timestamps.popleft()

        self._timestamps.append(self._clock())
        return slept


def _classify_exception(exc: BaseException) -> RewriteStatus:
    """Classify an API exception into timeout, rate_limit, quota_exhausted, or api_error."""
    if isinstance(exc, TimeoutError):
        return "fallback:timeout"
    cls_name = type(exc).__name__.lower()
    msg = str(exc).lower()
    combined = f"{cls_name} {msg}"
    if any(tok in combined for tok in ("timeout", "timed out", "deadline", "readtimeout")):
        return "fallback:timeout"
    if any(
        tok in combined
        for tok in (
            "daily quota",
            "perday",
            "daily_requests",
            "quota_exhausted",
            "exceeded daily",
            "free tier daily",
        )
    ):
        return "fallback:quota_exhausted"
    if any(
        tok in combined
        for tok in (
            "429",
            "rate limit",
            "ratelimit",
            "resource_exhausted",
            "resourceexhausted",
            "quota",
        )
    ):
        return "fallback:rate_limit"
    return "fallback:api_error"


class NoOpQueryRewriter:
    """Pass-through rewriter used when ``query_rewrite.enabled`` is ``False``."""

    def __init__(self, *, status: RewriteStatus = "disabled") -> None:
        self._status: RewriteStatus = status

    def rewrite(self, claim: str) -> RewriteResult:
        return RewriteResult(
            text=normalize_text(claim),
            status=self._status,
            latency_ms=0.0,
            raw_output=None,
        )


class GeminiQueryRewriter:
    """Gemini Flash query rewriter with disk cache, rate limiting, and fail-safe fallback."""

    def __init__(
        self,
        cfg: QueryRewriteConfig,
        *,
        client: LowLevelGeminiClient | None = None,
        api_key: str | None = None,
        clock_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
        online: bool = False,
    ) -> None:
        self._cfg = cfg
        self._client = client
        self._is_mock_client = client is not None
        self._api_key = api_key
        self._clock = clock_fn
        self._sleep = sleep_fn
        self._online = online
        self._rate_limiter = SlidingWindowRateLimiter(
            cfg.rpm_limit, clock_fn=clock_fn, sleep_fn=sleep_fn
        )
        self._quota_tracker = DailyQuotaTracker(
            cfg.cache_dir,
            budget=cfg.daily_request_budget,
            timezone_str=cfg.quota_reset_tz,
        )
        self._system_instruction, self._user_template, self._prompt_sha256 = load_prompt_template(
            cfg.prompt_path
        )
        self._health_checked = False
        self._unhealthy = False

    def _get_or_init_client(self) -> LowLevelGeminiClient:
        if self._client is None:
            resolved_key = (self._api_key or os.getenv("GEMINI_API_KEY") or "").strip()
            if not resolved_key:
                raise RuntimeError("Missing GEMINI_API_KEY in environment.")
            self._api_key = resolved_key
            self._client = _GoogleGenAIClient(resolved_key)
        return self._client

    def _ensure_healthy(self) -> bool:
        """Run one-time startup health-check via ``models.get``."""
        if self._health_checked:
            return not self._unhealthy
        self._health_checked = True
        try:
            client = self._get_or_init_client()
            client.check_model(self._cfg.model)
            logger.info("Gemini health-check OK for model %s", self._cfg.model)
            return True
        except Exception as exc:
            safe_err = _redact_api_key(str(exc), self._api_key)
            if self._cfg.required:
                raise RuntimeError(
                    f"Gemini health-check failed for {self._cfg.model} (required=True): {safe_err}"
                ) from exc
            logger.warning(
                "Gemini health-check failed for %s (%s); falling back to original claims.",
                self._cfg.model,
                safe_err,
            )
            self._unhealthy = True
            return False

    def _cache_file(self, key: str) -> Path:
        return self._cfg.cache_dir / f"{key}.json"

    def _read_cache(self, key: str) -> tuple[str, str | None] | None:
        path = self._cache_file(key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            cached_sha = str(data.get("prompt_sha256", "")).strip()
            if cached_sha and cached_sha != self._prompt_sha256:
                return None
            text = str(data.get("text", "")).strip()
            raw_output = data.get("raw_output")
            if text:
                return text, (str(raw_output) if raw_output is not None else None)
        except Exception as exc:
            logger.warning("Ignoring corrupted rewrite cache file %s: %s", path, exc)
        return None

    def _write_cache(self, key: str, claim_nfc: str, text: str, raw_output: str) -> None:
        path = self._cache_file(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "claim": claim_nfc,
            "text": text,
            "raw_output": raw_output,
            "model": self._cfg.model,
            "prompt_version": self._cfg.prompt_version,
            "prompt_sha256": self._prompt_sha256,
            "created_at": datetime.now(UTC).isoformat(),
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def rewrite(self, claim: str, *, online: bool | None = None) -> RewriteResult:
        """Rewrite ``claim`` into newspaper search keywords, falling back safely on any error."""
        t0 = time.perf_counter()
        deadline_clock_start = self._clock()
        is_online = self._online if online is None else online
        claim_nfc = normalize_text(claim)
        if not self._cfg.enabled:
            return RewriteResult(
                text=claim_nfc,
                status="disabled",
                latency_ms=0.0,
                raw_output=None,
            )

        cache_key = compute_cache_key(
            claim_nfc,
            self._cfg.prompt_version,
            self._cfg.model,
            self._prompt_sha256,
        )
        cached = self._read_cache(cache_key)
        if cached is not None:
            cached_text, cached_raw = cached
            latency_ms = (time.perf_counter() - t0) * _MS_PER_SECOND
            return RewriteResult(
                text=cached_text,
                status="cache_hit",
                latency_ms=latency_ms,
                raw_output=cached_raw,
            )

        if not self._quota_tracker.can_request():
            latency_ms = (time.perf_counter() - t0) * _MS_PER_SECOND
            logger.warning(
                "Daily quota limit reached (%d req/day) or circuit broken; status=fallback:quota_exhausted",
                self._cfg.daily_request_budget,
            )
            return RewriteResult(
                text=claim_nfc,
                status="fallback:quota_exhausted",
                latency_ms=latency_ms,
                raw_output=None,
            )

        if not self._ensure_healthy():
            latency_ms = (time.perf_counter() - t0) * _MS_PER_SECOND
            logger.warning("rewrite_status=fallback:api_error (model unhealthy)")
            return RewriteResult(
                text=claim_nfc,
                status="fallback:api_error",
                latency_ms=latency_ms,
                raw_output=None,
            )

        client = self._get_or_init_client()
        user_content = self._user_template.replace("{claim}", claim_nfc)
        last_status: RewriteStatus = "fallback:api_error"
        effective_retries = self._cfg.max_retries_online if is_online else self._cfg.max_retries

        for attempt in range(effective_retries + 1):
            if (self._clock() - deadline_clock_start) >= self._cfg.total_deadline_s:
                last_status = "fallback:timeout"
                logger.warning(
                    "Rewrite total_deadline_s (%.2fs) exceeded before attempt %d; aborting.",
                    self._cfg.total_deadline_s,
                    attempt + 1,
                )
                break

            if attempt > 0:
                backoff_s = _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                if (self._clock() - deadline_clock_start + backoff_s) >= self._cfg.total_deadline_s:
                    last_status = "fallback:timeout"
                    logger.warning(
                        "Rewrite total_deadline_s (%.2fs) would be exceeded by backoff; aborting.",
                        self._cfg.total_deadline_s,
                    )
                    break
                self._sleep(backoff_s)

            self._rate_limiter.acquire()
            if (self._clock() - deadline_clock_start) >= self._cfg.total_deadline_s:
                last_status = "fallback:timeout"
                logger.warning(
                    "Rewrite total_deadline_s (%.2fs) exceeded after rate-limiter wait; aborting.",
                    self._cfg.total_deadline_s,
                )
                break

            remaining_s = max(
                0.1, self._cfg.total_deadline_s - (self._clock() - deadline_clock_start)
            )
            call_timeout_s = min(self._cfg.timeout_s, remaining_s)

            if not self._is_mock_client:
                self._quota_tracker.record_request(note=f"claim={claim_nfc[:40]}")
            try:
                raw_output = client.generate(
                    model_name=self._cfg.model,
                    system_instruction=self._system_instruction,
                    user_content=user_content,
                    temperature=self._cfg.temperature,
                    max_output_tokens=self._cfg.max_output_tokens,
                    timeout_s=call_timeout_s,
                )
            except Exception as exc:
                last_status = _classify_exception(exc)
                safe_err = _redact_api_key(str(exc), self._api_key)
                logger.warning(
                    "Gemini rewrite attempt %d/%d failed (%s): %s",
                    attempt + 1,
                    effective_retries + 1,
                    last_status,
                    safe_err,
                )
                if last_status == "fallback:quota_exhausted":
                    self._quota_tracker.trip_circuit_breaker()
                    logger.error(
                        "Circuit breaker tripped due to 429 quota exhaustion; halting retries."
                    )
                    break
                if is_online and last_status == "fallback:timeout":
                    logger.warning("Online mode: skipping retry on timeout (A19).")
                    break
                continue

            sanitized = sanitize_keywords(
                raw_output,
                source_claim=claim_nfc,
                max_keywords=self._cfg.max_keywords,
                max_chars=self._cfg.max_chars,
                forbidden_phrases=self._cfg.forbidden_phrases,
                drop_terms=self._cfg.drop_terms,
            )
            latency_ms = (time.perf_counter() - t0) * _MS_PER_SECOND
            if sanitized is None:
                logger.warning(
                    "rewrite_status=fallback:invalid_output | raw=%r",
                    raw_output[:120] if raw_output else "",
                )
                return RewriteResult(
                    text=claim_nfc,
                    status="fallback:invalid_output",
                    latency_ms=latency_ms,
                    raw_output=raw_output,
                )

            self._write_cache(cache_key, claim_nfc, sanitized, raw_output)
            return RewriteResult(
                text=sanitized,
                status="ok",
                latency_ms=latency_ms,
                raw_output=raw_output,
            )

        latency_ms = (time.perf_counter() - t0) * _MS_PER_SECOND
        logger.warning("rewrite_status=%s (retries exhausted)", last_status)
        return RewriteResult(
            text=claim_nfc,
            status=last_status,
            latency_ms=latency_ms,
            raw_output=None,
        )
