"""Text normalization and cleaning for Vietnamese KB articles.

All functions are pure. ``normalize_text`` is mandatory before segmentation,
embedding, or tokenization (copilot-instructions §4).
"""

from __future__ import annotations

import re
import unicodedata

# Control characters, EXCEPT tab (\x09) and newline (\x0a).
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Horizontal whitespace runs (space + tab).
_MULTI_HSPACE_RE = re.compile(r"[ \t]+")
# Three or more newlines -> paragraph break.
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

# Common news-site boilerplate. Anchored to whole lines.
_BOILERPLATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*\(?\s*xem\s+thêm.*$", re.IGNORECASE | re.UNICODE),
    re.compile(r"^\s*đọc\s+thêm\s*:.*$", re.IGNORECASE | re.UNICODE),
    re.compile(r"^\s*nguồn\s*:.*$", re.IGNORECASE | re.UNICODE),
    re.compile(r"^\s*ảnh\s*:.*$", re.IGNORECASE | re.UNICODE),
)


def normalize_text(text: str) -> str:
    """Normalize Vietnamese text.

    Steps (order matters):
        1. Unicode NFC (mandatory for Vietnamese diacritics).
        2. Strip control characters (except tab/newline).
        3. Normalize CRLF/CR to LF.
        4. Collapse horizontal whitespace runs to a single space.
        5. Collapse 3+ newlines to a single paragraph break (``\\n\\n``).
        6. Strip leading/trailing whitespace.

    Case is preserved. Does NOT perform word segmentation.

    Args:
        text: Raw input string.

    Returns:
        Normalized string (possibly empty).
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = _CTRL_RE.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _MULTI_HSPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def strip_boilerplate(text: str) -> str:
    """Remove lines matching common news-site boilerplate.

    Applies after :func:`normalize_text`. Idempotent.

    Args:
        text: Normalized text with ``\\n`` line separators.

    Returns:
        Text with boilerplate lines removed; remaining lines preserved in order.
    """
    if not text:
        return ""
    kept = [ln for ln in text.split("\n") if not any(p.match(ln) for p in _BOILERPLATE_PATTERNS)]
    return "\n".join(kept).strip()


def is_too_short(text: str, min_chars: int) -> bool:
    """Return True if ``len(text) < min_chars``.

    ``min_chars`` is required (no default) — callers must source it from config
    (copilot-instructions §3, "no magic numbers").
    """
    return len(text) < min_chars
