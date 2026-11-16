"""Phase 2.5 Task 6 — RSS Specific Text Cleaning (Pipeline D).

Implements Brief #3 §D1:
- Strips duplicate title on line 1 (matching after NFC, casefold, and punctuation strip).
- Fixes VnExpress ``location-stamp`` run-on (e.g. ``Hải PhòngHỏa hoạn...`` -> ``Hải Phòng - Hỏa hoạn...``).
- Keeps ``(Dân trí) - `` prefix consistent with static KB chunks (which preserve ``(Dân trí)`` 909 times).
- Integrates with standard ``normalize_text``, ``strip_boilerplate``, and length guards.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from vnfake.data.cleaning import normalize_text, strip_boilerplate

# Major Vietnamese provinces / cities appearing in VnExpress location stamps
VN_LOCATIONS: tuple[str, ...] = (
    "Hà Nội",
    "TP HCM",
    "TP.HCM",
    "TP Hồ Chí Minh",
    "Hải Phòng",
    "Đà Nẵng",
    "Cần Thơ",
    "Quảng Ninh",
    "Bình Dương",
    "Đồng Nai",
    "Nghệ An",
    "Thanh Hóa",
    "Huế",
    "Thừa Thiên Huế",
    "Khánh Hòa",
    "Lâm Đồng",
    "Bà Rịa - Vũng Tàu",
    "Bắc Giang",
    "Bắc Ninh",
    "Vĩnh Phúc",
    "Phú Thọ",
    "Nam Định",
    "Thái Bình",
    "Hải Dương",
    "Hưng Yên",
    "An Giang",
    "Kiên Giang",
    "Cà Mau",
    "Sóc Trăng",
    "Bạc Liêu",
    "Bến Tre",
    "Tiền Giang",
    "Long An",
    "Tây Ninh",
    "Bình Phước",
    "Đắk Lắk",
    "Gia Lai",
    "Kon Tum",
    "Đắk Nông",
    "Quảng Trị",
    "Quảng Bình",
    "Hà Tĩnh",
    "Ninh Bình",
    "Hà Nam",
    "Thái Nguyên",
    "Lạng Sơn",
    "Cao Bằng",
    "Hà Giang",
    "Tuyên Quang",
    "Yên Bái",
    "Lào Cai",
    "Sơn La",
    "Điện Biên",
    "Lai Châu",
    "Hòa Bình",
    "Quảng Nam",
    "Quảng Ngãi",
    "Bình Định",
    "Phú Yên",
    "Ninh Thuận",
    "Bình Thuận",
    "Hậu Giang",
    "Trà Vinh",
    "Vĩnh Long",
    "Đồng Tháp",
)

_VN_UPPERCASE_CHAR_CLASS = (
    r"[A-ZÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ]"
)


def fix_location_stamp_runon(text: str, locations: Sequence[str] = VN_LOCATIONS) -> str:
    """Insert ``' - '`` between run-on location stamp and following sentence.

    Example:
        ``Hải PhòngHỏa hoạn bùng lên...`` -> ``Hải Phòng - Hỏa hoạn bùng lên...``
        ``TP HCMDinh thự có...`` -> ``TP HCM - Dinh thự có...``
    """
    out = text
    for loc in locations:
        pattern = rf"(^|\n)({re.escape(loc)})({_VN_UPPERCASE_CHAR_CLASS})"
        out = re.sub(pattern, r"\1\2 - \3", out)
    return out


def strip_duplicate_title_line(text: str, title: str | None) -> str:
    """Drop the first line if it mirrors the article title (after normalization)."""
    if not title:
        return text

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""

    first_line_clean = re.sub(r"[^\w\s]", "", unicodedata.normalize("NFC", lines[0]).lower())
    first_line_clean = " ".join(first_line_clean.split())

    title_clean = re.sub(r"[^\w\s]", "", unicodedata.normalize("NFC", title).lower())
    title_clean = " ".join(title_clean.split())

    if first_line_clean == title_clean:
        return "\n".join(lines[1:]).strip()
    return text


def clean_rss_article_text(
    raw_text: str,
    title: str | None = None,
    *,
    min_chars: int = 100,
) -> str:
    """Complete cleaning pipeline for an extracted RSS article text.

    Pipeline:
    1. Unicode NFC normalization.
    2. Fix VnExpress / media location stamp run-on.
    3. Strip duplicate title on line 1.
    4. Strip residual boilerplate (read more, comment forms, etc.).
    5. Return empty string if length < min_chars.
    """
    if not raw_text or not raw_text.strip():
        return ""

    norm = normalize_text(raw_text)
    fixed_locations = fix_location_stamp_runon(norm)
    no_dup_title = strip_duplicate_title_line(fixed_locations, title)
    cleaned = strip_boilerplate(no_dup_title)

    if len(cleaned) < min_chars:
        return ""

    return cleaned
