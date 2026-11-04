"""Unit tests for vnfake.data.cleaning."""

from __future__ import annotations

from vnfake.data.cleaning import is_too_short, normalize_text, strip_boilerplate


def test_normalize_nfc() -> None:
    # decomposed "ế" = e + circumflex + acute
    decomposed = "e\u0302\u0301"
    assert normalize_text(decomposed) == "\u1ebf"


def test_normalize_strips_control_chars() -> None:
    assert normalize_text("a\x00b\x1fc") == "a b c"


def test_normalize_crlf_to_lf() -> None:
    assert normalize_text("a\r\nb") == "a\nb"


def test_normalize_collapses_horizontal_whitespace() -> None:
    assert normalize_text("a   b\t\tc") == "a b c"


def test_normalize_collapses_many_newlines() -> None:
    assert normalize_text("a\n\n\n\nb") == "a\n\nb"


def test_normalize_empty_and_whitespace_only() -> None:
    assert normalize_text("") == ""
    assert normalize_text("   \n\t") == ""


def test_normalize_preserves_case() -> None:
    assert normalize_text("Xin Chào") == "Xin Chào"


def test_is_too_short() -> None:
    assert is_too_short("abc", 5) is True
    assert is_too_short("abcde", 5) is False
    assert is_too_short("abcdef", 5) is False


def test_strip_boilerplate() -> None:
    text = "Nội dung chính.\nXem thêm: bài khác\nĐoạn 2.\nNguồn: VnExpress"
    out = strip_boilerplate(text)
    assert "Xem thêm" not in out
    assert "Nguồn:" not in out
    assert "Nội dung chính." in out
    assert "Đoạn 2." in out


def test_strip_boilerplate_idempotent() -> None:
    text = "A\nXem thêm: B\nC"
    once = strip_boilerplate(text)
    assert strip_boilerplate(once) == once
