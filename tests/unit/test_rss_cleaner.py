"""Unit tests for vnfake.data.rss_cleaner."""

from __future__ import annotations

from vnfake.data.rss_cleaner import (
    clean_rss_article_text,
    fix_location_stamp_runon,
    strip_duplicate_title_line,
)


def test_fix_location_stamp_runon() -> None:
    raw = "Hải PhòngHỏa hoạn thiêu rụi nhà xưởng trong đêm."
    expected = "Hải Phòng - Hỏa hoạn thiêu rụi nhà xưởng trong đêm."
    assert fix_location_stamp_runon(raw) == expected

    raw_tphcm = "TP HCMDinh thự Độc Lập mở cửa đón khách."
    assert fix_location_stamp_runon(raw_tphcm) == "TP HCM - Dinh thự Độc Lập mở cửa đón khách."

    # Normal text without runon should remain untouched
    normal = "Hà Nội - Chiều nay thời tiết mát mẻ."
    assert fix_location_stamp_runon(normal) == normal


def test_strip_duplicate_title_line() -> None:
    title = "Giá vàng SJC tiếp tục tăng mạnh"
    body_with_title = (
        "Giá vàng SJC tiếp tục tăng mạnh.\n"
        "Sáng nay, các doanh nghiệp vàng lớn niêm yết giá bán ra ở mức cao kỷ lục."
    )
    cleaned = strip_duplicate_title_line(body_with_title, title)
    assert cleaned == "Sáng nay, các doanh nghiệp vàng lớn niêm yết giá bán ra ở mức cao kỷ lục."

    # Different first line should remain untouched
    body_different = (
        "Thị trường tài chính biến động.\n"
        "Sáng nay, các doanh nghiệp vàng lớn niêm yết giá bán ra ở mức cao kỷ lục."
    )
    assert strip_duplicate_title_line(body_different, title) == body_different


def test_clean_rss_article_text_end_to_end() -> None:
    title = "Bão số 3 đổ bộ vào đất liền"
    raw = (
        "Bão số 3 đổ bộ vào đất liền!\n"
        "Đà NẵngGió giật mạnh cấp 11 kèm mưa lớn kéo dài từ rạng sáng nay "
        "khiến nhiều tuyến phố trung tâm ngập sâu. "
        "Các lực lượng cứu hộ đã túc trực 24/24 để hỗ trợ người dân di dời tài sản.\n"
        "Xem thêm: Tin tức thời tiết mới nhất."
    )
    cleaned = clean_rss_article_text(raw, title=title, min_chars=50)
    assert "Đà Nẵng - Gió giật mạnh" in cleaned
    assert not cleaned.startswith("Bão số 3")
    assert "Xem thêm" not in cleaned
