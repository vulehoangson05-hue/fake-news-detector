"""Unit tests for vnfake.data.dedup."""

from __future__ import annotations

from vnfake.data.dedup import (
    dedup_by_minhash,
    dedup_by_url,
    dedup_vifactcheck_splits,
)
from vnfake.schemas.kb import KBRecord
from vnfake.schemas.vifactcheck import ViFactCheckSample


def _kb(rec_id: str, text: str, url: str) -> KBRecord:
    return KBRecord(
        id=rec_id,
        title=f"t-{rec_id}",
        text=text,
        label="real",
        url=url,
        crawl_date="2024-01-01",
    )


def _vfc(stmt: str, label: int = 0) -> ViFactCheckSample:
    return ViFactCheckSample(statement=stmt, evidence="e", labels=label)


def test_dedup_by_url_keeps_first() -> None:
    recs = [_kb("1", "a", "u1"), _kb("2", "b", "u1"), _kb("3", "c", "u2")]
    out = dedup_by_url(recs)
    assert [r.id for r in out] == ["1", "3"]


def test_dedup_by_minhash_removes_near_duplicate() -> None:
    text = "Đây là nội dung bài viết về một sự kiện chính trị rất quan trọng hôm nay."
    recs = [
        _kb("1", text, "u1"),
        _kb("2", text, "u2"),  # identical content
        _kb("3", "Hoàn toàn khác biệt: thể thao, bóng đá, cầu thủ.", "u3"),
    ]
    out = dedup_by_minhash(recs, threshold=0.8, num_perm=64, shingle_n=3)
    assert len(out) == 2
    assert {r.id for r in out} == {"1", "3"}


def test_dedup_by_minhash_keeps_distinct() -> None:
    recs = [
        _kb("1", "Chủ đề kinh tế: lạm phát tăng cao.", "u1"),
        _kb("2", "Chủ đề thể thao: đội tuyển thắng trận.", "u2"),
    ]
    out = dedup_by_minhash(recs, threshold=0.8, num_perm=64, shingle_n=3)
    assert len(out) == 2


def test_dedup_vifactcheck_splits_removes_cross_split() -> None:
    dev = [_vfc("S1")]
    test = [_vfc("S2")]
    train = [_vfc("S1"), _vfc("S2"), _vfc("S3"), _vfc("S3")]
    cleaned, out_dev, out_test = dedup_vifactcheck_splits(train, dev, test)
    assert [s.statement for s in cleaned] == ["S3"]
    assert [s.statement for s in out_dev] == ["S1"]
    assert [s.statement for s in out_test] == ["S2"]


def test_dedup_vifactcheck_splits_case_insensitive() -> None:
    dev = [_vfc("Hello World")]
    test: list[ViFactCheckSample] = []
    train = [_vfc("HELLO WORLD")]
    cleaned, _, _ = dedup_vifactcheck_splits(train, dev, test)
    assert cleaned == []


def test_dedup_vifactcheck_splits_preserves_dev_test() -> None:
    dev = [_vfc("D1"), _vfc("D1")]  # internal dup allowed (protected)
    test = [_vfc("T1")]
    train: list[ViFactCheckSample] = []
    _, out_dev, out_test = dedup_vifactcheck_splits(train, dev, test)
    assert len(out_dev) == 2  # untouched
    assert len(out_test) == 1
