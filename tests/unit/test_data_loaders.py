"""Unit tests for vnfake.data.loaders."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vnfake.data.loaders import iter_kb_records, load_split


def _kb_dict(i: int, **overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": f"doc-{i}",
        "title": f"Tiêu đề {i}",
        "text": f"Nội dung bài {i}.",
        "label": "real",
        "url": f"https://example.com/{i}",
        "date": "2024-01-01",
    }
    base.update(overrides)
    return base


def test_iter_kb_records_all_valid(tmp_path: Path) -> None:
    src = tmp_path / "kb.jsonl"
    rej = tmp_path / "rejected.jsonl"
    src.write_text(
        "\n".join(json.dumps(_kb_dict(i), ensure_ascii=False) for i in range(3)) + "\n",
        encoding="utf-8",
    )
    records = list(iter_kb_records(src, rej))
    assert len(records) == 3
    assert [r.id for r in records] == ["doc-0", "doc-1", "doc-2"]
    assert rej.read_text(encoding="utf-8") == ""


def test_iter_kb_records_bad_json_written_to_rejected(tmp_path: Path) -> None:
    src = tmp_path / "kb.jsonl"
    rej = tmp_path / "rejected.jsonl"
    src.write_text(
        json.dumps(_kb_dict(0)) + "\n" + "{not-json\n" + json.dumps(_kb_dict(1)) + "\n",
        encoding="utf-8",
    )
    records = list(iter_kb_records(src, rej))
    assert len(records) == 2
    rejected = [json.loads(ln) for ln in rej.read_text(encoding="utf-8").splitlines()]
    assert len(rejected) == 1
    assert rejected[0]["line_no"] == 2
    assert "json_error" in str(rejected[0]["error"])


def test_iter_kb_records_schema_invalid_written_to_rejected(tmp_path: Path) -> None:
    src = tmp_path / "kb.jsonl"
    rej = tmp_path / "rejected.jsonl"
    bad = _kb_dict(0, label="fake")
    src.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    records = list(iter_kb_records(src, rej))
    assert records == []
    assert rej.read_text(encoding="utf-8").strip() != ""


def test_iter_kb_records_auto_id_when_missing(tmp_path: Path) -> None:
    src = tmp_path / "kb.jsonl"
    rej = tmp_path / "rejected.jsonl"
    row = _kb_dict(0)
    del row["id"]
    src.write_text(json.dumps(row) + "\n", encoding="utf-8")
    records = list(iter_kb_records(src, rej))
    assert len(records) == 1
    assert records[0].id  # non-empty auto-generated


def test_iter_kb_records_skips_blank_lines(tmp_path: Path) -> None:
    src = tmp_path / "kb.jsonl"
    rej = tmp_path / "rejected.jsonl"
    src.write_text("\n\n" + json.dumps(_kb_dict(0)) + "\n\n", encoding="utf-8")
    records = list(iter_kb_records(src, rej))
    assert len(records) == 1


def test_iter_kb_records_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list(iter_kb_records(tmp_path / "missing.jsonl", tmp_path / "rej.jsonl"))


def _vfc_dict(label: int = 0) -> dict[str, object]:
    return {"Statement": "A", "Context": "ctx", "labels": label, "Evidence": "ev"}


def test_load_split_train_ok(tmp_path: Path) -> None:
    (tmp_path / "train.jsonl").write_text(json.dumps(_vfc_dict(1)) + "\n", encoding="utf-8")
    samples = load_split("train", tmp_path)
    assert len(samples) == 1
    assert samples[0].labels == 1


def test_load_split_test_blocked_without_final(tmp_path: Path) -> None:
    (tmp_path / "test.jsonl").write_text(json.dumps(_vfc_dict()) + "\n", encoding="utf-8")
    with pytest.raises(PermissionError):
        load_split("test", tmp_path)


def test_load_split_test_allowed_with_final(tmp_path: Path) -> None:
    (tmp_path / "test.jsonl").write_text(json.dumps(_vfc_dict(2)) + "\n", encoding="utf-8")
    samples = load_split("test", tmp_path, allow_test=True)
    assert len(samples) == 1
    assert samples[0].labels == 2


def test_load_split_writes_rejected(tmp_path: Path) -> None:
    rej = tmp_path / "rej.jsonl"
    (tmp_path / "dev.jsonl").write_text(
        json.dumps(_vfc_dict(0)) + "\n" + "{bad\n", encoding="utf-8"
    )
    samples = load_split("dev", tmp_path, rejected_path=rej)
    assert len(samples) == 1
    assert rej.read_text(encoding="utf-8").strip() != ""
