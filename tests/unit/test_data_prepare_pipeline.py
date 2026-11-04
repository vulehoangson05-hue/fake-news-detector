"""Smoke test for vnfake.data.prepare.run_prepare with tiny fixtures."""

from __future__ import annotations

import json
from pathlib import Path

from vnfake.config import (
    AppConfig,
    ChunkConfig,
    DedupConfig,
    PathsConfig,
    PreprocessingConfig,
)
from vnfake.data.prepare import run_prepare


class _FakeSegmenter:
    def segment(self, text: str) -> str:
        return text.replace(" ", "_")

    def segment_batch(self, texts: list[str]) -> list[str]:
        return [self.segment(t) for t in texts]


def _kb_dict(i: int, text: str) -> dict[str, object]:
    return {
        "id": f"doc-{i}",
        "title": f"Tiêu đề {i}",
        "text": text,
        "label": "real",
        "url": f"https://example.com/{i}",
        "date": "2024-01-01",
    }


def _vfc(stmt: str, label: int = 0) -> dict[str, object]:
    return {"Statement": stmt, "Context": "", "labels": label, "Evidence": "ev"}


def _make_cfg(root: Path) -> AppConfig:
    return AppConfig(
        seed=42,
        label_map={0: "real", 1: "fake", 2: "unverifiable"},
        paths=PathsConfig(
            kb_dantri=root / "kb" / "dantri.jsonl",
            kb_vnexpress=root / "kb" / "vnexpress.jsonl",
            vifactcheck_dir=root / "vifactcheck",
            interim_dir=root / "interim",
            processed_dir=root / "processed",
            artifacts_dir=root / "artifacts",
        ),
        preprocessing=PreprocessingConfig(
            min_chars=10,
            chunk=ChunkConfig(window=2, overlap=0),
            dedup=DedupConfig(minhash_threshold=0.9, minhash_num_perm=32, minhash_shingle_n=2),
        ),
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )


def test_run_prepare_end_to_end(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    _write_jsonl(
        cfg.paths.kb_dantri,
        [
            _kb_dict(0, "Câu một. Câu hai. Câu ba. Câu bốn."),
            _kb_dict(1, "Quá ngắn."),
        ],
    )
    _write_jsonl(cfg.paths.kb_vnexpress, [_kb_dict(2, "Đoạn văn dài hơn mười ký tự.")])
    _write_jsonl(cfg.paths.vifactcheck_dir / "train.jsonl", [_vfc("S1"), _vfc("S2"), _vfc("S3")])
    _write_jsonl(cfg.paths.vifactcheck_dir / "dev.jsonl", [_vfc("S2")])
    _write_jsonl(cfg.paths.vifactcheck_dir / "test.jsonl", [_vfc("S3")])

    stats = run_prepare(cfg, segmenter=_FakeSegmenter(), skip_test=False)

    assert stats["kb_raw_clean"] == 2
    assert stats["train_input"] == 3
    assert stats["train_clean"] == 1
    assert stats["dev_clean"] == 1
    assert stats["test_clean"] == 1

    chunks_file = cfg.paths.processed_dir / "kb_chunks.jsonl"
    assert chunks_file.exists()
    lines = chunks_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == stats["kb_chunks"] >= 2
    first = json.loads(lines[0])
    assert first["source"] in {"dantri", "vnexpress"}
    assert first["text_segmented"].count("_") > 0

    assert (cfg.paths.interim_dir / "rejected_kb_dantri.jsonl").exists()


def test_run_prepare_skip_test(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    _write_jsonl(cfg.paths.kb_dantri, [_kb_dict(0, "Nội dung đủ dài để qua filter.")])
    _write_jsonl(cfg.paths.kb_vnexpress, [])
    _write_jsonl(cfg.paths.vifactcheck_dir / "train.jsonl", [_vfc("A")])
    _write_jsonl(cfg.paths.vifactcheck_dir / "dev.jsonl", [_vfc("B")])
    _write_jsonl(cfg.paths.vifactcheck_dir / "test.jsonl", [_vfc("C")])

    stats = run_prepare(cfg, segmenter=None, skip_test=True, skip_segmentation=True)

    assert stats["test_clean"] == 0
    assert not (cfg.paths.processed_dir / "test_clean.jsonl").exists()
