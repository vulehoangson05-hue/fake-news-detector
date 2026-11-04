"""Data preparation pipeline (consumed by ``scripts/02_prepare_data.py``)."""

from __future__ import annotations

import json
from pathlib import Path

from vnfake.config import AppConfig
from vnfake.data.chunking import Chunk, chunk_record
from vnfake.data.cleaning import is_too_short, normalize_text, strip_boilerplate
from vnfake.data.dedup import dedup_by_minhash, dedup_by_url, dedup_vifactcheck_splits
from vnfake.data.loaders import iter_kb_records, load_split
from vnfake.data.segmentation import SegmenterProtocol
from vnfake.schemas.kb import KBRecord
from vnfake.schemas.vifactcheck import ViFactCheckSample
from vnfake.utils.logging import get_logger

logger = get_logger(__name__)


def _clean_record(record: KBRecord, min_chars: int) -> KBRecord | None:
    clean_text = strip_boilerplate(normalize_text(record.text))
    if is_too_short(clean_text, min_chars):
        return None
    return record.model_copy(update={"text": clean_text, "title": normalize_text(record.title)})


def _load_and_clean_kb(cfg: AppConfig, rejected_dir: Path) -> list[tuple[KBRecord, str]]:
    sources = [("dantri", cfg.paths.kb_dantri), ("vnexpress", cfg.paths.kb_vnexpress)]
    out: list[tuple[KBRecord, str]] = []
    for name, path in sources:
        kept = 0
        for rec in iter_kb_records(path, rejected_dir / f"rejected_kb_{name}.jsonl"):
            clean = _clean_record(rec, cfg.preprocessing.min_chars)
            if clean is not None:
                out.append((clean, name))
                kept += 1
        logger.info("KB[%s]: %d cleaned records", name, kept)
    return out


def _dedup_kb(
    cfg: AppConfig, records_with_source: list[tuple[KBRecord, str]]
) -> list[tuple[KBRecord, str]]:
    doc_id_to_source = {rec.id: src for rec, src in records_with_source}
    after_url = dedup_by_url([rec for rec, _ in records_with_source])
    d = cfg.preprocessing.dedup
    after_minhash = dedup_by_minhash(
        after_url,
        threshold=d.minhash_threshold,
        num_perm=d.minhash_num_perm,
        shingle_n=d.minhash_shingle_n,
    )
    return [(rec, doc_id_to_source.get(rec.id, "unknown")) for rec in after_minhash]


def _chunk_kb(
    cfg: AppConfig,
    records_with_source: list[tuple[KBRecord, str]],
    segmenter: SegmenterProtocol | None,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for rec, src in records_with_source:
        chunks.extend(
            chunk_record(
                rec,
                source=src,
                segmenter=segmenter,
                window=cfg.preprocessing.chunk.window,
                overlap=cfg.preprocessing.chunk.overlap,
            )
        )
    return chunks


def _write_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")


def _write_samples(samples: list[ViFactCheckSample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.model_dump(by_alias=True), ensure_ascii=False) + "\n")


def run_prepare(
    cfg: AppConfig,
    *,
    segmenter: SegmenterProtocol | None,
    skip_test: bool = False,
    skip_segmentation: bool = False,
) -> dict[str, int]:
    """Run the full pipeline. Returns a dict of counts."""
    if skip_segmentation:
        logger.warning("--skip-segmentation: text_segmented will be empty strings")
        segmenter = None

    interim, processed = cfg.paths.interim_dir, cfg.paths.processed_dir
    interim.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    raw_kb = _load_and_clean_kb(cfg, interim)
    deduped_kb = _dedup_kb(cfg, raw_kb)
    chunks = _chunk_kb(cfg, deduped_kb, segmenter)
    _write_chunks(chunks, processed / "kb_chunks.jsonl")
    logger.info(
        "KB: raw+clean=%d -> dedup=%d -> chunks=%d",
        len(raw_kb),
        len(deduped_kb),
        len(chunks),
    )

    train = load_split(
        "train",
        cfg.paths.vifactcheck_dir,
        rejected_path=interim / "rejected_vifactcheck_train.jsonl",
    )
    dev = load_split(
        "dev", cfg.paths.vifactcheck_dir, rejected_path=interim / "rejected_vifactcheck_dev.jsonl"
    )
    test = (
        load_split(
            "test",
            cfg.paths.vifactcheck_dir,
            allow_test=True,
            rejected_path=interim / "rejected_vifactcheck_test.jsonl",
        )
        if not skip_test
        else []
    )
    train_clean, dev_clean, test_clean = dedup_vifactcheck_splits(train, dev, test)
    _write_samples(train_clean, processed / "train_clean.jsonl")
    _write_samples(dev_clean, processed / "dev_clean.jsonl")
    if not skip_test:
        _write_samples(test_clean, processed / "test_clean.jsonl")

    stats = {
        "kb_raw_clean": len(raw_kb),
        "kb_after_dedup": len(deduped_kb),
        "kb_chunks": len(chunks),
        "train_input": len(train),
        "train_clean": len(train_clean),
        "dev_clean": len(dev_clean),
        "test_clean": len(test_clean) if not skip_test else 0,
    }
    logger.info("Prepare done: %s", stats)
    return stats
