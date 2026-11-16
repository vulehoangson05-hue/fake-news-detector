"""Prepare Article-Disjoint Dataset Splits (Phase 4, Brief #5 Bản vá 1 §11.3 Task 4.0).

Combines train + dev (5,781 rows), groups claims by Context hash, and splits by article:
- 20% articles -> Holdout set H (unseen articles for generalization evaluation)
- 10% articles -> Validation set V (for epoch selection & temperature calibration)
- Remaining 70% -> Train set (strictly article-disjoint from H and V)

Saves to:
- ``data/interim/article_disjoint/selected_train.jsonl``
- ``data/interim/article_disjoint/selected_val.jsonl``
- ``data/interim/article_disjoint/selected_holdout.jsonl``
- ``data/interim/article_disjoint/manifest.json``
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from vnfake.config import load_config
from vnfake.utils.logging import get_logger

logger = get_logger(__name__)


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    with p.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(p: Path, records: list[dict[str, Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.strip().encode("utf-8")).hexdigest()


def main() -> None:
    cfg = load_config()
    proc_dir = cfg.paths.processed_dir
    interim_dir = cfg.paths.interim_dir

    raw_train = read_jsonl(proc_dir / "train_clean.jsonl")
    raw_dev = read_jsonl(proc_dir / "dev_clean.jsonl")
    sel_train = read_jsonl(interim_dir / "selected_train.jsonl")
    sel_dev = read_jsonl(interim_dir / "selected_dev.jsonl")

    assert len(raw_train) == len(sel_train), f"Train mismatch: {len(raw_train)} vs {len(sel_train)}"
    assert len(raw_dev) == len(sel_dev), f"Dev mismatch: {len(raw_dev)} vs {len(sel_dev)}"

    all_raw = raw_train + raw_dev
    all_sel = sel_train + sel_dev
    total_samples = len(all_raw)
    logger.info("Combined train + dev: %d samples", total_samples)

    # Group sample indices by context hash
    ctx_to_indices: dict[str, list[int]] = {}
    for i, r in enumerate(all_raw):
        ctx_hash = sha256_text(r["Context"])
        ctx_to_indices.setdefault(ctx_hash, []).append(i)

    unique_articles = sorted(ctx_to_indices.keys())
    total_articles = len(unique_articles)
    logger.info("Total unique articles (Contexts): %d", total_articles)

    # Split articles randomly with seed 42
    rng = random.Random(cfg.seed)
    rng.shuffle(unique_articles)

    n_h = int(total_articles * 0.20)  # 20%
    n_v = int(total_articles * 0.10)  # 10%

    h_article_set = set(unique_articles[:n_h])
    v_article_set = set(unique_articles[n_h : n_h + n_v])
    train_article_set = set(unique_articles[n_h + n_v :])

    h_indices = [i for a in h_article_set for i in ctx_to_indices[a]]
    v_indices = [i for a in v_article_set for i in ctx_to_indices[a]]
    train_indices = [i for a in train_article_set for i in ctx_to_indices[a]]

    # Re-index sample_idx deterministically for each split
    def _prepare_split(indices: list[int]) -> list[dict[str, Any]]:
        # Sort by original index for reproducibility
        sorted_idxs = sorted(indices)
        records: list[dict[str, Any]] = []
        for new_idx, orig_i in enumerate(sorted_idxs):
            rec = dict(all_sel[orig_i])
            rec["sample_idx"] = new_idx
            rec["original_combined_idx"] = orig_i
            rec["article_hash"] = sha256_text(all_raw[orig_i]["Context"])
            records.append(rec)
        return records

    disjoint_train = _prepare_split(train_indices)
    disjoint_val = _prepare_split(v_indices)
    disjoint_holdout = _prepare_split(h_indices)

    out_dir = interim_dir / "article_disjoint"
    write_jsonl(out_dir / "selected_train.jsonl", disjoint_train)
    write_jsonl(out_dir / "selected_val.jsonl", disjoint_val)
    write_jsonl(out_dir / "selected_holdout.jsonl", disjoint_holdout)

    def _stats(records: list[dict[str, Any]], n_arts: int) -> dict[str, Any]:
        lbls = [r["labels"] for r in records]
        n = len(lbls)
        return {
            "num_articles": n_arts,
            "num_samples": n,
            "pct_total_samples": round(n / total_samples * 100.0, 2),
            "real_count": lbls.count(0),
            "fake_count": lbls.count(1),
            "unverifiable_count": lbls.count(2),
            "real_pct": round(lbls.count(0) / n * 100.0, 2),
            "fake_pct": round(lbls.count(1) / n * 100.0, 2),
            "unverifiable_pct": round(lbls.count(2) / n * 100.0, 2),
        }

    manifest = {
        "seed": cfg.seed,
        "total_articles": total_articles,
        "total_samples": total_samples,
        "splits": {
            "train": _stats(disjoint_train, len(train_article_set)),
            "validation_v": _stats(disjoint_val, len(v_article_set)),
            "holdout_h": _stats(disjoint_holdout, len(h_article_set)),
        },
    }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Saved article-disjoint datasets to %s", out_dir)
    logger.info("Train: %d articles, %d samples", len(train_article_set), len(disjoint_train))
    logger.info("Val V: %d articles, %d samples", len(v_article_set), len(disjoint_val))
    logger.info("Holdout H: %d articles, %d samples", len(h_article_set), len(disjoint_holdout))


if __name__ == "__main__":
    main()
