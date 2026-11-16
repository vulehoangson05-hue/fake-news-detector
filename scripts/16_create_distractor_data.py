"""Generate Distractor / Unlinked Evidence datasets (Brief #5 §2 Task 4.1).

Creates:
1. ``data/interim/distractor_dev.jsonl`` (1,446 samples: 723 random + 723 hard BM25)
2. ``data/interim/distractor_train.jsonl`` (1,266 samples: 633 random + 633 hard BM25)
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from vnfake.config import load_config
from vnfake.data.distractors import (
    DistractorPool,
    generate_dev_distractors,
    generate_same_article_distractors,
    generate_train_distractors,
)
from vnfake.data.segmentation import get_segmenter
from vnfake.utils.logging import get_logger

logger = get_logger(__name__)


def read_jsonl(p: Path) -> list[dict]:
    with p.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(p: Path, records: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    cfg = load_config()
    proc_dir = cfg.paths.processed_dir
    interim_dir = cfg.paths.interim_dir

    train_records = read_jsonl(proc_dir / "train_clean.jsonl")
    dev_records = read_jsonl(proc_dir / "dev_clean.jsonl")

    segmenter = get_segmenter(cfg.segmentation.backend, str(cfg.segmentation.vncorenlp_dir))

    logger.info("Building Dev Distractor Pool from dev contexts...")
    dev_pool = DistractorPool.from_records(dev_records)
    logger.info(
        "Dev pool articles: %d, indexed sentences: %d", len(dev_pool.articles), dev_pool.num_docs
    )

    logger.info("Generating Dev Distractors (1 random + 1 hard BM25 per dev claim)...")
    dev_pairs = generate_dev_distractors(dev_records, dev_pool, seed=cfg.seed, segmenter=segmenter)
    logger.info("Generated %d dev distractor pairs", len(dev_pairs))

    dev_out = interim_dir / "distractor_dev.jsonl"
    write_jsonl(dev_out, [asdict(p) for p in dev_pairs])
    logger.info("Saved %s", dev_out)

    logger.info("Generating Same-Article Distractors for Dev (1 per claim, containment < 0.3)...")
    same_article_pairs = generate_same_article_distractors(
        dev_records,
        segmenter=segmenter,
        max_containment=0.3,
    )
    logger.info("Generated %d same-article distractor pairs", len(same_article_pairs))
    same_art_out = interim_dir / "distractor_same_article_dev.jsonl"
    write_jsonl(same_art_out, [asdict(p) for p in same_article_pairs])
    logger.info("Saved %s", same_art_out)

    train_out = interim_dir / "distractor_train.jsonl"
    if not train_out.exists():
        logger.info("Building Train Distractor Pool from train contexts...")
        train_pool = DistractorPool.from_records(train_records)
        logger.info(
            "Train pool articles: %d, indexed sentences: %d",
            len(train_pool.articles),
            train_pool.num_docs,
        )

        # We want negative pairs to account for 20% in V-D: 1,266 / (5058 + 1266) = 20.02%
        target_train_count = 1266
        logger.info(
            "Generating Train Distractors (%d pairs: 50%% random, 50%% hard BM25)...",
            target_train_count,
        )
        train_pairs = generate_train_distractors(
            train_records,
            train_pool,
            target_count=target_train_count,
            seed=cfg.seed,
            segmenter=segmenter,
        )
        logger.info("Generated %d train distractor pairs", len(train_pairs))
        write_jsonl(train_out, [asdict(p) for p in train_pairs])
        logger.info("Saved %s", train_out)
    else:
        logger.info("Train distractors already exist at %s (skipping)", train_out)


if __name__ == "__main__":
    main()
