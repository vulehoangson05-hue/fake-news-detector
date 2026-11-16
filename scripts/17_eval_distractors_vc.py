"""Evaluate V-C on dev distractor / unlinked evidence dataset (Brief #5 §2.1 & §11.4).

Measures prediction distribution of V-C (seeds 42, 43, 44) across:
1. All raw unlinked pairs (1,446 pairs: 723 random + 723 hard BM25)
2. Random unlinked pairs (723 pairs)
3. Raw Hard BM25 pairs (723 pairs)
4. Filtered Hard BM25 pairs (containment < 0.3, excluding 11 pairs)
5. All filtered unlinked pairs (1,435 pairs: 723 random + 712 hard BM25)
6. Same-article different-sentence pairs (723 pairs)
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HOME", r"D:\hf-cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

from vnfake.config import load_config
from vnfake.data.dataset import ModelType
from vnfake.models.classifier import VerifierClassifier
from vnfake.models.tokenization import encode_verifier_dataset
from vnfake.utils.logging import get_logger

logger = get_logger(__name__)


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    with p.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _dist(subset_preds: np.ndarray) -> dict[str, Any]:
    n = len(subset_preds)
    if n == 0:
        return {
            "real_pct": 0.0,
            "fake_pct": 0.0,
            "unverifiable_pct": 0.0,
            "counts": {"real": 0, "fake": 0, "unverifiable": 0, "total": 0},
        }
    n_real = int(np.sum(subset_preds == 0))
    n_fake = int(np.sum(subset_preds == 1))
    n_unv = int(np.sum(subset_preds == 2))
    return {
        "real_pct": round(n_real / n * 100.0, 2),
        "fake_pct": round(n_fake / n * 100.0, 2),
        "unverifiable_pct": round(n_unv / n * 100.0, 2),
        "counts": {"real": n_real, "fake": n_fake, "unverifiable": n_unv, "total": n},
    }


def main() -> None:
    cfg = load_config()
    distractor_dev_path = cfg.paths.interim_dir / "distractor_dev.jsonl"
    same_art_dev_path = cfg.paths.interim_dir / "distractor_same_article_dev.jsonl"

    distractor_rows = read_jsonl(distractor_dev_path)
    same_art_rows = read_jsonl(same_art_dev_path)

    logger.info("Loaded %d dev distractor pairs from %s", len(distractor_rows), distractor_dev_path)
    logger.info("Loaded %d same-article distractor pairs from %s", len(same_art_rows), same_art_dev_path)

    # Subsets of distractor_dev
    rand_indices = [
        i for i, r in enumerate(distractor_rows) if r.get("distractor_type") == "random"
    ]
    hard_raw_indices = [
        i for i, r in enumerate(distractor_rows) if r.get("distractor_type") == "hard_bm25"
    ]
    hard_filtered_indices = [
        i
        for i in hard_raw_indices
        if float(distractor_rows[i].get("containment_with_gold", 0.0)) < 0.3
    ]
    all_filtered_indices = sorted(rand_indices + hard_filtered_indices)

    logger.info(
        "Indices: Random=%d, Hard Raw=%d, Hard Filtered (<0.3)=%d (dropped %d), All Filtered=%d",
        len(rand_indices),
        len(hard_raw_indices),
        len(hard_filtered_indices),
        len(hard_raw_indices) - len(hard_filtered_indices),
        len(all_filtered_indices),
    )

    # Find V-C run checkpoints
    seeds = (42, 43, 44)
    models_dir = cfg.paths.artifacts_dir / "models"
    run_dirs: dict[int, Path] = {}
    for d in sorted(models_dir.glob("phobert_v-c_*")):
        rep_file = d / "eval_report.json"
        if rep_file.exists():
            data = json.loads(rep_file.read_text(encoding="utf-8"))
            s = int(data.get("seed", -1))
            if s in seeds and s not in run_dirs:
                run_dirs[s] = d

    # Encode with tokenizer from seed 42
    init_verifier = VerifierClassifier(
        run_dirs[42] / "checkpoint",
        cfg.label_map,
        local_files_only=True,
    )
    tokenizer = init_verifier.load_tokenizer()
    model_type: ModelType = "phobert"
    max_length = cfg.classifier.max_len_phobert

    distractor_ds = encode_verifier_dataset(
        [r["statement_segmented"] for r in distractor_rows],
        [r["distractor_evidence_segmented"] for r in distractor_rows],
        [r.get("target_label", 2) for r in distractor_rows],
        tokenizer=tokenizer,
        model_type=model_type,
        max_length=max_length,
        label_map=cfg.label_map,
        statement_only=False,
        presegmented=True,
    )

    same_art_ds = encode_verifier_dataset(
        [r["statement_segmented"] for r in same_art_rows],
        [r["distractor_evidence_segmented"] for r in same_art_rows],
        [r.get("target_label", 2) for r in same_art_rows],
        tokenizer=tokenizer,
        model_type=model_type,
        max_length=max_length,
        label_map=cfg.label_map,
        statement_only=False,
        presegmented=True,
    )

    del init_verifier
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results: dict[str, Any] = {"seeds": {}, "summary": {}}

    categories = [
        ("all_raw", "all_raw_pairs"),
        ("random", "random_pairs"),
        ("hard_raw", "hard_bm25_raw_pairs"),
        ("hard_filtered", "hard_bm25_filtered_pairs"),
        ("all_filtered", "all_filtered_pairs"),
        ("same_article", "same_article_pairs"),
    ]

    agg: dict[str, dict[str, list[float]]] = {
        cat_key: {"unverifiable": [], "real": [], "fake": []} for _, cat_key in categories
    }

    for s in seeds:
        ckpt_dir = run_dirs[s] / "checkpoint"
        logger.info("Evaluating V-C Seed %d...", s)
        verifier = VerifierClassifier(ckpt_dir, cfg.label_map, local_files_only=True)

        logits_dist, _ = verifier.predict_logits(distractor_ds, batch_size=16)
        preds_dist = np.argmax(logits_dist, axis=-1)

        logits_same, _ = verifier.predict_logits(same_art_ds, batch_size=16)
        preds_same = np.argmax(logits_same, axis=-1)

        seed_dists: dict[str, Any] = {
            "all_raw": _dist(preds_dist),
            "random": _dist(preds_dist[rand_indices]),
            "hard_raw": _dist(preds_dist[hard_raw_indices]),
            "hard_filtered": _dist(preds_dist[hard_filtered_indices]),
            "all_filtered": _dist(preds_dist[all_filtered_indices]),
            "same_article": _dist(preds_same),
        }

        results["seeds"][str(s)] = seed_dists

        for short_name, cat_key in categories:
            d = seed_dists[short_name]
            agg[cat_key]["unverifiable"].append(d["unverifiable_pct"])
            agg[cat_key]["real"].append(d["real_pct"])
            agg[cat_key]["fake"].append(d["fake_pct"])

        logger.info(
            "Seed %d -> Hard Raw: %.2f%% unv | Hard Filtered: %.2f%% unv | Same Article: %.2f%% unv",
            s,
            seed_dists["hard_raw"]["unverifiable_pct"],
            seed_dists["hard_filtered"]["unverifiable_pct"],
            seed_dists["same_article"]["unverifiable_pct"],
        )

        del verifier
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for _, cat_key in categories:
        u_mean = float(np.mean(agg[cat_key]["unverifiable"]))
        u_std = float(np.std(agg[cat_key]["unverifiable"]))
        r_mean = float(np.mean(agg[cat_key]["real"]))
        f_mean = float(np.mean(agg[cat_key]["fake"]))
        results["summary"][cat_key] = {
            "unverifiable_pct": round(u_mean, 2),
            "unverifiable_std": round(u_std, 2),
            "real_pct": round(r_mean, 2),
            "fake_pct": round(f_mean, 2),
        }

    out_file = cfg.paths.artifacts_dir / "distractor_vc_eval.json"
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved updated distractor evaluation report to %s", out_file)


if __name__ == "__main__":
    main()
