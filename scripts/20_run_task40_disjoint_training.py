"""Task 4.0 Runner: Article-Disjoint Training and Evaluation (Brief #5 Bản vá 1 §11.3).

Executes:
1. Prepares article-disjoint splits (Holdout 20%, Val 10%, Train 70%) if not already present.
2. Trains Variant V-C (seed 42, fp32) on disjoint train set.
3. Evaluates V-C on Holdout H (both gold and selected evidence).
4. Trains Baseline (claim-only, seed 42, fp32) on disjoint train set and evaluates on Holdout H
   to isolate article-memorization from evidence-reading.
5. Saves comparative analysis to ``artifacts/task40_disjoint_report.json``.
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
from typing import Any

os.environ.setdefault("HF_HOME", r"D:\hf-cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch

from vnfake.config import load_config
from vnfake.training.train import train_and_evaluate_verifier
from vnfake.utils.logging import get_logger

logger = get_logger(__name__)


def _clean_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    cfg = load_config()
    disjoint_dir = cfg.paths.interim_dir / "article_disjoint"
    train_split = disjoint_dir / "selected_train.jsonl"
    val_split = disjoint_dir / "selected_val.jsonl"
    holdout_split = disjoint_dir / "selected_holdout.jsonl"

    if not (train_split.exists() and val_split.exists() and holdout_split.exists()):
        logger.info("Article-disjoint splits not found. Preparing now...")
        subprocess.run(
            [sys.executable, "scripts/19_prepare_article_disjoint_data.py"],
            check=True,
        )

    logger.info("=== TASK 4.0: STARTING ARTICLE-DISJOINT V-C TRAINING (SEED 42) ===")
    _clean_gpu()
    vc_run = train_and_evaluate_verifier(
        cfg,
        variant="V-C",
        seed=42,
        train_split_path=train_split,
        val_split_path=val_split,
        holdout_split_path=holdout_split,
        run_id="phobert_vc_disjoint_s42",
    )

    logger.info("=== TASK 4.0: STARTING ARTICLE-DISJOINT BASELINE TRAINING (SEED 42) ===")
    _clean_gpu()
    base_run = train_and_evaluate_verifier(
        cfg,
        variant="baseline",
        seed=42,
        train_split_path=train_split,
        val_split_path=val_split,
        holdout_split_path=holdout_split,
        run_id="phobert_baseline_disjoint_s42",
    )

    # Compile comparative report
    manifest = json.loads((disjoint_dir / "manifest.json").read_text(encoding="utf-8"))

    report: dict[str, Any] = {
        "manifest": manifest,
        "runs": {
            "v_c": {
                "run_id": vc_run["run_id"],
                "val_gold_f1": vc_run["dev_gold_calibrated"]["macro_f1"],
                "val_selected_f1": vc_run["dev_selected_calibrated"]["macro_f1"],
                "holdout_h": vc_run.get("holdout_h"),
            },
            "baseline": {
                "run_id": base_run["run_id"],
                "val_f1": base_run["dev_gold_calibrated"]["macro_f1"],
                "holdout_h": base_run.get("holdout_h"),
            },
        },
    }

    out_file = cfg.paths.artifacts_dir / "task40_disjoint_report.json"
    out_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved Task 4.0 Disjoint Report to %s", out_file)


if __name__ == "__main__":
    main()
