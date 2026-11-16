"""Gate G2 Verification Script (Brief #5 §1).

Audits context & evidence leakage across train/dev/test splits:
1. Counts unique Context and Evidence in train, dev, test.
2. Measures % of dev samples whose Context or Evidence appears in train.
3. For test: measures Context overlap strictly via SHA-256 hashes (NO labels read, NO models run).
4. If overlap is significant: computes V-C (3 seeds) Macro-F1 on dev_subset_unseen
   (dev samples whose Context is NOT in train).
5. Saves report to ``artifacts/g2_context_overlap_report.json``.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from vnfake.config import load_config
from vnfake.training.train import load_selected_split
from vnfake.utils.logging import get_logger

logger = get_logger(__name__)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.strip().encode("utf-8")).hexdigest()


def _mean_std(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    m = sum(values) / n
    if n == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in values) / (n - 1)
    return m, math.sqrt(var)


def main() -> None:
    cfg = load_config()
    proc_dir = cfg.paths.processed_dir

    train_path = proc_dir / "train_clean.jsonl"
    dev_path = proc_dir / "dev_clean.jsonl"
    test_path = proc_dir / "test_clean.jsonl"

    def read_jsonl(p: Path) -> list[dict[str, Any]]:
        with p.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    train_rows = read_jsonl(train_path)
    dev_rows = read_jsonl(dev_path)
    test_rows = read_jsonl(test_path)

    logger.info("Rows: train=%d, dev=%d, test=%d", len(train_rows), len(dev_rows), len(test_rows))

    # Contexts & Evidences
    train_contexts = [r["Context"].strip() for r in train_rows if "Context" in r]
    dev_contexts = [r["Context"].strip() for r in dev_rows if "Context" in r]
    # For test: ONLY extract Context SHA-256, NEVER touch labels
    test_context_hashes = [sha256_text(r["Context"]) for r in test_rows if "Context" in r]

    train_evidences = [r["Evidence"].strip() for r in train_rows if "Evidence" in r]
    dev_evidences = [r["Evidence"].strip() for r in dev_rows if "Evidence" in r]

    unique_ctx_train = set(train_contexts)
    unique_ctx_dev = set(dev_contexts)
    unique_ctx_test_hashes = set(test_context_hashes)

    train_ctx_hashes = {sha256_text(c) for c in train_contexts}
    dev_ctx_hashes = {sha256_text(c) for c in dev_contexts}
    all_seen_ctx_hashes = train_ctx_hashes | dev_ctx_hashes

    unique_evid_train = set(train_evidences)
    unique_evid_dev = set(dev_evidences)

    logger.info(
        "Unique Contexts: train=%d, dev=%d, test=%d",
        len(unique_ctx_train),
        len(unique_ctx_dev),
        len(unique_ctx_test_hashes),
    )
    logger.info("Unique Evidences: train=%d, dev=%d", len(unique_evid_train), len(unique_evid_dev))

    # Dev overlap with train
    dev_ctx_in_train_mask = [c in unique_ctx_train for c in dev_contexts]
    dev_ctx_overlap_count = sum(dev_ctx_in_train_mask)
    dev_ctx_overlap_pct = round(dev_ctx_overlap_count / len(dev_rows) * 100.0, 2)

    dev_evid_in_train_mask = [e in unique_evid_train for e in dev_evidences]
    dev_evid_overlap_count = sum(dev_evid_in_train_mask)
    dev_evid_overlap_pct = round(dev_evid_overlap_count / len(dev_rows) * 100.0, 2)

    # Test overlap with train / (train+dev) strictly by hash
    test_ctx_in_train_count = sum(h in train_ctx_hashes for h in test_context_hashes)
    test_ctx_in_train_pct = round(test_ctx_in_train_count / len(test_rows) * 100.0, 2)

    test_ctx_in_seen_count = sum(h in all_seen_ctx_hashes for h in test_context_hashes)
    test_ctx_in_seen_pct = round(test_ctx_in_seen_count / len(test_rows) * 100.0, 2)

    logger.info(
        "Dev Context in Train: %d / %d (%.2f%%)",
        dev_ctx_overlap_count,
        len(dev_rows),
        dev_ctx_overlap_pct,
    )
    logger.info(
        "Dev Evidence in Train: %d / %d (%.2f%%)",
        dev_evid_overlap_count,
        len(dev_rows),
        dev_evid_overlap_pct,
    )
    logger.info(
        "Test Context in Train (by hash): %d / %d (%.2f%%)",
        test_ctx_in_train_count,
        len(test_rows),
        test_ctx_in_train_pct,
    )
    logger.info(
        "Test Context in Train+Dev (by hash): %d / %d (%.2f%%)",
        test_ctx_in_seen_count,
        len(test_rows),
        test_ctx_in_seen_pct,
    )

    # Identify unseen dev subset
    dev_unseen_indices = [i for i, in_train in enumerate(dev_ctx_in_train_mask) if not in_train]
    logger.info(
        "Dev unseen Context samples: %d / %d (%.2f%%)",
        len(dev_unseen_indices),
        len(dev_rows),
        round(len(dev_unseen_indices) / len(dev_rows) * 100.0, 2),
    )

    # Evaluate V-C (3 seeds) on dev_unseen subset
    unseen_eval_results: dict[str, Any] = {}
    if dev_unseen_indices:
        dev_records = load_selected_split(cfg.paths.interim_dir / "selected_dev.jsonl")
        unseen_records = [dev_records[i] for i in dev_unseen_indices]

        # Read saved predictions from artifacts/g1_seed_predictions.json if available
        g1_file = cfg.paths.artifacts_dir / "g1_seed_predictions.json"
        if g1_file.exists():
            g1_data = json.loads(g1_file.read_text(encoding="utf-8"))
            seeds = (42, 43, 44)

            gold_f1s: list[float] = []
            gold_accs: list[float] = []
            sel_f1s: list[float] = []
            sel_accs: list[float] = []

            for s in seeds:
                preds_g = np.array(g1_data["seeds"][str(s)]["predictions_gold"])[dev_unseen_indices]
                preds_s = np.array(g1_data["seeds"][str(s)]["predictions_selected"])[
                    dev_unseen_indices
                ]
                true_y = np.array([r.label for r in unseen_records])

                from vnfake.data.dataset import build_class_mapping

                class_names, src_to_cls = build_class_mapping(cfg.label_map)
                mapped_y = np.array([src_to_cls[y] for y in true_y])
                num_classes = len(class_names)

                def _eval_preds(
                    preds: np.ndarray, y: np.ndarray, n_classes: int = num_classes
                ) -> tuple[float, float, list[list[int]]]:
                    acc = float(np.mean(preds == y))
                    cm = [[0] * n_classes for _ in range(n_classes)]
                    for t, p in zip(y.tolist(), preds.tolist(), strict=True):
                        cm[t][p] += 1
                    f1s = []
                    for c in range(n_classes):
                        tp = cm[c][c]
                        fp = sum(cm[r][c] for r in range(n_classes) if r != c)
                        fn = sum(cm[c][r] for r in range(n_classes) if r != c)
                        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                        f1s.append(f1)
                    macro_f1 = float(sum(f1s) / max(len(f1s), 1))
                    return round(acc, 4), round(macro_f1, 4), cm

                acc_g, f1_g, cm_g = _eval_preds(preds_g, mapped_y)
                acc_s, f1_s, cm_s = _eval_preds(preds_s, mapped_y)

                gold_f1s.append(f1_g)
                gold_accs.append(acc_g)
                sel_f1s.append(f1_s)
                sel_accs.append(acc_s)

                logger.info(
                    "Seed %d Dev Unseen: Gold Macro-F1=%.4f (Acc=%.4f) | Selected Macro-F1=%.4f (Acc=%.4f)",
                    s,
                    f1_g,
                    acc_g,
                    f1_s,
                    acc_s,
                )
                unseen_eval_results[str(s)] = {
                    "gold_macro_f1": f1_g,
                    "gold_accuracy": acc_g,
                    "gold_confusion_matrix": cm_g,
                    "selected_macro_f1": f1_s,
                    "selected_accuracy": acc_s,
                    "selected_confusion_matrix": cm_s,
                }

            g_f1_m, g_f1_std = _mean_std(gold_f1s)
            g_acc_m, g_acc_std = _mean_std(gold_accs)
            s_f1_m, s_f1_std = _mean_std(sel_f1s)
            s_acc_m, s_acc_std = _mean_std(sel_accs)

            logger.info("Dev Unseen Gold Macro-F1: %.4f ± %.4f", g_f1_m, g_f1_std)
            logger.info("Dev Unseen Selected Macro-F1: %.4f ± %.4f", s_f1_m, s_f1_std)
            unseen_eval_results["summary"] = {
                "num_unseen_samples": len(unseen_records),
                "gold_macro_f1_mean": round(g_f1_m, 4),
                "gold_macro_f1_std": round(g_f1_std, 4),
                "gold_accuracy_mean": round(g_acc_m, 4),
                "gold_accuracy_std": round(g_acc_std, 4),
                "selected_macro_f1_mean": round(s_f1_m, 4),
                "selected_macro_f1_std": round(s_f1_std, 4),
                "selected_accuracy_mean": round(s_acc_m, 4),
                "selected_accuracy_std": round(s_acc_std, 4),
            }

    report = {
        "unique_contexts": {
            "train": len(unique_ctx_train),
            "dev": len(unique_ctx_dev),
            "test": len(unique_ctx_test_hashes),
        },
        "unique_evidences": {
            "train": len(unique_evid_train),
            "dev": len(unique_evid_dev),
        },
        "dev_overlap_with_train": {
            "context_count": dev_ctx_overlap_count,
            "context_pct": dev_ctx_overlap_pct,
            "evidence_count": dev_evid_overlap_count,
            "evidence_pct": dev_evid_overlap_pct,
        },
        "test_overlap_strictly_by_context_hash": {
            "in_train_count": test_ctx_in_train_count,
            "in_train_pct": test_ctx_in_train_pct,
            "in_seen_count": test_ctx_in_seen_count,
            "in_seen_pct": test_ctx_in_seen_pct,
        },
        "dev_unseen_evaluation": unseen_eval_results,
    }

    out_file = cfg.paths.artifacts_dir / "g2_context_overlap_report.json"
    out_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved G2 report to %s", out_file)


if __name__ == "__main__":
    main()
