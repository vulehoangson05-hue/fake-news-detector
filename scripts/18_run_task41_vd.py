"""Task 4.1 Runner: Train and Evaluate Variant V-D (Brief #5 §2).

Executes:
1. Train V-D on Seed 42 in fp32 on GTX 1660.
2. Evaluate pre-defined selection rule (Brief #5 §2.3):
   (a) avg_macro_f1(gold, selected) >= V-C_avg - 0.01 (>= 0.7682)
   (b) distractor unverifiable_pct >= 80.0%
3. If PASSED:
   Train remaining seeds (43, 44) for V-D.
   Compute 3-seed mean ± std.
   Conclude: SELECT V-D.
4. If FAILED:
   Conclude: RETAIN V-C + RELEVANCE GATE.
5. Saves results to ``artifacts/vd_campaign_report.json``.
"""

from __future__ import annotations

import gc
import json
import math
import os
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


def _mean_std(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    m = sum(values) / n
    if n == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in values) / (n - 1)
    return m, math.sqrt(var)


def run_vd_campaign() -> dict[str, Any]:
    cfg = load_config()
    models_dir = cfg.paths.artifacts_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Train V-D on seed 42
    logger.info("=== TASK 4.1: TRAINING VARIANT V-D (SEED 42) ===")
    _clean_gpu()
    vd_s42 = train_and_evaluate_verifier(cfg, variant="V-D", seed=42)

    g_f1_42 = float(vd_s42["dev_gold_calibrated"]["macro_f1"])
    s_f1_42 = float(vd_s42["dev_selected_calibrated"]["macro_f1"])
    avg_f1_42 = (g_f1_42 + s_f1_42) / 2.0

    dist_info = vd_s42.get("dev_distractor") or {}
    unv_pct_42 = float(dist_info.get("unverifiable_pct", 0.0))

    vc_avg_f1 = 0.7782  # V-C seed 42 average Macro-F1 (0.8062 + 0.7501) / 2
    f1_threshold = vc_avg_f1 - 0.01  # 0.7682

    pass_f1 = avg_f1_42 >= f1_threshold
    pass_unv = unv_pct_42 >= 80.0
    selected_vd = pass_f1 and pass_unv

    logger.info("--- V-D Seed 42 Results ---")
    logger.info("Dev Gold Macro-F1: %.4f", g_f1_42)
    logger.info("Dev Selected Macro-F1: %.4f", s_f1_42)
    logger.info(
        "Average Macro-F1: %.4f (threshold >= %.4f -> %s)",
        avg_f1_42,
        f1_threshold,
        "PASS" if pass_f1 else "FAIL",
    )
    logger.info(
        "Distractor Unverifiable: %.2f%% (threshold >= 80.0%% -> %s)",
        unv_pct_42,
        "PASS" if pass_unv else "FAIL",
    )
    logger.info(
        "Decision: %s",
        "SELECT V-D (training seeds 43, 44)" if selected_vd else "RETAIN V-C + RELEVANCE GATE",
    )

    runs: dict[int, dict[str, Any]] = {42: vd_s42}

    if selected_vd:
        for s in (43, 44):
            logger.info("=== TASK 4.1: TRAINING VARIANT V-D (SEED %d) ===", s)
            _clean_gpu()
            runs[s] = train_and_evaluate_verifier(cfg, variant="V-D", seed=s)

        gold_f1s = [float(runs[s]["dev_gold_calibrated"]["macro_f1"]) for s in (42, 43, 44)]
        gold_accs = [float(runs[s]["dev_gold_calibrated"]["accuracy"]) for s in (42, 43, 44)]
        sel_f1s = [float(runs[s]["dev_selected_calibrated"]["macro_f1"]) for s in (42, 43, 44)]
        sel_accs = [float(runs[s]["dev_selected_calibrated"]["accuracy"]) for s in (42, 43, 44)]
        unv_pcts = [
            float((runs[s].get("dev_distractor") or {}).get("unverifiable_pct", 0.0))
            for s in (42, 43, 44)
        ]

        summary = {
            "chosen_variant": "V-D",
            "rule_evaluation": {
                "rule_a_f1": {"threshold": f1_threshold, "actual": avg_f1_42, "passed": pass_f1},
                "rule_b_unv": {"threshold": 80.0, "actual": unv_pct_42, "passed": pass_unv},
            },
            "seeds": {str(s): runs[s]["run_id"] for s in (42, 43, 44)},
            "dev_gold_macro_f1": {
                "mean": round(_mean_std(gold_f1s)[0], 4),
                "std": round(_mean_std(gold_f1s)[1], 4),
            },
            "dev_gold_accuracy": {
                "mean": round(_mean_std(gold_accs)[0], 4),
                "std": round(_mean_std(gold_accs)[1], 4),
            },
            "dev_selected_macro_f1": {
                "mean": round(_mean_std(sel_f1s)[0], 4),
                "std": round(_mean_std(sel_f1s)[1], 4),
            },
            "dev_selected_accuracy": {
                "mean": round(_mean_std(sel_accs)[0], 4),
                "std": round(_mean_std(sel_accs)[1], 4),
            },
            "distractor_unverifiable_pct": {
                "mean": round(_mean_std(unv_pcts)[0], 2),
                "std": round(_mean_std(unv_pcts)[1], 2),
            },
        }
    else:
        summary = {
            "chosen_variant": "V-C",
            "reason": "V-D failed selection rule (avg F1 drop > 0.01 or distractor unv < 80%)",
            "rule_evaluation": {
                "rule_a_f1": {"threshold": f1_threshold, "actual": avg_f1_42, "passed": pass_f1},
                "rule_b_unv": {"threshold": 80.0, "actual": unv_pct_42, "passed": pass_unv},
            },
            "vd_s42": {
                "gold_macro_f1": g_f1_42,
                "selected_macro_f1": s_f1_42,
                "average_macro_f1": avg_f1_42,
                "distractor_unverifiable_pct": unv_pct_42,
            },
        }

    out_file = cfg.paths.artifacts_dir / "vd_campaign_report.json"
    out_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved Task 4.1 report to %s", out_file)
    return summary


if __name__ == "__main__":
    run_vd_campaign()
