"""Offline Mini-E2E on 30 Cached Tavily Spike Claims (Brief #5 §11.4, §12.3.5, §12.4).

Uses the EXACT same unified evidence concatenation + BM25 sentence selection as the live pipeline
(``TavilyWebSearcher.search_cached_only`` + ``select_evidence_from_web_hits``), then runs:
1. Relevance scoring ``r`` via BAAI/bge-reranker-v2-m3
2. Pinned V-C 3-seed ensemble probability calibration (SHA-256 verified)
3. DecisionGate (tau_rel, tau_conf) with ``show_probs`` enforcement
4. Comprehensive reporting per §12.4:
   - Overall accuracy vs always-unverifiable baseline (33.33%)
   - Coverage (% real/fake predictions)
   - Selective Accuracy (accuracy within affirmed real/fake predictions)
   - False Affirmation Rate (wrong real/fake predictions / 30)
   - Hit (n=9) vs Miss (n=21) and True Label breakdowns
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from typing import Any

os.environ.setdefault("HF_HOME", r"D:\hf-cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

from vnfake.config import load_config, load_search_config
from vnfake.data.cleaning import normalize_text
from vnfake.pipeline.decision import (
    REASON_CONFIRMED,
    REASON_LOW_CONFIDENCE,
    REASON_LOW_RELEVANCE,
    DecisionGate,
    load_vc_ensemble,
)
from vnfake.retrieval.evidence_selector import compute_3gram_containment
from vnfake.retrieval.reranker import BgeReranker
from vnfake.retrieval.web_search import TavilyWebSearcher, select_evidence_from_web_hits
from vnfake.utils.logging import get_logger

logger = get_logger(__name__)


def _dist_summary(rows: list[dict[str, Any]], verdict_key: str) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"count": 0, "unverifiable_pct": 0.0, "real_pct": 0.0, "fake_pct": 0.0}
    n_unv = sum(1 for r in rows if r[verdict_key] == "unverifiable")
    n_real = sum(1 for r in rows if r[verdict_key] == "real")
    n_fake = sum(1 for r in rows if r[verdict_key] == "fake")
    return {
        "count": n,
        "unverifiable_pct": round(n_unv / n * 100.0, 2),
        "real_pct": round(n_real / n * 100.0, 2),
        "fake_pct": round(n_fake / n * 100.0, 2),
        "counts": {"unverifiable": n_unv, "real": n_real, "fake": n_fake},
    }


def _selective_metrics(rows: list[dict[str, Any]], verdict_key: str) -> dict[str, Any]:
    """Compute Overall Acc, Coverage, Selective Acc, and False Affirmation Rate (§12.4)."""
    n = len(rows)
    if n == 0:
        return {}
    correct_all = sum(1 for r in rows if r[verdict_key] == r["true_label"])
    affirmed = [r for r in rows if r[verdict_key] in {"real", "fake"}]
    n_aff = len(affirmed)
    correct_aff = sum(1 for r in affirmed if r[verdict_key] == r["true_label"])
    wrong_aff = n_aff - correct_aff

    return {
        "num_claims": n,
        "overall_accuracy_pct": round(correct_all / n * 100.0, 2),
        "correct_all_count": correct_all,
        "coverage_count": n_aff,
        "coverage_pct": round(n_aff / n * 100.0, 2),
        "abstain_count": n - n_aff,
        "abstain_pct": round((n - n_aff) / n * 100.0, 2),
        "selective_accuracy_pct": round(correct_aff / n_aff * 100.0, 2) if n_aff > 0 else 0.0,
        "correct_affirmed_count": correct_aff,
        "false_affirmation_count": wrong_aff,
        "false_affirmation_rate_pct": round(wrong_aff / n * 100.0, 2),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline mini-E2E on 30 cached Tavily spike claims.")
    p.add_argument("--tau-rel", type=float, default=None, help="Override tau_rel from config")
    p.add_argument("--tau-conf", type=float, default=None, help="Override tau_conf from config")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    search_cfg = load_search_config()
    spike_path = cfg.paths.artifacts_dir / "tavily_spike_report.json"
    if not spike_path.exists():
        raise FileNotFoundError(f"Missing spike report at {spike_path}")

    spike_data = json.loads(spike_path.read_text(encoding="utf-8"))
    claims = spike_data.get("variants", {}).get("Q1", {}).get("claims", [])
    logger.info("Loaded %d cached Tavily Q1 claims from %s", len(claims), spike_path)

    tau_rel = float(args.tau_rel) if args.tau_rel is not None else float(cfg.decision.tau_rel)
    tau_conf = float(args.tau_conf) if args.tau_conf is not None else float(cfg.decision.tau_conf)

    logger.info("DecisionGate thresholds: tau_rel=%.4f | tau_conf=%.4f", tau_rel, tau_conf)
    gate = DecisionGate(tau_rel=tau_rel, tau_conf=tau_conf, class_names=cfg.label_map)

    # 1. Gather evidence using exact same live-equivalent offline selector
    searcher = TavilyWebSearcher(search_cfg)
    claim_items: list[dict[str, Any]] = []

    for c in claims:
        stmt = normalize_text(str(c["statement"]))
        gold_ev = normalize_text(str(c.get("gold_evidence", "")))
        true_label = str(c.get("label", "unverifiable")).strip()
        is_hit = bool(c.get("hit_ge_05", False))
        claim_id = str(c.get("claim_id", ""))
        query = str(c.get("query") or stmt)

        res = searcher.search_cached_only(query, variant="Q1")
        hits = res.hits if res is not None else []
        sel_ev = select_evidence_from_web_hits(stmt, hits, top_k=3, max_tokens=150)

        evidence_text = sel_ev.text.strip()
        ev_items = [{"text": it.text, "url": it.url, "domain": it.domain} for it in sel_ev.items]
        if not evidence_text:
            fallback_title = normalize_text(str(c.get("best_title", "")))
            evidence_text = fallback_title
            if fallback_title:
                ev_items = [
                    {"text": fallback_title, "url": str(c.get("best_url", "")), "domain": ""}
                ]

        containment = compute_3gram_containment(gold_ev, evidence_text) if gold_ev and evidence_text else 0.0
        claim_items.append(
            {
                "claim_id": claim_id,
                "statement": stmt,
                "gold_evidence": gold_ev,
                "true_label": true_label,
                "is_hit": is_hit,
                "containment_with_gold": round(containment, 4),
                "evidence_text": evidence_text,
                "evidence_items": ev_items,
                "evidence_num_sentences": len(ev_items),
                "evidence_word_count": sel_ev.word_count,
            }
        )

    # 2. Score relevance r with BAAI/bge-reranker-v2-m3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Computing relevance scores with BgeReranker on %s...", device)
    reranker = BgeReranker(
        cfg.retrieval.reranker.model_name,
        batch_size=16,
        use_fp16=False,
        max_length=512,
        device=device,
    )
    pairs = [(item["statement"], item["evidence_text"]) for item in claim_items]
    r_list = reranker.score_pairs(pairs)
    for item, r_val in zip(claim_items, r_list, strict=True):
        item["relevance_score"] = round(r_val, 4)

    # Free reranker VRAM before loading verifier ensemble
    del reranker
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. Predict calibrated probabilities with pinned V-C 3-seed ensemble
    logger.info("Evaluating pinned V-C 3-seed ensemble on %s...", device)
    ensemble = load_vc_ensemble(cfg, device=device, verify_sha256=True)
    statements = [item["statement"] for item in claim_items]
    evidences = [item["evidence_text"] for item in claim_items]
    ensemble_probs = ensemble.predict_probs_batch(statements, evidences, presegmented=False, batch_size=16)

    # 4. Apply Decision Gate and compare with raw argmax
    results: list[dict[str, Any]] = []
    class_names = [cfg.label_map[k] for k in sorted(cfg.label_map.keys())]

    for item, probs in zip(claim_items, ensemble_probs, strict=True):
        r_val = float(item["relevance_score"])
        raw_idx = int(np.argmax(probs))
        raw_verdict = class_names[raw_idx]
        raw_conf = float(probs[raw_idx])
        dec = gate.decide(r_val, probs)

        res_item = dict(item)
        res_item.update(
            {
                "raw_verdict": raw_verdict,
                "raw_confidence": round(raw_conf, 4),
                "ensemble_probs": {c: round(float(probs[i]), 4) for i, c in enumerate(class_names)},
                "gated_verdict": dec.verdict,
                "show_probs": dec.show_probs,
                "passed_relevance": dec.passed_relevance,
                "passed_confidence": dec.passed_confidence,
                "decision_reason": dec.reason,
                "is_correct_raw": raw_verdict == item["true_label"],
                "is_correct_gated": dec.verdict == item["true_label"],
            }
        )
        results.append(res_item)

    ensemble.unload()
    del ensemble
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 5. Aggregate metrics (§12.4 compliant)
    hits = [r for r in results if r["is_hit"]]
    misses = [r for r in results if not r["is_hit"]]
    n_always_unv_correct = sum(1 for r in results if r["true_label"] == "unverifiable")

    report = {
        "num_claims": len(results),
        "hit_claims_count": len(hits),
        "miss_claims_count": len(misses),
        "tau_rel": tau_rel,
        "tau_conf": tau_conf,
        "always_unverifiable_baseline_accuracy_pct": round(
            n_always_unv_correct / max(len(results), 1) * 100.0, 2
        ),
        "overall_summary": {
            "raw_verifier": _selective_metrics(results, "raw_verdict"),
            "gated_decision": _selective_metrics(results, "gated_verdict"),
        },
        "hit_group": {
            "count": len(hits),
            "avg_relevance": round(float(np.mean([r["relevance_score"] for r in hits])), 4),
            "raw_verifier_dist": _dist_summary(hits, "raw_verdict"),
            "gated_decision_dist": _dist_summary(hits, "gated_verdict"),
            "selective_metrics_raw": _selective_metrics(hits, "raw_verdict"),
            "selective_metrics_gated": _selective_metrics(hits, "gated_verdict"),
            "reasons": {
                REASON_LOW_RELEVANCE: sum(1 for r in hits if r["decision_reason"] == REASON_LOW_RELEVANCE),
                REASON_LOW_CONFIDENCE: sum(1 for r in hits if r["decision_reason"] == REASON_LOW_CONFIDENCE),
                REASON_CONFIRMED: sum(1 for r in hits if r["decision_reason"] == REASON_CONFIRMED),
            },
        },
        "miss_group": {
            "count": len(misses),
            "avg_relevance": round(float(np.mean([r["relevance_score"] for r in misses])), 4),
            "raw_verifier_dist": _dist_summary(misses, "raw_verdict"),
            "gated_decision_dist": _dist_summary(misses, "gated_verdict"),
            "abstain_rate_gated_pct": round(
                sum(1 for r in misses if r["gated_verdict"] == "unverifiable") / max(len(misses), 1) * 100.0,
                2,
            ),
            "false_affirmations_raw": sum(1 for r in misses if r["raw_verdict"] in {"real", "fake"}),
            "false_affirmations_gated": sum(1 for r in misses if r["gated_verdict"] in {"real", "fake"}),
            "reasons": {
                REASON_LOW_RELEVANCE: sum(1 for r in misses if r["decision_reason"] == REASON_LOW_RELEVANCE),
                REASON_LOW_CONFIDENCE: sum(1 for r in misses if r["decision_reason"] == REASON_LOW_CONFIDENCE),
                REASON_CONFIRMED: sum(1 for r in misses if r["decision_reason"] == REASON_CONFIRMED),
            },
        },
        "by_true_label": {
            lbl: {
                "count": len([r for r in results if r["true_label"] == lbl]),
                "raw_dist": _dist_summary([r for r in results if r["true_label"] == lbl], "raw_verdict"),
                "gated_dist": _dist_summary([r for r in results if r["true_label"] == lbl], "gated_verdict"),
            }
            for lbl in ("real", "fake", "unverifiable")
        },
        "claims": results,
    }

    out_path = cfg.paths.artifacts_dir / "mini_e2e_tavily_offline.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved offline mini-E2E report to %s", out_path)


if __name__ == "__main__":
    main()
