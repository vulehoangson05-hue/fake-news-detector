"""Task 4.2 Runner: Tune Relevance Gate, Confidence Threshold & Ensemble ECE (Brief #5 §3, §11.5, §12.3).

Evaluates BAAI/bge-reranker-v2-m3 + pinned V-C 3-seed ensemble on:
- Positive pairs: (claim, gold evidence) + (claim, selected evidence with containment >= 0.5)
- Negative pairs (3 types):
  (i)   random from other article (723 pairs)
  (ii)  hard BM25 from other article with containment < 0.3 (712 pairs)
  (iii) same-article different-sentence with containment < 0.3 (723 pairs)

Produces `artifacts/decision_gate_tuning.json` containing:
- Separate & combined ROC-AUC metrics
- Multi-threshold operating table (tau_rel in {0.5, 0.6, 0.7, 0.7559, 0.8, 0.9}) (§12.3.1)
- Same-article before vs after DecisionGate distributions (§12.3.2)
- tau_conf threshold sweep ({0.50, 0.55, 0.60, 0.65}) across dev & Tavily 30 claims (§12.3.3)
- Unified 30-claim cached Tavily sanity check (§12.3.5)
- 3-seed V-C Ensemble in-sample ECE on dev gold & dev selected (§12.3.6)
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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve

from vnfake.config import load_config, load_search_config
from vnfake.data.cleaning import normalize_text
from vnfake.pipeline.decision import DecisionGate, load_vc_ensemble
from vnfake.retrieval.evidence_selector import (
    compute_3gram_containment,
    select_evidence_from_context,
)
from vnfake.retrieval.reranker import BgeReranker
from vnfake.retrieval.web_search import TavilyWebSearcher, select_evidence_from_web_hits
from vnfake.training.metrics import compute_ece
from vnfake.utils.logging import get_logger

logger = get_logger(__name__)

_TAU_REL_SWEEP = (0.50, 0.60, 0.70, 0.7559, 0.80, 0.90)
_TAU_CONF_SWEEP = (0.50, 0.55, 0.60, 0.65)


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    with p.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_unified_tavily_spike_items(
    spike_path: Path,
) -> list[dict[str, Any]]:
    """Load 30 Q1 spike claims with exact live-equivalent BM25 sourced evidence selection."""
    spike_data = json.loads(spike_path.read_text(encoding="utf-8"))
    claims_q1 = spike_data.get("variants", {}).get("Q1", {}).get("claims", [])
    search_cfg = load_search_config()
    searcher = TavilyWebSearcher(search_cfg)

    items: list[dict[str, Any]] = []
    for c in claims_q1:
        stmt = normalize_text(str(c["statement"]))
        gold_ev = normalize_text(str(c.get("gold_evidence", "")))
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
                    {
                        "text": fallback_title,
                        "url": str(c.get("best_url", "")),
                        "domain": "",
                    }
                ]

        containment = compute_3gram_containment(gold_ev, evidence_text) if gold_ev and evidence_text else 0.0
        items.append(
            {
                "claim_id": str(c.get("claim_id", "")),
                "statement": stmt,
                "gold_evidence": gold_ev,
                "true_label": str(c.get("label", "unverifiable")).strip(),
                "is_hit": bool(c.get("hit_ge_05", False)),
                "query": query,
                "evidence_text": evidence_text,
                "evidence_items": ev_items,
                "evidence_num_sentences": len(ev_items),
                "evidence_word_count": sel_ev.word_count,
                "containment_with_gold": round(containment, 4),
            }
        )
    return items


def _eval_ensemble_probs(
    probs: np.ndarray,
    true_labels: list[int],
) -> dict[str, Any]:
    y_true = np.asarray(true_labels, dtype=np.int64)
    preds = np.argmax(probs, axis=-1)
    acc = float(accuracy_score(y_true, preds))
    mf1 = float(f1_score(y_true, preds, average="macro"))
    ece = float(compute_ece(probs, y_true))
    return {
        "num_samples": len(y_true),
        "accuracy": round(acc, 4),
        "macro_f1": round(mf1, 4),
        "ece_in_sample": round(ece, 4),
    }


def _gate_dist(
    probs: np.ndarray,
    r_scores: list[float],
    tau_rel: float,
    tau_conf: float,
    class_names: list[str],
) -> dict[str, Any]:
    gate = DecisionGate(tau_rel=tau_rel, tau_conf=tau_conf, class_names=class_names)
    counts = {"real": 0, "fake": 0, "unverifiable": 0}
    reason_counts = {"low_rel": 0, "low_conf": 0, "confirmed": 0}
    n = len(r_scores)
    for r_val, p_vec in zip(r_scores, probs, strict=True):
        dec = gate.decide(r_val, p_vec)
        counts[dec.verdict] += 1
        if not dec.passed_relevance:
            reason_counts["low_rel"] += 1
        elif not dec.passed_confidence:
            reason_counts["low_conf"] += 1
        else:
            reason_counts["confirmed"] += 1
    return {
        "count": n,
        "unverifiable_pct": round(counts["unverifiable"] / max(n, 1) * 100.0, 2),
        "real_pct": round(counts["real"] / max(n, 1) * 100.0, 2),
        "fake_pct": round(counts["fake"] / max(n, 1) * 100.0, 2),
        "leakage_real_fake_pct": round((counts["real"] + counts["fake"]) / max(n, 1) * 100.0, 2),
        "counts": counts,
        "reasons": reason_counts,
    }


def main() -> None:
    cfg = load_config()
    proc_dir = cfg.paths.processed_dir
    interim_dir = cfg.paths.interim_dir

    dev_records = read_jsonl(proc_dir / "dev_clean.jsonl")
    dist_dev_records = read_jsonl(interim_dir / "distractor_dev.jsonl")
    same_art_records = read_jsonl(interim_dir / "distractor_same_article_dev.jsonl")
    label_str_to_int = {v: k for k, v in cfg.label_map.items()}
    class_names = [cfg.label_map[i] for i in sorted(cfg.label_map)]

    logger.info(
        "Loaded %d dev records, %d distractor records, %d same-article records",
        len(dev_records),
        len(dist_dev_records),
        len(same_art_records),
    )

    # 1. Prepare Positive & Negative Pairs + Selected Texts
    dev_stmts = [normalize_text(str(r["Statement"])) for r in dev_records]
    dev_golds = [normalize_text(str(r["Evidence"])) for r in dev_records]
    dev_true_labels = [int(r["labels"]) for r in dev_records]

    dev_selected_texts: list[str] = []
    pos_pairs: list[list[str]] = [[st, g] for st, g in zip(dev_stmts, dev_golds, strict=True)]
    selected_pos_count = 0

    for st, g, r in zip(dev_stmts, dev_golds, dev_records, strict=True):
        sel = select_evidence_from_context(st, str(r.get("Context", "")))
        sel_text = normalize_text(sel.text)
        dev_selected_texts.append(sel_text)
        if compute_3gram_containment(g, sel_text) >= 0.5:
            pos_pairs.append([st, sel_text])
            selected_pos_count += 1

    neg_random: list[list[str]] = []
    neg_hard_bm25: list[list[str]] = []
    for r in dist_dev_records:
        stmt = normalize_text(str(r["statement"]))
        ev = normalize_text(str(r["distractor_evidence"]))
        dtype = r.get("distractor_type")
        c_gold = float(r.get("containment_with_gold", 0.0))
        if dtype == "random":
            neg_random.append([stmt, ev])
        elif dtype == "hard_bm25" and c_gold < 0.3:
            neg_hard_bm25.append([stmt, ev])

    same_stmts = [normalize_text(str(r["statement"])) for r in same_art_records]
    same_evids = [normalize_text(str(r["distractor_evidence"])) for r in same_art_records]
    neg_same_article: list[list[str]] = [
        [st, ev] for st, ev in zip(same_stmts, same_evids, strict=True)
    ]

    # Load unified 30 cached Tavily claims
    spike_path = cfg.paths.artifacts_dir / "tavily_spike_report.json"
    tavily_items = load_unified_tavily_spike_items(spike_path) if spike_path.exists() else []

    # 2. Score / Load Reranker scores
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scores_cache_file = cfg.paths.artifacts_dir / "cache" / "dev_reranker_scores.json"
    scores_cache_file.parent.mkdir(parents=True, exist_ok=True)

    reranker_wrapper: BgeReranker | None = None
    if scores_cache_file.exists():
        logger.info("Loading cached dev reranker scores from %s...", scores_cache_file)
        cached_scores = json.loads(scores_cache_file.read_text(encoding="utf-8"))
        pos_scores = [float(s) for s in cached_scores["pos_scores"]]
        rand_scores = [float(s) for s in cached_scores["rand_scores"]]
        hard_scores = [float(s) for s in cached_scores["hard_scores"]]
        same_scores = [float(s) for s in cached_scores["same_scores"]]
    else:
        reranker_wrapper = BgeReranker(
            cfg.retrieval.reranker.model_name,
            batch_size=32,
            use_fp16=False,
            max_length=512,
            device=device,
        )
        pos_scores = reranker_wrapper.score_pairs(pos_pairs)
        rand_scores = reranker_wrapper.score_pairs(neg_random)
        hard_scores = reranker_wrapper.score_pairs(neg_hard_bm25)
        same_scores = reranker_wrapper.score_pairs(neg_same_article)
        scores_cache_file.write_text(
            json.dumps(
                {
                    "pos_scores": pos_scores,
                    "rand_scores": rand_scores,
                    "hard_scores": hard_scores,
                    "same_scores": same_scores,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    assert len(same_scores) == len(same_art_records)

    # Always score unified 30 Tavily claims cleanly (cached if unchanged, takes ~1.5s)
    tavily_r_scores: list[float] = []
    if tavily_items:
        if reranker_wrapper is None:
            reranker_wrapper = BgeReranker(
                cfg.retrieval.reranker.model_name,
                batch_size=16,
                use_fp16=False,
                max_length=512,
                device=device,
            )
        tavily_pairs = [(it["statement"], it["evidence_text"]) for it in tavily_items]
        tavily_r_scores = reranker_wrapper.score_pairs(tavily_pairs)
        # Also score dev_selected for completeness in threshold study
        dev_sel_scores = cached_scores.get("dev_selected_scores") if scores_cache_file.exists() else None
        if dev_sel_scores is None:
            logger.info("Scoring 723 dev_selected pairs with reranker for tau_conf conditional stats...")
            dev_sel_r_scores = reranker_wrapper.score_pairs(
                list(zip(dev_stmts, dev_selected_texts, strict=True))
            )
            if scores_cache_file.exists():
                cached_scores["dev_selected_scores"] = dev_sel_r_scores
                scores_cache_file.write_text(json.dumps(cached_scores, indent=2), encoding="utf-8")
        else:
            dev_sel_r_scores = [float(s) for s in dev_sel_scores]
    else:
        dev_sel_r_scores = pos_scores[: len(dev_records)]

    # Free reranker VRAM before loading PhoBERT Ensemble!
    if reranker_wrapper is not None:
        del reranker_wrapper
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 3. ROC-AUC & Multi-Threshold Sweep (§12.3.1)
    def _calc_auc(p_scores: list[float], n_scores: list[float]) -> dict[str, Any]:
        y_true = [1] * len(p_scores) + [0] * len(n_scores)
        y_pred = p_scores + n_scores
        auc = float(roc_auc_score(y_true, y_pred))
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))
        return {
            "roc_auc": round(auc, 4),
            "optimal_tau": round(float(thresholds[best_idx]), 4),
            "youden_j": round(float(j_scores[best_idx]), 4),
            "tpr_at_tau": round(float(tpr[best_idx]), 4),
            "fpr_at_tau": round(float(fpr[best_idx]), 4),
            "pos_score_mean": round(float(np.mean(p_scores)), 4),
            "pos_score_std": round(float(np.std(p_scores)), 4),
            "neg_score_mean": round(float(np.mean(n_scores)), 4),
            "neg_score_std": round(float(np.std(n_scores)), 4),
        }

    auc_random = _calc_auc(pos_scores, rand_scores)
    auc_hard = _calc_auc(pos_scores, hard_scores)
    auc_same = _calc_auc(pos_scores, same_scores)
    combined_neg_scores = rand_scores + hard_scores + same_scores
    auc_combined = _calc_auc(pos_scores, combined_neg_scores)

    gold_pos_scores = pos_scores[: len(dev_records)]
    operating_table: list[dict[str, Any]] = []
    for tau in _TAU_REL_SWEEP:
        tpr_all = sum(1 for s in pos_scores if s >= tau) / len(pos_scores)
        tpr_gold = sum(1 for s in gold_pos_scores if s >= tau) / len(gold_pos_scores)
        fpr_rand = sum(1 for s in rand_scores if s >= tau) / len(rand_scores)
        fpr_hard = sum(1 for s in hard_scores if s >= tau) / len(hard_scores)
        fpr_same = sum(1 for s in same_scores if s >= tau) / len(same_scores)
        fpr_comb = sum(1 for s in combined_neg_scores if s >= tau) / len(combined_neg_scores)
        operating_table.append(
            {
                "tau_rel": tau,
                "tpr_pos_all_pct": round(tpr_all * 100.0, 2),
                "tpr_pos_gold_pct": round(tpr_gold * 100.0, 2),
                "fpr_random_pct": round(fpr_rand * 100.0, 2),
                "fpr_hard_bm25_pct": round(fpr_hard * 100.0, 2),
                "fpr_same_article_pct": round(fpr_same * 100.0, 2),
                "fpr_combined_pct": round(fpr_comb * 100.0, 2),
                "youden_j_combined": round(tpr_all - fpr_comb, 4),
            }
        )

    # 4. Run / Load 3-seed V-C Ensemble probabilities (§12.3.2, §12.3.3, §12.3.6)
    ens_cache_file = cfg.paths.artifacts_dir / "cache" / "dev_vc_ensemble_probs.npz"
    if ens_cache_file.exists():
        logger.info("Loading cached V-C ensemble probabilities from %s", ens_cache_file)
        npz_data = np.load(ens_cache_file)
        probs_gold = npz_data["probs_gold"]
        probs_selected = npz_data["probs_selected"]
        probs_same = npz_data["probs_same"]
        probs_tavily = npz_data["probs_tavily"]
    else:
        logger.info("Running pinned V-C 3-seed ensemble (SHA-256 verified) on %s...", device)
        ensemble = load_vc_ensemble(cfg, device=device, verify_sha256=True)
        probs_gold = ensemble.predict_probs_batch(dev_stmts, dev_golds, presegmented=False, batch_size=16)
        probs_selected = ensemble.predict_probs_batch(dev_stmts, dev_selected_texts, presegmented=False, batch_size=16)
        probs_same = ensemble.predict_probs_batch(same_stmts, same_evids, presegmented=False, batch_size=16)
        if tavily_items:
            probs_tavily = ensemble.predict_probs_batch(
                [it["statement"] for it in tavily_items],
                [it["evidence_text"] for it in tavily_items],
                presegmented=False,
                batch_size=16,
            )
        else:
            probs_tavily = np.zeros((0, 3), dtype=np.float64)

        np.savez(
            ens_cache_file,
            probs_gold=probs_gold,
            probs_selected=probs_selected,
            probs_same=probs_same,
            probs_tavily=probs_tavily,
        )
        ensemble.unload()
        del ensemble
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 12.3.6: Ensemble In-Sample ECE & Macro-F1
    ensemble_dev_metrics = {
        "dev_gold": _eval_ensemble_probs(probs_gold, dev_true_labels),
        "dev_selected": _eval_ensemble_probs(probs_selected, dev_true_labels),
    }
    logger.info(
        "Ensemble In-Sample ECE: gold=%.4f (F1=%.4f) | selected=%.4f (F1=%.4f)",
        ensemble_dev_metrics["dev_gold"]["ece_in_sample"],
        ensemble_dev_metrics["dev_gold"]["macro_f1"],
        ensemble_dev_metrics["dev_selected"]["ece_in_sample"],
        ensemble_dev_metrics["dev_selected"]["macro_f1"],
    )

    # 12.3.2: Same-Article Distractor Gating Evaluation (before vs after gate)
    same_raw_preds = np.argmax(probs_same, axis=-1)
    same_before_counts = {
        c_name: int(np.sum(same_raw_preds == idx)) for idx, c_name in enumerate(class_names)
    }
    n_same = len(same_art_records)
    same_before = {
        "count": n_same,
        "unverifiable_pct": round(same_before_counts["unverifiable"] / n_same * 100.0, 2),
        "real_pct": round(same_before_counts["real"] / n_same * 100.0, 2),
        "fake_pct": round(same_before_counts["fake"] / n_same * 100.0, 2),
        "leakage_real_fake_pct": round(
            (same_before_counts["real"] + same_before_counts["fake"]) / n_same * 100.0, 2
        ),
        "counts": same_before_counts,
    }
    chosen_tau_rel = auc_combined["optimal_tau"]
    same_after_by_tau_rel = {
        str(t): _gate_dist(probs_same, same_scores, tau_rel=t, tau_conf=0.50, class_names=class_names)
        for t in _TAU_REL_SWEEP
    }
    same_after_at_opt_by_conf = {
        str(tc): _gate_dist(probs_same, same_scores, tau_rel=chosen_tau_rel, tau_conf=tc, class_names=class_names)
        for tc in _TAU_CONF_SWEEP
    }

    # 12.3.3: tau_conf activation frequency sweep on dev & Tavily 30
    def _conf_sweep(
        probs: np.ndarray,
        r_vals: list[float] | None,
        true_lbls: list[int] | None,
    ) -> list[dict[str, Any]]:
        max_p = np.max(probs, axis=-1)
        preds = np.argmax(probs, axis=-1)
        n = len(max_p)
        rows: list[dict[str, Any]] = []
        pass_rel_mask = (
            np.asarray([r >= chosen_tau_rel for r in r_vals], dtype=bool)
            if r_vals is not None
            else np.ones(n, dtype=bool)
        )
        n_pass_rel = int(np.sum(pass_rel_mask))

        for tc in _TAU_CONF_SWEEP:
            low_all = int(np.sum(max_p < tc))
            low_after_rel = int(np.sum((max_p < tc) & pass_rel_mask))

            row: dict[str, Any] = {
                "tau_conf": tc,
                "low_conf_all_count": low_all,
                "low_conf_all_pct": round(low_all / max(n, 1) * 100.0, 2),
                "pass_rel_count": n_pass_rel,
                "low_conf_after_rel_count": low_after_rel,
                "low_conf_after_rel_pct_of_pass_rel": round(
                    low_after_rel / max(n_pass_rel, 1) * 100.0, 2
                ),
            }
            if true_lbls is not None:
                y_arr = np.asarray(true_lbls, dtype=np.int64)
                unv_idx = label_str_to_int["unverifiable"]
                # Gated verdict on full set (with relevance gate if r_vals given, plus conf gate)
                gated_preds = np.where((~pass_rel_mask) | (max_p < tc), unv_idx, preds)
                row["gated_accuracy_pct"] = round(float(np.mean(gated_preds == y_arr)) * 100.0, 2)
                row["gated_macro_f1"] = round(float(f1_score(y_arr, gated_preds, average="macro")), 4)
            rows.append(row)
        return rows

    tau_conf_study = {
        "tau_rel_fixed": chosen_tau_rel,
        "dev_gold": _conf_sweep(probs_gold, gold_pos_scores, dev_true_labels),
        "dev_selected": _conf_sweep(probs_selected, dev_sel_r_scores, dev_true_labels),
        "same_article": _conf_sweep(probs_same, same_scores, None),
        "tavily_30": _conf_sweep(
            probs_tavily,
            tavily_r_scores,
            [label_str_to_int[it["true_label"]] for it in tavily_items],
        )
        if tavily_items
        else [],
    }

    # Sanity check summary on 30 Tavily claims (exact unified values matching script 22)
    tavily_sanity: dict[str, Any] = {}
    if tavily_items:
        hit_scores = [r for it, r in zip(tavily_items, tavily_r_scores, strict=True) if it["is_hit"]]
        miss_scores = [r for it, r in zip(tavily_items, tavily_r_scores, strict=True) if not it["is_hit"]]
        claim_rows = [
            {
                "claim_id": it["claim_id"],
                "label": it["true_label"],
                "is_hit": it["is_hit"],
                "r_score": round(r_s, 4),
                "passes_tau_opt": r_s >= chosen_tau_rel,
                "passes_tau_05": r_s >= 0.50,
            }
            for it, r_s in zip(tavily_items, tavily_r_scores, strict=True)
        ]
        tavily_sanity = {
            "num_claims": len(tavily_items),
            "hit_claims_count": len(hit_scores),
            "miss_claims_count": len(miss_scores),
            "hit_r_mean": round(float(np.mean(hit_scores)), 4) if hit_scores else 0.0,
            "miss_r_mean": round(float(np.mean(miss_scores)), 4) if miss_scores else 0.0,
            "hit_pass_pct_at_optimal_tau": round(
                sum(1 for s in hit_scores if s >= chosen_tau_rel) / max(len(hit_scores), 1) * 100.0,
                2,
            ),
            "miss_pass_pct_at_optimal_tau": round(
                sum(1 for s in miss_scores if s >= chosen_tau_rel) / max(len(miss_scores), 1) * 100.0,
                2,
            ),
            "claims": claim_rows,
        }
        logger.info(
            "Unified Tavily Sanity Check: Hit r_mean=%.4f | Miss r_mean=%.4f",
            tavily_sanity["hit_r_mean"],
            tavily_sanity["miss_r_mean"],
        )

    report = {
        "model_name": cfg.retrieval.reranker.model_name,
        "auc_by_negative_type": {
            "random": auc_random,
            "hard_bm25_filtered": auc_hard,
            "same_article": auc_same,
            "combined": auc_combined,
        },
        "operating_table_tau_rel": operating_table,
        "ensemble_dev_metrics": ensemble_dev_metrics,
        "same_article_gating": {
            "before_gate": same_before,
            "after_gate_by_tau_rel_at_conf_050": same_after_by_tau_rel,
            "after_gate_at_opt_tau_rel_by_tau_conf": same_after_at_opt_by_conf,
        },
        "tau_conf_activation_study": tau_conf_study,
        "chosen_tau_rel": chosen_tau_rel,
        "chosen_tau_conf": cfg.decision.tau_conf,
        "tavily_sanity": tavily_sanity,
    }

    out_file = cfg.paths.artifacts_dir / "decision_gate_tuning.json"
    out_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved upgraded Decision Gate tuning report to %s", out_file)


if __name__ == "__main__":
    main()
