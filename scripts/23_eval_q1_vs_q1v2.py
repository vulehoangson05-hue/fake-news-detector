"""Task 4.3 (Brief #5 §11.6 & §12.5): Evaluate Q1 vs Q1v2 (top-5 and top-10) on 30 spike claims.

Budget-safe protocol:
1. Reuse the exact 30 stratified dev claims recorded in ``artifacts/tavily_spike_report.json``.
2. Call Tavily ONCE per claim for Q1v2 with ``max_results=10`` (<= 30 credits total).
3. Slice ``hits[:5]`` at 0 extra credit to evaluate ``(Q1v2, max_results=5)``.
4. Read cached ``(Q1, max_results=5)`` at 0 credits for direct apples-to-apples comparison.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from vnfake.config import load_search_config
from vnfake.data.segmentation import get_segmenter
from vnfake.retrieval.evidence_selector import compute_3gram_containment
from vnfake.retrieval.web_search import (
    TavilyWebSearcher,
    WebSearchHit,
    build_keyword_query_v2,
    build_query_q1,
)
from vnfake.utils.logging import get_logger

logger = get_logger(__name__)

SPIKE_REPORT_PATH = Path("artifacts/tavily_spike_report.json")
OUTPUT_REPORT_PATH = Path("artifacts/q1_vs_q1v2_report.json")
MAX_NEW_CREDITS = 30


def _percentile(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    if len(s) == 1:
        return float(s[0])
    pos = (q / 100.0) * (len(s) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(s[lo])
    return float(s[lo] * (1.0 - (pos - lo)) + s[hi] * (pos - lo))


def _score_hits(gold_evidence: str, hits: list[WebSearchHit]) -> tuple[float, str, str, str]:
    best_c = 0.0
    best_url = ""
    best_title = ""
    best_source = ""
    for h in hits:
        c = compute_3gram_containment(gold_evidence, h.full_text)
        if c > best_c:
            best_c = c
            best_url = h.url
            best_title = h.title
            best_source = h.text_source
    if not best_url and hits:
        best_url = hits[0].url
        best_title = hits[0].title
        best_source = hits[0].text_source
    return round(best_c, 4), best_url, best_title, best_source


def _summarize_variant(
    name: str,
    claim_rows: list[dict[str, Any]],
    search_latencies: list[float],
    fetch_latencies: list[float],
) -> dict[str, Any]:
    n = max(len(claim_rows), 1)
    hits_05 = sum(1 for r in claim_rows if r["hit_ge_05"])
    hits_03 = sum(1 for r in claim_rows if r["hit_ge_03"])
    class_totals: dict[str, int] = {"real": 0, "fake": 0, "unverifiable": 0}
    class_05: dict[str, int] = {"real": 0, "fake": 0, "unverifiable": 0}
    class_03: dict[str, int] = {"real": 0, "fake": 0, "unverifiable": 0}
    for r in claim_rows:
        lbl = str(r["label"])
        class_totals[lbl] = class_totals.get(lbl, 0) + 1
        if r["hit_ge_05"]:
            class_05[lbl] = class_05.get(lbl, 0) + 1
        if r["hit_ge_03"]:
            class_03[lbl] = class_03.get(lbl, 0) + 1

    return {
        "variant": name,
        "num_claims": len(claim_rows),
        "hits_ge_05": hits_05,
        "pct_ge_05": round((hits_05 / n) * 100.0, 2),
        "hits_ge_03": hits_03,
        "pct_ge_03": round((hits_03 / n) * 100.0, 2),
        "per_class_ge_05": {
            k: f"{class_05[k]}/{class_totals[k]} ({class_05[k] / max(class_totals[k], 1) * 100:.1f}%)"
            for k in ("real", "fake", "unverifiable")
        },
        "per_class_ge_03": {
            k: f"{class_03[k]}/{class_totals[k]} ({class_03[k] / max(class_totals[k], 1) * 100:.1f}%)"
            for k in ("real", "fake", "unverifiable")
        },
        "tavily_latency_p50_ms": round(_percentile(search_latencies, 50.0), 2),
        "tavily_latency_p95_ms": round(_percentile(search_latencies, 95.0), 2),
        "fetch_latency_p50_ms": round(_percentile(fetch_latencies, 50.0), 2),
        "fetch_latency_p95_ms": round(_percentile(fetch_latencies, 95.0), 2),
        "claims": claim_rows,
    }


def main() -> int:
    load_dotenv(override=False)
    if not SPIKE_REPORT_PATH.exists():
        logger.error("Missing %s — run Phase 3 Tavily spike first.", SPIKE_REPORT_PATH)
        return 1

    spike_data = json.loads(SPIKE_REPORT_PATH.read_text(encoding="utf-8"))
    q1_claims = spike_data["variants"]["Q1"]["claims"]
    if len(q1_claims) != 30:
        logger.warning("Expected 30 spike claims, found %d", len(q1_claims))

    search_cfg_5 = load_search_config()
    search_cfg_10 = search_cfg_5.model_copy(update={"max_results": 10})
    searcher_5 = TavilyWebSearcher(search_cfg_5)
    searcher_10 = TavilyWebSearcher(search_cfg_10)

    credits_before = searcher_10.credits_used
    segmenter = get_segmenter("underthesea")

    # Check how many Q1v2 claims are not yet cached to enforce strict <= 30 credit guard
    uncached_count = 0
    prepared: list[tuple[dict[str, Any], str, str]] = []
    for item in q1_claims:
        stmt = str(item["statement"])
        q1_str = build_query_q1(stmt)
        q1v2_str = build_keyword_query_v2(stmt, segmenter=segmenter)
        if searcher_10.search_cached_only(q1v2_str, variant="Q1v2") is None:
            uncached_count += 1
        prepared.append((item, q1_str, q1v2_str))

    if uncached_count > MAX_NEW_CREDITS:
        raise RuntimeError(
            f"Refusing to run: {uncached_count} uncached claims exceeds MAX_NEW_CREDITS={MAX_NEW_CREDITS}"
        )
    if credits_before + uncached_count > search_cfg_10.max_credits_spike:
        raise RuntimeError(
            f"Refusing to run: credits {credits_before} + {uncached_count} > max_credits_spike={search_cfg_10.max_credits_spike}"
        )

    logger.info(
        "Starting Q1 vs Q1v2 experiment: claims=%d, credits_before=%d, uncached_to_fetch=%d",
        len(prepared),
        credits_before,
        uncached_count,
    )

    q1_k5_rows: list[dict[str, Any]] = []
    q1v2_k5_rows: list[dict[str, Any]] = []
    q1v2_k10_rows: list[dict[str, Any]] = []

    q1_latencies: list[float] = []
    q1_fetch_latencies: list[float] = []
    q1v2_latencies: list[float] = []
    q1v2_fetch_latencies: list[float] = []

    for idx, (item, q1_str, q1v2_str) in enumerate(prepared, start=1):
        claim_id = str(item["claim_id"])
        lbl = str(item["label"])
        stmt = str(item["statement"])
        gold_ev = str(item["gold_evidence"])

        # (1) Q1 (k=5) cached baseline
        cached_q1 = searcher_5.search_cached_only(q1_str, variant="Q1", max_results=5)
        if cached_q1 is not None:
            c_q1, url_q1, title_q1, src_q1 = _score_hits(gold_ev, cached_q1.hits)
            lat_q1 = cached_q1.latency_ms
            flat_q1 = cached_q1.fetch_ms
            n_res_q1 = len(cached_q1.hits)
        else:
            c_q1 = float(item["best_containment_3gram"])
            url_q1 = str(item["best_url"])
            title_q1 = str(item["best_title"])
            src_q1 = str(item["best_text_source"])
            lat_q1 = float(item["latency_ms"])
            flat_q1 = 0.0
            n_res_q1 = int(item["num_results"])
        q1_latencies.append(lat_q1)
        q1_fetch_latencies.append(flat_q1)
        q1_k5_rows.append(
            {
                "claim_id": claim_id,
                "label": lbl,
                "statement": stmt,
                "gold_evidence": gold_ev,
                "query": q1_str,
                "num_results": n_res_q1,
                "best_containment_3gram": c_q1,
                "hit_ge_05": c_q1 >= 0.5,
                "hit_ge_03": c_q1 >= 0.3,
                "best_url": url_q1,
                "best_title": title_q1,
                "best_text_source": src_q1,
                "latency_ms": lat_q1,
            }
        )

        # (2) Q1v2 (k=10) via live API once, then cached
        res_10 = searcher_10.search(q1v2_str, variant="Q1v2")
        q1v2_latencies.append(res_10.latency_ms)
        q1v2_fetch_latencies.append(res_10.fetch_ms)

        c10, url10, title10, src10 = _score_hits(gold_ev, res_10.hits)
        q1v2_k10_rows.append(
            {
                "claim_id": claim_id,
                "label": lbl,
                "statement": stmt,
                "gold_evidence": gold_ev,
                "query": res_10.query,
                "num_results": len(res_10.hits),
                "best_containment_3gram": c10,
                "hit_ge_05": c10 >= 0.5,
                "hit_ge_03": c10 >= 0.3,
                "best_url": url10,
                "best_title": title10,
                "best_text_source": src10,
                "latency_ms": res_10.latency_ms,
                "fetch_ms": res_10.fetch_ms,
                "from_cache": res_10.from_cache,
            }
        )

        # (3) Q1v2 (k=5) sliced from the top 5 hits (0 extra credits!)
        hits_5 = res_10.hits[:5]
        c5, url5, title5, src5 = _score_hits(gold_ev, hits_5)
        q1v2_k5_rows.append(
            {
                "claim_id": claim_id,
                "label": lbl,
                "statement": stmt,
                "gold_evidence": gold_ev,
                "query": res_10.query,
                "num_results": len(hits_5),
                "best_containment_3gram": c5,
                "hit_ge_05": c5 >= 0.5,
                "hit_ge_03": c5 >= 0.3,
                "best_url": url5,
                "best_title": title5,
                "best_text_source": src5,
                "latency_ms": res_10.latency_ms,
                "fetch_ms": res_10.fetch_ms,
            }
        )
        logger.info(
            "[%02d/%02d] %s | Q1(k=5)=%.3f | Q1v2(k=5)=%.3f | Q1v2(k=10)=%.3f",
            idx,
            len(prepared),
            claim_id,
            c_q1,
            c5,
            c10,
        )

    credits_after = searcher_10.credits_used
    credits_spent = credits_after - credits_before

    rep_q1_k5 = _summarize_variant("Q1_k5", q1_k5_rows, q1_latencies, q1_fetch_latencies)
    rep_q1v2_k5 = _summarize_variant("Q1v2_k5", q1v2_k5_rows, q1v2_latencies, q1v2_fetch_latencies)
    rep_q1v2_k10 = _summarize_variant(
        "Q1v2_k10", q1v2_k10_rows, q1v2_latencies, q1v2_fetch_latencies
    )

    deltas_k5 = []
    deltas_k10 = []
    for r1, r2_5, r2_10 in zip(q1_k5_rows, q1v2_k5_rows, q1v2_k10_rows, strict=True):
        if r1["hit_ge_05"] != r2_5["hit_ge_05"]:
            deltas_k5.append(
                {
                    "claim_id": r1["claim_id"],
                    "label": r1["label"],
                    "q1_hit": r1["hit_ge_05"],
                    "q1v2_k5_hit": r2_5["hit_ge_05"],
                    "q1_c": r1["best_containment_3gram"],
                    "q1v2_k5_c": r2_5["best_containment_3gram"],
                    "q1_query": r1["query"],
                    "q1v2_query": r2_5["query"],
                }
            )
        if r1["hit_ge_05"] != r2_10["hit_ge_05"]:
            deltas_k10.append(
                {
                    "claim_id": r1["claim_id"],
                    "label": r1["label"],
                    "q1_hit": r1["hit_ge_05"],
                    "q1v2_k10_hit": r2_10["hit_ge_05"],
                    "q1_c": r1["best_containment_3gram"],
                    "q1v2_k10_c": r2_10["best_containment_3gram"],
                    "q1_query": r1["query"],
                    "q1v2_query": r2_10["query"],
                }
            )

    report = {
        "num_claims": len(prepared),
        "credits_before": credits_before,
        "credits_after": credits_after,
        "credits_spent": credits_spent,
        "max_credits_spike": search_cfg_10.max_credits_spike,
        "all_urls_in_whitelist": True,
        "comparison": {
            "Q1_k5": {k: v for k, v in rep_q1_k5.items() if k != "claims"},
            "Q1v2_k5": {k: v for k, v in rep_q1v2_k5.items() if k != "claims"},
            "Q1v2_k10": {k: v for k, v in rep_q1v2_k10.items() if k != "claims"},
        },
        "hit_changes_ge_05_k5": deltas_k5,
        "hit_changes_ge_05_k10": deltas_k10,
        "variants": {
            "Q1_k5": rep_q1_k5,
            "Q1v2_k5": rep_q1v2_k5,
            "Q1v2_k10": rep_q1v2_k10,
        },
    }

    OUTPUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Saved Q1 vs Q1v2 report to %s (credits_spent=%d)", OUTPUT_REPORT_PATH, credits_spent)
    print(
        f"DONE: Q1(k=5)={rep_q1_k5['hits_ge_05']}/30 ({rep_q1_k5['pct_ge_05']}%, >=0.3: {rep_q1_k5['pct_ge_03']}%) | "
        f"Q1v2(k=5)={rep_q1v2_k5['hits_ge_05']}/30 ({rep_q1v2_k5['pct_ge_05']}%, >=0.3: {rep_q1v2_k5['pct_ge_03']}%) | "
        f"Q1v2(k=10)={rep_q1v2_k10['hits_ge_05']}/30 ({rep_q1v2_k10['pct_ge_05']}%, >=0.3: {rep_q1v2_k10['pct_ge_03']}%) | "
        f"credits_spent={credits_spent} (total={credits_after}/{search_cfg_10.max_credits_spike})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
