"""Phase 2.5 Task 5a (Brief #3 Phần B) — Clean k-sweep at fp32, batch_size=4.

Evaluates full dev (723 claims) across k in {20, 30, 50} under identical
conditions (dtype=fp32, batch_size=4, max_length=512, seed=42, no Gemini) to
separate the effect of retrieve_k from the Phase 2 fp16 vs fp32 confound.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch

from vnfake.config import AppConfig, load_config
from vnfake.evaluation.retrieval_eval import (
    EvalQuery,
    _percentile,
    compute_ranking_metrics,
    hit_rate_at_k,
    iter_eval_queries,
    make_proxy_matcher,
    resolve_collection_name,
)
from vnfake.retrieval.embedder import BgeM3Embedder
from vnfake.retrieval.reranker import BgeReranker
from vnfake.retrieval.vector_store import QdrantStore
from vnfake.utils.logging import get_logger
from vnfake.utils.seed import set_seed

logger = get_logger(__name__)

_DEFAULT_OUT_JSON = Path("artifacts/profiling/k_sweep_fp32_b4.json")
_DEFAULT_K_VALUES: tuple[int, ...] = (20, 30, 50)
_MS_PER_SECOND = 1000.0
_PERCENTILE_P50 = 50.0
_PERCENTILE_P95 = 95.0
_TOP_K_RERANK = 5


@dataclass(frozen=True, slots=True)
class KSweepEntryResult:
    """Aggregated metrics for one retrieve_k value on full dev."""

    retrieve_k: int
    rerank_k: int
    dtype: str
    batch_size: int
    max_length: int
    num_queries: int
    # Pre-rerank metrics
    pre_hit_at_1_count: int
    pre_hit_at_1_rate: float
    pre_hit_at_3_count: int
    pre_hit_at_3_rate: float
    pre_hit_at_5_count: int
    pre_hit_at_5_rate: float
    ceiling_hit_at_k_count: int
    ceiling_hit_at_k_rate: float
    # Post-rerank metrics
    post_hit_at_1_count: int
    post_hit_at_1_rate: float
    post_hit_at_3_count: int
    post_hit_at_3_rate: float
    post_hit_at_5_count: int
    post_hit_at_5_rate: float
    post_mrr_at_5: float
    post_ndcg_at_5: float
    # Latency
    search_p50_ms: float
    search_p95_ms: float
    rerank_p50_ms: float
    rerank_p95_ms: float
    rerank_mean_ms: float


@dataclass(frozen=True, slots=True)
class KSweepReport:
    """Complete k-sweep report across k in {20, 30, 50}."""

    timestamp_utc: str
    collection: str
    collection_point_count: int
    seed: int
    num_queries: int
    dtype: str
    batch_size: int
    max_length: int
    embed_total_ms: float
    embed_per_query_ms: float
    results_by_k: list[KSweepEntryResult]
    top5_set_overlap: dict[str, float]
    top1_exact_agreement: dict[str, float]


def compute_top5_set_overlap(
    top5_ids_a: Sequence[Sequence[str]],
    top5_ids_b: Sequence[Sequence[str]],
    *,
    top_n: int = _TOP_K_RERANK,
) -> float:
    """Mean fraction of shared chunk_ids in top-5 between two k settings."""
    if not top5_ids_a or len(top5_ids_a) != len(top5_ids_b):
        return 0.0
    scores: list[float] = []
    for a, b in zip(top5_ids_a, top5_ids_b, strict=True):
        set_a = set(a[:top_n])
        set_b = set(b[:top_n])
        denom = max(1, min(top_n, max(len(set_a), len(set_b))))
        scores.append(len(set_a & set_b) / float(denom))
    return sum(scores) / len(scores)


def compute_top1_agreement(
    top5_ids_a: Sequence[Sequence[str]],
    top5_ids_b: Sequence[Sequence[str]],
) -> float:
    """Fraction of queries where the #1 reranked chunk_id is identical."""
    if not top5_ids_a or len(top5_ids_a) != len(top5_ids_b):
        return 0.0
    matches = sum(
        1 for a, b in zip(top5_ids_a, top5_ids_b, strict=True) if a and b and a[0] == b[0]
    )
    return matches / len(top5_ids_a)


def run_k_sweep(
    cfg: AppConfig,
    *,
    k_values: Sequence[int] = _DEFAULT_K_VALUES,
    batch_size: int = 4,
    dtype: str = "fp32",
    max_length: int = 512,
) -> KSweepReport:
    """Execute clean k-sweep on full dev for k in k_values."""
    for k in k_values:
        if k < 20:
            raise ValueError(f"Hard guardrail violated: retrieve_k={k} < 20 is forbidden.")

    set_seed(cfg.seed)
    dev_path = cfg.paths.processed_dir / "dev_clean.jsonl"
    queries: list[EvalQuery] = list(iter_eval_queries(dev_path))
    n_queries = len(queries)
    logger.info("Loaded %d dev queries from %s", n_queries, dev_path)

    collection = resolve_collection_name(cfg)
    store = QdrantStore.from_env(
        dense_name=cfg.retrieval.dense_vector_name,
        sparse_name=cfg.retrieval.sparse_vector_name,
    )
    point_count = int(store._client.count(collection_name=collection).count)
    logger.info("Verified Qdrant collection %s has %d points", collection, point_count)

    # Step 1: Batch-encode all dev claims once and unload embedder to free VRAM.
    statements = [q.statement for q in queries]
    embedder = BgeM3Embedder(
        model_name=cfg.retrieval.embedding.model_name,
        dim=cfg.retrieval.embedding.dim,
        use_fp16=cfg.retrieval.embedding.use_fp16,
        weight_threshold=cfg.retrieval.sparse.weight_threshold,
    )
    logger.info("Batch-encoding %d dev queries with bge-m3...", n_queries)
    t_emb0 = time.perf_counter()
    dense_vecs = embedder.embed_dense(statements, batch_size=cfg.retrieval.embedding.batch_size)
    sparse_vecs = embedder.embed_sparse(statements, batch_size=cfg.retrieval.sparse.batch_size)
    embed_total_ms = (time.perf_counter() - t_emb0) * _MS_PER_SECOND
    embedder.unload()
    logger.info("Finished embedding in %.2f ms; unloaded bge-m3 from VRAM.", embed_total_ms)

    # Step 2: Initialize BgeReranker (fp32, batch_size=4) and warm up.
    use_fp16 = dtype == "fp16"
    reranker = BgeReranker(
        model_name=cfg.retrieval.reranker.model_name,
        batch_size=batch_size,
        use_fp16=use_fp16,
        max_length=max_length,
    )
    # Warm up model weights and CUDA kernels on first query before timing k-sweep
    warmup_cands = store.search_hybrid(
        collection,
        dense_query=dense_vecs[0],
        sparse_query=sparse_vecs[0],
        top_k=20,
        dense_prefetch_k=cfg.retrieval.hybrid.dense_prefetch_k,
        sparse_prefetch_k=cfg.retrieval.hybrid.sparse_prefetch_k,
    )
    _ = reranker.rerank(queries[0].statement, warmup_cands, top_k=_TOP_K_RERANK)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    logger.info("Reranker initialized and warmed up (dtype=%s, batch_size=%d).", dtype, batch_size)

    matcher = make_proxy_matcher()
    results_by_k: list[KSweepEntryResult] = []
    top5_ids_by_k: dict[int, list[list[str]]] = {}

    for k in k_values:
        logger.info(
            "=== Running k-sweep for k=%d (n=%d, dtype=%s, batch=%d) ===",
            k,
            n_queries,
            dtype,
            batch_size,
        )
        search_latencies: list[float] = []
        rerank_latencies: list[float] = []
        pre_relevances: list[list[bool]] = []
        post_relevances: list[list[bool]] = []
        top5_ids_for_k: list[list[str]] = []

        for idx, (query, d_vec, s_vec) in enumerate(
            zip(queries, dense_vecs, sparse_vecs, strict=True), start=1
        ):
            t0 = time.perf_counter()
            candidates = store.search_hybrid(
                collection,
                dense_query=d_vec,
                sparse_query=s_vec,
                top_k=k,
                dense_prefetch_k=cfg.retrieval.hybrid.dense_prefetch_k,
                sparse_prefetch_k=cfg.retrieval.hybrid.sparse_prefetch_k,
            )
            search_ms = (time.perf_counter() - t0) * _MS_PER_SECOND
            search_latencies.append(search_ms)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            reranked = reranker.rerank(query.statement, candidates, top_k=_TOP_K_RERANK)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            rerank_ms = (time.perf_counter() - t1) * _MS_PER_SECOND
            rerank_latencies.append(rerank_ms)

            pre_rels = [matcher(query, c) for c in candidates]
            post_rels = [matcher(query, c) for c in reranked]
            pre_relevances.append(pre_rels)
            post_relevances.append(post_rels)
            top5_ids_for_k.append([c.chunk_id for c in reranked])

            if idx % 100 == 0 or idx == n_queries:
                logger.info(
                    "[k=%d] Processed %d/%d claims | running rerank p50=%.2f ms, p95=%.2f ms",
                    k,
                    idx,
                    n_queries,
                    _percentile(rerank_latencies, _PERCENTILE_P50),
                    _percentile(rerank_latencies, _PERCENTILE_P95),
                )

        top5_ids_by_k[k] = top5_ids_for_k

        pre_hit_1_cnt = int(sum(hit_rate_at_k(r, 1) for r in pre_relevances))
        pre_hit_3_cnt = int(sum(hit_rate_at_k(r, 3) for r in pre_relevances))
        pre_hit_5_cnt = int(sum(hit_rate_at_k(r, 5) for r in pre_relevances))
        ceil_hit_k_cnt = int(sum(hit_rate_at_k(r, k) for r in pre_relevances))

        post_hit_1_cnt = int(sum(hit_rate_at_k(r, 1) for r in post_relevances))
        post_hit_3_cnt = int(sum(hit_rate_at_k(r, 3) for r in post_relevances))
        post_hit_5_cnt = int(sum(hit_rate_at_k(r, 5) for r in post_relevances))
        post_metrics = compute_ranking_metrics(post_relevances, (1, 3, 5))

        entry = KSweepEntryResult(
            retrieve_k=k,
            rerank_k=_TOP_K_RERANK,
            dtype=dtype,
            batch_size=batch_size,
            max_length=max_length,
            num_queries=n_queries,
            pre_hit_at_1_count=pre_hit_1_cnt,
            pre_hit_at_1_rate=pre_hit_1_cnt / n_queries,
            pre_hit_at_3_count=pre_hit_3_cnt,
            pre_hit_at_3_rate=pre_hit_3_cnt / n_queries,
            pre_hit_at_5_count=pre_hit_5_cnt,
            pre_hit_at_5_rate=pre_hit_5_cnt / n_queries,
            ceiling_hit_at_k_count=ceil_hit_k_cnt,
            ceiling_hit_at_k_rate=ceil_hit_k_cnt / n_queries,
            post_hit_at_1_count=post_hit_1_cnt,
            post_hit_at_1_rate=post_hit_1_cnt / n_queries,
            post_hit_at_3_count=post_hit_3_cnt,
            post_hit_at_3_rate=post_hit_3_cnt / n_queries,
            post_hit_at_5_count=post_hit_5_cnt,
            post_hit_at_5_rate=post_hit_5_cnt / n_queries,
            post_mrr_at_5=post_metrics["mrr@5"],
            post_ndcg_at_5=post_metrics["ndcg@5"],
            search_p50_ms=_percentile(search_latencies, _PERCENTILE_P50),
            search_p95_ms=_percentile(search_latencies, _PERCENTILE_P95),
            rerank_p50_ms=_percentile(rerank_latencies, _PERCENTILE_P50),
            rerank_p95_ms=_percentile(rerank_latencies, _PERCENTILE_P95),
            rerank_mean_ms=sum(rerank_latencies) / len(rerank_latencies),
        )
        results_by_k.append(entry)
        logger.info(
            "Done k=%d | Post Hit@1=%d/%d (%.4f), Hit@3=%d/%d (%.4f), Hit@5=%d/%d (%.4f) | Ceiling Hit@%d=%d/%d (%.4f) | Rerank p50=%.2f ms, p95=%.2f ms",
            k,
            entry.post_hit_at_1_count,
            n_queries,
            entry.post_hit_at_1_rate,
            entry.post_hit_at_3_count,
            n_queries,
            entry.post_hit_at_3_rate,
            entry.post_hit_at_5_count,
            n_queries,
            entry.post_hit_at_5_rate,
            k,
            entry.ceiling_hit_at_k_count,
            n_queries,
            entry.ceiling_hit_at_k_rate,
            entry.rerank_p50_ms,
            entry.rerank_p95_ms,
        )

    overlaps: dict[str, float] = {}
    top1_agreements: dict[str, float] = {}
    sorted_ks = list(k_values)
    for i in range(len(sorted_ks)):
        for j in range(i + 1, len(sorted_ks)):
            ka, kb = sorted_ks[i], sorted_ks[j]
            pair_key = f"k={ka}_vs_k={kb}"
            overlaps[pair_key] = compute_top5_set_overlap(top5_ids_by_k[ka], top5_ids_by_k[kb])
            top1_agreements[pair_key] = compute_top1_agreement(top5_ids_by_k[ka], top5_ids_by_k[kb])

    return KSweepReport(
        timestamp_utc=datetime.now(UTC).isoformat(),
        collection=collection,
        collection_point_count=point_count,
        seed=cfg.seed,
        num_queries=n_queries,
        dtype=dtype,
        batch_size=batch_size,
        max_length=max_length,
        embed_total_ms=embed_total_ms,
        embed_per_query_ms=embed_total_ms / max(1, n_queries),
        results_by_k=results_by_k,
        top5_set_overlap=overlaps,
        top1_exact_agreement=top1_agreements,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for Phase 2.5 Task 5a clean k-sweep."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    parser = argparse.ArgumentParser(
        description="Phase 2.5 Task 5a — Clean k-sweep on full dev at fp32, batch_size=4"
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=list(_DEFAULT_K_VALUES),
        help="List of retrieve_k values to sweep (default: 20 30 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Reranker batch size (default: 4)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Reranker dtype (default: fp32)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT_JSON,
        help=f"Output JSON artifact path (default: {_DEFAULT_OUT_JSON})",
    )
    args = parser.parse_args(argv)

    cfg = load_config()
    report = run_k_sweep(
        cfg,
        k_values=args.k,
        batch_size=args.batch_size,
        dtype=args.dtype,
        max_length=cfg.retrieval.reranker.max_length,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Saved k-sweep artifact to %s", args.out)
    return 0
