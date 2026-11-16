"""Phase 2.5 Task 1 — CLI wrapper for reranker latency profiling."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import torch

from vnfake.config import load_config
from vnfake.evaluation.rerank_profiler import (
    ConfigProfileResult,
    collect_environment_info,
    compute_mean_top_k_overlap,
    load_reranker_for_dtype,
    precompute_candidates,
    profile_configuration,
    sample_dev_claims,
    save_profiling_artifact,
)
from vnfake.evaluation.retrieval_eval import iter_eval_queries
from vnfake.retrieval.embedder import BgeM3Embedder
from vnfake.retrieval.vector_store import QdrantStore
from vnfake.utils.logging import get_logger
from vnfake.utils.seed import set_seed

logger = get_logger(__name__)

_DEFAULT_N_CLAIMS = 100
_DEFAULT_WARMUP = 3


def _parse_int_list(raw_values: list[str] | None, default: list[int]) -> list[int]:
    if not raw_values:
        return default
    out: list[int] = []
    for item in raw_values:
        for part in item.split(","):
            stripped = part.strip()
            if stripped:
                out.append(int(stripped))
    return out or default


def _parse_str_list(raw_values: list[str] | None, default: list[str]) -> list[str]:
    if not raw_values:
        return default
    out: list[str] = []
    for item in raw_values:
        for part in item.split(","):
            stripped = part.strip()
            if stripped:
                out.append(stripped)
    return out or default


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for rerank latency profiling."""
    parser = argparse.ArgumentParser(
        description="Phase 2.5 Task 1 — Profile BGE reranker latency and rank stability"
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to YAML config")
    parser.add_argument(
        "--k",
        nargs="+",
        default=None,
        help="retrieve_k values to profile (default: from config, or e.g. --k 50 20)",
    )
    parser.add_argument(
        "--n-claims",
        type=int,
        default=_DEFAULT_N_CLAIMS,
        help="Number of dev claims to sample (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        nargs="+",
        default=None,
        help="Batch size(s) to profile (default: from config, or e.g. --batch-size 1 4 8 16 32)",
    )
    parser.add_argument(
        "--dtype",
        nargs="+",
        default=None,
        help="Dtype(s) to profile: fp32, fp16, bf16 (default: from config)",
    )
    parser.add_argument(
        "--max-length",
        nargs="+",
        default=None,
        help="Max token length(s) to profile (default: from config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for sampling dev claims (default: cfg.seed)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default: artifacts/profiling/rerank_<timestamp>.json)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=_DEFAULT_WARMUP,
        help="Number of warm-up claims excluded from stats (default: 3)",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.seed
    set_seed(seed)

    default_dtype = "fp16" if cfg.retrieval.reranker.use_fp16 else "fp32"
    k_values = _parse_int_list(args.k, [cfg.retrieval.retrieve_k])
    batch_sizes = _parse_int_list(args.batch_size, [cfg.retrieval.reranker.batch_size])
    dtypes = _parse_str_list(args.dtype, [default_dtype])
    max_lengths = _parse_int_list(args.max_length, [cfg.retrieval.reranker.max_length])

    env_info = collect_environment_info()
    logger.info("Environment: %s", env_info)

    dev_path = cfg.paths.processed_dir / "dev_clean.jsonl"
    all_queries = list(iter_eval_queries(dev_path))
    sampled = sample_dev_claims(all_queries, n_claims=args.n_claims, seed=seed)
    claim_ids = [cid for cid, _ in sampled]

    store = QdrantStore.from_env(
        dense_name=cfg.retrieval.dense_vector_name,
        sparse_name=cfg.retrieval.sparse_vector_name,
    )
    embedder = BgeM3Embedder(
        model_name=cfg.retrieval.embedding.model_name,
        dim=cfg.retrieval.embedding.dim,
        use_fp16=cfg.retrieval.embedding.use_fp16,
        weight_threshold=cfg.retrieval.sparse.weight_threshold,
    )
    max_k = max(k_values)
    claims_with_candidates = precompute_candidates(
        cfg,
        sampled,
        embedder=embedder,
        store=store,
        max_k=max_k,
    )
    # Free embedding model from VRAM before profiling reranker in isolation
    del embedder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results: list[ConfigProfileResult] = []
    baseline_by_k: dict[int, dict[str, list[str]]] = {}

    for dtype_name in dtypes:
        loaded_pair = load_reranker_for_dtype(
            cfg.retrieval.reranker.model_name,
            dtype_name,
            env_info.device,
        )
        try:
            for k in k_values:
                oom_reached = False
                for max_len in max_lengths:
                    for bsize in batch_sizes:
                        if oom_reached:
                            logger.info(
                                "Skipping (k=%d, batch=%d, dtype=%s) after earlier OOM",
                                k,
                                bsize,
                                dtype_name,
                            )
                            continue
                        logger.info(
                            "Profiling k=%d | batch=%d | dtype=%s | max_length=%d ...",
                            k,
                            bsize,
                            dtype_name,
                            max_len,
                        )
                        res = profile_configuration(
                            cfg.retrieval.reranker.model_name,
                            claims_with_candidates,
                            retrieve_k=k,
                            rerank_k=cfg.retrieval.rerank_k,
                            batch_size=bsize,
                            dtype_name=dtype_name,
                            max_length=max_len,
                            device=env_info.device,
                            warmup_claims=args.warmup,
                            baseline_top5=baseline_by_k.get(k),
                            loaded_pair=loaded_pair,
                        )
                        if (
                            res.status == "ok"
                            and k not in baseline_by_k
                            and dtype_name == "fp32"
                            and bsize == cfg.retrieval.reranker.batch_size
                            and max_len == cfg.retrieval.reranker.max_length
                        ):
                            baseline_by_k[k] = res.top5_ids_per_claim
                        if res.status == "oom":
                            oom_reached = True
                        results.append(res)
        finally:
            del loaded_pair
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    for k in k_values:
        if k not in baseline_by_k:
            for r in results:
                if r.retrieve_k == k and r.status == "ok":
                    baseline_by_k[k] = r.top5_ids_per_claim
                    break

    # Backfill overlap against the resolved per-k baseline
    final_results: list[ConfigProfileResult] = []
    for r in results:
        base_map = baseline_by_k.get(r.retrieve_k)
        overlap = (
            compute_mean_top_k_overlap(base_map, r.top5_ids_per_claim, k=cfg.retrieval.rerank_k)
            if (r.status == "ok" and base_map is not None)
            else None
        )
        final_results.append(
            ConfigProfileResult(
                retrieve_k=r.retrieve_k,
                batch_size=r.batch_size,
                dtype=r.dtype,
                max_length=r.max_length,
                status=r.status,
                error_message=r.error_message,
                num_claims_measured=r.num_claims_measured,
                warmup_claims_skipped=r.warmup_claims_skipped,
                p50_ms=r.p50_ms,
                p95_ms=r.p95_ms,
                mean_ms=r.mean_ms,
                mean_tokenize_ms=r.mean_tokenize_ms,
                mean_forward_ms=r.mean_forward_ms,
                mean_postprocess_ms=r.mean_postprocess_ms,
                ms_per_pair=r.ms_per_pair,
                overlap_top5_vs_baseline=overlap,
                top5_ids_per_claim=r.top5_ids_per_claim,
            )
        )

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = cfg.paths.artifacts_dir / "profiling"
    out_path = args.out or (out_dir / f"rerank_{ts}.json")
    claim_ids_path = out_dir / f"rerank_subset_n{len(claim_ids)}_seed{seed}.json"
    save_profiling_artifact(
        out_path,
        env_info=env_info,
        seed=seed,
        claim_ids_path=claim_ids_path,
        claim_ids=claim_ids,
        results=final_results,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
