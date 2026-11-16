"""Phase 2.5 Task 5 — CLI runner for C1 A/B testing on 60 human-labeled dev claims."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from vnfake.config import load_config
from vnfake.evaluation.ab_testing import (
    compute_metrics_and_write_report,
    export_label_pack,
    generate_blind_label_pack,
    run_c1_ab_pipeline,
    stratified_sample_dev,
)
from vnfake.retrieval.embedder import BgeM3Embedder
from vnfake.retrieval.query_rewriter import GeminiQueryRewriter
from vnfake.retrieval.reranker import BgeReranker
from vnfake.retrieval.vector_store import QdrantStore
from vnfake.utils.logging import get_logger
from vnfake.utils.seed import set_seed

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Run Phase 2.5 C1 A/B testing on 60 human-labeled claims."""
    parser = argparse.ArgumentParser(
        description="Phase 2.5 C1 — A/B Testing Query Rewrite on 60 dev claims"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base.yaml"),
        help="Path to base config (default: configs/base.yaml)",
    )
    parser.add_argument(
        "--dev-clean",
        type=Path,
        default=Path("data/processed/dev_clean.jsonl"),
        help="Path to dev_clean.jsonl (default: data/processed/dev_clean.jsonl)",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=20,
        help="Number of claims to sample per class (default: 20 -> 60 total)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic sampling seed (default: 42)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/ab_testing"),
        help="Output directory for human labeling pack (default: artifacts/ab_testing)",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("docs/REPORT_ab_c1.md"),
        help="Output path for C1 markdown report (default: docs/REPORT_ab_c1.md)",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(args.seed)

    # Resolve collection from index_manifest.json
    manifest_path = cfg.paths.artifacts_dir / "index_manifest.json"
    if not manifest_path.exists():
        logger.error("index_manifest.json not found at %s", manifest_path)
        return 1
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    collection = manifest_data.get("collection")
    if not collection:
        logger.error("No 'collection' in %s", manifest_path)
        return 1

    prompt_path = cfg.query_rewrite.prompt_path
    prompt_bytes = prompt_path.read_bytes() if prompt_path.exists() else b""
    prompt_sha256 = hashlib.sha256(prompt_bytes).hexdigest()

    logger.info(
        "Starting C1 A/B testing: collection=%s | prompt_version=%s | sha256=%s | seed=%d",
        collection,
        cfg.query_rewrite.prompt_version,
        prompt_sha256[:12],
        args.seed,
    )

    # 1. Stratified sample dev claims
    sampled_claims = stratified_sample_dev(
        args.dev_clean,
        seed=args.seed,
        per_class=args.per_class,
    )
    logger.info("Sampled %d claims (20 Real, 20 Fake, 20 Unverifiable)", len(sampled_claims))

    # 2. Initialize QdrantStore, Embedder, Reranker, Rewriter
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

    reranker = BgeReranker(
        model_name=cfg.retrieval.reranker.model_name,
        max_length=cfg.retrieval.reranker.max_length,
        batch_size=cfg.retrieval.reranker.batch_size,
        use_fp16=cfg.retrieval.reranker.use_fp16,
    )

    # Enable rewriter for the evaluation pipeline
    rewrite_cfg = cfg.query_rewrite.model_copy(update={"enabled": True})
    rewriter = GeminiQueryRewriter(rewrite_cfg)

    # 3. Run pipeline on the 3 branches
    records = run_c1_ab_pipeline(
        cfg,
        store=store,
        embedder=embedder,
        reranker=reranker,
        collection=collection,
        rewriter=rewriter,
        sampled_claims=sampled_claims,
        retrieve_k=cfg.retrieval.retrieve_k,  # 30
        top_k=cfg.retrieval.rerank_k,  # 5
    )

    # 4. Generate blind labeling pack
    items, unblind_map = generate_blind_label_pack(records, seed=args.seed)
    export_label_pack(items, unblind_map, args.out_dir)

    # 5. Compute metrics & write report
    manifest_c1_path = args.out_dir / "manifest_c1.json"
    compute_metrics_and_write_report(
        records,
        out_path=args.report_out,
        manifest_path=manifest_c1_path,
        prompt_version=cfg.query_rewrite.prompt_version,
        prompt_sha256=prompt_sha256,
        model_name=cfg.query_rewrite.model,
        retrieve_k=cfg.retrieval.retrieve_k,
        rerank_k=cfg.retrieval.rerank_k,
    )

    logger.info("C1 A/B testing completed successfully! Report: %s", args.report_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
