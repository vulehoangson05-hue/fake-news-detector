"""Phase 2 — CLI script to test end-to-end hybrid retrieval + reranking on KB."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from vnfake.config import load_config
from vnfake.retrieval.embedder import BgeM3Embedder
from vnfake.retrieval.reranker import BgeReranker
from vnfake.retrieval.vector_store import QdrantStore

DEFAULT_QUERY = "Ngân hàng Nhà nước triển khai bán vàng miếng SJC bình ổn thị trường qua các ngân hàng thương mại"


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for testing hybrid search and reranking."""
    parser = argparse.ArgumentParser(
        description="Phase 2 — Test hybrid search (Qdrant RRF) + BGE reranker"
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=DEFAULT_QUERY,
        help="Câu truy vấn tiếng Việt (mặc định: câu thời sự mẫu)",
    )
    parser.add_argument(
        "--query",
        "-q",
        dest="query_opt",
        type=str,
        default=None,
        help="Tùy chọn truyền câu truy vấn qua flag --query",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Đường dẫn file cấu hình YAML (mặc định: configs/base.yaml)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Số lượng kết quả sau rerank (mặc định: cfg.retrieval.rerank_k)",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    query: str = args.query_opt if args.query_opt is not None else args.query
    top_k: int = args.top_k if args.top_k is not None else cfg.retrieval.rerank_k

    manifest_path = cfg.paths.artifacts_dir / "index_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest file not found: {manifest_path}. Run scripts/03_index_kb.py first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    collection = str(manifest["collection"])

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
        batch_size=cfg.retrieval.reranker.batch_size,
        use_fp16=cfg.retrieval.reranker.use_fp16,
        max_length=cfg.retrieval.reranker.max_length,
    )

    # 1. Embedding (Dense + Sparse)
    t0 = time.perf_counter()
    dense_vec = embedder.embed_dense([query], batch_size=cfg.retrieval.embedding.batch_size)[0]
    sparse_vec = embedder.embed_sparse([query], batch_size=cfg.retrieval.sparse.batch_size)[0]
    embed_ms = (time.perf_counter() - t0) * 1000.0

    # 2. Hybrid Search (Qdrant server-side RRF)
    t1 = time.perf_counter()
    candidates = store.search_hybrid(
        collection,
        dense_query=dense_vec,
        sparse_query=sparse_vec,
        top_k=cfg.retrieval.retrieve_k,
        dense_prefetch_k=cfg.retrieval.hybrid.dense_prefetch_k,
        sparse_prefetch_k=cfg.retrieval.hybrid.sparse_prefetch_k,
    )
    search_ms = (time.perf_counter() - t1) * 1000.0

    # 3. Reranking
    t2 = time.perf_counter()
    results = reranker.rerank(query, candidates, top_k=top_k)
    rerank_ms = (time.perf_counter() - t2) * 1000.0

    total_ms = embed_ms + search_ms + rerank_ms

    print("=" * 80)
    print(f"Collection : {collection}")
    print(f"Query      : {query}")
    print("-" * 80)
    print("Latency Breakdown:")
    print(f"  - Embedding (Dense + Sparse) : {embed_ms:8.2f} ms")
    print(f"  - Hybrid Search (Qdrant RRF) : {search_ms:8.2f} ms (candidates={len(candidates)})")
    print(f"  - Reranking (Top {top_k:<2d})         : {rerank_ms:8.2f} ms")
    print(f"  - Total                      : {total_ms:8.2f} ms")
    print("=" * 80)
    print(f"TOP {len(results)} RESULTS:")

    for idx, hit in enumerate(results, start=1):
        snippet = hit.text.replace("\n", " ").strip()
        if len(snippet) > 300:
            snippet = f"{snippet[:297]}..."
        print(f"\n[{idx}] Score  : {hit.score:.4f}")
        print(f"    Source : {hit.source} | Doc ID: {hit.doc_id} | Chunk ID: {hit.chunk_id}")
        if hit.title:
            print(f"    Title  : {hit.title}")
        print(f"    Text   : {snippet}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
