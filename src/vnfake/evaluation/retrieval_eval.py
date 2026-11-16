"""Phase 2 — Automated retrieval evaluation using proxy metrics (DATA_AUDIT.md §6).

Evaluates Hybrid Search (Qdrant server-side RRF) and Cross-Encoder Reranking on
the cleaned dev split (``dev_clean.jsonl``). Because ViFactCheck gold evidence
rarely appears verbatim in the Dân Trí / VnExpress KB, primary evaluation uses
extensible proxy relevance matchers (chunk_id/source match + lexical n-gram
containment) alongside strict gold-evidence containment for drift monitoring.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from vnfake.config import AppConfig, load_config
from vnfake.data.cleaning import normalize_text
from vnfake.retrieval.embedder import BgeM3Embedder, EmbedderProtocol
from vnfake.retrieval.reranker import BgeReranker, RerankerProtocol
from vnfake.retrieval.vector_store import QdrantStore, ScoredChunk
from vnfake.schemas.vifactcheck import ViFactCheckSample
from vnfake.utils.logging import get_logger
from vnfake.utils.seed import set_seed

logger = get_logger(__name__)

_MANIFEST_FILENAME = "index_manifest.json"
_DEV_CLEAN_FILENAME = "dev_clean.jsonl"
_DEFAULT_REPORT_PATH = Path("docs/REPORT_retrieval.md")
_DEFAULT_NGRAM_SIZE = 2
_DEFAULT_PROXY_THRESHOLD = 0.30
_DEFAULT_GOLD_NGRAM_SIZE = 3
_DEFAULT_GOLD_THRESHOLD = 0.50
_MS_PER_SECOND = 1000.0
_PERCENTILE_P50 = 50.0
_PERCENTILE_P95 = 95.0
_PERCENTILE_DIVISOR = 100.0
_LOG_BASE_2 = 2.0
_RANK_OFFSET = 1
_DCG_RANK_OFFSET = 2


@dataclass(frozen=True, slots=True)
class EvalQuery:
    """One evaluation query loaded from a ViFactCheck split."""

    statement: str
    evidence: str
    label: int
    gold_chunk_ids: frozenset[str] = field(default_factory=frozenset)
    gold_sources: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True, slots=True)
class QueryEvalRecord:
    """Retrieved and reranked results for a single :class:`EvalQuery`."""

    query: EvalQuery
    retrieved: list[ScoredChunk]
    reranked: list[ScoredChunk]
    search_ms: float
    rerank_ms: float


@dataclass(frozen=True, slots=True)
class RetrievalEvalReport:
    """Aggregated evaluation metrics and latency breakdown."""

    collection: str
    num_queries: int
    retrieve_k: int
    rerank_k: int
    embed_total_ms: float
    embed_per_query_ms: float
    search_p50_ms: float
    search_p95_ms: float
    rerank_p50_ms: float
    rerank_p95_ms: float
    proxy_hybrid_metrics: dict[str, float]
    proxy_rerank_metrics: dict[str, float]
    gold_hybrid_metrics: dict[str, float]
    gold_rerank_metrics: dict[str, float]
    label_breakdown: dict[str, dict[str, float]]
    mean_top1_hybrid_score: float
    mean_top1_rerank_score: float


RelevanceMatcher = Callable[[EvalQuery, ScoredChunk], bool]


def extract_word_ngrams(text: str, n: int) -> set[str]:
    """Extract lowercased word ``n``-grams from ``text``."""
    if n < 1:
        raise ValueError(f"ngram size n must be >= 1, got {n}")
    tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    if not tokens:
        return set()
    if len(tokens) < n:
        return {" ".join(tokens)}
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def ngram_containment(source_text: str, candidate_text: str, n: int) -> float:
    """Compute fraction of ``source_text`` word ``n``-grams present in ``candidate_text``."""
    src_ngrams = extract_word_ngrams(source_text, n)
    if not src_ngrams:
        return 0.0
    cand_ngrams = extract_word_ngrams(candidate_text, n)
    if not cand_ngrams:
        return 0.0
    return len(src_ngrams & cand_ngrams) / len(src_ngrams)


def match_explicit_metadata(query: EvalQuery, chunk: ScoredChunk) -> bool:
    """Check if ``chunk`` matches explicit ``gold_chunk_ids`` or ``gold_sources``."""
    if query.gold_chunk_ids and (
        chunk.chunk_id in query.gold_chunk_ids or chunk.doc_id in query.gold_chunk_ids
    ):
        return True
    return bool(query.gold_sources and chunk.source in query.gold_sources)


def make_proxy_matcher(
    *,
    ngram_n: int = _DEFAULT_NGRAM_SIZE,
    threshold: float = _DEFAULT_PROXY_THRESHOLD,
    include_evidence: bool = True,
) -> RelevanceMatcher:
    """Build an extensible proxy relevance matcher (DATA_AUDIT.md §6).

    A chunk is considered relevant if:
        1. It matches explicit ``gold_chunk_ids`` / ``gold_sources`` on the query, OR
        2. Its word ``ngram_n`` containment with ``query.statement`` (or
           ``query.evidence`` when ``include_evidence=True``) is ``>= threshold``.
    """

    def _matcher(query: EvalQuery, chunk: ScoredChunk) -> bool:
        if match_explicit_metadata(query, chunk):
            return True
        chunk_body = f"{chunk.title}\n{chunk.text}" if chunk.title else chunk.text
        stmt_score = ngram_containment(query.statement, chunk_body, ngram_n)
        if stmt_score >= threshold:
            return True
        if include_evidence and query.evidence:
            ev_score = ngram_containment(query.evidence, chunk_body, ngram_n)
            if ev_score >= threshold:
                return True
        return False

    return _matcher


def hit_rate_at_k(relevances: Sequence[bool], k: int) -> float:
    """Return 1.0 if any item in ``relevances[:k]`` is True, else 0.0."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    return 1.0 if any(relevances[:k]) else 0.0


def mrr_at_k(relevances: Sequence[bool], k: int) -> float:
    """Return Reciprocal Rank within top-``k`` (0.0 if no relevant hit)."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    for idx, is_rel in enumerate(relevances[:k]):
        if is_rel:
            return 1.0 / float(idx + _RANK_OFFSET)
    return 0.0


def ndcg_at_k(relevances: Sequence[bool], k: int) -> float:
    """Return binary nDCG@k for ``relevances[:k]``."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    sliced = [1 if r else 0 for r in relevances[:k]]
    num_rel = sum(sliced)
    if num_rel == 0:
        return 0.0
    dcg = sum(
        rel / math.log(idx + _DCG_RANK_OFFSET, _LOG_BASE_2)
        for idx, rel in enumerate(sliced)
        if rel > 0
    )
    idcg = sum(1.0 / math.log(idx + _DCG_RANK_OFFSET, _LOG_BASE_2) for idx in range(num_rel))
    return dcg / idcg if idcg > 0.0 else 0.0


def compute_ranking_metrics(
    relevances_per_query: Sequence[Sequence[bool]],
    k_values: Sequence[int],
) -> dict[str, float]:
    """Compute mean ``hit_rate@k``, ``mrr@k``, and ``ndcg@k`` across queries."""
    n = len(relevances_per_query)
    if n == 0:
        return {
            metric_key: 0.0
            for k in k_values
            for metric_key in (f"hit_rate@{k}", f"mrr@{k}", f"ndcg@{k}")
        }
    out: dict[str, float] = {}
    for k in k_values:
        out[f"hit_rate@{k}"] = sum(hit_rate_at_k(r, k) for r in relevances_per_query) / n
        out[f"mrr@{k}"] = sum(mrr_at_k(r, k) for r in relevances_per_query) / n
        out[f"ndcg@{k}"] = sum(ndcg_at_k(r, k) for r in relevances_per_query) / n
    return out


def _percentile(values: Sequence[float], pct: float) -> float:
    """Compute linear-interpolated percentile on a non-empty list of floats."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (pct / _PERCENTILE_DIVISOR) * (len(sorted_vals) - 1)
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return sorted_vals[lower]
    weight = pos - lower
    return sorted_vals[lower] * (1.0 - weight) + sorted_vals[upper] * weight


def iter_eval_queries(path: Path) -> Iterator[EvalQuery]:
    """Stream and validate :class:`EvalQuery` objects from a JSONL split file."""
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
                sample = ViFactCheckSample.model_validate(obj)
            except (json.JSONDecodeError, ValidationError) as exc:
                logger.warning("Skipping invalid sample at %s:%d — %s", path.name, lineno, exc)
                continue
            raw_chunk_ids = obj.get("chunk_ids") or obj.get("gold_chunk_ids") or []
            chunk_ids = (
                frozenset(str(cid) for cid in raw_chunk_ids)
                if isinstance(raw_chunk_ids, list)
                else frozenset()
            )
            raw_sources = obj.get("sources") or obj.get("gold_sources") or []
            sources = (
                frozenset(str(src) for src in raw_sources)
                if isinstance(raw_sources, list)
                else frozenset()
            )
            yield EvalQuery(
                statement=normalize_text(sample.statement),
                evidence=normalize_text(sample.evidence),
                label=sample.labels,
                gold_chunk_ids=chunk_ids,
                gold_sources=sources,
            )


def resolve_collection_name(cfg: AppConfig, override: str | None = None) -> str:
    """Resolve Qdrant collection name from ``override`` or ``index_manifest.json``."""
    if override:
        return override
    manifest_path = cfg.paths.artifacts_dir / _MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Index manifest not found at {manifest_path}. Run scripts/03_index_kb.py first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    collection = str(manifest.get("collection", "")).strip()
    if not collection:
        raise ValueError(f"Missing 'collection' key in {manifest_path}")
    return collection


def _execute_queries(
    cfg: AppConfig,
    collection: str,
    queries: Sequence[EvalQuery],
    embedder: EmbedderProtocol,
    store: QdrantStore,
    reranker: RerankerProtocol,
) -> tuple[list[QueryEvalRecord], float]:
    """Batch-embed all statements, then run hybrid search and reranking."""
    if not queries:
        return [], 0.0

    statements = [q.statement for q in queries]
    logger.info("Batch-encoding %d queries (dense + sparse)...", len(statements))
    t_embed = time.perf_counter()
    dense_vecs = embedder.embed_dense(statements, batch_size=cfg.retrieval.embedding.batch_size)
    sparse_vecs = embedder.embed_sparse(statements, batch_size=cfg.retrieval.sparse.batch_size)
    embed_total_ms = (time.perf_counter() - t_embed) * _MS_PER_SECOND
    if hasattr(embedder, "unload"):
        embedder.unload()

    records: list[QueryEvalRecord] = []
    retrieve_k = cfg.retrieval.retrieve_k
    rerank_k = cfg.retrieval.rerank_k

    for query, d_vec, s_vec in zip(queries, dense_vecs, sparse_vecs, strict=True):
        t0 = time.perf_counter()
        candidates = store.search_hybrid(
            collection,
            dense_query=d_vec,
            sparse_query=s_vec,
            top_k=retrieve_k,
            dense_prefetch_k=cfg.retrieval.hybrid.dense_prefetch_k,
            sparse_prefetch_k=cfg.retrieval.hybrid.sparse_prefetch_k,
        )
        search_ms = (time.perf_counter() - t0) * _MS_PER_SECOND

        t1 = time.perf_counter()
        reranked = reranker.rerank(query.statement, candidates, top_k=rerank_k)
        rerank_ms = (time.perf_counter() - t1) * _MS_PER_SECOND

        records.append(
            QueryEvalRecord(
                query=query,
                retrieved=candidates,
                reranked=reranked,
                search_ms=search_ms,
                rerank_ms=rerank_ms,
            )
        )
    return records, embed_total_ms


def _compute_label_breakdown(
    records: Sequence[QueryEvalRecord],
    matcher: RelevanceMatcher,
    label_map: dict[int, str],
    rerank_k: int,
) -> dict[str, dict[str, float]]:
    """Stratify proxy Hit Rate@K and MRR@K by ground-truth ViFactCheck label."""
    grouped: dict[str, list[list[bool]]] = {name: [] for name in label_map.values()}
    for rec in records:
        label_name = label_map.get(rec.query.label, str(rec.query.label))
        rels = [matcher(rec.query, c) for c in rec.reranked]
        grouped.setdefault(label_name, []).append(rels)

    breakdown: dict[str, dict[str, float]] = {}
    for label_name, rel_lists in grouped.items():
        metrics = compute_ranking_metrics(rel_lists, [rerank_k])
        metrics["count"] = float(len(rel_lists))
        breakdown[label_name] = metrics
    return breakdown


def evaluate_retrieval(
    cfg: AppConfig,
    *,
    store: QdrantStore,
    embedder: EmbedderProtocol,
    reranker: RerankerProtocol,
    dev_path: Path | None = None,
    collection: str | None = None,
    limit: int | None = None,
    proxy_matcher: RelevanceMatcher | None = None,
    gold_matcher: RelevanceMatcher | None = None,
) -> RetrievalEvalReport:
    """Run end-to-end retrieval + reranker evaluation on the dev split."""
    resolved_dev = dev_path or (cfg.paths.processed_dir / _DEV_CLEAN_FILENAME)
    resolved_collection = resolve_collection_name(cfg, collection)
    queries = list(iter_eval_queries(resolved_dev))
    if limit is not None and limit > 0:
        queries = queries[:limit]

    p_matcher = proxy_matcher or make_proxy_matcher()
    g_matcher = gold_matcher or make_proxy_matcher(
        ngram_n=_DEFAULT_GOLD_NGRAM_SIZE,
        threshold=_DEFAULT_GOLD_THRESHOLD,
        include_evidence=True,
    )

    records, embed_total_ms = _execute_queries(
        cfg, resolved_collection, queries, embedder, store, reranker
    )
    n = len(records)
    rerank_k = cfg.retrieval.rerank_k
    retrieve_k = cfg.retrieval.retrieve_k
    k_rerank_eval = sorted({1, min(3, rerank_k), rerank_k})
    k_hybrid_eval = sorted({*k_rerank_eval, retrieve_k})

    proxy_hybrid_rels = [[p_matcher(r.query, c) for c in r.retrieved] for r in records]
    proxy_rerank_rels = [[p_matcher(r.query, c) for c in r.reranked] for r in records]
    gold_hybrid_rels = [[g_matcher(r.query, c) for c in r.retrieved] for r in records]
    gold_rerank_rels = [[g_matcher(r.query, c) for c in r.reranked] for r in records]

    search_latencies = [r.search_ms for r in records]
    rerank_latencies = [r.rerank_ms for r in records]
    top1_hybrid = [r.retrieved[0].score for r in records if r.retrieved]
    top1_rerank = [r.reranked[0].score for r in records if r.reranked]

    return RetrievalEvalReport(
        collection=resolved_collection,
        num_queries=n,
        retrieve_k=retrieve_k,
        rerank_k=rerank_k,
        embed_total_ms=embed_total_ms,
        embed_per_query_ms=(embed_total_ms / n) if n else 0.0,
        search_p50_ms=_percentile(search_latencies, _PERCENTILE_P50),
        search_p95_ms=_percentile(search_latencies, _PERCENTILE_P95),
        rerank_p50_ms=_percentile(rerank_latencies, _PERCENTILE_P50),
        rerank_p95_ms=_percentile(rerank_latencies, _PERCENTILE_P95),
        proxy_hybrid_metrics=compute_ranking_metrics(proxy_hybrid_rels, k_hybrid_eval),
        proxy_rerank_metrics=compute_ranking_metrics(proxy_rerank_rels, k_rerank_eval),
        gold_hybrid_metrics=compute_ranking_metrics(gold_hybrid_rels, k_hybrid_eval),
        gold_rerank_metrics=compute_ranking_metrics(gold_rerank_rels, k_rerank_eval),
        label_breakdown=_compute_label_breakdown(records, p_matcher, cfg.label_map, rerank_k),
        mean_top1_hybrid_score=(sum(top1_hybrid) / len(top1_hybrid)) if top1_hybrid else 0.0,
        mean_top1_rerank_score=(sum(top1_rerank) / len(top1_rerank)) if top1_rerank else 0.0,
    )


def format_markdown_report(report: RetrievalEvalReport) -> str:
    """Render :class:`RetrievalEvalReport` as a Vietnamese Markdown document."""
    rk = report.rerank_k
    tk = report.retrieve_k
    k_list = sorted({1, min(3, rk), rk})
    lines: list[str] = [
        "# Báo cáo Đánh giá Retrieval & Reranking (Phase 2 — M2)",
        "",
        f"- **Thời gian tạo:** `{datetime.now(UTC).isoformat()}`",
        f"- **Collection Qdrant:** `{report.collection}`",
        f"- **Số lượng mẫu đánh giá (dev):** `{report.num_queries}`",
        f"- **Cấu hình:** `retrieve_k = {tk}`, `rerank_k = {rk}`",
        "",
        "## 1. Hiệu năng Truy vấn (Latency)",
        "",
        "| Bước xử lý | Chi tiết | Thời gian (ms) |",
        "|---|---|---:|",
        f"| **Embedding (Batch)** | Trung bình / câu (`bge-m3` dense + sparse) | `{report.embed_per_query_ms:.2f}` |",
        f"| **Hybrid Search (Qdrant RRF)** | p50 (`top-{tk}`) | `{report.search_p50_ms:.2f}` |",
        f"| **Hybrid Search (Qdrant RRF)** | **p95** (Exit criteria < 300 ms) | **`{report.search_p95_ms:.2f}`** |",
        f"| **Reranking (`bge-reranker-v2-m3`)** | p50 (`{tk} -> {rk}`) | `{report.rerank_p50_ms:.2f}` |",
        f"| **Reranking (`bge-reranker-v2-m3`)** | p95 (`{tk} -> {rk}`) | `{report.rerank_p95_ms:.2f}` |",
        "",
        "## 2. Kết quả Proxy Metric (Chính — theo `docs/DATA_AUDIT.md §6`)",
        "",
        "Do tập KB (Dân Trí / VnExpress) và ViFactCheck khác nguồn biên tập, chỉ số chính dùng **Proxy Relevance** (khớp `chunk_id`/`source` hoặc độ phủ word n-gram giữa Claim/Evidence và Chunk $\\ge 0.30$).",
        "",
        "| Giai đoạn | "
        + " | ".join(f"Hit Rate@{k} | MRR@{k} | nDCG@{k}" for k in k_list)
        + f" | Hit Rate@{tk} |",
        "|---|" + "---:|" * (len(k_list) * 3 + 1),
    ]

    hybrid_cells = [
        f"{report.proxy_hybrid_metrics.get(f'hit_rate@{k}', 0.0):.4f} | "
        f"{report.proxy_hybrid_metrics.get(f'mrr@{k}', 0.0):.4f} | "
        f"{report.proxy_hybrid_metrics.get(f'ndcg@{k}', 0.0):.4f}"
        for k in k_list
    ]
    rerank_cells = [
        f"{report.proxy_rerank_metrics.get(f'hit_rate@{k}', 0.0):.4f} | "
        f"{report.proxy_rerank_metrics.get(f'mrr@{k}', 0.0):.4f} | "
        f"{report.proxy_rerank_metrics.get(f'ndcg@{k}', 0.0):.4f}"
        for k in k_list
    ]
    hr_tk = report.proxy_hybrid_metrics.get(f"hit_rate@{tk}", 0.0)
    lines.append(f"| **Hybrid RRF (Pre-Rerank)** | {' | '.join(hybrid_cells)} | {hr_tk:.4f} |")
    lines.append(f"| **Hybrid + Reranker** | {' | '.join(rerank_cells)} | — |")

    lines.extend(
        [
            "",
            "## 3. Phân tích theo Nhãn gốc (Post-Rerank Proxy Metric)",
            "",
            f"| Nhãn (`label_map`) | Số mẫu | Hit Rate@{rk} | MRR@{rk} | nDCG@{rk} |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label_name, stats in report.label_breakdown.items():
        cnt = int(stats.get("count", 0.0))
        hr = stats.get(f"hit_rate@{rk}", 0.0)
        mrr = stats.get(f"mrr@{rk}", 0.0)
        ndcg = stats.get(f"ndcg@{rk}", 0.0)
        lines.append(f"| `{label_name}` | {cnt} | {hr:.4f} | {mrr:.4f} | {ndcg:.4f} |")

    lines.extend(
        [
            "",
            "## 4. Chỉ số Theo dõi Drift (Strict Gold 3-gram Containment $\\ge 0.50$)",
            "",
            f"- **Hybrid RRF Hit Rate@{tk}:** `{report.gold_hybrid_metrics.get(f'hit_rate@{tk}', 0.0):.4f}`",
            f"- **Hybrid RRF Hit Rate@{rk}:** `{report.gold_hybrid_metrics.get(f'hit_rate@{rk}', 0.0):.4f}`",
            f"- **Hybrid + Reranker Hit Rate@{rk}:** `{report.gold_rerank_metrics.get(f'hit_rate@{rk}', 0.0):.4f}`",
            f"- **Hybrid + Reranker MRR@{rk}:** `{report.gold_rerank_metrics.get(f'mrr@{rk}', 0.0):.4f}`",
            f"- **Mean Top-1 Score (Hybrid / Rerank):** `{report.mean_top1_hybrid_score:.4f}` / `{report.mean_top1_rerank_score:.4f}`",
            "",
        ]
    )
    return "\n".join(lines)


def log_report_summary(report: RetrievalEvalReport) -> None:
    """Log evaluation summary to console via structured logger."""
    rk = report.rerank_k
    tk = report.retrieve_k
    logger.info(
        "Retrieval Eval Summary | collection=%s | queries=%d | search_p50=%.2fms | search_p95=%.2fms | rerank_p50=%.2fms",
        report.collection,
        report.num_queries,
        report.search_p50_ms,
        report.search_p95_ms,
        report.rerank_p50_ms,
    )
    logger.info(
        "Proxy Metrics (Pre-Rerank)  | HR@%d=%.4f | MRR@%d=%.4f | nDCG@%d=%.4f | HR@%d=%.4f",
        rk,
        report.proxy_hybrid_metrics.get(f"hit_rate@{rk}", 0.0),
        rk,
        report.proxy_hybrid_metrics.get(f"mrr@{rk}", 0.0),
        rk,
        report.proxy_hybrid_metrics.get(f"ndcg@{rk}", 0.0),
        tk,
        report.proxy_hybrid_metrics.get(f"hit_rate@{tk}", 0.0),
    )
    logger.info(
        "Proxy Metrics (Post-Rerank) | HR@%d=%.4f | MRR@%d=%.4f | nDCG@%d=%.4f | top1_score=%.4f",
        rk,
        report.proxy_rerank_metrics.get(f"hit_rate@{rk}", 0.0),
        rk,
        report.proxy_rerank_metrics.get(f"mrr@{rk}", 0.0),
        rk,
        report.proxy_rerank_metrics.get(f"ndcg@{rk}", 0.0),
        report.mean_top1_rerank_score,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for Phase 2 retrieval evaluation."""
    parser = argparse.ArgumentParser(
        description="Phase 2 — Evaluate Hybrid Search + Reranker on ViFactCheck dev split"
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to YAML config")
    parser.add_argument(
        "--dev-path",
        type=Path,
        default=None,
        help="Override dev JSONL path (default: <processed_dir>/dev_clean.jsonl)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Override Qdrant collection name (default: from index_manifest.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_REPORT_PATH,
        help="Markdown report output path (default: docs/REPORT_retrieval.md)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of dev samples to evaluate",
    )
    parser.add_argument(
        "--retrieve-k",
        type=int,
        default=None,
        help="Override cfg.retrieval.retrieve_k via CLI without modifying base.yaml",
    )
    parser.add_argument(
        "--rerank-batch-size",
        type=int,
        default=None,
        help="Override cfg.retrieval.reranker.batch_size via CLI",
    )
    parser.add_argument(
        "--rerank-dtype",
        choices=["fp32", "fp16"],
        default=None,
        help="Override reranker fp16/fp32 mode via CLI",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to save evaluation report as JSON (without overwriting markdown)",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.retrieve_k is not None:
        cfg.retrieval.retrieve_k = args.retrieve_k
    if args.rerank_batch_size is not None:
        cfg.retrieval.reranker.batch_size = args.rerank_batch_size
    if args.rerank_dtype is not None:
        cfg.retrieval.reranker.use_fp16 = args.rerank_dtype == "fp16"
    set_seed(cfg.seed)

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

    report = evaluate_retrieval(
        cfg,
        store=store,
        embedder=embedder,
        reranker=reranker,
        dev_path=args.dev_path,
        collection=args.collection,
        limit=args.limit,
    )
    log_report_summary(report)

    if args.json_out is not None:
        from dataclasses import asdict

        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(asdict(report), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Saved retrieval evaluation JSON to %s", args.json_out)
    else:
        md_content = format_markdown_report(report)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(md_content, encoding="utf-8")
        logger.info("Saved retrieval evaluation report to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
