"""Phase 2.5 Task 5 — A/B Testing Query Rewrite on 60 Human-Labeled Dev Claims.

Implements BRIEF #3 §C1:
- Selects 60 stratified claims (20 real, 20 fake, 20 unverifiable; seed=42) from ``dev_clean.jsonl``.
- Runs 3 branches at ``retrieve_k = 30``:
    1. ``original``: baseline hybrid retrieval without rewrite.
    2. ``rewritten_only``: hybrid retrieval using Gemini-rewritten keywords.
    3. ``original+rewritten``: reciprocal rank fusion (RRF) of original and rewritten candidates.
- Invariant: Reranker ALWAYS scores against the original claim (ARCHITECTURE §5.3.1).
- Generates blind, shuffled human-labeling pack (.json, .csv, .md) for PM.
- Computes proxy retrieval metrics (Hit@1, Hit@3, Hit@5, MRR@5) and latency p50/p95.
- Writes comprehensive report to ``docs/REPORT_ab_c1.md``.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vnfake.config import AppConfig
from vnfake.evaluation.retrieval_eval import EvalQuery, make_proxy_matcher
from vnfake.retrieval.embedder import EmbedderProtocol
from vnfake.retrieval.query_rewriter import (
    QueryRewriter,
)
from vnfake.retrieval.reranker import RerankerProtocol
from vnfake.retrieval.service import RetrievalService
from vnfake.retrieval.vector_store import QdrantStore, ScoredChunk
from vnfake.schemas.vifactcheck import ViFactCheckSample
from vnfake.utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_SEED = 42
_DEFAULT_PER_CLASS = 20
_MS_PER_SECOND = 1000.0


@dataclass(frozen=True, slots=True)
class BranchResult:
    """Retrieval outcome for one branch on a single claim."""

    branch_name: str
    query_used: str
    rewrite_status: str
    candidates_count: int
    evidences: list[ScoredChunk]
    proxy_hits: dict[str, bool]  # e.g. {"hit@1": True, "hit@3": True, "hit@5": True}
    retrieve_ms: float
    rerank_ms: float
    total_ms: float


@dataclass(frozen=True, slots=True)
class ClaimABRecord:
    """Full evaluation record for one claim across all 3 branches."""

    claim_id: str
    sample_index: int
    statement: str
    gold_label: int  # 0=real, 1=fake, 2=unverifiable
    gold_evidence: str
    rewritten_query: str
    rewrite_status: str
    rewrite_latency_ms: float
    branch_original: BranchResult
    branch_rewritten_only: BranchResult
    branch_fused: BranchResult


@dataclass(frozen=True, slots=True)
class BlindLabelItem:
    """Anonymized candidate evidence chunk for human labeling."""

    label_item_id: str
    claim_id: str
    statement: str
    chunk_id: str
    source: str
    title: str
    text: str
    url: str
    human_verdict: str = ""  # To be filled by human labeler: 1=relevant, 0=irrelevant


def stratified_sample_dev(
    dev_clean_path: Path,
    *,
    seed: int = _DEFAULT_SEED,
    per_class: int = _DEFAULT_PER_CLASS,
) -> list[tuple[int, ViFactCheckSample]]:
    """Sample ``per_class`` samples per label (0, 1, 2) deterministically."""
    if not dev_clean_path.exists():
        raise FileNotFoundError(f"Dev clean file not found: {dev_clean_path}")

    records: list[tuple[int, ViFactCheckSample]] = []
    with dev_clean_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            raw = line.strip()
            if not raw:
                continue
            item = ViFactCheckSample.model_validate_json(raw)
            records.append((idx, item))

    by_label: dict[int, list[tuple[int, ViFactCheckSample]]] = {0: [], 1: [], 2: []}
    for idx, item in records:
        if item.labels in by_label:
            by_label[item.labels].append((idx, item))

    rng = random.Random(seed)
    sampled: list[tuple[int, ViFactCheckSample]] = []
    for label in (0, 1, 2):
        pool = by_label[label]
        if len(pool) < per_class:
            raise ValueError(
                f"Label {label} has only {len(pool)} samples, fewer than required {per_class}"
            )
        chosen = rng.sample(pool, per_class)
        sampled.extend(chosen)

    # Sort deterministically by original sample index
    sampled.sort(key=lambda t: t[0])
    return sampled


def _compute_proxy_hits(
    query: EvalQuery,
    evidences: Sequence[ScoredChunk],
    matcher: Any,
) -> dict[str, bool]:
    """Compute Hit@1, Hit@3, Hit@5 against proxy relevance matcher."""
    is_hit_1 = False
    is_hit_3 = False
    is_hit_5 = False
    for rank, chunk in enumerate(evidences[:5]):
        if matcher(query, chunk):
            if rank < 1:
                is_hit_1 = True
            if rank < 3:
                is_hit_3 = True
            if rank < 5:
                is_hit_5 = True
            break
    return {"hit@1": is_hit_1, "hit@3": is_hit_3, "hit@5": is_hit_5}


def run_c1_ab_pipeline(
    cfg: AppConfig,
    *,
    store: QdrantStore,
    embedder: EmbedderProtocol,
    reranker: RerankerProtocol,
    collection: str,
    rewriter: QueryRewriter,
    sampled_claims: Sequence[tuple[int, ViFactCheckSample]],
    retrieve_k: int = 30,
    top_k: int = 5,
) -> list[ClaimABRecord]:
    """Execute the 3 branches (original, rewritten_only, original+rewritten) on sampled claims."""
    # Build RetrievalService configured with rewriter enabled
    cfg_rewrite = cfg.model_copy(deep=True)
    cfg_rewrite.query_rewrite.enabled = True
    cfg_baseline = cfg.model_copy(deep=True)
    cfg_baseline.query_rewrite.enabled = False

    retrieval_service = RetrievalService(
        cfg_rewrite,
        store=store,
        embedder=embedder,
        reranker=reranker,
        collection=collection,
        rewriter=rewriter,
    )
    # Baseline service without rewriter
    baseline_service = RetrievalService(
        cfg_baseline,
        store=store,
        embedder=embedder,
        reranker=reranker,
        collection=collection,
        rewriter=None,
    )

    proxy_matcher = make_proxy_matcher()
    records: list[ClaimABRecord] = []

    for i, (orig_idx, sample) in enumerate(sampled_claims):
        claim_id = f"c1_dev_{orig_idx:03d}"
        claim_text = sample.statement
        gold_label = sample.labels
        gold_evidence = sample.evidence
        logger.info(
            "Evaluating claim %d/%d [%s] (label=%d): %s",
            i + 1,
            len(sampled_claims),
            claim_id,
            gold_label,
            claim_text[:60],
        )

        eval_query = EvalQuery(
            statement=claim_text,
            evidence=gold_evidence,
            label=gold_label,
        )

        # 1. Branch: original (no rewrite)
        res_orig = baseline_service.search(
            claim_text,
            retrieve_k=retrieve_k,
            top_k=top_k,
        )
        hits_orig = _compute_proxy_hits(eval_query, res_orig.evidences, proxy_matcher)
        branch_orig = BranchResult(
            branch_name="original",
            query_used=res_orig.claim,
            rewrite_status=res_orig.rewrite_status,
            candidates_count=len(res_orig.candidates),
            evidences=res_orig.evidences,
            proxy_hits=hits_orig,
            retrieve_ms=res_orig.retrieve_ms,
            rerank_ms=res_orig.rerank_ms,
            total_ms=res_orig.total_ms,
        )

        # 2. Branch: rewritten_only
        res_rew = retrieval_service.search(
            claim_text,
            retrieve_k=retrieve_k,
            top_k=top_k,
            fusion="rewritten_only",
        )
        hits_rew = _compute_proxy_hits(eval_query, res_rew.evidences, proxy_matcher)
        branch_rew = BranchResult(
            branch_name="rewritten_only",
            query_used=res_rew.rewrite_result.text,
            rewrite_status=res_rew.rewrite_status,
            candidates_count=len(res_rew.candidates),
            evidences=res_rew.evidences,
            proxy_hits=hits_rew,
            retrieve_ms=res_rew.retrieve_ms,
            rerank_ms=res_rew.rerank_ms,
            total_ms=res_rew.total_ms,
        )

        # 3. Branch: original+rewritten (RRF fused)
        res_fused = retrieval_service.search(
            claim_text,
            retrieve_k=retrieve_k,
            top_k=top_k,
            fusion="original+rewritten",
        )
        hits_fused = _compute_proxy_hits(eval_query, res_fused.evidences, proxy_matcher)
        branch_fused = BranchResult(
            branch_name="original+rewritten",
            query_used=f"{res_fused.claim} || {res_fused.rewrite_result.text}",
            rewrite_status=res_fused.rewrite_status,
            candidates_count=len(res_fused.candidates),
            evidences=res_fused.evidences,
            proxy_hits=hits_fused,
            retrieve_ms=res_fused.retrieve_ms,
            rerank_ms=res_fused.rerank_ms,
            total_ms=res_fused.total_ms,
        )

        records.append(
            ClaimABRecord(
                claim_id=claim_id,
                sample_index=orig_idx,
                statement=claim_text,
                gold_label=gold_label,
                gold_evidence=gold_evidence,
                rewritten_query=res_rew.rewrite_result.text,
                rewrite_status=res_rew.rewrite_status,
                rewrite_latency_ms=res_rew.rewrite_result.latency_ms,
                branch_original=branch_orig,
                branch_rewritten_only=branch_rew,
                branch_fused=branch_fused,
            )
        )

    return records


def generate_blind_label_pack(
    records: Sequence[ClaimABRecord],
    *,
    seed: int = _DEFAULT_SEED,
) -> tuple[list[BlindLabelItem], dict[str, dict[str, Any]]]:
    """Build anonymized, shuffled candidate pool for human labeling.

    Each (claim_id, chunk_id) appears strictly ONCE across the union of top-5 from
    all 3 branches. System codes are stripped from label items and kept strictly in
    unblind_map. Full chunk text is preserved (A12).
    """
    rng = random.Random(seed)
    items: list[BlindLabelItem] = []
    unblind_map: dict[str, dict[str, Any]] = {}

    item_counter = 1
    for rec in records:
        # Collect unique chunks across all 3 branches, tracking branch membership
        chunk_map: dict[str, ScoredChunk] = {}
        chunk_branches: dict[str, list[str]] = {}

        for branch in (rec.branch_original, rec.branch_rewritten_only, rec.branch_fused):
            for chunk in branch.evidences[:5]:
                if chunk.chunk_id not in chunk_map:
                    chunk_map[chunk.chunk_id] = chunk
                    chunk_branches[chunk.chunk_id] = []
                if branch.branch_name not in chunk_branches[chunk.chunk_id]:
                    chunk_branches[chunk.chunk_id].append(branch.branch_name)

        # Shuffle candidates within the claim so presentation order is random
        claim_chunk_ids = list(chunk_map.keys())
        rng.shuffle(claim_chunk_ids)

        for cid in claim_chunk_ids:
            chunk = chunk_map[cid]
            label_item_id = f"item_{item_counter:04d}"
            item_counter += 1

            items.append(
                BlindLabelItem(
                    label_item_id=label_item_id,
                    claim_id=rec.claim_id,
                    statement=rec.statement,
                    chunk_id=chunk.chunk_id,
                    source=chunk.source,
                    title=chunk.title,
                    text=chunk.text,
                    url=chunk.url,
                    human_verdict="",
                )
            )
            unblind_map[label_item_id] = {
                "claim_id": rec.claim_id,
                "chunk_id": chunk.chunk_id,
                "branches": chunk_branches[cid],
            }

    return items, unblind_map


def export_label_pack(
    items: Sequence[BlindLabelItem],
    unblind_map: dict[str, dict[str, Any]],
    out_dir: Path,
) -> None:
    """Export blind human-labeling pack to JSON, CSV, and Markdown without system codes."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. JSON
    json_path = out_dir / "human_label_pack_60.json"
    json_data = [asdict(item) for item in items]
    json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")

    # 2. Secret unblind key (saved separately for verification after labeling)
    unblind_path = out_dir / "unblind_key_do_not_open_before_labeling.json"
    unblind_path.write_text(json.dumps(unblind_map, indent=2, ensure_ascii=False), encoding="utf-8")

    # 3. CSV for PM labeling in spreadsheet
    csv_path = out_dir / "human_label_pack_60.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "label_item_id",
                "claim_id",
                "statement",
                "chunk_id",
                "source",
                "title",
                "text",
                "url",
                "human_verdict_1_or_0",
            ]
        )
        for it in items:
            writer.writerow(
                [
                    it.label_item_id,
                    it.claim_id,
                    it.statement,
                    it.chunk_id,
                    it.source,
                    it.title,
                    it.text,
                    it.url,
                    "",  # empty cell for PM to fill
                ]
            )

    # 4. Markdown file with instructions and readable formatting
    md_path = out_dir / "human_label_pack_60.md"
    lines = [
        "# Gói Gán Nhãn Đánh Giá Retrieval (60 Claims Dev) — Phase 2.5 C1",
        "",
        "> **Hướng dẫn dành cho PM / Người thẩm định:**",
        "> - Tập dữ liệu gồm 60 claim đại diện (20 Real, 20 Fake, 20 Unverifiable).",
        "> - Mỗi claim đi kèm danh sách các đoạn trích (evidence chunks) gộp từ cả 3 nhánh hệ thống (đã được khử trùng và xáo trộn ngẫu nhiên). Mỗi chunk chỉ xuất hiện đúng 1 lần cho mỗi claim.",
        "> - **Tiêu chí gán nhãn cột `Đánh giá (1/0)`:**",
        ">   - Điền **1** nếu chunk nói về **đúng sự kiện/thực thể** của claim và **đủ để giúp xác minh** (ủng hộ hoặc bác bỏ).",
        ">   - Điền **0** nếu còn lại, kể cả cùng chủ đề rộng nhưng không đủ để xác minh.",
        "",
        "| Item ID | Claim ID | Phát biểu (Claim) | Nguồn | Tiêu đề bài viết | Trích đoạn bằng chứng (Toàn văn) | URL | Đánh giá (1/0) |",
        "|---|---|---|---|---|---|---|:---:|",
    ]
    for it in items:
        stmt_esc = it.statement.replace("|", "\\|")
        title_esc = it.title.replace("|", "\\|")
        text_esc = it.text.replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| `{it.label_item_id}` | `{it.claim_id}` | {stmt_esc} | `{it.source}` | {title_esc} | {text_esc} | [Link]({it.url}) | [ ] |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    # 5. README instructions in the artifact directory
    readme_path = out_dir / "README.md"
    readme_path.write_text(
        "# Hướng dẫn Gán Nhãn Gói Human Labeling Phase 2.5 C1\n\n"
        "## 1. Quy tắc gán nhãn:\n"
        "- Mở file `human_label_pack_60.csv` trong Excel / Google Sheets.\n"
        "- Tại cột `human_verdict_1_or_0`:\n"
        "  - Điền **1** nếu chunk nói về **đúng sự kiện/thực thể** của claim và **đủ để giúp xác minh** (ủng hộ hoặc bác bỏ).\n"
        "  - Điền **0** nếu còn lại, kể cả cùng chủ đề rộng nhưng không đủ để xác minh.\n\n"
        "## 2. Bảo mật nhãn mù:\n"
        "- File không chứa bất kỳ cột mã hệ thống nào (`system_code`).\n"
        "- Thứ tự các chunk trong mỗi claim đã được xáo trộn ngẫu nhiên (seed=42).\n"
        "- Khóa giải mã được lưu riêng tại `unblind_key_do_not_open_before_labeling.json` "
        "và chỉ được mở sau khi hoàn tất gán nhãn.\n",
        encoding="utf-8",
    )

    logger.info(
        "Exported %d labeling items to %s (JSON, CSV, MD, README)",
        len(items),
        out_dir,
    )


def compute_metrics_and_write_report(
    records: Sequence[ClaimABRecord],
    *,
    out_path: Path,
    manifest_path: Path,
    prompt_version: str,
    prompt_sha256: str,
    model_name: str,
    retrieve_k: int,
    rerank_k: int,
) -> None:
    """Calculate branch proxy metrics, latencies, and format markdown report."""
    n = len(records)
    if n == 0:
        return

    # Hit rates per branch
    def _hit_rate(branch_fn: Any, k_name: str) -> float:
        hits = sum(1 for r in records if branch_fn(r).proxy_hits[k_name])
        return hits / n

    hit1_orig = _hit_rate(lambda r: r.branch_original, "hit@1")
    hit3_orig = _hit_rate(lambda r: r.branch_original, "hit@3")
    hit5_orig = _hit_rate(lambda r: r.branch_original, "hit@5")

    hit1_rew = _hit_rate(lambda r: r.branch_rewritten_only, "hit@1")
    hit3_rew = _hit_rate(lambda r: r.branch_rewritten_only, "hit@3")
    hit5_rew = _hit_rate(lambda r: r.branch_rewritten_only, "hit@5")

    hit1_fused = _hit_rate(lambda r: r.branch_fused, "hit@1")
    hit3_fused = _hit_rate(lambda r: r.branch_fused, "hit@3")
    hit5_fused = _hit_rate(lambda r: r.branch_fused, "hit@5")

    # Rewrite status breakdown
    status_counts = Counter(r.rewrite_status for r in records)
    latencies_rewrite = [r.rewrite_latency_ms for r in records if r.rewrite_latency_ms > 0]
    latencies_rewrite.sort()
    rew_p50 = latencies_rewrite[len(latencies_rewrite) // 2] if latencies_rewrite else 0.0
    idx95 = min(len(latencies_rewrite) - 1, math.ceil(len(latencies_rewrite) * 0.95) - 1)
    rew_p95 = latencies_rewrite[idx95] if latencies_rewrite else 0.0

    # Pairwise branch divergence (A12.3: số claim mà top-5 sau rerank khác nhau)
    diff_orig_rew = sum(
        1
        for r in records
        if [c.chunk_id for c in r.branch_original.evidences[:5]]
        != [c.chunk_id for c in r.branch_rewritten_only.evidences[:5]]
    )
    diff_orig_fused = sum(
        1
        for r in records
        if [c.chunk_id for c in r.branch_original.evidences[:5]]
        != [c.chunk_id for c in r.branch_fused.evidences[:5]]
    )
    diff_rew_fused = sum(
        1
        for r in records
        if [c.chunk_id for c in r.branch_rewritten_only.evidences[:5]]
        != [c.chunk_id for c in r.branch_fused.evidences[:5]]
    )

    # Write manifest
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "task": "Phase 2.5 Task 5 / C1 A/B testing",
        "num_claims": n,
        "per_class": n // 3,
        "prompt_version": prompt_version,
        "prompt_sha256": prompt_sha256,
        "model": model_name,
        "retrieve_k": retrieve_k,
        "rerank_k": rerank_k,
        "proxy_hit_rates": {
            "original": {"hit@1": hit1_orig, "hit@3": hit3_orig, "hit@5": hit5_orig},
            "rewritten_only": {"hit@1": hit1_rew, "hit@3": hit3_rew, "hit@5": hit5_rew},
            "original+rewritten": {
                "hit@1": hit1_fused,
                "hit@3": hit3_fused,
                "hit@5": hit5_fused,
            },
        },
        "branch_divergence_top5_claims": {
            "original_vs_rewritten_only": diff_orig_rew,
            "original_vs_fused": diff_orig_fused,
            "rewritten_only_vs_fused": diff_rew_fused,
        },
        "rewrite_status": dict(status_counts),
        "rewrite_latency_p50_ms": rew_p50,
        "rewrite_latency_p95_ms": rew_p95,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    # Markdown Report
    lines = [
        "# Báo cáo Thử nghiệm A/B Query Rewrite (C1: 60 Claims Dev) — Phase 2.5",
        "",
        f"- **Thời gian thực hiện:** `{manifest['timestamp_utc']}`",
        f"- **Model LLM:** `{model_name}`",
        f"- **Phiên bản Prompt:** `{prompt_version}` (`sha256`: `{prompt_sha256}`)",
        f"- **Số lượng claim:** `{n}` (20 Real, 20 Fake, 20 Unverifiable; seed={_DEFAULT_SEED})",
        f"- **Tham số Retrieval:** `retrieve_k = {retrieve_k}`, `rerank_k = {rerank_k}`",
        "",
        "## 1. Bảng Tổng Hợp Proxy Metric (Hit Rate trên 60 claims dev)",
        "",
        "| Nhánh Thử Nghiệm | Hit@1 (Proxy) | Hit@3 (Proxy) | Hit@5 (Proxy) | Số claim trúng @5 |",
        "|---|---:|---:|---:|---:|",
        f"| **1. `original` (Baseline)** | `{hit1_orig * 100:.2f}%` | `{hit3_orig * 100:.2f}%` | `{hit5_orig * 100:.2f}%` | `{int(hit5_orig * n)}/{n}` |",
        f"| **2. `rewritten_only`** | `{hit1_rew * 100:.2f}%` | `{hit3_rew * 100:.2f}%` | `{hit5_rew * 100:.2f}%` | `{int(hit5_rew * n)}/{n}` |",
        f"| **3. `original+rewritten` (RRF)** | `{hit1_fused * 100:.2f}%` | `{hit3_fused * 100:.2f}%` | `{hit5_fused * 100:.2f}%` | `{int(hit5_fused * n)}/{n}` |",
        "",
        "## 2. Độ Khác Biệt Giữa Các Nhánh (Branch Divergence Top-5)",
        "",
        f"- **`original` vs `rewritten_only`:** khác nhau ở `{diff_orig_rew}/{n}` claim (`{diff_orig_rew / n * 100:.1f}%`).",
        f"- **`original` vs `original+rewritten`:** khác nhau ở `{diff_orig_fused}/{n}` claim (`{diff_orig_fused / n * 100:.1f}%`).",
        f"- **`rewritten_only` vs `original+rewritten`:** khác nhau ở `{diff_rew_fused}/{n}` claim (`{diff_rew_fused / n * 100:.1f}%`).",
        "",
        "> [!IMPORTANT]",
        "> Các chỉ số trên là **Proxy Metric** (đo độ trùng lặp từ khóa n-gram với gold evidence/statement).",
        "> Quyết định chính thức của Cổng C1 phụ thuộc vào **kết quả gán nhãn thủ công (Human Labeling)** của PM trên file `artifacts/ab_testing/human_label_pack_60.csv`.",
        "",
        "## 3. Thống Kê Rewrite & Quota Gemini",
        "",
        f"- **Phân bố trạng thái Rewrite:** `{dict(status_counts)}`",
        f"- **Tỉ lệ Rewrite Thành công (`ok` / `cache_hit`):** `{((status_counts.get('ok', 0) + status_counts.get('cache_hit', 0)) / n) * 100:.1f}%`",
        f"- **Độ trễ Rewrite p50:** `{rew_p50:.2f} ms`",
        f"- **Độ trễ Rewrite p95:** `{rew_p95:.2f} ms`",
        "",
        "## 4. Gói Gán Nhãn Mù (Human Labeling Pack)",
        "",
        "- File dữ liệu gán nhãn JSON: `artifacts/ab_testing/human_label_pack_60.json`",
        "- File bảng tính gán nhãn CSV: `artifacts/ab_testing/human_label_pack_60.csv`",
        "- File hướng dẫn & bảng Markdown: `artifacts/ab_testing/human_label_pack_60.md`",
        "- Khóa giải mã nhánh (anonymization key): `artifacts/ab_testing/unblind_key_do_not_open_before_labeling.json`",
        "",
        "## 5. Chi tiết 60 Claims Dev và Trạng thái Rewrite",
        "",
        "| ID | Nhãn Gốc | Claim Gốc | Rewritten Query (v3) | Trạng thái | Rewrite ≠ Claim |",
        "|---|---|---|---|---|:---:|",
    ]
    for r in records:
        label_str = {0: "REAL", 1: "FAKE", 2: "UNVERIFIABLE"}.get(r.gold_label, str(r.gold_label))
        cl_esc = r.statement.replace("|", "\\|")
        rw_esc = r.rewritten_query.replace("|", "\\|")
        is_diff = "Có" if r.statement.strip() != r.rewritten_query.strip() else "Không"
        lines.append(
            f"| `{r.claim_id}` | `{label_str}` | {cl_esc} | `{rw_esc}` | `{r.rewrite_status}` | {is_diff} |"
        )

    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote C1 A/B report to %s", out_path)
