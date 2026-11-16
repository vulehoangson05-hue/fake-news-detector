"""Phase 2.5 Task 1 — Reranker latency profiling and rank-stability analysis.

Measures per-claim latency breakdown (tokenize, forward, post-process, ms/pair),
p50/p95 across (retrieve_k, batch_size, dtype, max_length) configurations on a
reproducible dev subset, and top-k overlap against the baseline configuration.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from vnfake.config import AppConfig
from vnfake.evaluation.retrieval_eval import (
    EvalQuery,
    resolve_collection_name,
)
from vnfake.retrieval.embedder import EmbedderProtocol
from vnfake.retrieval.vector_store import QdrantStore, ScoredChunk
from vnfake.utils.logging import get_logger

logger = get_logger(__name__)

_MIN_RETRIEVE_K = 20
_DEFAULT_WARMUP_CLAIMS = 3
_DEFAULT_SUBSET_SIZE = 100
_DEFAULT_TOP_K_OVERLAP = 5
_MS_PER_SECOND = 1000.0
_BYTES_PER_GB = 1e9
_PERCENTILE_P50 = 50.0
_PERCENTILE_P95 = 95.0
_PERCENTILE_DIVISOR = 100.0
_CLAIM_HASH_LEN = 8


@dataclass(frozen=True, slots=True)
class EnvironmentInfo:
    """Hardware and library environment metadata for reproducible profiling."""

    timestamp_utc: str
    os_platform: str
    cpu_model: str
    cpu_logical_cores: int
    torch_num_threads: int
    device: str
    gpu_name: str
    gpu_vram_gb: float
    cuda_available: bool
    bf16_supported: bool
    python_version: str
    torch_version: str
    transformers_version: str
    flagembedding_version: str
    sentence_transformers_version: str


@dataclass(frozen=True, slots=True)
class ClaimTiming:
    """Per-claim timing breakdown in milliseconds."""

    claim_id: str
    tokenize_ms: float
    forward_ms: float
    postprocess_ms: float
    total_ms: float
    num_pairs: int
    top_chunk_ids: list[str]


@dataclass(frozen=True, slots=True)
class ConfigProfileResult:
    """Aggregated profiling result for one (k, batch_size, dtype, max_length) run."""

    retrieve_k: int
    batch_size: int
    dtype: str
    max_length: int
    status: str
    error_message: str | None
    num_claims_measured: int
    warmup_claims_skipped: int
    p50_ms: float
    p95_ms: float
    mean_ms: float
    mean_tokenize_ms: float
    mean_forward_ms: float
    mean_postprocess_ms: float
    ms_per_pair: float
    overlap_top5_vs_baseline: float | None
    top5_ids_per_claim: dict[str, list[str]]


def compute_percentile(values: Sequence[float], pct: float) -> float:
    """Compute linear-interpolated percentile on ``values``."""
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


def compute_top_k_overlap(
    baseline_ids: Sequence[str],
    candidate_ids: Sequence[str],
    k: int = _DEFAULT_TOP_K_OVERLAP,
) -> float:
    """Compute set-overlap ratio in top-``k`` between baseline and candidate."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    base_slice = list(baseline_ids[:k])
    cand_slice = list(candidate_ids[:k])
    if not base_slice and not cand_slice:
        return 1.0
    denom = min(k, max(len(base_slice), len(cand_slice)))
    if denom == 0:
        return 1.0
    return len(set(base_slice) & set(cand_slice)) / float(denom)


def compute_mean_top_k_overlap(
    baseline_map: dict[str, list[str]],
    candidate_map: dict[str, list[str]],
    k: int = _DEFAULT_TOP_K_OVERLAP,
) -> float:
    """Compute mean top-``k`` overlap across all common ``claim_id`` keys."""
    common_keys = [cid for cid in baseline_map if cid in candidate_map]
    if not common_keys:
        return 0.0
    total = sum(
        compute_top_k_overlap(baseline_map[cid], candidate_map[cid], k=k) for cid in common_keys
    )
    return total / len(common_keys)


def make_claim_id(index: int, statement: str) -> str:
    """Create a deterministic human-readable claim identifier."""
    digest = hashlib.sha1(statement.encode("utf-8")).hexdigest()[:_CLAIM_HASH_LEN]
    return f"dev_{index:04d}_{digest}"


def sample_dev_claims(
    queries: Sequence[EvalQuery],
    n_claims: int,
    seed: int,
) -> list[tuple[str, EvalQuery]]:
    """Deterministically sample ``n_claims`` from ``queries`` using ``seed``."""
    indexed = [(make_claim_id(i, q.statement), q) for i, q in enumerate(queries)]
    if n_claims >= len(indexed):
        return indexed
    rng = random.Random(seed)
    return rng.sample(indexed, k=n_claims)


def _pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def collect_environment_info() -> EnvironmentInfo:
    """Gather CPU, GPU, VRAM, and library version metadata."""
    cuda_ok = torch.cuda.is_available()
    device = "cuda" if cuda_ok else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "CPU only"
    vram_gb = (
        round(torch.cuda.get_device_properties(0).total_memory / _BYTES_PER_GB, 2)
        if cuda_ok
        else 0.0
    )
    bf16_ok = bool(torch.cuda.is_bf16_supported()) if cuda_ok else False
    cpu_name = (
        platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown") or "unknown"
    )
    return EnvironmentInfo(
        timestamp_utc=datetime.now(UTC).isoformat(),
        os_platform=platform.platform(),
        cpu_model=cpu_name,
        cpu_logical_cores=os.cpu_count() or 1,
        torch_num_threads=torch.get_num_threads(),
        device=device,
        gpu_name=gpu_name,
        gpu_vram_gb=vram_gb,
        cuda_available=cuda_ok,
        bf16_supported=bf16_ok,
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        transformers_version=_pkg_version("transformers"),
        flagembedding_version=_pkg_version("FlagEmbedding"),
        sentence_transformers_version=_pkg_version("sentence-transformers"),
    )


def _sync_device(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _resolve_torch_dtype(dtype_name: str, device: str) -> torch.dtype | None:
    """Return target ``torch.dtype`` or ``None`` if unsupported on current device."""
    norm = dtype_name.lower().strip()
    if norm == "fp32":
        return torch.float32
    if not device.startswith("cuda"):
        return None
    if norm == "fp16":
        return torch.float16
    if norm == "bf16":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else None
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _profile_single_claim(
    claim_id: str,
    statement: str,
    candidates: Sequence[ScoredChunk],
    *,
    tokenizer: object,
    model: torch.nn.Module,
    device: str,
    batch_size: int,
    max_length: int,
    rerank_k: int,
) -> ClaimTiming:
    """Run tokenized cross-encoder scoring for one claim with fine-grained timers."""
    if not candidates:
        return ClaimTiming(claim_id, 0.0, 0.0, 0.0, 0.0, 0, [])

    pairs = [[statement, c.text] for c in candidates]
    tokenize_ms = 0.0
    forward_ms = 0.0
    postprocess_ms = 0.0
    all_scores: list[float] = []

    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start : start + batch_size]

            t0 = time.perf_counter()
            inputs = tokenizer(  # type: ignore[operator]
                batch_pairs,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tokenize_ms += (time.perf_counter() - t0) * _MS_PER_SECOND

            _sync_device(device)
            t1 = time.perf_counter()
            inputs_on_dev = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs_on_dev, return_dict=True).logits.view(-1).float()
            _sync_device(device)
            forward_ms += (time.perf_counter() - t1) * _MS_PER_SECOND

            t2 = time.perf_counter()
            probs = torch.sigmoid(logits).cpu().tolist()
            if isinstance(probs, float):
                all_scores.append(float(probs))
            else:
                all_scores.extend(float(x) for x in probs)
            postprocess_ms += (time.perf_counter() - t2) * _MS_PER_SECOND

    t3 = time.perf_counter()
    scored_pairs = sorted(
        zip(candidates, all_scores, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    top_ids = [c.chunk_id for c, _ in scored_pairs[:rerank_k]]
    postprocess_ms += (time.perf_counter() - t3) * _MS_PER_SECOND

    total_ms = tokenize_ms + forward_ms + postprocess_ms
    return ClaimTiming(
        claim_id=claim_id,
        tokenize_ms=tokenize_ms,
        forward_ms=forward_ms,
        postprocess_ms=postprocess_ms,
        total_ms=total_ms,
        num_pairs=len(pairs),
        top_chunk_ids=top_ids,
    )


def load_reranker_for_dtype(
    model_name: str,
    dtype_name: str,
    device: str,
) -> tuple[object, torch.nn.Module] | None:
    """Load tokenizer and sequence-classification model for ``dtype_name`` once."""
    torch_dtype = _resolve_torch_dtype(dtype_name, device)
    if torch_dtype is None:
        return None
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        dtype=torch_dtype,
    ).to(device)
    model.eval()
    return tokenizer, model


def profile_configuration(
    model_name: str,
    claims_with_candidates: Sequence[tuple[str, str, list[ScoredChunk]]],
    *,
    retrieve_k: int,
    rerank_k: int,
    batch_size: int,
    dtype_name: str,
    max_length: int,
    device: str,
    warmup_claims: int = _DEFAULT_WARMUP_CLAIMS,
    baseline_top5: dict[str, list[str]] | None = None,
    loaded_pair: tuple[object, torch.nn.Module] | None = None,
) -> ConfigProfileResult:
    """Profile one `(retrieve_k, batch_size, dtype, max_length)` configuration."""
    if retrieve_k < _MIN_RETRIEVE_K:
        raise ValueError(
            f"Hard constraint violated: retrieve_k={retrieve_k} < {_MIN_RETRIEVE_K} is forbidden."
        )

    torch_dtype = _resolve_torch_dtype(dtype_name, device)
    if torch_dtype is None:
        return ConfigProfileResult(
            retrieve_k=retrieve_k,
            batch_size=batch_size,
            dtype=dtype_name,
            max_length=max_length,
            status="not_applicable",
            error_message=f"dtype {dtype_name} not applicable on {device}",
            num_claims_measured=0,
            warmup_claims_skipped=0,
            p50_ms=0.0,
            p95_ms=0.0,
            mean_ms=0.0,
            mean_tokenize_ms=0.0,
            mean_forward_ms=0.0,
            mean_postprocess_ms=0.0,
            ms_per_pair=0.0,
            overlap_top5_vs_baseline=None,
            top5_ids_per_claim={},
        )

    owns_model = loaded_pair is None
    model: torch.nn.Module | None = None
    try:
        if loaded_pair is not None:
            tokenizer, model = loaded_pair
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                dtype=torch_dtype,
            ).to(device)
            model.eval()

        timings: list[ClaimTiming] = []
        top5_map: dict[str, list[str]] = {}
        for claim_id, statement, candidates in claims_with_candidates:
            sliced_candidates = candidates[:retrieve_k]
            timing = _profile_single_claim(
                claim_id,
                statement,
                sliced_candidates,
                tokenizer=tokenizer,
                model=model,
                device=device,
                batch_size=batch_size,
                max_length=max_length,
                rerank_k=rerank_k,
            )
            timings.append(timing)
            top5_map[claim_id] = timing.top_chunk_ids

        measured = timings[warmup_claims:] if len(timings) > warmup_claims else timings
        skipped = len(timings) - len(measured)
        totals = [t.total_ms for t in measured]
        toks = [t.tokenize_ms for t in measured]
        fwds = [t.forward_ms for t in measured]
        posts = [t.postprocess_ms for t in measured]
        total_pairs = sum(t.num_pairs for t in measured)
        n_m = len(measured)

        overlap = (
            1.0
            if baseline_top5 is None
            else compute_mean_top_k_overlap(baseline_top5, top5_map, k=rerank_k)
        )

        return ConfigProfileResult(
            retrieve_k=retrieve_k,
            batch_size=batch_size,
            dtype=dtype_name,
            max_length=max_length,
            status="ok",
            error_message=None,
            num_claims_measured=n_m,
            warmup_claims_skipped=skipped,
            p50_ms=compute_percentile(totals, _PERCENTILE_P50),
            p95_ms=compute_percentile(totals, _PERCENTILE_P95),
            mean_ms=(sum(totals) / n_m) if n_m else 0.0,
            mean_tokenize_ms=(sum(toks) / n_m) if n_m else 0.0,
            mean_forward_ms=(sum(fwds) / n_m) if n_m else 0.0,
            mean_postprocess_ms=(sum(posts) / n_m) if n_m else 0.0,
            ms_per_pair=(sum(totals) / total_pairs) if total_pairs else 0.0,
            overlap_top5_vs_baseline=overlap,
            top5_ids_per_claim=top5_map,
        )
    except RuntimeError as exc:
        is_oom = "out of memory" in str(exc).lower()
        status = "oom" if is_oom else "error"
        logger.warning(
            "Config (k=%d, batch=%d, dtype=%s, max_len=%d) failed with %s: %s",
            retrieve_k,
            batch_size,
            dtype_name,
            max_length,
            status,
            exc,
        )
        return ConfigProfileResult(
            retrieve_k=retrieve_k,
            batch_size=batch_size,
            dtype=dtype_name,
            max_length=max_length,
            status=status,
            error_message=str(exc).splitlines()[0] if str(exc) else status,
            num_claims_measured=0,
            warmup_claims_skipped=0,
            p50_ms=0.0,
            p95_ms=0.0,
            mean_ms=0.0,
            mean_tokenize_ms=0.0,
            mean_forward_ms=0.0,
            mean_postprocess_ms=0.0,
            ms_per_pair=0.0,
            overlap_top5_vs_baseline=None,
            top5_ids_per_claim={},
        )
    finally:
        if owns_model:
            del model
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()


def precompute_candidates(
    cfg: AppConfig,
    sampled_claims: Sequence[tuple[str, EvalQuery]],
    *,
    embedder: EmbedderProtocol,
    store: QdrantStore,
    max_k: int,
    collection: str | None = None,
) -> list[tuple[str, str, list[ScoredChunk]]]:
    """Embed ``sampled_claims`` once, fetch top-``max_k`` hybrid candidates, and free embedder."""
    resolved_collection = resolve_collection_name(cfg, collection)
    statements = [q.statement for _, q in sampled_claims]
    logger.info("Encoding %d sampled claims for candidate precomputation...", len(statements))
    dense_vecs = embedder.embed_dense(statements, batch_size=cfg.retrieval.embedding.batch_size)
    sparse_vecs = embedder.embed_sparse(statements, batch_size=cfg.retrieval.sparse.batch_size)

    out: list[tuple[str, str, list[ScoredChunk]]] = []
    for (claim_id, query), d_vec, s_vec in zip(
        sampled_claims, dense_vecs, sparse_vecs, strict=True
    ):
        hits = store.search_hybrid(
            resolved_collection,
            dense_query=d_vec,
            sparse_query=s_vec,
            top_k=max_k,
            dense_prefetch_k=max(cfg.retrieval.hybrid.dense_prefetch_k, max_k),
            sparse_prefetch_k=max(cfg.retrieval.hybrid.sparse_prefetch_k, max_k),
        )
        out.append((claim_id, query.statement, hits))
    return out


def save_profiling_artifact(
    output_path: Path,
    *,
    env_info: EnvironmentInfo,
    seed: int,
    claim_ids_path: Path,
    claim_ids: Sequence[str],
    results: Sequence[ConfigProfileResult],
) -> None:
    """Persist raw profiling JSON and the list of sampled ``claim_id`` values."""
    claim_ids_path.parent.mkdir(parents=True, exist_ok=True)
    claim_ids_path.write_text(
        json.dumps(
            {"seed": seed, "n_claims": len(claim_ids), "claim_ids": list(claim_ids)}, indent=2
        ),
        encoding="utf-8",
    )

    serializable_results = []
    for r in results:
        d = asdict(r)
        d.pop("top5_ids_per_claim", None)
        serializable_results.append(d)

    payload = {
        "environment": asdict(env_info),
        "seed": seed,
        "claim_ids_file": str(claim_ids_path),
        "num_sampled_claims": len(claim_ids),
        "results": serializable_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved profiling artifact to %s", output_path)
