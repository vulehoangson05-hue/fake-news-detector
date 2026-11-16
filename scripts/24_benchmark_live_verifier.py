"""Task 4.3 (Brief #5 §4.8 & §12.5): Benchmark LiveVerifier step-by-step latency & Peak VRAM.

Runs strictly offline over the 30 cached Q1v2 spike claims (0 Tavily credits spent)
in both ``fp32`` (``use_fp16=False``) and ``fp16`` (``use_fp16=True``) sequentially on GTX 1660.
Records per-stage p50/p95 latencies, verdict/mode counts, Peak VRAM (MB), and asserts that
``vnfake.retrieval.embedder`` (BAAI/bge-m3) is never imported during the cached/live path.
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HOME", r"D:\hf-cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

from vnfake.config import load_config, load_search_config
from vnfake.pipeline.live_verifier import LiveVerifier
from vnfake.retrieval.embedder import BgeM3Embedder
from vnfake.retrieval.web_search import TavilyWebSearcher
from vnfake.utils.logging import get_logger

logger = get_logger(__name__)

Q1V2_REPORT_PATH = Path("artifacts/q1_vs_q1v2_report.json")
FALLBACK_SPIKE_PATH = Path("artifacts/tavily_spike_report.json")
OUTPUT_PATH = Path("artifacts/live_verifier_benchmark.json")


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


def _summarize_stage(vals: list[float]) -> dict[str, float]:
    return {
        "mean_ms": round(sum(vals) / max(len(vals), 1), 2),
        "p50_ms": round(_percentile(vals, 50.0), 2),
        "p95_ms": round(_percentile(vals, 95.0), 2),
        "max_ms": round(max(vals) if vals else 0.0, 2),
    }


def _benchmark_mode(
    *,
    use_fp16: bool,
    claims: list[dict[str, Any]],
    device: str,
) -> dict[str, Any]:
    cfg = load_config()
    search_cfg = load_search_config()
    searcher = TavilyWebSearcher(search_cfg)
    credits_before = searcher.credits_used

    if device.startswith("cuda") and torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline_alloc_mb = torch.cuda.memory_allocated() / (1024**2)
    else:
        baseline_alloc_mb = 0.0

    verifier = LiveVerifier(
        cfg,
        search_cfg,
        device=device,
        use_fp16=use_fp16,
        demo_mode=True,
        short_circuit_low_relevance=True,
        searcher=searcher,
    )

    bge_m3_calls = {"init": 0, "load": 0}
    orig_init = BgeM3Embedder.__init__
    orig_load = BgeM3Embedder._load

    def _spy_init(self: Any, *args: Any, **kwargs: Any) -> Any:
        bge_m3_calls["init"] += 1
        return orig_init(self, *args, **kwargs)

    def _spy_load(self: Any) -> Any:
        bge_m3_calls["load"] += 1
        return orig_load(self)

    BgeM3Embedder.__init__ = _spy_init  # type: ignore[method-assign]
    BgeM3Embedder._load = _spy_load  # type: ignore[method-assign]

    t_warm0 = time.perf_counter()
    verifier.warmup()
    warmup_ms = round((time.perf_counter() - t_warm0) * 1000.0, 2)

    if device.startswith("cuda") and torch.cuda.is_available():
        after_warmup_alloc_mb = torch.cuda.memory_allocated() / (1024**2)
        after_warmup_peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    else:
        after_warmup_alloc_mb = 0.0
        after_warmup_peak_mb = 0.0

    stage_keys = ("query_ms", "search_ms", "fetch_ms", "select_ms", "rerank_ms", "verify_ms", "total_ms")
    stage_vals: dict[str, list[float]] = {k: [] for k in stage_keys}
    verify_nonzero_vals: list[float] = []
    mode_counts: dict[str, int] = {}
    verdict_counts: dict[str, int] = {}
    sample_outputs: list[dict[str, Any]] = []

    try:
        for item in claims:
            stmt = str(item["statement"])
            res = verifier.verify(stmt)
            lat_dict = res.latency_ms.model_dump()
            for k in stage_keys:
                stage_vals[k].append(float(lat_dict[k]))
            if float(lat_dict["verify_ms"]) > 0.0:
                verify_nonzero_vals.append(float(lat_dict["verify_ms"]))

            mode_counts[res.mode] = mode_counts.get(res.mode, 0) + 1
            verdict_counts[res.verdict] = verdict_counts.get(res.verdict, 0) + 1
            if len(sample_outputs) < 5:
                sample_outputs.append(res.model_dump())

        if device.startswith("cuda") and torch.cuda.is_available():
            peak_alloc_mb = torch.cuda.max_memory_allocated() / (1024**2)
            peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024**2)
        else:
            peak_alloc_mb = 0.0
            peak_reserved_mb = 0.0
    finally:
        BgeM3Embedder.__init__ = orig_init  # type: ignore[method-assign]
        BgeM3Embedder._load = orig_load  # type: ignore[method-assign]
        credits_after = searcher.credits_used
        verifier.unload()
        gc.collect()
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    bge_m3_loaded = bge_m3_calls["init"] > 0 or bge_m3_calls["load"] > 0
    if bge_m3_loaded or mode_counts.get("fallback", 0) > 0:
        raise RuntimeError(
            f"Invariant violated: bge_m3_calls={bge_m3_calls}, mode_counts={mode_counts}"
        )

    return {
        "precision": "fp16" if use_fp16 else "fp32",
        "device": device,
        "num_claims": len(claims),
        "credits_spent": credits_after - credits_before,
        "warmup_ms": warmup_ms,
        "vram_mb": {
            "baseline_allocated_mb": round(baseline_alloc_mb, 2),
            "after_warmup_allocated_mb": round(after_warmup_alloc_mb, 2),
            "after_warmup_peak_mb": round(after_warmup_peak_mb, 2),
            "peak_allocated_mb": round(peak_alloc_mb, 2),
            "peak_reserved_mb": round(peak_reserved_mb, 2),
        },
        "latency_stages_ms": {k: _summarize_stage(v) for k, v in stage_vals.items()},
        "verify_ms_when_executed": {
            "executed_count": len(verify_nonzero_vals),
            **_summarize_stage(verify_nonzero_vals),
        },
        "mode_counts": mode_counts,
        "verdict_counts": verdict_counts,
        "bge_m3_loaded": bge_m3_loaded,
        "bge_m3_calls": bge_m3_calls,
        "sample_outputs": sample_outputs,
    }


def main() -> int:
    source_path = Q1V2_REPORT_PATH if Q1V2_REPORT_PATH.exists() else FALLBACK_SPIKE_PATH
    if not source_path.exists():
        logger.error("Neither %s nor %s exists.", Q1V2_REPORT_PATH, FALLBACK_SPIKE_PATH)
        return 1

    raw = json.loads(source_path.read_text(encoding="utf-8"))
    claims = (
        raw["variants"]["Q1v2_k5"]["claims"]
        if "Q1v2_k5" in raw.get("variants", {})
        else raw["variants"]["Q1"]["claims"]
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Benchmarking LiveVerifier on device=%s across %d cached claims", device, len(claims))

    fp32_res = _benchmark_mode(use_fp16=False, claims=claims, device=device)
    logger.info(
        "[fp32] peak_vram=%.1f MB | total p50=%.2f ms, p95=%.2f ms | modes=%s | verdicts=%s",
        fp32_res["vram_mb"]["peak_allocated_mb"],
        fp32_res["latency_stages_ms"]["total_ms"]["p50_ms"],
        fp32_res["latency_stages_ms"]["total_ms"]["p95_ms"],
        fp32_res["mode_counts"],
        fp32_res["verdict_counts"],
    )

    fp16_res: dict[str, Any] | None = None
    if device.startswith("cuda"):
        fp16_res = _benchmark_mode(use_fp16=True, claims=claims, device=device)
        logger.info(
            "[fp16] peak_vram=%.1f MB | total p50=%.2f ms, p95=%.2f ms | modes=%s | verdicts=%s",
            fp16_res["vram_mb"]["peak_allocated_mb"],
            fp16_res["latency_stages_ms"]["total_ms"]["p50_ms"],
            fp16_res["latency_stages_ms"]["total_ms"]["p95_ms"],
            fp16_res["mode_counts"],
            fp16_res["verdict_counts"],
        )

    out = {
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu",
        "num_claims": len(claims),
        "fp32": fp32_res,
        "fp16": fp16_res,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved live verifier benchmark to %s", OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
