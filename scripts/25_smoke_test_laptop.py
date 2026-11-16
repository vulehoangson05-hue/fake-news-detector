"""Task 4.4 (Brief #5 §6.3) — Automated Smoke Test for Laptop RTX 3050 (4 GB VRAM) & PC.

Single-command verification script for PM / Tech Lead:
1. Detects GPU hardware (``torch.cuda.get_device_name(0)``, total VRAM MB, compute capability).
2. Runs ``LiveVerifier`` + ``app.demo`` rendering functions across the 6 cached demo claims
   in BOTH ``fp16`` and ``fp32`` modes sequentially (0 Tavily credits, 0 network calls).
3. Verifies UI rendering rules (including mandatory §12.5.3 probability hiding when ``show_probs=False``)
   and ``create_demo()`` Gradio Blocks construction.
4. Records Peak VRAM (allocated & reserved MB) and step-by-step latency (p50/p95 ms) to
   ``artifacts/laptop_smoke_test.json``.
"""

from __future__ import annotations

import gc
import json
import math
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

for _cand in (Path(r"D:\hf-cache"), Path.home() / ".cache" / "huggingface"):
    if _cand.exists():
        os.environ.setdefault("HF_HOME", str(_cand))
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        break

import torch  # noqa: E402

# Ensure repo root is on sys.path so `import app.demo` works when invoked directly
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.demo import (  # noqa: E402
    DEFAULT_EXAMPLES,
    create_demo,
    render_evidence_markdown,
    render_metrics_markdown,
    render_probabilities,
    render_verdict_badge,
)
from vnfake.config import load_config, load_search_config  # noqa: E402
from vnfake.pipeline.live_verifier import LiveVerifier  # noqa: E402
from vnfake.retrieval.embedder import BgeM3Embedder  # noqa: E402
from vnfake.retrieval.web_search import TavilyWebSearcher  # noqa: E402
from vnfake.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

OUTPUT_PATH = Path("artifacts/laptop_smoke_test.json")
LAPTOP_VRAM_LIMIT_MB = 4096.0


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


def _summarize_latencies(vals: list[float]) -> dict[str, float]:
    return {
        "mean_ms": round(sum(vals) / max(len(vals), 1), 2),
        "p50_ms": round(_percentile(vals, 50.0), 2),
        "p95_ms": round(_percentile(vals, 95.0), 2),
        "max_ms": round(max(vals) if vals else 0.0, 2),
    }


def inspect_hardware() -> dict[str, Any]:
    """Inspect CPU/GPU hardware and VRAM capacity."""
    info: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": "CPU",
        "total_vram_mb": 0.0,
        "compute_capability": "N/A",
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        major, minor = torch.cuda.get_device_capability(0)
        info["device_name"] = torch.cuda.get_device_name(0)
        info["total_vram_mb"] = round(props.total_memory / (1024**2), 2)
        info["compute_capability"] = f"{major}.{minor}"
    return info


def run_precision_smoke(
    *,
    use_fp16: bool,
    claims: list[str],
    device: str,
) -> dict[str, Any]:
    """Run LiveVerifier + app.demo renderers in the specified precision mode."""
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

    stage_keys = (
        "query_ms",
        "search_ms",
        "fetch_ms",
        "select_ms",
        "rerank_ms",
        "verify_ms",
        "total_ms",
    )
    stage_vals: dict[str, list[float]] = {k: [] for k in stage_keys}
    render_checks: list[dict[str, Any]] = []

    try:
        # Also verify Gradio Blocks construction with this verifier instance
        demo_blocks = create_demo(verifier)
        assert demo_blocks is not None

        for stmt in claims:
            res = verifier.verify(stmt)
            lat_dict = res.latency_ms.model_dump()
            for k in stage_keys:
                stage_vals[k].append(float(lat_dict[k]))

            # Exercise all UI renderers from app/demo.py
            badge_html = render_verdict_badge(res.verdict, res.reason)
            probs_dict, warn_msg = render_probabilities(
                res.probs, res.show_probs, res.reason
            )
            ev_md = render_evidence_markdown(res.evidence)
            met_md = render_metrics_markdown(
                res, cfg.decision.tau_rel, cfg.decision.tau_conf
            )

            # Enforce §12.5.3 invariant on UI rendering
            if res.show_probs:
                if probs_dict is None or warn_msg is not None:
                    raise RuntimeError("UI failed to render probabilities when show_probs=True")
            else:
                if probs_dict is not None or not warn_msg:
                    raise RuntimeError("UI leaked probabilities when show_probs=False (§12.5.3)")

            render_checks.append(
                {
                    "claim": stmt[:80],
                    "verdict": res.verdict,
                    "mode": res.mode,
                    "show_probs": res.show_probs,
                    "relevance": res.relevance,
                    "confidence": res.confidence,
                    "total_ms": res.latency_ms.total_ms,
                    "badge_rendered": bool(badge_html),
                    "evidence_rendered": bool(ev_md),
                    "metrics_rendered": bool(met_md),
                }
            )

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
    if bge_m3_loaded:
        raise RuntimeError(f"BAAI/bge-m3 was unexpectedly loaded: {bge_m3_calls}")

    return {
        "precision": "fp16" if use_fp16 else "fp32",
        "device": device,
        "num_claims": len(claims),
        "credits_spent": credits_after - credits_before,
        "warmup_ms": warmup_ms,
        "vram_mb": {
            "baseline_allocated_mb": round(baseline_alloc_mb, 2),
            "peak_allocated_mb": round(peak_alloc_mb, 2),
            "peak_reserved_mb": round(peak_reserved_mb, 2),
            "fits_4gb_vram_budget": round(peak_alloc_mb, 2) <= LAPTOP_VRAM_LIMIT_MB,
        },
        "latency_stages_ms": {k: _summarize_latencies(v) for k, v in stage_vals.items()},
        "bge_m3_loaded": bge_m3_loaded,
        "ui_render_checks_passed": True,
        "claims_summary": render_checks,
    }


def main() -> int:
    hw = inspect_hardware()
    device = "cuda" if hw["cuda_available"] else "cpu"
    claims = [str(row[0]) for row in DEFAULT_EXAMPLES]

    logger.info(
        "Running Laptop/PC Smoke Test on %s (VRAM=%.1f MB, device=%s, claims=%d)",
        hw["device_name"],
        hw["total_vram_mb"],
        device,
        len(claims),
    )

    results: dict[str, Any] = {
        "hardware": hw,
        "modes": {},
    }

    # Test fp16 first (primary mode for 4GB Laptop RTX 3050), then fp32
    modes_to_run = [True, False] if device.startswith("cuda") else [False]
    for use_fp16 in modes_to_run:
        mode_key = "fp16" if use_fp16 else "fp32"
        res = run_precision_smoke(use_fp16=use_fp16, claims=claims, device=device)
        results["modes"][mode_key] = res
        logger.info(
            "[%s] Peak VRAM=%.2f MB (reserved=%.2f MB, fits_4gb=%s) | total p50=%.2f ms, p95=%.2f ms",
            mode_key,
            res["vram_mb"]["peak_allocated_mb"],
            res["vram_mb"]["peak_reserved_mb"],
            res["vram_mb"]["fits_4gb_vram_budget"],
            res["latency_stages_ms"]["total_ms"]["p50_ms"],
            res["latency_stages_ms"]["total_ms"]["p95_ms"],
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved smoke test report to %s", OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
