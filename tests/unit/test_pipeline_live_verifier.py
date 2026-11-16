"""Unit tests for Phase 4 Task 4.3 LiveVerifier pipeline."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vnfake.config import AppConfig, DecisionConfig, PathsConfig, SearchConfig
from vnfake.pipeline.decision import REASON_CONFIRMED, REASON_LOW_CONFIDENCE, REASON_LOW_RELEVANCE
from vnfake.pipeline.live_verifier import EvidenceItem, LiveVerifier, LiveVerifyResult
from vnfake.retrieval.query_rewriter import RewriteResult
from vnfake.retrieval.web_search import TavilyWebSearcher, build_keyword_query_v2


class _FakeTransport:
    def __init__(self, *, should_fail: bool = False, delay_s: float = 0.0) -> None:
        self.calls = 0
        self.should_fail = should_fail
        self.delay_s = delay_s

    def search_post(self, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        self.calls += 1
        if self.delay_s > 0:
            time.sleep(self.delay_s)
        if self.should_fail:
            raise RuntimeError("Simulated Tavily network failure")
        return {
            "results": [
                {
                    "title": "Bộ Giáo dục công bố lịch thi THPT",
                    "url": "https://vnexpress.net/lich-thi-thpt-2026-123456.html",
                    "content": "Lịch thi tốt nghiệp THPT năm 2026 diễn ra ngày 25-6.",
                    "raw_content": (
                        "Bộ Giáo dục và Đào tạo chính thức công bố lịch thi tốt nghiệp THPT "
                        "năm 2026 vào ngày 25-6 trên toàn quốc với hơn một triệu thí sinh tham dự. "
                        * 6
                    ),
                    "score": 0.94,
                },
                {
                    "title": "Thông tin bổ sung kỳ thi",
                    "url": "https://tuoitre.vn/ky-thi-thpt-2026-789.htm",
                    "content": "Thí sinh làm thủ tục dự thi ngày 24-6.",
                    "raw_content": (
                        "Các sở giáo dục địa phương đã chuẩn bị đầy đủ điểm thi và phương án "
                        "coi thi an toàn cho kỳ thi tốt nghiệp THPT năm 2026 vào cuối tháng 6. "
                        * 6
                    ),
                    "score": 0.85,
                },
            ]
        }


class _FakeReranker:
    def __init__(self, score: float = 0.92) -> None:
        self.score = score
        self.calls = 0

    def score_pair(self, query: str, text: str) -> float:
        self.calls += 1
        return self.score


class _FakeEnsemble:
    def __init__(self, probs: list[float] | None = None) -> None:
        self.probs = probs or [0.88, 0.08, 0.04]
        self.calls = 0

    def predict_single(
        self,
        statement: str,
        evidence: str,
        *,
        presegmented: bool = False,
    ) -> np.ndarray:
        self.calls += 1
        return np.asarray(self.probs, dtype=np.float64)


def _make_configs(tmp_path: Path) -> tuple[AppConfig, SearchConfig]:
    paths = PathsConfig(
        kb_dantri=tmp_path,
        kb_vnexpress=tmp_path,
        vifactcheck_dir=tmp_path,
        interim_dir=tmp_path,
        processed_dir=tmp_path,
        artifacts_dir=tmp_path / "artifacts",
    )
    app_cfg = AppConfig(
        paths=paths,
        label_map={0: "real", 1: "fake", 2: "unverifiable"},
        decision=DecisionConfig(tau_rel=0.7559, tau_conf=0.55, live_deadline_s=20.0),
    )
    search_cfg = SearchConfig(
        provider="tavily",
        search_depth="basic",
        max_results=5,
        topic="general",
        cache_dir=tmp_path / "web_cache",
        max_credits_spike=20,
        timeout_s=5.0,
        rate_limit_s=1.0,
        user_agent="VNFakeUnit/1.0",
        include_domains=["vnexpress.net", "tuoitre.vn"],
    )
    return app_cfg, search_cfg


def test_live_verifier_normal_path_and_no_kb_loaded(tmp_path: Path) -> None:
    app_cfg, search_cfg = _make_configs(tmp_path)
    transport = _FakeTransport()
    searcher = TavilyWebSearcher(
        search_cfg, api_key="tvly-test", transport=transport, page_fetcher=lambda _u: ""
    )
    kb_called = False

    def _forbid_kb(_claim: str) -> list[EvidenceItem]:
        nonlocal kb_called
        kb_called = True
        raise AssertionError("Local KB / bge-m3 MUST NOT be called during live path!")

    verifier = LiveVerifier(
        app_cfg,
        search_cfg,
        device="cpu",
        searcher=searcher,
        reranker=_FakeReranker(score=0.91),
        ensemble=_FakeEnsemble(probs=[0.88, 0.08, 0.04]),
        kb_fallback_fn=_forbid_kb,
    )

    stmt = "Bộ Giáo dục và Đào tạo công bố lịch thi tốt nghiệp THPT năm 2026 ngày 25-6."
    res = verifier.verify(stmt)

    assert isinstance(res, LiveVerifyResult)
    assert res.verdict == "real"
    assert res.mode == "live"
    assert res.show_probs is True
    assert res.reason == REASON_CONFIRMED
    assert res.relevance == pytest.approx(0.91, abs=1e-4)
    assert res.confidence == pytest.approx(0.88, abs=1e-4)
    assert res.probs == {"real": 0.88, "fake": 0.08, "unverifiable": 0.04}
    assert res.query == build_keyword_query_v2(stmt, segmenter=verifier._segmenter)
    assert not kb_called
    assert 1 <= len(res.evidence) <= 3
    assert all(ev.url.startswith("https://") and ev.domain in {"vnexpress.net", "tuoitre.vn"} for ev in res.evidence)
    assert res.latency_ms.total_ms >= 0.0
    assert res.latency_ms.select_ms >= 0.0
    assert res.latency_ms.rerank_ms >= 0.0
    assert res.latency_ms.verify_ms >= 0.0


def test_live_verifier_gates_hide_probs(tmp_path: Path) -> None:
    app_cfg, search_cfg = _make_configs(tmp_path)
    searcher = TavilyWebSearcher(
        search_cfg, api_key="tvly-test", transport=_FakeTransport(), page_fetcher=lambda _u: ""
    )

    # 1. Blocked by relevance gate (r < tau_rel) -> short-circuits ensemble, show_probs=False
    ens_low_rel = _FakeEnsemble(probs=[0.90, 0.05, 0.05])
    lv_rel = LiveVerifier(
        app_cfg,
        search_cfg,
        device="cpu",
        searcher=searcher,
        reranker=_FakeReranker(score=0.35),
        ensemble=ens_low_rel,
    )
    res_rel = lv_rel.verify("Bộ Giáo dục công bố lịch thi THPT 2026.")
    assert res_rel.verdict == "unverifiable"
    assert res_rel.show_probs is False
    assert res_rel.reason == REASON_LOW_RELEVANCE
    assert ens_low_rel.calls == 0

    # 2. Passed relevance, blocked by confidence gate (max prob < tau_conf) -> show_probs=False
    ens_low_conf = _FakeEnsemble(probs=[0.42, 0.43, 0.15])
    lv_conf = LiveVerifier(
        app_cfg,
        search_cfg,
        device="cpu",
        searcher=searcher,
        reranker=_FakeReranker(score=0.90),
        ensemble=ens_low_conf,
    )
    res_conf = lv_conf.verify("Bộ Giáo dục công bố lịch thi THPT 2026.")
    assert res_conf.verdict == "unverifiable"
    assert res_conf.show_probs is False
    assert res_conf.reason == REASON_LOW_CONFIDENCE
    assert ens_low_conf.calls == 1


def test_live_verifier_demo_mode_prefers_cache_and_fallback_chain(tmp_path: Path) -> None:
    app_cfg, search_cfg = _make_configs(tmp_path)
    stmt = "Bộ Giáo dục và Đào tạo công bố lịch thi tốt nghiệp THPT năm 2026 ngày 25-6."

    # Warm disk cache via one live call
    warm_searcher = TavilyWebSearcher(
        search_cfg, api_key="tvly-test", transport=_FakeTransport(), page_fetcher=lambda _u: ""
    )
    warm_res = LiveVerifier(
        app_cfg,
        search_cfg,
        device="cpu",
        searcher=warm_searcher,
        reranker=_FakeReranker(0.89),
        ensemble=_FakeEnsemble([0.05, 0.87, 0.08]),
    ).verify(stmt)
    assert warm_res.mode == "live"
    assert warm_res.verdict == "fake"

    # DEMO_MODE + failing network transport MUST serve from cache with 0 network calls
    failing_transport = _FakeTransport(should_fail=True)
    cached_searcher = TavilyWebSearcher(
        search_cfg, api_key="tvly-test", transport=failing_transport, page_fetcher=lambda _u: ""
    )
    demo_res = LiveVerifier(
        app_cfg,
        search_cfg,
        device="cpu",
        demo_mode=True,
        searcher=cached_searcher,
        reranker=_FakeReranker(0.89),
        ensemble=_FakeEnsemble([0.05, 0.87, 0.08]),
    ).verify(stmt)
    assert demo_res.mode == "cache"
    assert demo_res.verdict == "fake"
    assert demo_res.show_probs is True
    assert failing_transport.calls == 0

    # Cache miss + Tavily fails -> falls back to Local KB
    uncached_stmt = "Thủ tướng Phạm Minh Chính dự hội nghị chuyển đổi số quốc gia năm 2026."
    kb_searcher = TavilyWebSearcher(
        search_cfg, api_key="tvly-test", transport=_FakeTransport(should_fail=True), page_fetcher=lambda _u: ""
    )
    kb_lv = LiveVerifier(
        app_cfg,
        search_cfg,
        device="cpu",
        searcher=kb_searcher,
        reranker=_FakeReranker(0.86),
        ensemble=_FakeEnsemble([0.81, 0.11, 0.08]),
        kb_fallback_fn=lambda _c: [
            EvidenceItem(
                text="Thủ tướng Phạm Minh Chính chủ trì hội nghị chuyển đổi số quốc gia tại Hà Nội.",
                url="https://baochinhphu.vn/chuyen-doi-so-2026.htm",
                domain="baochinhphu.vn",
            )
        ],
    )
    kb_res = kb_lv.verify(uncached_stmt)
    assert kb_res.mode == "fallback"
    assert kb_res.verdict == "real"
    assert kb_res.evidence and kb_res.evidence[0].domain == "baochinhphu.vn"

    # Tavily fails + cache miss + KB empty -> graceful unverifiable
    empty_kb_lv = LiveVerifier(
        app_cfg,
        search_cfg,
        device="cpu",
        searcher=kb_searcher,
        reranker=_FakeReranker(0.86),
        ensemble=_FakeEnsemble([0.81, 0.11, 0.08]),
        kb_fallback_fn=lambda _c: [],
    )
    fail_res = empty_kb_lv.verify("Một nhận định hoàn toàn không có trong cache hay KB.")
    assert fail_res.mode == "fallback"
    assert fail_res.verdict == "unverifiable"
    assert fail_res.show_probs is False
    assert "không" in fail_res.reason.lower()


def test_live_verifier_deadline_and_optional_rewrite_and_fp16(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app_cfg, search_cfg = _make_configs(tmp_path)
    searcher = TavilyWebSearcher(
        search_cfg, api_key="tvly-test", transport=_FakeTransport(), page_fetcher=lambda _u: ""
    )

    # 1. Deadline exceeded
    lv_timeout = LiveVerifier(
        app_cfg,
        search_cfg,
        device="cpu",
        total_deadline_s=-0.01,
        searcher=searcher,
        reranker=_FakeReranker(0.9),
        ensemble=_FakeEnsemble([0.9, 0.05, 0.05]),
    )
    res_to = lv_timeout.verify("Bộ Giáo dục công bố lịch thi THPT năm 2026.")
    assert res_to.verdict == "unverifiable"
    assert res_to.show_probs is False
    assert res_to.mode == "fallback"
    assert "deadline" in res_to.reason.lower() or "thời gian" in res_to.reason.lower()

    # 2. Optional QueryRewriter wiring + use_fp16 propagation
    class _MockRewriter:
        def rewrite(self, claim: str) -> RewriteResult:
            return RewriteResult(
                text="lịch thi THPT 2026 Bộ Giáo dục",
                status="ok",
                latency_ms=5.0,
                raw_output="lịch thi THPT 2026 Bộ Giáo dục",
            )

    captured: dict[str, Any] = {}

    class _SpyReranker:
        def __init__(self, model_name: str, **kwargs: Any) -> None:
            captured["reranker_fp16"] = kwargs.get("use_fp16")

        def score_pair(self, query: str, text: str) -> float:
            return 0.90

    def _spy_load_ens(cfg: Any, device: str = "cuda", **kwargs: Any) -> _FakeEnsemble:
        captured["ens_fp16"] = kwargs.get("use_fp16")
        return _FakeEnsemble([0.85, 0.10, 0.05])

    monkeypatch.setattr("vnfake.pipeline.live_verifier.BgeReranker", _SpyReranker)
    monkeypatch.setattr("vnfake.pipeline.live_verifier.load_vc_ensemble", _spy_load_ens)

    lv_fp16 = LiveVerifier(
        app_cfg,
        search_cfg,
        device="cuda",
        use_fp16=True,
        searcher=searcher,
        query_rewriter=_MockRewriter(),
    )
    res_fp16 = lv_fp16.verify("Bộ Giáo dục công bố lịch thi THPT 2026.", use_llm_rewrite=True)
    assert res_fp16.query == "lịch thi THPT 2026 Bộ Giáo dục"
    assert captured["reranker_fp16"] is True
    assert captured["ens_fp16"] is True
