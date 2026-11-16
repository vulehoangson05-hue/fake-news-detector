"""Unit tests for app.demo (Brief #5 §6.1, §6.2 & §12.5.3)."""

from __future__ import annotations

import gradio as gr

from app.demo import (
    create_demo,
    render_evidence_markdown,
    render_metrics_markdown,
    render_probabilities,
    render_verdict_badge,
)
from vnfake.pipeline.live_verifier import (
    EvidenceItem,
    LatencyBreakdownMs,
    LiveVerifyResult,
)


def test_render_verdict_badge() -> None:
    real_html = render_verdict_badge("real", "mô hình xác nhận")
    assert "THẬT" in real_html
    assert "REAL" in real_html
    assert "mô hình xác nhận" in real_html

    fake_html = render_verdict_badge("fake", "mô hình xác nhận")
    assert "GIẢ" in fake_html
    assert "FAKE" in fake_html

    unv_html = render_verdict_badge("unverifiable", "không đủ bằng chứng liên quan")
    assert "CHƯA XÁC ĐỊNH" in unv_html
    assert "UNVERIFIABLE" in unv_html
    assert "không đủ bằng chứng liên quan" in unv_html


def test_render_probabilities_mandatory_gating_rule() -> None:
    # 1. When show_probs is True: must return 3-class probabilities
    probs = {"real": 0.854, "fake": 0.086, "unverifiable": 0.060}
    label_dict, warning = render_probabilities(probs, show_probs=True, reason="mô hình xác nhận")
    assert warning is None
    assert label_dict is not None
    assert label_dict["Thật (Real)"] == 0.854
    assert label_dict["Giả (Fake)"] == 0.086
    assert label_dict["Chưa xác định (Unverifiable)"] == 0.060

    # 2. When show_probs is False: MUST hide probabilities (Brief #5 §12.5.3)
    hidden_dict, hidden_warn = render_probabilities(
        probs, show_probs=False, reason="không đủ bằng chứng liên quan"
    )
    assert hidden_dict is None
    assert hidden_warn is not None
    assert "Đã ẩn phân bố xác suất" in hidden_warn
    assert "không đủ bằng chứng liên quan" in hidden_warn
    assert "§12.5.3" in hidden_warn

    # 3. None probs also handled safely
    none_dict, none_warn = render_probabilities(None, show_probs=False, reason="claim rỗng")
    assert none_dict is None
    assert none_warn is not None


def test_render_evidence_markdown() -> None:
    # Empty evidence
    empty_md = render_evidence_markdown([])
    assert "Không tìm thấy" in empty_md

    # Populated evidence with clickable links
    items = [
        EvidenceItem(
            text="Thông báo số 95 của Văn phòng Chính phủ.",
            url="https://vanban.chinhphu.vn/doc/123",
            domain="vanban.chinhphu.vn",
        ),
        EvidenceItem(
            text="Thủ tướng làm việc tại Hòa Bình.",
            url="https://vnexpress.net/thoi-su/hoa-binh.html",
            domain="vnexpress.net",
        ),
    ]
    md = render_evidence_markdown(items)
    assert "Thông báo số 95" in md
    assert "[vanban.chinhphu.vn](https://vanban.chinhphu.vn/doc/123)" in md
    assert "[vnexpress.net](https://vnexpress.net/thoi-su/hoa-binh.html)" in md


def test_render_metrics_markdown() -> None:
    res = LiveVerifyResult(
        claim="Man Utd chơi không tốt trong trận tứ kết FA Cup với Fulham.",
        query="Man Utd chơi không tốt trận tứ kết FA Cup Fulham",
        verdict="real",
        probs={"real": 0.82, "fake": 0.12, "unverifiable": 0.06},
        show_probs=True,
        relevance=0.9947,
        confidence=0.8186,
        evidence=[],
        latency_ms=LatencyBreakdownMs(
            query_ms=1.5,
            search_ms=10.2,
            fetch_ms=0.0,
            select_ms=25.4,
            rerank_ms=85.1,
            verify_ms=45.3,
            total_ms=167.5,
        ),
        mode="cache",
        reason="mô hình xác nhận",
    )
    md = render_metrics_markdown(res, tau_rel=0.7559, tau_conf=0.55)
    assert "0.9947" in md
    assert "0.8186" in md
    assert "CACHE" in md
    assert "167.5 ms" in md
    assert "Q1v2" in md


def test_create_demo_blocks_construction() -> None:
    demo = create_demo()
    assert isinstance(demo, gr.Blocks)


def test_run_demo_verification_workflow_and_error_handling() -> None:
    from types import SimpleNamespace

    from app.demo import run_demo_verification

    fake_cfg = SimpleNamespace(decision=SimpleNamespace(tau_rel=0.7559, tau_conf=0.55))

    class _DummyVerifier:
        def __init__(self, *, show_probs: bool = True, raise_err: bool = False) -> None:
            self.cfg = fake_cfg
            self.demo_mode = False
            self._show = show_probs
            self._raise = raise_err

        def verify(self, claim: str) -> LiveVerifyResult:
            if self._raise:
                raise RuntimeError("Simulated internal failure")
            return LiveVerifyResult(
                claim=claim,
                query="test query",
                verdict="real" if self._show else "unverifiable",
                probs={"real": 0.85, "fake": 0.10, "unverifiable": 0.05},
                show_probs=self._show,
                relevance=0.95 if self._show else 0.12,
                confidence=0.85 if self._show else 0.0,
                evidence=[
                    EvidenceItem(
                        text="Câu bằng chứng mẫu.",
                        url="https://vnexpress.net/a.html",
                        domain="vnexpress.net",
                    )
                ],
                latency_ms=LatencyBreakdownMs(total_ms=120.0),
                mode="cache",
                reason="mô hình xác nhận" if self._show else "không đủ bằng chứng liên quan",
            )

    # 1. Empty claim
    b_empty, p_empty, w_empty, _, _ = run_demo_verification(
        _DummyVerifier(), "   ", True  # type: ignore[arg-type]
    )
    assert "Vui lòng nhập" in b_empty
    assert p_empty["visible"] is False
    assert w_empty["visible"] is False

    # 2. Passed gates (show_probs=True)
    v_ok = _DummyVerifier(show_probs=True)
    b_ok, p_ok, w_ok, ev_ok, met_ok = run_demo_verification(
        v_ok, "Nhận định hợp lệ", True  # type: ignore[arg-type]
    )
    assert v_ok.demo_mode is True
    assert "THẬT" in b_ok
    assert p_ok["visible"] is True
    assert p_ok["value"]["Thật (Real)"] == 0.85
    assert w_ok["visible"] is False
    assert "vnexpress.net" in ev_ok
    assert "120.0 ms" in met_ok

    # 3. Blocked by gate (show_probs=False) -> probabilities hidden, warning shown
    v_blocked = _DummyVerifier(show_probs=False)
    b_blk, p_blk, w_blk, _, _ = run_demo_verification(
        v_blocked, "Nhận định bị chặn", False  # type: ignore[arg-type]
    )
    assert "CHƯA XÁC ĐỊNH" in b_blk
    assert p_blk["visible"] is False
    assert p_blk["value"] is None
    assert w_blk["visible"] is True
    assert "Đã ẩn phân bố xác suất" in w_blk["value"]

    # 4. Graceful exception handling (no traceback crash)
    v_err = _DummyVerifier(raise_err=True)
    b_err, p_err, _, _, _ = run_demo_verification(
        v_err, "Nhận định lỗi", True  # type: ignore[arg-type]
    )
    assert "CHƯA XÁC ĐỊNH" in b_err
    assert "sự cố xử lý" in b_err
    assert p_err["visible"] is False

