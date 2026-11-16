"""Task 4.4 (Brief #5 §6.1, §6.2, §12.5.3) — Vietnamese Fake News Detector Demo (Gradio).

Provides an interactive Vietnamese web interface for the Phase 4 live verification pipeline:
- Statement input, DEMO_MODE toggle (prioritizes disk cache, 0 credits), progress indicators.
- Prominent Vietnamese verdict: THẬT (real) / GIẢ (fake) / CHƯA XÁC ĐỊNH (unverifiable).
- Mandatory gating rule (§12.5.3):
    * If show_probs == True: displays calibrated class probabilities.
    * If show_probs == False: HIDES model probabilities with an explicit alert.
- Relevance score against tau_rel, confidence against tau_conf, mode, and step-by-step latency.
- Sourced evidence items with clickable news links ([domain](url)).
- Mandatory disclaimer (§6.2) on model confidence vs absolute truth.
"""

from __future__ import annotations

import argparse
import html
import sys
from typing import Any, cast

import gradio as gr

from vnfake.config import load_config, load_search_config
from vnfake.pipeline.live_verifier import EvidenceItem, LiveVerifier, LiveVerifyResult
from vnfake.utils.logging import get_logger

logger = get_logger(__name__)

# 6 diverse cached examples for zero-credit instantaneous response in DEMO_MODE
DEFAULT_EXAMPLES: list[list[Any]] = [
    [
        "Tại buổi làm việc với lãnh đạo tỉnh Hòa Bình, văn phòng Chính phủ vừa ban hành "
        "Thông báo số 95 ngày 25/3/2023 kết luận của Thủ tướng Chính phủ Phạm Minh Chính.",
        True,
    ],
    [
        "Man Utd chơi không tốt trong trận tứ kết FA Cup với Fulham.",
        True,
    ],
    [
        "Theo Bộ trưởng Bộ Thể thao Nga Oleg Matytsin, các điều kiện do Ủy ban Olympic quốc tế (IOC) "
        "đưa ra đối với các vận động viên Nga và Belarus là phù hợp và cho rằng đó là việc phân biệt đối xử đối với các vận động viên.",
        True,
    ],
    [
        "13 người trên ôtô khách và tài xế xe tải bị thương, được người dân phá cửa kính đưa ra ngoài.",
        True,
    ],
    [
        "Trong 56 cháu có biểu hiện đau bụng, buồn nôn có 41 cháu đang nhập viện khám sàng lọc, "
        "số còn lại đã được phụ huynh đón về tiếp tục theo dõi sức khỏe tại nhà với sự theo dõi "
        "của bộ y tế vì bệnh viện thiếu giường.",
        True,
    ],
    [
        "ChatGPT giúp người dùng có thể trao đổi trực tiếp và gián tiếp thông qua hình thức hỏi đáp "
        "là chính với những có sở dữ liệu có sẵn được thu nhập kể từ năm 2021 trở về trước.",
        True,
    ],
]

VERDICT_MAP: dict[str, tuple[str, str, str]] = {
    "real": ("THẬT", "🟢", "#2e7d32"),
    "fake": ("GIẢ", "🔴", "#c62828"),
    "unverifiable": ("CHƯA XÁC ĐỊNH", "🟡", "#ef6c00"),
}


def render_verdict_badge(verdict: str, reason: str) -> str:
    """Format verdict and reason into styled HTML/Markdown."""
    label_vi, emoji, color = VERDICT_MAP.get(
        verdict.lower(), ("CHƯA XÁC ĐỊNH", "🟡", "#ef6c00")
    )
    safe_reason = html.escape(reason)
    return (
        f"<div style='border: 2px solid {color}; border-radius: 8px; padding: 14px 18px; "
        f"background-color: {color}15; margin-bottom: 10px;'>"
        f"<div style='font-size: 1.35em; font-weight: 700; color: {color}; margin-bottom: 4px;'>"
        f"{emoji} KẾT LUẬN: {label_vi} ({verdict.upper()})</div>"
        f"<div style='font-size: 0.95em; color: #424242;'>"
        f"<strong>Lý do:</strong> {safe_reason}</div>"
        f"</div>"
    )


def render_probabilities(
    probs: dict[str, float] | None,
    show_probs: bool,
    reason: str,
) -> tuple[dict[str, float] | None, str | None]:
    """Format model probabilities obeying Brief #5 §12.5.3 gating rule.

    Returns:
        (probs_dict, None) if show_probs is True.
        (None, warning_message) if show_probs is False (probabilities strictly hidden).
    """
    if not show_probs or probs is None:
        warning_msg = (
            f"⚠️ **Đã ẩn phân bố xác suất bộ đọc do bằng chứng chưa vượt qua cổng kiểm định** "
            f"*(Lý do: {reason})*.\n\n"
            f"> Theo quy định an toàn hệ thống (§12.5.3), mô hình chỉ hiển thị độ tin cậy "
            f"khi bằng chứng thu thập được đáp ứng đồng thời ngưỡng liên quan (τ_rel) "
            f"và ngưỡng tin cậy (τ_conf)."
        )
        return None, warning_msg

    p_real = float(probs.get("real", 0.0))
    p_fake = float(probs.get("fake", 0.0))
    p_unv = float(probs.get("unverifiable", 0.0))

    label_probs = {
        "Thật (Real)": round(p_real, 4),
        "Giả (Fake)": round(p_fake, 4),
        "Chưa xác định (Unverifiable)": round(p_unv, 4),
    }
    return label_probs, None


def render_evidence_markdown(evidence: list[EvidenceItem]) -> str:
    """Format selected evidence sentences with clickable source links."""
    if not evidence:
        return "*Không tìm thấy câu bằng chứng phù hợp từ kết quả tra cứu báo chí.*"

    lines: list[str] = ["### 📰 Bằng chứng đã chọn lọc (BM25 top-3 sentences):"]
    for idx, item in enumerate(evidence, start=1):
        safe_text = html.escape(item.text).strip()
        domain = item.domain or "nguồn báo"
        if item.url and item.url.startswith("http"):
            src_link = f"[{domain}]({item.url})"
        else:
            src_link = f"`{domain}`"
        lines.append(f"{idx}. “*{safe_text}*” — **Nguồn:** {src_link}")
    return "\n\n".join(lines)


def render_metrics_markdown(
    res: LiveVerifyResult,
    tau_rel: float,
    tau_conf: float,
) -> str:
    """Format relevance, confidence, mode, and step-by-step latency breakdown."""
    lat = res.latency_ms
    rel_status = "ĐẠT" if res.relevance >= tau_rel else "DƯỚI NGƯỠNG"
    conf_status = "ĐẠT" if res.confidence >= tau_conf else "DƯỚI NGƯỠNG"

    return (
        f"### ⚙️ Thông số vận hành & Độ trễ:\n\n"
        f"- **Điểm liên quan (BGE Reranker):** `{res.relevance:.4f}` / `τ_rel = {tau_rel:.4f}` ({rel_status})\n"
        f"- **Độ tin cậy mô hình:** `{res.confidence:.4f}` / `τ_conf = {tau_conf:.4f}` ({conf_status})\n"
        f"- **Chế độ thực thi:** `{res.mode.upper()}` (Tavily `{res.mode}`)\n"
        f"- **Truy vấn từ khóa sinh ra (Q1v2):** `{res.query}`\n"
        f"- **Tổng độ trễ:** **`{lat.total_ms:.1f} ms`**\n\n"
        f"| Giai đoạn | Thời gian | Mô tả |\n"
        f"|---|---:|---|\n"
        f"| 1. Query Builder (Q1v2) | `{lat.query_ms:.1f} ms` | Tách từ & lọc stopword bảo toàn phủ định |\n"
        f"| 2. Web Search & API | `{lat.search_ms:.1f} ms` | Tra cứu Tavily hoặc đọc cache đĩa |\n"
        f"| 3. Page Fetch (Parallel) | `{lat.fetch_ms:.1f} ms` | Tải trang song song (tối đa 3 trang) |\n"
        f"| 4. Evidence Selector | `{lat.select_ms:.1f} ms` | BM25 sentence selection (top-3, <=150 từ) |\n"
        f"| 5. Relevance Reranker | `{lat.rerank_ms:.1f} ms` | Cross-encoder BGE-Reranker-v2-m3 |\n"
        f"| 6. Verifier Ensemble | `{lat.verify_ms:.1f} ms` | PhoBERT 3-seed calibrated ensemble |\n"
    )


def run_demo_verification(
    verifier: LiveVerifier,
    claim_text: str,
    use_demo_mode: bool,
    progress: Any = None,
) -> tuple[str, dict[str, Any], dict[str, Any], str, str]:
    """Execute verification and return Gradio component updates (testable without UI server)."""
    text = (claim_text or "").strip()
    if not text:
        empty_badge = render_verdict_badge(
            "unverifiable", "Vui lòng nhập nội dung nhận định cần kiểm chứng."
        )
        return (
            empty_badge,
            gr.update(value=None, visible=False),
            gr.update(value="", visible=False),
            "*Chưa có nội dung để tìm kiếm.*",
            "*Chưa có thông số thực thi.*",
        )

    verifier.demo_mode = use_demo_mode

    if callable(progress):
        progress(0.1, desc="Chuẩn hóa nhận định & sinh từ khóa (Q1v2)...")
    try:
        if callable(progress):
            progress(0.3, desc="Tìm kiếm trên báo chí chính thống & kiểm tra cache...")
            progress(0.6, desc="Chọn lọc câu bằng chứng liên quan nhất (BM25)...")
            progress(0.8, desc="Chấm điểm liên quan (BGE Reranker)...")
        res = verifier.verify(text)
        if callable(progress):
            progress(0.95, desc="Kiểm định cổng quyết định đa ngưỡng...")
    except Exception as exc:
        logger.exception("Unexpected error in live verification UI: %s", exc)
        err_badge = render_verdict_badge(
            "unverifiable", f"Hệ thống gặp sự cố xử lý: {html.escape(str(exc))}"
        )
        return (
            err_badge,
            gr.update(value=None, visible=False),
            gr.update(value="", visible=False),
            "*Không thể thu thập bằng chứng do lỗi hệ thống.*",
            "*Không có thông số đo lường.*",
        )

    if callable(progress):
        progress(1.0, desc="Hoàn tất!")
    v_badge = render_verdict_badge(res.verdict, res.reason)
    probs_dict, warn_text = render_probabilities(res.probs, res.show_probs, res.reason)
    evidence_md = render_evidence_markdown(res.evidence)
    metrics_md = render_metrics_markdown(
        res, verifier.cfg.decision.tau_rel, verifier.cfg.decision.tau_conf
    )

    if res.show_probs and probs_dict is not None:
        probs_update = gr.update(value=probs_dict, visible=True)
        warn_update = gr.update(value="", visible=False)
    else:
        probs_update = gr.update(value=None, visible=False)
        warn_update = gr.update(value=warn_text or "", visible=True)

    return v_badge, probs_update, warn_update, evidence_md, metrics_md


_LOW_VRAM_THRESHOLD_GB = 5.0


def _resolve_default_fp16(
    config_fp16: bool,
    *,
    force_fp16: bool = False,
    device: str | None = None,
) -> bool:
    """Default to fp16=True when running on a CUDA GPU with <= 5 GB VRAM (e.g. RTX 3050 4GB)."""
    if force_fp16 or config_fp16:
        return True
    try:
        import torch

        if (device is None or device.startswith("cuda")) and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total_vram_gb = float(props.total_memory) / float(1024**3)
            if total_vram_gb <= _LOW_VRAM_THRESHOLD_GB:
                return True
    except Exception:
        pass
    return False


def _warmup_verifier(verifier: LiveVerifier) -> None:
    """Eagerly warm up word segmenter, BGE reranker, and 3-seed PhoBERT ensemble before UI launch."""
    seg = getattr(verifier, "_segmenter", None)
    if seg is not None:
        seg_fn = getattr(seg, "segment", None)
        if callable(seg_fn):
            seg_fn("Khởi động bộ tách từ tiếng Việt.")
    warmup_fn = getattr(verifier, "warmup", None)
    if callable(warmup_fn):
        warmup_fn()


def create_demo(verifier: LiveVerifier | None = None) -> gr.Blocks:
    """Build the Gradio demo application Blocks interface."""
    active_verifier: LiveVerifier | None = verifier

    def get_verifier() -> LiveVerifier:
        nonlocal active_verifier
        if active_verifier is None:
            cfg = load_config()
            search_cfg = load_search_config()
            use_fp16 = _resolve_default_fp16(cfg.decision.use_fp16)
            active_verifier = LiveVerifier(cfg, search_cfg, use_fp16=use_fp16)
            _warmup_verifier(active_verifier)
        return active_verifier

    with gr.Blocks(
        title="VNFake — Hệ thống Kiểm chứng Tin tức Tiếng Việt",
    ) as demo:
        gr.Markdown(
            "# 🔍 VNFake — Hệ thống Kiểm chứng Tin tức & Nhận định Tiếng Việt\n"
            "Mô hình PhoBERT Classifier kết hợp Tìm kiếm Báo chí Chính thống (Tavily) "
            "& Cổng Quyết định Đa ngưỡng (**Brief #5 Phase 4**)."
        )

        gr.Markdown(
            "> ⚠️ **Lưu ý (§6.2):** *Xác suất hiển thị là độ tin cậy của mô hình dựa trên "
            "bằng chứng tìm được từ báo chí chính thống, không phải xác suất sự thật tuyệt đối.*"
        )

        with gr.Row():
            with gr.Column(scale=5):
                claim_input = gr.Textbox(
                    label="Nhận định (Claim) cần kiểm chứng",
                    placeholder="Nhập nhận định thời sự cần kiểm tra (tiếng Việt)...",
                    lines=3,
                    max_lines=6,
                )
                with gr.Row():
                    demo_mode_cb = gr.Checkbox(
                        label="DEMO_MODE (ưu tiên cache trên đĩa, phản hồi tức thì < 1s, 0 credit)",
                        value=True,
                    )
                with gr.Row():
                    verify_btn = gr.Button("🔍 Kiểm chứng", variant="primary", scale=2)
                    clear_btn = gr.Button("🗑️ Xóa", scale=1)

            with gr.Column(scale=5):
                verdict_output = gr.Markdown(
                    value="*Kết quả kiểm chứng sẽ hiển thị tại đây sau khi bấm 'Kiểm chứng'.*",
                    label="Phán quyết",
                )
                probs_output = gr.Label(
                    label="Độ tin cậy mô hình 3 lớp (Đã hiệu chuẩn nhiệt độ)",
                    visible=False,
                )
                warning_output = gr.Markdown(
                    visible=False,
                )

        with gr.Row():
            with gr.Column(scale=5):
                evidence_output = gr.Markdown(
                    label="Bằng chứng từ báo chí",
                    value="*Chưa có dữ liệu bằng chứng.*",
                )
            with gr.Column(scale=5):
                metrics_output = gr.Markdown(
                    label="Thông số kỹ thuật & Độ trễ",
                    value="*Chưa có thông số độ trễ.*",
                )

        gr.Markdown("### 💡 Nhận định mẫu (Bấm để thử nghiệm tức thì ở DEMO_MODE):")
        gr.Examples(
            examples=DEFAULT_EXAMPLES,
            inputs=[claim_input, demo_mode_cb],
            label="Mẫu nhận định phân tầng đã có sẵn trong cache",
        )

        def on_verify(
            claim_text: str,
            use_demo_mode: bool,
            progress: gr.Progress = gr.Progress(),  # noqa: B008
        ) -> tuple[str, Any, Any, str, str]:
            return run_demo_verification(
                get_verifier(), claim_text, use_demo_mode, progress=progress
            )

        def on_clear() -> tuple[str, str, Any, Any, str, str]:
            return (
                "",
                "*Kết quả kiểm chứng sẽ hiển thị tại đây sau khi bấm 'Kiểm chứng'.*",
                gr.update(value=None, visible=False),
                gr.update(value="", visible=False),
                "*Chưa có dữ liệu bằng chứng.*",
                "*Chưa có thông số độ trễ.*",
            )

        verify_btn.click(
            fn=on_verify,
            inputs=[claim_input, demo_mode_cb],
            outputs=[
                verdict_output,
                probs_output,
                warning_output,
                evidence_output,
                metrics_output,
            ],
        )
        clear_btn.click(
            fn=on_clear,
            inputs=[],
            outputs=[
                claim_input,
                verdict_output,
                probs_output,
                warning_output,
                evidence_output,
                metrics_output,
            ],
        )

    return cast(gr.Blocks, demo)


def main() -> int:
    parser = argparse.ArgumentParser(description="VNFake Live Verification Gradio Demo")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to listen on")
    parser.add_argument("--share", action="store_true", help="Create public gradio.live share link")
    parser.add_argument("--fp16", action="store_true", help="Force fp16 mode for lower VRAM (laptop 4GB)")
    parser.add_argument("--device", type=str, default=None, help="Inference device (cuda / cpu)")
    args = parser.parse_args()

    cfg = load_config()
    search_cfg = load_search_config()
    use_fp16 = _resolve_default_fp16(
        cfg.decision.use_fp16,
        force_fp16=bool(args.fp16),
        device=args.device,
    )
    verifier = LiveVerifier(
        cfg,
        search_cfg,
        device=args.device,
        use_fp16=use_fp16,
    )
    logger.info("Warming up models (underthesea, BGE reranker, 3-seed PhoBERT ensemble)...")
    _warmup_verifier(verifier)
    demo = create_demo(verifier)
    logger.info("Launching Gradio demo on http://%s:%d (fp16=%s)", args.host, args.port, use_fp16)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    sys.exit(main())
