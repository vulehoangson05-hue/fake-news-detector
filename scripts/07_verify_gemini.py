"""Phase 2.5 Task 2 — CLI wrapper for verifying Gemini Flash-Lite availability."""

from __future__ import annotations

import argparse
from pathlib import Path

from vnfake.evaluation.gemini_verify import (
    DEFAULT_GEMINI_MODEL,
    GeminiVerifyReport,
    GoogleGenAIAdapter,
    ProbeResult,
    format_gemini_verify_markdown,
    load_gemini_api_key,
    run_gemini_verification,
    validate_model_name,
)
from vnfake.utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_REPORT_OUT = Path("docs/REPORT_gemini_verify.md")


class _MissingKeyAdapter:
    """Fallback adapter when ``GEMINI_API_KEY`` is missing in the environment."""

    def get_model_info(self, model_name: str) -> str:
        raise RuntimeError("GEMINI_API_KEY is not set in environment or .env")

    def generate_keywords(
        self,
        model_name: str,
        prompt: str,
        *,
        temperature: float,
        max_output_tokens: int,
        timeout_s: float = 10.0,
    ) -> str:
        raise RuntimeError("GEMINI_API_KEY is not set in environment or .env")


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for Phase 2.5 Task 2 Gemini verification."""
    parser = argparse.ArgumentParser(
        description="Phase 2.5 Task 2 — Verify gemini-3.1-flash-lite API availability"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_GEMINI_MODEL,
        help=f"Gemini model ID to verify (default: {DEFAULT_GEMINI_MODEL})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_REPORT_OUT,
        help="Path to output Markdown report (default: docs/REPORT_gemini_verify.md)",
    )
    args = parser.parse_args(argv)

    validate_model_name(args.model)
    api_key = load_gemini_api_key()
    if api_key is None:
        logger.error("GEMINI_API_KEY is missing from environment / .env.")
        adapter = _MissingKeyAdapter()
        report: GeminiVerifyReport = run_gemini_verification(
            adapter,
            model_name=args.model,
            api_key_for_redaction=None,
        )
    else:
        try:
            real_adapter = GoogleGenAIAdapter(api_key=api_key)
            report = run_gemini_verification(
                real_adapter,
                model_name=args.model,
                api_key_for_redaction=api_key,
            )
        except Exception as exc:
            err_msg = f"SDK initialization error: {exc}"
            logger.error(err_msg)
            report = GeminiVerifyReport(
                timestamp_utc="",
                model_name=args.model,
                model_get_ok=False,
                model_get_status=err_msg,
                probes=[
                    ProbeResult(
                        case_id="init",
                        claim="",
                        is_injection_probe=False,
                        latency_ms=0.0,
                        raw_output="",
                        is_compliant=False,
                        compliance_note=err_msg,
                        error=err_msg,
                    )
                ],
                free_tier_quota_note="cần PM đọc thủ công",
                overall_available=False,
            )

    md = format_gemini_verify_markdown(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    logger.info(
        "Gemini verification finished | model=%s | model_get_ok=%s | available=%s | report=%s",
        report.model_name,
        report.model_get_ok,
        report.overall_available,
        args.out,
    )
    return 0 if report.overall_available else 1


if __name__ == "__main__":
    raise SystemExit(main())
