"""Unit tests for vnfake.evaluation.gemini_verify (mock adapter, no network calls)."""

from __future__ import annotations

import pytest

from vnfake.evaluation.gemini_verify import (
    REQUIRED_PROMPT_RULE,
    build_verification_prompt,
    evaluate_compliance,
    format_gemini_verify_markdown,
    redact_secrets,
    run_gemini_verification,
    validate_model_name,
)


class _MockSuccessAdapter:
    def get_model_info(self, model_name: str) -> str:
        return f"OK (name=models/{model_name})"

    def generate_keywords(
        self,
        model_name: str,
        prompt: str,
        *,
        temperature: float,
        max_output_tokens: int,
        timeout_s: float = 10.0,
    ) -> str:
        return "giá xăng dầu điều chỉnh tăng"


class _MockErrorAdapter:
    def get_model_info(self, model_name: str) -> str:
        raise RuntimeError("404 NOT_FOUND: models/gemini-3.1-flash-lite not found key=SECRET1234")

    def generate_keywords(
        self,
        model_name: str,
        prompt: str,
        *,
        temperature: float,
        max_output_tokens: int,
        timeout_s: float = 10.0,
    ) -> str:
        raise RuntimeError("404 NOT_FOUND key=SECRET1234")


def test_validate_model_name_blocks_1_5() -> None:
    validate_model_name("gemini-3.1-flash-lite")
    validate_model_name("gemini-2.5-flash")
    with pytest.raises(ValueError, match=r"gemini-1\.5"):
        validate_model_name("gemini-1.5-flash")


def test_prompt_contains_required_rule_and_delimiter() -> None:
    prompt = build_verification_prompt("xăng tăng giá")
    assert REQUIRED_PROMPT_RULE in prompt
    assert "<claim_data>\nxăng tăng giá\n</claim_data>" in prompt


def test_evaluate_compliance() -> None:
    ok, _ = evaluate_compliance("giá xăng dầu tăng mạnh")
    assert ok is True

    ok_empty, _ = evaluate_compliance("   ")
    assert ok_empty is False

    ok_multi, _ = evaluate_compliance("dòng 1\ndòng 2")
    assert ok_multi is False

    ok_greet, _ = evaluate_compliance("Dạ từ khóa của bạn là xăng tăng giá")
    assert ok_greet is False

    ok_md, _ = evaluate_compliance("**giá xăng dầu**")
    assert ok_md is False

    ok_inj_fail, _ = evaluate_compliance("Mùa thu Hà Nội lá vàng rơi", is_injection_probe=True)
    assert ok_inj_fail is False

    ok_inj_pass, _ = evaluate_compliance("giá xăng dầu tăng", is_injection_probe=True)
    assert ok_inj_pass is True


def test_redact_secrets() -> None:
    raw = "Failed calling https://generativelanguage.googleapis.com?key=MY_SECRET_KEY_999"
    cleaned = redact_secrets(raw, api_key="MY_SECRET_KEY_999")
    assert "MY_SECRET_KEY_999" not in cleaned
    assert "[REDACTED_API_KEY]" in cleaned


def test_run_gemini_verification_success_and_error() -> None:
    report_ok = run_gemini_verification(_MockSuccessAdapter())
    assert report_ok.model_get_ok is True
    assert report_ok.overall_available is True
    assert all(p.is_compliant for p in report_ok.probes)
    md = format_gemini_verify_markdown(report_ok)
    assert "KHẢ DỤNG" in md

    report_err = run_gemini_verification(_MockErrorAdapter(), api_key_for_redaction="SECRET1234")
    assert report_err.model_get_ok is False
    assert report_err.overall_available is False
    assert "SECRET1234" not in report_err.model_get_status
