"""Unit tests for Phase 2.5 Task 3: LLM Query Rewrite, Sanitizer, Validator, and RetrievalService."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import pytest
from pydantic import ValidationError

from vnfake.config import AppConfig, PathsConfig, QueryRewriteConfig, load_config
from vnfake.retrieval.embedder import SparseVectorData
from vnfake.retrieval.query_rewriter import (
    REQUIRED_PROMPT_RULE,
    GeminiQueryRewriter,
    NoOpQueryRewriter,
    SlidingWindowRateLimiter,
    load_prompt_template,
    sanitize_keywords,
)
from vnfake.retrieval.service import RetrievalService
from vnfake.retrieval.vector_store import ScoredChunk


def _make_cfg(
    tmp_path: Path,
    *,
    enabled: bool = True,
    fusion: Literal["rewritten_only", "original+rewritten"] = "rewritten_only",
    max_retries: int = 2,
    rpm_limit: int = 10,
    required: bool = False,
) -> AppConfig:
    base = load_config()
    qr = base.query_rewrite.model_copy(
        update={
            "enabled": enabled,
            "fusion": fusion,
            "max_retries": max_retries,
            "rpm_limit": rpm_limit,
            "required": required,
            "cache_dir": tmp_path / "cache",
        }
    )
    return base.model_copy(update={"query_rewrite": qr})


class _FakeGeminiClient:
    """In-memory mock of LowLevelGeminiClient for unit testing without network."""

    def __init__(
        self,
        responses: Sequence[str | Exception] = ("giá xăng tăng điều chỉnh giá",),
        *,
        health_error: Exception | None = None,
    ) -> None:
        self._responses = list(responses)
        self._health_error = health_error
        self.check_calls = 0
        self.generate_calls = 0
        self.last_system_instruction: str | None = None
        self.last_user_content: str | None = None

    def check_model(self, model_name: str) -> None:
        self.check_calls += 1
        if self._health_error is not None:
            raise self._health_error

    def generate(
        self,
        *,
        model_name: str,
        system_instruction: str,
        user_content: str,
        temperature: float,
        max_output_tokens: int,
        timeout_s: float,
    ) -> str:
        self.generate_calls += 1
        self.last_system_instruction = system_instruction
        self.last_user_content = user_content
        if not self._responses:
            return "từ khóa mặc định"
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _RecordingEmbedder:
    """Fake embedder that records all texts passed to embed_dense/embed_sparse."""

    dim: int = 4

    def __init__(self) -> None:
        self.dense_inputs: list[str] = []
        self.sparse_inputs: list[str] = []

    def embed_dense(self, texts: Sequence[str], *, batch_size: int = 16) -> list[list[float]]:
        self.dense_inputs.extend(texts)
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    def embed_sparse(self, texts: Sequence[str], *, batch_size: int = 16) -> list[SparseVectorData]:
        self.sparse_inputs.extend(texts)
        return [SparseVectorData(indices=[1, 2], values=[0.5, 0.5]) for _ in texts]


class _FakeQdrantStore:
    """Fake store returning deterministic ScoredChunks per query call."""

    def __init__(self, batches: Sequence[Sequence[ScoredChunk]]) -> None:
        self._batches = [list(b) for b in batches]
        self.call_count = 0

    def search_hybrid(
        self,
        name: str,
        dense_query: Sequence[float],
        sparse_query: SparseVectorData,
        top_k: int,
        *,
        dense_prefetch_k: int,
        sparse_prefetch_k: int,
        with_payload: bool = True,
    ) -> list[ScoredChunk]:
        idx = min(self.call_count, len(self._batches) - 1)
        self.call_count += 1
        return list(self._batches[idx][:top_k])


class _RecordingReranker:
    """Fake reranker that records the exact query string it receives."""

    def __init__(self) -> None:
        self.received_queries: list[str] = []
        self.received_candidate_ids: list[list[str]] = []

    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        self.received_queries.append(query)
        self.received_candidate_ids.append([c.chunk_id for c in candidates])
        return list(candidates[:top_k])


def _make_chunk(cid: str, score: float = 0.5) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=cid,
        doc_id=f"doc_{cid}",
        chunk_idx=0,
        title="Tiêu đề",
        text="Nội dung bài viết",
        text_segmented="Nội_dung bài_viết",
        url=f"https://vnexpress.net/{cid}.html",
        crawl_date="2026-11-08",
        source="vnexpress",
        score=score,
    )


def test_config_defaults_match_base_yaml() -> None:
    """Verify default values in config.py match configs/base.yaml (Brief #2 §6)."""
    yaml_cfg = load_config()
    code_default_cfg = AppConfig(
        paths=PathsConfig(
            kb_dantri=Path("data/raw/kb/dantri.jsonl"),
            kb_vnexpress=Path("data/raw/kb/vnexpress.jsonl"),
            vifactcheck_dir=Path("data/raw/vifactcheck"),
            interim_dir=Path("data/interim"),
            processed_dir=Path("data/processed"),
            artifacts_dir=Path("artifacts"),
        ),
        label_map={0: "real", 1: "fake", 2: "unverifiable"},
    )
    assert yaml_cfg.retrieval.retrieve_k == code_default_cfg.retrieval.retrieve_k == 30
    assert (
        yaml_cfg.retrieval.reranker.batch_size
        == code_default_cfg.retrieval.reranker.batch_size
        == 4
    )
    assert yaml_cfg.retrieval.reranker.dtype == code_default_cfg.retrieval.reranker.dtype == "fp32"
    assert (
        yaml_cfg.retrieval.reranker.use_fp16
        is code_default_cfg.retrieval.reranker.use_fp16
        is False
    )
    assert yaml_cfg.query_rewrite.model_dump() == code_default_cfg.query_rewrite.model_dump()


def test_query_rewrite_model_validator() -> None:
    """Validator must accept Flash family models and reject gemini-1.5-* or non-Flash."""
    cfg_ok = QueryRewriteConfig(model="gemini-3.1-flash-lite")
    assert cfg_ok.model == "gemini-3.1-flash-lite"

    cfg_ok2 = QueryRewriteConfig(model="gemini-2.5-flash")
    assert cfg_ok2.model == "gemini-2.5-flash"

    for invalid_model in (
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-1.5-pro",
        "gemini-2.5-pro",
        "gpt-4o-mini",
        "claude-3-haiku",
    ):
        with pytest.raises(ValidationError):
            QueryRewriteConfig(model=invalid_model)


def test_prompt_template_contains_required_rule_and_no_dev_leakage() -> None:
    """Template must contain REQUIRED_PROMPT_RULE, delimiter, and zero numbers in few-shots (A5)."""
    import re

    prompt_path = Path("configs/prompts/query_rewrite.txt")
    sys_inst, body, prompt_sha = load_prompt_template(prompt_path)
    assert sys_inst == REQUIRED_PROMPT_RULE
    assert len(prompt_sha) == 64
    assert REQUIRED_PROMPT_RULE in body
    assert "<claim_data>" in body and "</claim_data>" in body and "{claim}" in body
    # Ensure few-shot examples before the final <claim_data>{claim}</claim_data> have no multi-digit numbers
    few_shot_section = body.split("Bây giờ hãy trích xuất từ khóa")[0]
    assert re.findall(r"\d{2,}", few_shot_section) == []


@pytest.mark.parametrize(
    ("raw_input", "expected"),
    [
        ("- giá xăng, tăng giá xăng, điều chỉnh giá", "giá xăng tăng giá xăng điều chỉnh giá"),
        ("1. **Ngân hàng Nhà nước** `vàng miếng SJC`", "Ngân hàng Nhà nước vàng miếng SJC"),
        ('"mức phạt vượt đèn đỏ xe máy năm nay"', "mức phạt vượt đèn đỏ xe máy năm nay"),
        ("Từ khóa: học sinh Hà Nội nghỉ học, siêu bão", "học sinh Hà Nội nghỉ học siêu bão"),
        ('Keywords: "giá điện sinh hoạt tăng"', "giá điện sinh hoạt tăng"),
        ("giá xăng tăng mạnh\nGhi chú: dòng thứ hai", "giá xăng tăng mạnh"),
    ],
)
def test_sanitize_keywords_valid_cases(raw_input: str, expected: str) -> None:
    forbidden = QueryRewriteConfig().forbidden_phrases
    out = sanitize_keywords(raw_input, max_keywords=15, max_chars=256, forbidden_phrases=forbidden)
    assert out == expected


def test_sanitize_keywords_alien_numbers_filtered_and_fallback(tmp_path: Path) -> None:
    """Verify A5.3: multi-digit numbers not in claim are dropped; empty result triggers fallback."""
    claim_no_year = "vượt đèn đỏ xe máy năm nay bị phạt tới 6 triệu đồng đúng không mọi người"
    # "2024" is an alien 4-digit number not in claim -> token "2024" dropped; "6" is 1 digit -> kept
    cleaned = sanitize_keywords(
        "vượt đèn đỏ xe máy phạt 6 triệu đồng mức phạt vi phạm giao thông 2024",
        source_claim=claim_no_year,
    )
    assert cleaned == "vượt đèn đỏ xe máy phạt 6 triệu đồng mức phạt vi phạm giao thông"

    # Multi-digit number present in claim ("2024", "60") -> kept
    claim_with_num = "nghị định 168 năm 2024 phạt 60 triệu đồng"
    cleaned_kept = sanitize_keywords(
        "nghị định 168 năm 2024 phạt 60 triệu đồng",
        source_claim=claim_with_num,
    )
    assert cleaned_kept == "nghị định 168 năm 2024 phạt 60 triệu đồng"

    # Output consists ONLY of alien multi-digit numbers -> returns None -> fallback:invalid_output
    assert sanitize_keywords("2024 2025", source_claim="giá xăng tăng mạnh") is None

    cfg = _make_cfg(tmp_path, enabled=True)
    client = _FakeGeminiClient(["2024 2025"])
    rewriter = GeminiQueryRewriter(cfg.query_rewrite, client=client, sleep_fn=lambda _: None)
    res = rewriter.rewrite("giá xăng tăng mạnh")
    assert res.status == "fallback:invalid_output"
    assert res.text == "giá xăng tăng mạnh"


@pytest.mark.parametrize(
    "invalid_input",
    [
        "",
        "   \n  ",
        '""',
        "Từ khóa: ",
        "Xin chào bạn, dưới đây là từ khóa: giá xăng tăng",
        "Dạ vâng, giá xăng sắp tăng mạnh",
        "Giải thích: tin đồn này không có căn cứ",
        "từ " * 120,  # exceeds max_chars=256
        "Hà Nội mùa thu\nCây cơm nguội vàng\nCây bàng lá đỏ\nNằm kề bên nhau",  # 4-line poem after injection
        "Đây là bài thơ về mùa thu Hà Nội tặng bạn",
    ],
)
def test_sanitize_keywords_rejects_invalid_outputs(invalid_input: str) -> None:
    forbidden = QueryRewriteConfig().forbidden_phrases
    out = sanitize_keywords(
        invalid_input, max_keywords=15, max_chars=256, forbidden_phrases=forbidden
    )
    assert out is None


def test_sanitize_keywords_truncates_to_max_keywords() -> None:
    out = sanitize_keywords("một hai ba bốn năm sáu bảy", max_keywords=4, max_chars=256)
    assert out == "một hai ba bốn"


def test_rewriter_ok_and_disk_cache_hit_and_resume(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, enabled=True)
    client = _FakeGeminiClient(["giá xăng, tăng giá xăng, bình ổn giá"])
    rewriter = GeminiQueryRewriter(cfg.query_rewrite, client=client, sleep_fn=lambda _: None)

    claim = "mấy bữa nay đồn xăng tăng giá mạnh hả"
    res1 = rewriter.rewrite(claim)
    assert res1.status == "ok"
    assert res1.text == "giá xăng tăng giá xăng bình ổn giá"
    assert client.generate_calls == 1

    # Second call on same instance -> cache_hit without calling API
    res2 = rewriter.rewrite(claim)
    assert res2.status == "cache_hit"
    assert res2.text == "giá xăng tăng giá xăng bình ổn giá"
    assert client.generate_calls == 1

    # New instance pointing to same cache_dir (resume) -> cache_hit without calling check_model or generate
    client2 = _FakeGeminiClient([RuntimeError("Should not be called")])
    rewriter_resumed = GeminiQueryRewriter(
        cfg.query_rewrite, client=client2, sleep_fn=lambda _: None
    )
    res3 = rewriter_resumed.rewrite(claim)
    assert res3.status == "cache_hit"
    assert res3.text == "giá xăng tăng giá xăng bình ổn giá"
    assert client2.generate_calls == 0


@pytest.mark.parametrize(
    ("exc", "expected_status"),
    [
        (TimeoutError("Request timed out after 10s"), "fallback:timeout"),
        (RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded"), "fallback:rate_limit"),
        (
            RuntimeError("503 Service Unavailable ?key=AIzaSySecretKey123456789012345"),
            "fallback:api_error",
        ),
    ],
)
def test_rewriter_fallbacks_do_not_cache(
    tmp_path: Path, exc: Exception, expected_status: str
) -> None:
    cfg = _make_cfg(tmp_path, enabled=True, max_retries=1)
    client = _FakeGeminiClient([exc, exc])
    sleeps: list[float] = []
    rewriter = GeminiQueryRewriter(cfg.query_rewrite, client=client, sleep_fn=sleeps.append)

    claim = "tin đồn thất thiệt về bão"
    res = rewriter.rewrite(claim)
    assert res.status == expected_status
    assert res.text == claim
    assert client.generate_calls == 2
    assert sleeps == [1.0]
    assert list((tmp_path / "cache").glob("*.json")) == []


def test_rewriter_invalid_output_fallback_does_not_cache(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, enabled=True, max_retries=2)
    client = _FakeGeminiClient(["Dạ vâng xin chào bạn, đây là bài thơ mùa thu Hà Nội"])
    rewriter = GeminiQueryRewriter(cfg.query_rewrite, client=client, sleep_fn=lambda _: None)

    claim = "claim thử nghiệm"
    res = rewriter.rewrite(claim)
    assert res.status == "fallback:invalid_output"
    assert res.text == claim
    assert client.generate_calls == 1
    assert list((tmp_path / "cache").glob("*.json")) == []


def test_rewriter_health_check_failure_behavior(tmp_path: Path) -> None:
    # required=False -> graceful fallback
    cfg_opt = _make_cfg(tmp_path, enabled=True, required=False)
    client_bad = _FakeGeminiClient(health_error=RuntimeError("404 Model not found"))
    rewriter_opt = GeminiQueryRewriter(cfg_opt.query_rewrite, client=client_bad)
    res = rewriter_opt.rewrite("claim kiểm tra")
    assert res.status == "fallback:api_error"
    assert res.text == "claim kiểm tra"

    # required=True -> fail fast
    cfg_req = _make_cfg(tmp_path, enabled=True, required=True)
    client_bad2 = _FakeGeminiClient(health_error=RuntimeError("404 Model not found"))
    rewriter_req = GeminiQueryRewriter(cfg_req.query_rewrite, client=client_bad2)
    with pytest.raises(RuntimeError, match="required=True"):
        rewriter_req.rewrite("claim kiểm tra")


def test_sliding_window_rate_limiter() -> None:
    now = [100.0]
    slept: list[float] = []

    def fake_clock() -> float:
        return now[0]

    def fake_sleep(seconds: float) -> None:
        slept.append(seconds)
        now[0] += seconds

    limiter = SlidingWindowRateLimiter(rpm_limit=2, clock_fn=fake_clock, sleep_fn=fake_sleep)
    assert limiter.acquire() == 0.0
    now[0] += 10.0
    assert limiter.acquire() == 0.0
    # 3rd request at t=110.0 should wait 50.0s until t=160.0
    waited = limiter.acquire()
    assert pytest.approx(waited, abs=1e-6) == 50.0
    assert len(slept) == 1


def test_retrieval_service_reranker_always_receives_original_claim(tmp_path: Path) -> None:
    """Verify RetrievalService passes rewritten text ONLY to retrieval and original claim to reranker."""
    claim = "mấy bữa nay đồn xăng tăng giá mạnh hông"
    c1, c2, c3 = _make_chunk("c1", 0.9), _make_chunk("c2", 0.8), _make_chunk("c3", 0.7)

    # 1. Disabled mode (regression test)
    cfg_off = _make_cfg(tmp_path, enabled=False)
    embedder_off = _RecordingEmbedder()
    store_off = _FakeQdrantStore([[c1, c2]])
    reranker_off = _RecordingReranker()
    svc_off = RetrievalService(
        cfg_off,
        store=store_off,  # type: ignore[arg-type]
        embedder=embedder_off,
        reranker=reranker_off,
        collection="test_col",
        rewriter=NoOpQueryRewriter(),
    )
    out_off = svc_off.search(claim)
    assert out_off.rewrite_status == "disabled"
    assert embedder_off.dense_inputs == [claim]
    assert reranker_off.received_queries == [claim]

    # 2. Enabled mode (rewritten_only)
    cfg_on = _make_cfg(tmp_path, enabled=True, fusion="rewritten_only")
    client = _FakeGeminiClient(["giá xăng dầu điều chỉnh tăng"])
    rewriter = GeminiQueryRewriter(cfg_on.query_rewrite, client=client)
    embedder_on = _RecordingEmbedder()
    store_on = _FakeQdrantStore([[c1, c2]])
    reranker_on = _RecordingReranker()
    svc_on = RetrievalService(
        cfg_on,
        store=store_on,  # type: ignore[arg-type]
        embedder=embedder_on,
        reranker=reranker_on,
        collection="test_col",
        rewriter=rewriter,
    )
    out_on = svc_on.search(claim)
    assert out_on.rewrite_status == "ok"
    assert embedder_on.dense_inputs == ["giá xăng dầu điều chỉnh tăng"]
    assert reranker_on.received_queries == [claim]

    # 3. Enabled mode (original+rewritten RRF fusion)
    cfg_fuse = _make_cfg(tmp_path, enabled=True, fusion="original+rewritten")
    embedder_fuse = _RecordingEmbedder()
    store_fuse = _FakeQdrantStore([[c1, c2], [c2, c3]])
    reranker_fuse = _RecordingReranker()
    svc_fuse = RetrievalService(
        cfg_fuse,
        store=store_fuse,  # type: ignore[arg-type]
        embedder=embedder_fuse,
        reranker=reranker_fuse,
        collection="test_col",
        rewriter=rewriter,
    )
    out_fuse = svc_fuse.search(claim)
    assert out_fuse.rewrite_status == "cache_hit"
    assert embedder_fuse.dense_inputs == [claim, "giá xăng dầu điều chỉnh tăng"]
    assert reranker_fuse.received_queries == [claim]
    # c2 appears in both lists so RRF ranks c2 first
    assert reranker_fuse.received_candidate_ids[0][0] == "c2"


def test_k_sweep_overlap_and_guardrail(tmp_path: Path) -> None:
    """Verify k_sweep overlap helpers and hard guardrail against retrieve_k < 20."""
    from vnfake.evaluation.k_sweep import (
        compute_top1_agreement,
        compute_top5_set_overlap,
        run_k_sweep,
    )

    a = [["c1", "c2", "c3", "c4", "c5"], ["c10", "c20", "c30", "c40", "c50"]]
    b = [["c1", "c2", "c3", "c4", "c99"], ["c10", "c20", "c30", "c40", "c50"]]
    assert pytest.approx(compute_top5_set_overlap(a, b)) == 0.9
    assert pytest.approx(compute_top1_agreement(a, b)) == 1.0

    cfg = _make_cfg(tmp_path)
    with pytest.raises(ValueError, match="retrieve_k=10 < 20"):
        run_k_sweep(cfg, k_values=[10, 20])


def test_rewriter_online_no_retry_on_timeout_and_total_deadline(tmp_path: Path) -> None:
    """Verify A19: online mode never retries on timeout, and total_deadline_s caps total wait."""
    cfg = _make_cfg(tmp_path, enabled=True, max_retries=2)
    qr_cfg = cfg.query_rewrite.model_copy(update={"max_retries_online": 0, "total_deadline_s": 5.0})

    # 1. Online mode: even if client has 2 responses, timeout on attempt 1 aborts immediately
    client_online = _FakeGeminiClient([TimeoutError("ReadTimeout"), "giá xăng dầu điều chỉnh tăng"])
    rewriter_online = GeminiQueryRewriter(qr_cfg, client=client_online, online=True)
    res_online = rewriter_online.rewrite("giá xăng tăng")
    assert res_online.status == "fallback:timeout"
    assert client_online.generate_calls == 1

    # 2. Total deadline exceeded during backoff in batch mode -> aborts before second call
    clock_now = [0.0]

    def fake_clock() -> float:
        return clock_now[0]

    def fake_sleep(sec: float) -> None:
        clock_now[0] += sec

    qr_deadline = cfg.query_rewrite.model_copy(update={"max_retries": 2, "total_deadline_s": 0.8})
    client_deadline = _FakeGeminiClient([RuntimeError("503 Service Unavailable"), "giá xăng tăng"])
    rewriter_deadline = GeminiQueryRewriter(
        qr_deadline,
        client=client_deadline,
        clock_fn=fake_clock,
        sleep_fn=fake_sleep,
        online=False,
    )
    # Next backoff is 1.0s >= total_deadline_s (0.8s) -> aborts before second call
    res_deadline = rewriter_deadline.rewrite("giá xăng tăng mạnh")
    assert res_deadline.status == "fallback:timeout"
    assert client_deadline.generate_calls == 1
