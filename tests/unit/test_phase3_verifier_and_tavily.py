"""Unit tests for Phase 3 Verifier modules and Tavily Web Search spike (Brief #4)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from vnfake.config import SearchConfig
from vnfake.models.calibration import TemperatureScaler
from vnfake.models.classifier import (
    VerifierClassifier,
    build_hf_label_maps,
    collate_pair_batch,
    infer_model_type,
)
from vnfake.models.tokenization import (
    encode_verifier_dataset,
    measure_pair_token_lengths,
)
from vnfake.retrieval.evidence_selector import (
    compute_3gram_containment,
    select_evidence_from_context,
)
from vnfake.retrieval.web_search import (
    TavilyCreditTracker,
    TavilyWebSearcher,
    build_query_q0,
    build_query_q1,
    is_allowed_domain,
    run_tavily_spike,
    sample_stratified_dev_claims,
)
from vnfake.schemas.vifactcheck import ViFactCheckSample
from vnfake.training.metrics import (
    compute_ece,
    evaluate_verifier_logits,
    softmax_with_temperature,
)


class _DummySegmenter:
    def segment(self, text: str) -> str:
        return text.replace("Việt Nam", "Việt_Nam")

    def segment_batch(self, texts: Sequence[str]) -> list[str]:
        return [self.segment(t) for t in texts]


class _DummyTokenizer:
    def __call__(
        self,
        text: str,
        text_pair: str | None = None,
        *,
        max_length: int,
        truncation: bool | str,
        padding: bool | str,
    ) -> Mapping[str, Sequence[int]]:
        toks_a = text.split()
        toks_b = text_pair.split() if text_pair else []
        if truncation == "only_second" and len(toks_a) + len(toks_b) + 3 > max_length:
            keep_b = max(0, max_length - len(toks_a) - 3)
            toks_b = toks_b[:keep_b]
        elif truncation is True and len(toks_a) + len(toks_b) + 2 > max_length:
            toks_a = toks_a[: max(1, max_length - 2)]
        total = len(toks_a) + len(toks_b) + (3 if text_pair is not None else 2)
        return {
            "input_ids": list(range(1, total + 1)),
            "attention_mask": [1] * total,
        }


def test_evidence_selector_bm25_and_containment() -> None:
    context = (
        "Hôm nay thời tiết tại Hà Nội rất đẹp. "
        "Bộ Giáo dục và Đào tạo công bố lịch thi tốt nghiệp THPT năm 2026 vào ngày 25-6. "
        "Nhiều thí sinh đang tích cực ôn tập. "
        "Các trường đại học cũng đã lên phương án tuyển sinh riêng."
    )
    claim = "Bộ Giáo dục và Đào tạo công bố lịch thi tốt nghiệp THPT năm 2026 vào ngày 25-6."
    gold = "Bộ Giáo dục và Đào tạo công bố lịch thi tốt nghiệp THPT năm 2026 vào ngày 25-6."

    selected = select_evidence_from_context(claim, context, top_k=2, max_tokens=50)
    assert selected.method == "bm25"
    assert selected.num_sentences <= 2
    assert selected.word_count <= 50
    assert selected.sentence_indices == sorted(selected.sentence_indices)
    assert compute_3gram_containment(gold, selected.text) >= 0.9


def test_token_length_measurement_and_encoding() -> None:
    tok = _DummyTokenizer()
    seg = _DummySegmenter()
    label_map = {0: "real", 1: "fake", 2: "unverifiable"}
    statements = ["Đội tuyển Việt Nam chiến thắng", "Giá vàng tăng mạnh hôm nay"]
    evidences = ["Đội tuyển Việt Nam thắng đậm 3-0", "Giá vàng SJC tăng 500 nghìn đồng"]
    labels = [0, 1]

    stats = measure_pair_token_lengths(
        statements,
        evidences,
        tokenizer=tok,
        model_type="phobert",
        segmenter=seg,
    )
    assert stats.count == 2
    assert stats.p50 > 0
    assert stats.max_len >= stats.p50

    ds_pair = encode_verifier_dataset(
        statements,
        evidences,
        labels,
        tokenizer=tok,
        model_type="phobert",
        max_length=16,
        label_map=label_map,
        segmenter=seg,
        statement_only=False,
    )
    assert len(ds_pair) == 2
    assert int(ds_pair[0]["labels"].item()) == 0

    ds_stmt = encode_verifier_dataset(
        statements,
        evidences,
        labels,
        tokenizer=tok,
        model_type="phobert",
        max_length=16,
        label_map=label_map,
        segmenter=seg,
        statement_only=True,
    )
    assert len(ds_stmt) == 2
    assert ds_stmt[0]["input_ids"].shape[0] < ds_pair[0]["input_ids"].shape[0]


def test_classifier_helpers_and_collator() -> None:
    assert infer_model_type("vinai/phobert-base-v2") == "phobert"
    assert infer_model_type("xlm-roberta-base") == "xlmr"
    with pytest.raises(ValueError, match="Cannot infer model_type"):
        infer_model_type("bert-base-uncased")

    label_map = {0: "real", 1: "fake", 2: "unverifiable"}
    names, id2label, label2id = build_hf_label_maps(label_map)
    assert names == ["real", "fake", "unverifiable"]
    assert id2label[1] == "fake"
    assert label2id["unverifiable"] == 2

    clf = VerifierClassifier("vinai/phobert-base-v2", label_map, device="cpu")
    assert clf.num_labels == 3
    assert clf.class_names == ["real", "fake", "unverifiable"]

    batch = [
        {
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1]),
            "labels": torch.tensor(0),
        },
        {
            "input_ids": torch.tensor([4, 5, 6, 7, 8]),
            "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
            "labels": torch.tensor(2),
        },
    ]
    collated = collate_pair_batch(batch, pad_token_id=1)
    assert collated["input_ids"].shape == (2, 5)
    assert int(collated["input_ids"][0, 4].item()) == 1
    assert int(collated["attention_mask"][0, 4].item()) == 0


def test_metrics_and_temperature_scaling(tmp_path: Path) -> None:
    label_map = {0: "real", 1: "fake", 2: "unverifiable"}
    logits = np.array(
        [
            [5.0, -1.0, -1.0],
            [-1.0, 4.5, -1.0],
            [-1.0, -1.0, 4.0],
            [3.0, 1.0, -1.0],
        ],
        dtype=np.float32,
    )
    labels = [0, 1, 2, 1]  # last sample is overconfidently wrong

    report = evaluate_verifier_logits(logits, labels, label_map)
    assert report.accuracy == 0.75
    assert 0.0 < report.macro_f1 <= 1.0
    assert report.num_samples == 4

    scaler = TemperatureScaler()
    cal = scaler.fit(logits, labels)
    assert cal.temperature > 0.0
    assert cal.nll_after <= cal.nll_before + 1e-5

    probs = scaler.predict_proba(logits)
    assert probs.shape == (4, 3)
    assert np.allclose(np.sum(probs, axis=-1), 1.0)
    assert 0.0 <= compute_ece(probs, labels) <= 1.0

    save_path = tmp_path / "calibration.json"
    scaler.save(save_path, cal)
    loaded = TemperatureScaler.load(save_path)
    assert loaded.temperature == pytest.approx(scaler.temperature)

    with pytest.raises(ValueError, match="temperature must be > 0"):
        softmax_with_temperature(logits, temperature=0.0)


class _MockTavilyTransport:
    def __init__(self, url: str = "https://vnexpress.net/bai-bao-test-123.html") -> None:
        self.url = url
        self.calls = 0

    def search_post(self, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        self.calls += 1
        return {
            "results": [
                {
                    "title": "Bài báo kiểm thử trên VnExpress",
                    "url": self.url,
                    "content": "Đoạn tóm tắt ngắn từ Tavily.",
                    "raw_content": (
                        "Bộ Giáo dục và Đào tạo chính thức công bố lịch thi tốt nghiệp THPT "
                        "năm 2026 vào ngày 25-6 trên toàn quốc. " * 8
                    ),
                    "score": 0.95,
                }
            ]
        }


def test_tavily_queries_domains_credits_and_spike(tmp_path: Path) -> None:
    stmt = "Bộ Giáo dục và Đào tạo đã chính thức công bố lịch thi ngày 25/6/2026 tại Hà Nội."
    q0 = build_query_q0(stmt, max_chars=40)
    assert len(q0) <= 40
    q1 = build_query_q1(stmt)
    assert "25/6/2026" in q1
    assert "Giáo" in q1

    domains = ["vnexpress.net", "tuoitre.vn", "chinhphu.vn"]
    assert is_allowed_domain("https://vnexpress.net/thoi-su/abc.html", domains)
    assert is_allowed_domain("https://sub.chinhphu.vn/tin-tuc", domains)
    assert not is_allowed_domain("https://facebook.com/post/123", domains)

    tracker = TavilyCreditTracker(tmp_path / "cred_cache", max_credits=2)
    tracker.check_can_spend(1)
    assert tracker.record_spend("q1", cost=1) == 1
    assert tracker.record_spend("q2", cost=1) == 2
    with pytest.raises(RuntimeError, match="credit spike ceiling reached"):
        tracker.check_can_spend(1)

    search_cfg = SearchConfig(
        provider="tavily",
        search_depth="basic",
        max_results=5,
        topic="general",
        cache_dir=tmp_path / "tavily_cache",
        max_credits_spike=10,
        timeout_s=5.0,
        rate_limit_s=1.0,
        user_agent="VNFakeTest/1.0",
        include_domains=domains,
    )
    transport = _MockTavilyTransport()
    searcher = TavilyWebSearcher(
        search_cfg,
        api_key="tvly-dummy-key",
        transport=transport,
        page_fetcher=lambda _u: "",
    )

    res1 = searcher.search(stmt, variant="Q0")
    assert not res1.from_cache
    assert res1.credits_used == 1
    assert len(res1.hits) == 1
    assert res1.hits[0].text_source == "tavily_raw_content"

    # Second identical search must hit disk cache without spending credits
    res2 = searcher.search(stmt, variant="Q0")
    assert res2.from_cache
    assert res2.credits_used == 1
    assert transport.calls == 1

    # Domain violation check
    bad_searcher = TavilyWebSearcher(
        search_cfg,
        api_key="tvly-dummy-key",
        transport=_MockTavilyTransport(url="https://untrusted-site.org/news"),
        page_fetcher=lambda _u: "",
    )
    with pytest.raises(ValueError, match="violates include_domains whitelist"):
        bad_searcher.search("Truy vấn bất kỳ khác", variant="Q0")

    # Stratified sampling & spike runner
    samples = [
        ViFactCheckSample(
            Statement=f"Bộ Giáo dục và Đào tạo công bố lịch thi tốt nghiệp THPT năm 2026 vào ngày 25-6 (mẫu {i}).",
            Context="Ngữ cảnh bài báo dài " * 10,
            Evidence="Bộ Giáo dục và Đào tạo chính thức công bố lịch thi tốt nghiệp THPT năm 2026 vào ngày 25-6",
            labels=i % 3,
        )
        for i in range(6)
    ]
    strat = sample_stratified_dev_claims(samples, per_class=1, seed=42)
    assert len(strat) == 3
    spike_report = run_tavily_spike(
        strat,
        searcher,
        {0: "real", 1: "fake", 2: "unverifiable"},
    )
    assert spike_report["num_claims"] == 3
    assert spike_report["all_urls_in_whitelist"] is True
    assert spike_report["variants"]["Q0"]["hits_ge_05"] == 3


def test_distractor_pool_and_generation() -> None:
    from vnfake.data.distractors import DistractorPool, generate_dev_distractors
    from vnfake.training.train import _VALID_VARIANTS

    assert "V-D" in _VALID_VARIANTS

    records = [
        {
            "sample_idx": 0,
            "Statement": "Thủ tướng đi công tác tại Nhật Bản.",
            "Context": "Thủ tướng Phạm Minh Chính thăm Nhật Bản. Hai bên ký kết hợp tác kinh tế toàn diện.",
            "labels": 0,
        },
        {
            "sample_idx": 1,
            "Statement": "Thời tiết Hà Nội có mưa lớn ngập lụt.",
            "Context": "Không khí lạnh tràn về miền Bắc gây mưa to diện rộng tại Hà Nội. Nhiều tuyến phố ngập úng nghiêm trọng.",
            "labels": 1,
        },
    ]

    pool = DistractorPool.from_records(records)
    assert len(pool.articles) == 2

    pairs = generate_dev_distractors(records, pool, seed=42)
    assert len(pairs) == 4  # 2 claims * 2 distractors (random & hard_bm25)
    for p in pairs:
        assert p.target_label == 2
        assert p.distractor_evidence.strip() != ""


def test_build_keyword_query_v2_preserves_compounds_and_negation() -> None:
    from vnfake.retrieval.web_search import build_keyword_query_v2, build_query_q1v2

    assert build_query_q1v2 is build_keyword_query_v2

    # Three mandatory phrases from Brief #5 §11.6 (words "người", "việc", "không" must survive)
    q_gov = build_keyword_query_v2("Thủ tướng Chính phủ Phạm Minh Chính làm việc với lãnh đạo")
    assert "Thủ tướng" in q_gov
    assert "Chính phủ" in q_gov
    assert "Phạm Minh Chính" in q_gov
    assert "làm việc" in q_gov
    assert "lãnh đạo" in q_gov
    assert "với" not in q_gov.split()

    q_db = build_keyword_query_v2("cơ sở dữ liệu người dùng")
    assert "cơ sở dữ liệu" in q_db
    assert "người dùng" in q_db

    q_mu = build_keyword_query_v2("Man Utd chơi không tốt")
    assert "không" in q_mu.split()
    assert "Man" in q_mu
    assert "tốt" in q_mu

    # Additional negation markers and dates/numbers must be retained
    neg_stmt = "Doanh nghiệp chưa từng công bố 150 tỷ đồng ngày 25/6/2026 và đừng tin lời đồn chẳng có căn cứ."
    q_neg = build_keyword_query_v2(neg_stmt)
    for token in ("chưa", "từng", "chẳng", "đừng", "150", "25/6/2026"):
        assert token in q_neg

    # Empty / fallback safety & truncation
    assert build_keyword_query_v2("là của và với") != ""
    assert len(build_keyword_query_v2(neg_stmt, max_chars=25)) <= 25


def test_build_keyword_query_v2_filters_2syllable_stopwords() -> None:
    from vnfake.retrieval.web_search import build_keyword_query_v2

    # Narrative / connective 2-syllable stopwords must be filtered out
    stmt = (
        "Bộ Công an cho biết về việc điều tra đối với các đối tượng "
        "có liên quan cùng với việc thông qua nhằm mục đích trục lợi."
    )
    q = build_keyword_query_v2(stmt)

    # 2-syllable stopwords filtered
    for stop in ("cho biết", "về việc", "đối với", "liên quan", "cùng với", "thông qua", "nhằm mục đích"):
        assert stop not in q

    # Core content preserved
    assert "Bộ Công an" in q
    assert "điều tra" in q
    assert "đối tượng" in q
    assert "trục lợi" in q


def test_clean_web_text_and_tag_url_filtering(tmp_path: Path) -> None:
    from vnfake.retrieval.web_search import clean_web_text, is_tag_or_search_url

    # 1. Residual Markdown image/link syntax ("![M.U...", "[...]", "[]", "[text](url)")
    assert (
        clean_web_text("![M.U đã đưa ra quyết định sa thải HLV Ruben Amorim.")
        == "M.U đã đưa ra quyết định sa thải HLV Ruben Amorim."
    )
    assert (
        clean_web_text(
            "[...] AnhCựu hậu vệ Rio Ferdinand yêu cầu lãnh đạo [Man Utd](https://vnexpress.net/mu) giải thích. []"
        )
        == "Anh Cựu hậu vệ Rio Ferdinand yêu cầu lãnh đạo Man Utd giải thích."
    )

    # 2. Tag headers ("# Man Utd...") and newspaper title lines ("- Báo VnExpress...")
    assert clean_web_text("# Man Utd sắp sa thải HLV Carrick") == ""
    assert clean_web_text("Man Utd sa thải Carrick - Báo VnExpress Thể thao") == ""

    # 3. Glued category prefixes ("AnhMan Utd...") fixed while mixed-case whitelist preserved
    glued = (
        "AnhMan Utd quyết định sa thải HLV Ruben Amorim theo nguồn tin từ VnExpress, "
        "VietNamNet, YouTube, TikTok trên iPhone, iPad chạy JavaScript và WordPress."
    )
    cleaned_glued = clean_web_text(glued)
    assert "Anh Man Utd" in cleaned_glued
    for brand in (
        "VnExpress",
        "VietNamNet",
        "YouTube",
        "TikTok",
        "iPhone",
        "iPad",
        "JavaScript",
        "WordPress",
    ):
        assert brand in cleaned_glued

    # 4. Tag / topic / search URL detection & filtering in _build_hits_from_raw
    assert is_tag_or_search_url("https://thanhnien.vn/hlv-amorim-tags1697865.html")
    assert is_tag_or_search_url("https://vnexpress.net/tag/man-utd-123")
    assert is_tag_or_search_url("https://vnexpress.net/chu-de/cup-fa-943")
    assert is_tag_or_search_url("https://tuoitre.vn/tim-kiem/ket-qua.htm")
    assert is_tag_or_search_url("https://dantri.com.vn/search/tin-tuc.htm")
    assert not is_tag_or_search_url("https://vnexpress.net/man-utd-sa-thai-hlv-amorim-5002109.html")

    search_cfg = SearchConfig(
        provider="tavily",
        search_depth="basic",
        max_results=5,
        topic="general",
        cache_dir=tmp_path / "tag_filter_cache",
        max_credits_spike=10,
        timeout_s=5.0,
        rate_limit_s=1.0,
        user_agent="VNFakeTest/1.0",
        include_domains=["vnexpress.net", "thanhnien.vn"],
    )
    searcher = TavilyWebSearcher(
        search_cfg,
        api_key="tvly-dummy",
        page_fetcher=lambda _u: "",
    )
    hits, _ = searcher._build_hits_from_raw(
        {
            "results": [
                {
                    "title": "HLV Amorim tin tức mới nhất",
                    "url": "https://thanhnien.vn/hlv-amorim-tags1697865.html",
                    "content": "# Man Utd sa thải HLV Amorim",
                    "raw_content": "# Man Utd sa thải HLV Amorim\n" * 20,
                    "score": 0.99,
                },
                {
                    "title": "Man Utd sa thải HLV Amorim - Báo VnExpress Thể thao",
                    "url": "https://vnexpress.net/man-utd-sa-thai-hlv-amorim-5002109.html",
                    "content": "AnhMan Utd quyết định sa thải HLV Ruben Amorim sau 14 tháng gắn bó.",
                    "raw_content": "",
                    "score": 0.91,
                },
            ]
        },
        allow_network_fetch=False,
    )
    assert len(hits) == 1
    assert hits[0].url == "https://vnexpress.net/man-utd-sa-thai-hlv-amorim-5002109.html"
    assert "Anh Man Utd quyết định sa thải" in hits[0].full_text
    assert "- Báo VnExpress" not in hits[0].full_text


def test_select_evidence_with_sources_deduplication_and_short_filter() -> None:
    from vnfake.retrieval.evidence_selector import select_evidence_with_sources

    claim = "Man Utd quyết định sa thải HLV Ruben Amorim sau 14 tháng gắn bó."
    dup_sent = (
        "Ban lãnh đạo Man Utd chính thức đưa ra quyết định sa thải HLV Ruben Amorim sau 14 tháng gắn bó."
    )
    near_dup_sent = (
        "Ban lãnh đạo Man Utd chính thức đưa ra quyết định sa thải HLV Ruben Amorim sau 14 tháng gắn bó tại CLB."
    )
    other_sent = (
        "Cựu tiền vệ Darren Fletcher sẽ tạm thời nắm quyền dẫn dắt đội một trong vai trò huấn luyện viên tạm quyền."
    )

    sourced_contexts = [
        (f"Tin vắn.\n!@#$%^&*()\n{dup_sent}", "https://vnexpress.net/a.html", "vnexpress.net"),
        (f"{dup_sent}", "https://tuoitre.vn/b.html", "tuoitre.vn"),
        (f"{near_dup_sent}\n{other_sent}", "https://dantri.com.vn/c.html", "dantri.com.vn"),
    ]

    res = select_evidence_with_sources(claim, sourced_contexts, top_k=3, max_tokens=150)
    assert len(res.items) == 2
    texts = [it.text for it in res.items]
    assert dup_sent in texts
    assert other_sent in texts
    assert near_dup_sent not in texts



