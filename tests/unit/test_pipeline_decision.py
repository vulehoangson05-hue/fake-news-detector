"""Unit tests for vnfake.pipeline.decision and sourced selection (Brief #5 §3, §11.5, §12.3, §12.5.3)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from vnfake.config import AppConfig, DecisionConfig, PathsConfig, VerifierCheckpointSpec
from vnfake.pipeline.decision import (
    REASON_CONFIRMED,
    REASON_LOW_CONFIDENCE,
    REASON_LOW_RELEVANCE,
    DecisionGate,
    load_vc_ensemble,
)
from vnfake.retrieval.evidence_selector import select_evidence_with_sources


def test_decision_gate_low_relevance() -> None:
    gate = DecisionGate(tau_rel=0.50, tau_conf=0.60)
    probs = [0.10, 0.85, 0.05]  # Very high fake probability
    res = gate.decide(relevance_score=0.42, probs=probs)

    assert res.verdict == "unverifiable"
    assert res.label_idx == 2
    assert not res.passed_relevance
    assert not res.passed_confidence
    assert res.show_probs is False
    assert res.reason == REASON_LOW_RELEVANCE
    assert res.relevance_score == 0.42


def test_decision_gate_low_confidence() -> None:
    gate = DecisionGate(tau_rel=0.50, tau_conf=0.60)
    # Relevance passes (0.75 >= 0.50), but max probability is 0.45 (< 0.60)
    probs = [0.45, 0.35, 0.20]
    res = gate.decide(relevance_score=0.75, probs=probs)

    assert res.verdict == "unverifiable"
    assert res.label_idx == 2
    assert res.passed_relevance
    assert not res.passed_confidence
    assert res.show_probs is False
    assert res.reason == REASON_LOW_CONFIDENCE
    assert res.relevance_score == 0.75


def test_decision_gate_confirmed_real() -> None:
    gate = DecisionGate(tau_rel=0.50, tau_conf=0.60)
    probs = [0.78, 0.12, 0.10]
    res = gate.decide(relevance_score=0.88, probs=probs)

    assert res.verdict == "real"
    assert res.label_idx == 0
    assert res.passed_relevance
    assert res.passed_confidence
    assert res.show_probs is True
    assert res.reason == REASON_CONFIRMED
    assert res.probs["real"] == 0.78


def test_decision_gate_confirmed_unverifiable_shows_probs() -> None:
    gate = DecisionGate(tau_rel=0.50, tau_conf=0.60)
    probs = [0.10, 0.15, 0.75]
    res = gate.decide(relevance_score=0.88, probs=probs)

    assert res.verdict == "unverifiable"
    assert res.passed_relevance is True
    assert res.passed_confidence is True
    assert res.show_probs is True
    assert res.reason == REASON_CONFIRMED


def test_decision_gate_confirmed_fake() -> None:
    gate = DecisionGate(tau_rel=0.50, tau_conf=0.60)
    probs = [0.10, 0.82, 0.08]
    res = gate.decide(relevance_score=0.65, probs=probs)

    assert res.verdict == "fake"
    assert res.label_idx == 1
    assert res.passed_relevance
    assert res.passed_confidence
    assert res.reason == REASON_CONFIRMED


def test_decision_gate_invalid_shapes() -> None:
    gate = DecisionGate()
    with pytest.raises(ValueError, match="Expected 1D probability array"):
        gate.decide(0.8, [0.5, 0.5])  # Length 2 instead of 3

    with pytest.raises(ValueError, match="Expected 1D probability array"):
        gate.decide(0.8, np.ones((2, 3)))


def test_decision_gate_with_mapping_class_names() -> None:
    mapping = {0: "real", 1: "fake", 2: "unverifiable"}
    gate = DecisionGate(tau_rel=0.50, tau_conf=0.50, class_names=mapping)
    assert gate.class_names == ["real", "fake", "unverifiable"]
    res = gate.decide(relevance_score=0.90, probs=[0.1, 0.8, 0.1])
    assert res.verdict == "fake"
    assert res.label_idx == 1
    assert res.show_probs is True
    assert res.reason == REASON_CONFIRMED


def test_select_evidence_with_sources_retains_urls() -> None:
    stmt = "Bộ Giáo dục công bố lịch thi năm 2026 tại Hà Nội."
    contexts = [
        (
            "Hôm nay trời nắng đẹp. Bộ Giáo dục công bố lịch thi năm 2026 vào tháng 6.",
            "https://vnexpress.net/lich-thi-2026.html",
            "vnexpress.net",
        ),
        (
            "Học sinh tại Hà Nội đang ôn tập tích cực cho kỳ thi.",
            "https://tuoitre.vn/on-tap-2026.html",
            "tuoitre.vn",
        ),
    ]
    res = select_evidence_with_sources(stmt, contexts, top_k=2, max_tokens=150)
    assert res.items
    urls = {item.url for item in res.items}
    assert "https://vnexpress.net/lich-thi-2026.html" in urls
    for item in res.items:
        assert item.text in res.text


def test_load_vc_ensemble_pinned_sha256_verification(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    run_id = "phobert_v-c_s42_test"
    ckpt_dir = models_dir / run_id / "checkpoint"
    ckpt_dir.mkdir(parents=True)
    weights_file = ckpt_dir / "model.safetensors"
    weights_bytes = b"dummy-safetensors-weights"
    weights_file.write_bytes(weights_bytes)
    expected_sha = hashlib.sha256(weights_bytes).hexdigest()
    (models_dir / run_id / "temperature.json").write_text(
        json.dumps({"temperature": 1.35}), encoding="utf-8"
    )

    paths = PathsConfig(
        kb_dantri=tmp_path,
        kb_vnexpress=tmp_path,
        vifactcheck_dir=tmp_path,
        interim_dir=tmp_path,
        processed_dir=tmp_path,
        artifacts_dir=tmp_path,
    )
    label_map = {0: "real", 1: "fake", 2: "unverifiable"}

    # Empty checkpoints raises ValueError
    cfg_empty = AppConfig(paths=paths, label_map=label_map, decision=DecisionConfig(checkpoints=[]))
    with pytest.raises(ValueError, match="No pinned checkpoints"):
        load_vc_ensemble(cfg_empty, device="cpu")

    # Wrong SHA-256 raises ValueError
    cfg_bad = AppConfig(
        paths=paths,
        label_map=label_map,
        decision=DecisionConfig(
            checkpoints=[VerifierCheckpointSpec(seed=42, run_id=run_id, sha256="deadbeef")]
        ),
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_vc_ensemble(cfg_bad, device="cpu", verify_sha256=True)

    # Matching SHA-256 loads EnsembleVerifier properly
    cfg_ok = AppConfig(
        paths=paths,
        label_map=label_map,
        decision=DecisionConfig(
            checkpoints=[VerifierCheckpointSpec(seed=42, run_id=run_id, sha256=expected_sha)]
        ),
    )
    ens = load_vc_ensemble(cfg_ok, device="cpu", verify_sha256=True)
    assert ens.temperatures == [1.35]
    assert ens.checkpoint_dirs == [ckpt_dir]
