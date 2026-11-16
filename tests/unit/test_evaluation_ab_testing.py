"""Unit tests for vnfake.evaluation.ab_testing (deterministic sampling and blind packaging)."""

from __future__ import annotations

from pathlib import Path

import pytest

from vnfake.evaluation.ab_testing import (
    BranchResult,
    ClaimABRecord,
    export_label_pack,
    generate_blind_label_pack,
    stratified_sample_dev,
)
from vnfake.retrieval.vector_store import ScoredChunk
from vnfake.schemas.vifactcheck import ViFactCheckSample


@pytest.fixture
def mock_dev_file(tmp_path: Path) -> Path:
    dev_path = tmp_path / "mock_dev.jsonl"
    lines: list[str] = []
    # 5 samples per label (0, 1, 2)
    for lbl in (0, 1, 2):
        for i in range(5):
            s = ViFactCheckSample(
                statement=f"Phát biểu thử nghiệm số {i} lớp {lbl}",
                context="",
                labels=lbl,
                evidence=f"Bằng chứng số {i} lớp {lbl}",
            )
            lines.append(s.model_dump_json(by_alias=True))
    dev_path.write_text("\n".join(lines), encoding="utf-8")
    return dev_path


def test_stratified_sample_dev(mock_dev_file: Path) -> None:
    sampled = stratified_sample_dev(mock_dev_file, seed=42, per_class=2)
    assert len(sampled) == 6
    # 2 of each label
    labels = [s.labels for _, s in sampled]
    assert labels.count(0) == 2
    assert labels.count(1) == 2
    assert labels.count(2) == 2

    # Determinism with same seed
    sampled_again = stratified_sample_dev(mock_dev_file, seed=42, per_class=2)
    assert [idx for idx, _ in sampled] == [idx for idx, _ in sampled_again]


def test_stratified_sample_raises_on_insufficient_samples(mock_dev_file: Path) -> None:
    with pytest.raises(ValueError, match="fewer than required"):
        stratified_sample_dev(mock_dev_file, seed=42, per_class=10)


def test_blind_label_pack_dedup_and_no_system_code(tmp_path: Path) -> None:
    """Verify that candidate chunks are deduplicated per claim, have full text, and no system_code in CSV."""
    chunk_a = ScoredChunk(
        "c_a", "d_a", 0, "t_a", "Full text of chunk A", "", "u_a", "2026-11-08", "vnexpress", 0.9
    )
    chunk_b = ScoredChunk(
        "c_b", "d_b", 0, "t_b", "Full text of chunk B", "", "u_b", "2026-11-08", "vnexpress", 0.8
    )
    chunk_c = ScoredChunk(
        "c_c", "d_c", 0, "t_c", "Full text of chunk C", "", "u_c", "2026-11-08", "vnexpress", 0.7
    )

    b_orig = BranchResult(
        branch_name="original",
        query_used="claim text",
        rewrite_status="ok",
        candidates_count=2,
        evidences=[chunk_a, chunk_b],
        proxy_hits={"hit@1": False, "hit@3": False, "hit@5": False},
        retrieve_ms=10.0,
        rerank_ms=20.0,
        total_ms=30.0,
    )
    b_rew = BranchResult(
        branch_name="rewritten_only",
        query_used="rewritten query",
        rewrite_status="ok",
        candidates_count=2,
        evidences=[chunk_b, chunk_c],  # chunk_b is shared!
        proxy_hits={"hit@1": False, "hit@3": False, "hit@5": False},
        retrieve_ms=10.0,
        rerank_ms=20.0,
        total_ms=30.0,
    )
    b_fused = BranchResult(
        branch_name="original+rewritten",
        query_used="fused query",
        rewrite_status="ok",
        candidates_count=2,
        evidences=[chunk_a, chunk_c],  # chunk_a and chunk_c shared!
        proxy_hits={"hit@1": False, "hit@3": False, "hit@5": False},
        retrieve_ms=10.0,
        rerank_ms=20.0,
        total_ms=30.0,
    )

    rec = ClaimABRecord(
        claim_id="c1_dev_001",
        sample_index=0,
        statement="Test statement",
        gold_label=0,
        gold_evidence="Evidence",
        rewritten_query="rewritten query",
        rewrite_status="ok",
        rewrite_latency_ms=500.0,
        branch_original=b_orig,
        branch_rewritten_only=b_rew,
        branch_fused=b_fused,
    )

    items, unblind_map = generate_blind_label_pack([rec], seed=42)

    # 3 unique chunks (A, B, C)
    assert len(items) == 3
    chunk_ids = {it.chunk_id for it in items}
    assert chunk_ids == {"c_a", "c_b", "c_c"}

    # Full text preserved
    for it in items:
        assert it.text.startswith("Full text of chunk")

    # Unblind key records all branches
    for _item_id, meta in unblind_map.items():
        assert "branches" in meta
        assert "chunk_id" in meta
        assert "claim_id" in meta
        if meta["chunk_id"] == "c_b":
            assert set(meta["branches"]) == {"original", "rewritten_only"}
        elif meta["chunk_id"] == "c_a":
            assert set(meta["branches"]) == {"original", "original+rewritten"}
        elif meta["chunk_id"] == "c_c":
            assert set(meta["branches"]) == {"rewritten_only", "original+rewritten"}

    # Export to files and check CSV
    out_dir = tmp_path / "label_pack"
    export_label_pack(items, unblind_map, out_dir)

    csv_path = out_dir / "human_label_pack_60.csv"
    assert csv_path.exists()
    csv_content = csv_path.read_text(encoding="utf-8")
    header = csv_content.splitlines()[0]
    # No system_code column in CSV
    assert "system_code" not in header
    assert "text" in header
    assert "human_verdict_1_or_0" in header

    # Verify unblind key exists separately
    assert (out_dir / "unblind_key_do_not_open_before_labeling.json").exists()
    assert (out_dir / "README.md").exists()
