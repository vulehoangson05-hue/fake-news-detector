"""Unit tests for vnfake.evaluation.rerank_profiler (pure functions, no GPU/model load)."""

from __future__ import annotations

import pytest

from vnfake.evaluation.rerank_profiler import (
    compute_mean_top_k_overlap,
    compute_percentile,
    compute_top_k_overlap,
    make_claim_id,
    sample_dev_claims,
)
from vnfake.evaluation.retrieval_eval import EvalQuery


def test_compute_percentile() -> None:
    assert compute_percentile([], 50.0) == 0.0
    assert compute_percentile([42.0], 95.0) == pytest.approx(42.0)
    vals = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert compute_percentile(vals, 50.0) == pytest.approx(30.0)
    assert compute_percentile(vals, 95.0) == pytest.approx(48.0)


def test_compute_top_k_overlap() -> None:
    base = ["c1", "c2", "c3", "c4", "c5"]
    same = ["c5", "c4", "c3", "c2", "c1"]
    partial = ["c1", "c2", "x3", "x4", "x5"]
    disjoint = ["x1", "x2", "x3", "x4", "x5"]

    assert compute_top_k_overlap(base, same, k=5) == pytest.approx(1.0)
    assert compute_top_k_overlap(base, partial, k=5) == pytest.approx(0.4)
    assert compute_top_k_overlap(base, disjoint, k=5) == pytest.approx(0.0)
    assert compute_top_k_overlap([], [], k=5) == pytest.approx(1.0)

    with pytest.raises(ValueError):
        compute_top_k_overlap(base, same, k=0)


def test_compute_mean_top_k_overlap() -> None:
    base_map = {
        "q1": ["a", "b", "c", "d", "e"],
        "q2": ["1", "2", "3", "4", "5"],
    }
    cand_map = {
        "q1": ["a", "b", "c", "d", "e"],
        "q2": ["1", "2", "x", "y", "z"],
    }
    assert compute_mean_top_k_overlap(base_map, cand_map, k=5) == pytest.approx(0.7)
    assert compute_mean_top_k_overlap({}, cand_map, k=5) == pytest.approx(0.0)


def test_sample_dev_claims_reproducible() -> None:
    queries = [EvalQuery(statement=f"Claim số {i}", evidence="", label=0) for i in range(20)]
    s1 = sample_dev_claims(queries, n_claims=5, seed=42)
    s2 = sample_dev_claims(queries, n_claims=5, seed=42)
    s3 = sample_dev_claims(queries, n_claims=5, seed=99)

    assert [cid for cid, _ in s1] == [cid for cid, _ in s2]
    assert [cid for cid, _ in s1] != [cid for cid, _ in s3]
    assert make_claim_id(0, "test").startswith("dev_0000_")
