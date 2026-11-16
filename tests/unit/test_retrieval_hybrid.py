"""Unit tests for vnfake.retrieval.hybrid (RRF fusion)."""

from __future__ import annotations

import pytest

from vnfake.retrieval.hybrid import RankedItem, rrf_fuse


def test_rrf_single_list_preserves_order() -> None:
    a = [RankedItem("d1", 1.0), RankedItem("d2", 0.9), RankedItem("d3", 0.8)]
    out = rrf_fuse([a], k=60)
    assert [r.doc_id for r in out] == ["d1", "d2", "d3"]


def test_rrf_merges_two_lists() -> None:
    dense = [RankedItem("d1", 0.9), RankedItem("d2", 0.8)]
    sparse = [RankedItem("d2", 5.0), RankedItem("d3", 3.0)]
    out = rrf_fuse([dense, sparse], k=60)
    assert out[0].doc_id == "d2"
    assert {r.doc_id for r in out} == {"d1", "d2", "d3"}


def test_rrf_scores_formula() -> None:
    out = rrf_fuse([[RankedItem("a", 0.0)], [RankedItem("a", 0.0)]], k=60)
    assert out[0].score == pytest.approx(2 / 61)


def test_rrf_top_k_truncates() -> None:
    lists = [[RankedItem("d1", 1.0), RankedItem("d2", 1.0), RankedItem("d3", 1.0)]]
    assert len(rrf_fuse(lists, k=60, top_k=2)) == 2


def test_rrf_deduplicates_within_single_list() -> None:
    out = rrf_fuse([[RankedItem("d1", 1.0), RankedItem("d1", 0.5)]], k=60)
    assert len(out) == 1


def test_rrf_empty_inputs() -> None:
    assert rrf_fuse([]) == []
    assert rrf_fuse([[]]) == []


def test_rrf_rejects_bad_k() -> None:
    with pytest.raises(ValueError, match="k must be"):
        rrf_fuse([], k=0)


def test_rrf_rejects_bad_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        rrf_fuse([], top_k=0)
