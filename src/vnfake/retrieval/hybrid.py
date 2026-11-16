"""Reciprocal Rank Fusion (RRF) — client-side fallback / benchmark tool.

Production path uses server-side fusion in
``QdrantStore.search_hybrid`` (``Fusion.RRF``). This module is useful for
ablation (server-side vs client-side) and for tests without Qdrant.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RankedItem:
    """A document id with its score in one ranker's list."""

    doc_id: str
    score: float


def rrf_fuse(
    ranked_lists: Sequence[Sequence[RankedItem]],
    *,
    k: int = 60,
    top_k: int | None = None,
) -> list[RankedItem]:
    """Fuse ranked lists with Reciprocal Rank Fusion.

    Formula per document d::

        score(d) = sum over rankers r of 1 / (k + rank_r(d))

    where ``rank_r(d)`` is the 1-based rank of d in ranker r's list.

    Args:
        ranked_lists: Ordered results per ranker (best-first).
        k: RRF constant (typical 60).
        top_k: Optional truncation of fused output.

    Returns:
        Fused list, descending score; ties broken by ``doc_id``.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if top_k is not None and top_k < 1:
        raise ValueError(f"top_k must be >= 1 or None, got {top_k}")

    fused: dict[str, float] = {}
    for ranked in ranked_lists:
        seen_in_list: set[str] = set()
        for rank_zero_based, item in enumerate(ranked):
            if item.doc_id in seen_in_list:
                continue
            seen_in_list.add(item.doc_id)
            fused[item.doc_id] = fused.get(item.doc_id, 0.0) + 1.0 / (k + rank_zero_based + 1)

    ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    if top_k is not None:
        ordered = ordered[:top_k]
    return [RankedItem(doc_id=d, score=s) for d, s in ordered]
