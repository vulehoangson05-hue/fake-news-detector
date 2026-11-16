"""Tokenization and token-length profiling for Verifier models (Phase 3, Brief #4 §2.2-§2.5).

Rules enforced (copilot-instructions §4):
- PhoBERT inputs MUST be word-segmented (``_``-joined compounds), never lowercased.
- XLM-R uses raw NFC-normalized text without word segmentation.
- Truncation always prioritizes preserving ``Statement`` (``truncation="only_second"``)
  when an evidence pair is provided.
- Supports ``statement_only=True`` for the mandatory claim-only baseline.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from vnfake.data.cleaning import normalize_text
from vnfake.data.dataset import (
    ModelType,
    PairDataset,
    PairItem,
    TokenizerProtocol,
    build_class_mapping,
)
from vnfake.data.segmentation import SegmenterProtocol
from vnfake.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class TokenLengthStats:
    """Percentile statistics of tokenized sequence lengths."""

    count: int
    p50: float
    p95: float
    p99: float
    max_len: int
    mean: float
    pct_over_256: float


def _percentile(sorted_vals: Sequence[int], q: float) -> float:
    """Compute percentile ``q`` in ``[0, 100]`` using linear interpolation."""
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    if n == 1:
        return float(sorted_vals[0])
    pos = (q / 100.0) * (n - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_vals[lo])
    frac = pos - lo
    return float(sorted_vals[lo]) * (1.0 - frac) + float(sorted_vals[hi]) * frac


def measure_pair_token_lengths(
    statements: Sequence[str],
    evidences: Sequence[str],
    *,
    tokenizer: TokenizerProtocol,
    model_type: ModelType,
    segmenter: SegmenterProtocol | None = None,
    presegmented: bool = False,
) -> TokenLengthStats:
    """Measure untruncated token lengths of ``(Statement, evidence)`` pairs.

    Args:
        statements: Raw or pre-segmented statements.
        evidences: Raw or pre-segmented evidence texts.
        tokenizer: Hugging Face tokenizer.
        model_type: ``"phobert"`` or ``"xlmr"``.
        segmenter: Required if ``model_type == "phobert"`` and ``not presegmented``.
        presegmented: True if texts are already word-segmented.

    Returns:
        :class:`TokenLengthStats` with p50, p95, p99, max_len, mean, and % > 256.
    """
    if len(statements) != len(evidences):
        raise ValueError("statements and evidences must have the same length.")
    if not statements:
        return TokenLengthStats(
            count=0, p50=0.0, p95=0.0, p99=0.0, max_len=0, mean=0.0, pct_over_256=0.0
        )

    if model_type == "phobert" and not presegmented:
        if segmenter is None:
            raise ValueError("PhoBERT requires a segmenter when presegmented=False.")
        stmt_texts = segmenter.segment_batch([normalize_text(s) for s in statements])
        evid_texts = segmenter.segment_batch([normalize_text(e) for e in evidences])
    else:
        stmt_texts = [s if presegmented else normalize_text(s) for s in statements]
        evid_texts = [e if presegmented else normalize_text(e) for e in evidences]

    lengths: list[int] = []
    for st, ev in zip(stmt_texts, evid_texts, strict=True):
        enc = tokenizer(
            st,
            ev,
            max_length=4096,
            truncation=False,
            padding=False,
        )
        lengths.append(len(enc["input_ids"]))

    sorted_lens = sorted(lengths)
    n = len(sorted_lens)
    over_256 = sum(1 for x in sorted_lens if x > 256)
    return TokenLengthStats(
        count=n,
        p50=round(_percentile(sorted_lens, 50.0), 1),
        p95=round(_percentile(sorted_lens, 95.0), 1),
        p99=round(_percentile(sorted_lens, 99.0), 1),
        max_len=sorted_lens[-1],
        mean=round(sum(sorted_lens) / n, 1),
        pct_over_256=round((over_256 / n) * 100.0, 2),
    )


def encode_verifier_dataset(
    statements: Sequence[str],
    evidences: Sequence[str],
    labels: Sequence[int],
    *,
    tokenizer: TokenizerProtocol,
    model_type: ModelType,
    max_length: int,
    label_map: Mapping[int, str],
    segmenter: SegmenterProtocol | None = None,
    statement_only: bool = False,
    presegmented: bool = False,
) -> PairDataset:
    """Encode ``(Statement, evidence_text)`` or ``Statement``-only into a :class:`PairDataset`.

    Args:
        statements: Claim statements.
        evidences: Evidence texts (ignored when ``statement_only=True``).
        labels: Integer labels matching ``label_map`` keys.
        tokenizer: Loaded HF tokenizer.
        model_type: ``"phobert"`` or ``"xlmr"``.
        max_length: Max sequence length (default 256 for PhoBERT).
        label_map: Config label map ``{0: "real", 1: "fake", 2: "unverifiable"}``.
        segmenter: Required for PhoBERT when ``presegmented=False``.
        statement_only: If True, encodes ONLY ``Statement`` (mandatory baseline).
        presegmented: If True, inputs are already word-segmented.

    Returns:
        Tokenized :class:`PairDataset`.
    """
    if len(statements) != len(labels) or (not statement_only and len(evidences) != len(statements)):
        raise ValueError("Mismatched lengths between statements, evidences, and labels.")
    if not statements:
        return PairDataset([])

    _, src_to_class = build_class_mapping(label_map)

    if model_type == "phobert" and not presegmented:
        if segmenter is None:
            raise ValueError("PhoBERT requires a segmenter when presegmented=False.")
        stmt_texts = segmenter.segment_batch([normalize_text(s) for s in statements])
        evid_texts = (
            []
            if statement_only
            else segmenter.segment_batch([normalize_text(e) for e in evidences])
        )
    else:
        stmt_texts = [s if presegmented else normalize_text(s) for s in statements]
        evid_texts = (
            [] if statement_only else [e if presegmented else normalize_text(e) for e in evidences]
        )

    items: list[PairItem] = []
    for i, st in enumerate(stmt_texts):
        if statement_only:
            enc = tokenizer(
                st,
                None,
                max_length=max_length,
                truncation=True,
                padding=False,
            )
        else:
            ev = evid_texts[i]
            # Fallback if evidence is empty so truncation="only_second" never fails
            enc = tokenizer(
                st,
                ev if ev.strip() else ".",
                max_length=max_length,
                truncation="only_second",
                padding=False,
            )
        items.append(
            PairItem(
                input_ids=list(enc["input_ids"]),
                attention_mask=list(enc["attention_mask"]),
                token_type_ids=(list(enc["token_type_ids"]) if "token_type_ids" in enc else None),
                label=src_to_class[labels[i]],
            )
        )

    logger.info(
        "encode_verifier_dataset: model_type=%s n=%d max_length=%d statement_only=%s",
        model_type,
        len(items),
        max_length,
        statement_only,
    )
    return PairDataset(items)
