"""Deduplication utilities (URL exact + MinHash near-duplicate + leakage fix).

- ``dedup_by_url``: exact URL dedup for KB.
- ``dedup_by_minhash``: near-duplicate removal on ``title + "\\n" + text``.
- ``dedup_vifactcheck_splits``: removes train rows that duplicate dev/test
  (cross-split leakage) or other train rows (internal duplicates). Dev and
  test are preserved untouched.

NOTE: ``dedup_by_minhash`` is O(n) in practice but uses LSH; for 3 053 KB
records it completes in <1 s. For very large corpora, batch the insertion.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from datasketch import MinHash, MinHashLSH

from vnfake.schemas.kb import KBRecord
from vnfake.schemas.vifactcheck import ViFactCheckSample
from vnfake.utils.logging import get_logger

logger = get_logger(__name__)

# Named constants (overridable via config in callers — copilot-instructions §3).
DEFAULT_MINHASH_NUM_PERM = 128
DEFAULT_MINHASH_SHINGLE_N = 5
DEFAULT_MINHASH_THRESHOLD = 0.8

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def dedup_by_url(records: Sequence[KBRecord]) -> list[KBRecord]:
    """Remove exact duplicate URLs, keeping the first occurrence.

    Records with an empty ``url`` are dropped (``url`` is required for evidence
    citation per ARCHITECTURE §4.4; schema already enforces non-empty strings,
    this is a defensive guard).

    Args:
        records: KB records in file order.

    Returns:
        Deduplicated records.
    """
    seen: set[str] = set()
    kept: list[KBRecord] = []
    dropped_dup = dropped_empty = 0
    for r in records:
        if not r.url:
            dropped_empty += 1
            continue
        if r.url in seen:
            dropped_dup += 1
            continue
        seen.add(r.url)
        kept.append(r)
    logger.info(
        "dedup_by_url: kept=%d dropped_dup=%d dropped_empty=%d",
        len(kept),
        dropped_dup,
        dropped_empty,
    )
    return kept


def _shingles(text: str, n: int) -> set[str]:
    """Lowercased word-level n-gram shingles."""
    tokens = _WORD_RE.findall(text.lower())
    if not tokens:
        return set()
    if len(tokens) < n:
        return {" ".join(tokens)}
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _minhash(shingles: set[str], num_perm: int) -> MinHash:
    mh = MinHash(num_perm=num_perm)
    for sh in shingles:
        mh.update(sh.encode("utf-8"))
    return mh


def dedup_by_minhash(
    records: Sequence[KBRecord],
    *,
    threshold: float = DEFAULT_MINHASH_THRESHOLD,
    num_perm: int = DEFAULT_MINHASH_NUM_PERM,
    shingle_n: int = DEFAULT_MINHASH_SHINGLE_N,
) -> list[KBRecord]:
    """Near-duplicate removal via MinHash LSH. First occurrence wins.

    Comparison key: ``title + "\\n" + text`` (matches ARCHITECTURE §5.1 step 5
    embedding input).

    Args:
        records: KB records in file order.
        threshold: Jaccard similarity threshold for duplicates.
        num_perm: Number of MinHash permutations (accuracy vs. speed).
        shingle_n: Word n-gram size.

    Returns:
        Records with near-duplicates removed.
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept: list[KBRecord] = []
    dropped = 0
    for idx, r in enumerate(records):
        mh = _minhash(_shingles(f"{r.title}\n{r.text}", shingle_n), num_perm)
        if lsh.query(mh):
            dropped += 1
            continue
        lsh.insert(f"doc-{idx}", mh)
        kept.append(r)
    logger.info(
        "dedup_by_minhash: kept=%d dropped=%d (threshold=%.2f, num_perm=%d, n=%d)",
        len(kept),
        dropped,
        threshold,
        num_perm,
        shingle_n,
    )
    return kept


def _normalize_statement(text: str) -> str:
    """NFC-normalize, trim, lowercase (leakage key)."""
    return unicodedata.normalize("NFC", text).strip().casefold()


def dedup_vifactcheck_splits(
    train: Sequence[ViFactCheckSample],
    dev: Sequence[ViFactCheckSample],
    test: Sequence[ViFactCheckSample],
) -> tuple[list[ViFactCheckSample], list[ViFactCheckSample], list[ViFactCheckSample]]:
    """Remove leakage and internal duplicates from ``train`` only.

    A train row is dropped if its Statement (NFC + lowercase) matches:
        * any Statement in ``dev`` or ``test`` (cross-split leakage), OR
        * an earlier train row (internal duplicate — first occurrence wins).

    ``dev`` and ``test`` are returned unchanged (they are evaluation targets).

    Args:
        train: Original train split.
        dev: Dev split (protected).
        test: Test split (protected).

    Returns:
        ``(cleaned_train, dev, test)`` as fresh lists.
    """
    protected: set[str] = {_normalize_statement(s.statement) for s in dev}
    protected.update(_normalize_statement(s.statement) for s in test)

    seen: set[str] = set()
    cleaned_train: list[ViFactCheckSample] = []
    dropped_cross = dropped_internal = 0
    for s in train:
        key = _normalize_statement(s.statement)
        if key in protected:
            dropped_cross += 1
            continue
        if key in seen:
            dropped_internal += 1
            continue
        seen.add(key)
        cleaned_train.append(s)

    logger.info(
        "dedup_vifactcheck_splits: train %d -> %d (dropped cross_split=%d internal=%d)",
        len(train),
        len(cleaned_train),
        dropped_cross,
        dropped_internal,
    )
    return cleaned_train, list(dev), list(test)
