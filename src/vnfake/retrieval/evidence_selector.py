"""Sentence-level evidence selector from article context or web pages (Brief #4 §2.3).

Splits an article (``Context`` or fetched web text) into sentences, scores each
sentence against the ``Statement``, and selects the top-k sentences (default 3)
in their original article order while keeping total length <= ``max_tokens``
(~150 words/tokens).

Supports both:
- ``bm25`` (fast lexical unigram + bigram BM25; processes 5,000+ articles in seconds)
- ``reranker`` (cross-encoder ``bge-reranker-v2-m3`` when passed)
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from vnfake.data.chunking import split_sentences
from vnfake.data.cleaning import normalize_text
from vnfake.utils.logging import get_logger

logger = get_logger(__name__)

_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)
_BM25_K1 = 1.5
_BM25_B = 0.75
_BIGRAM_WEIGHT = 1.5
_DEFAULT_TOP_K_SENTENCES = 3
_DEFAULT_MAX_EVIDENCE_TOKENS = 150


class SentenceScorerProtocol(Protocol):
    """Optional cross-encoder scorer for ``(statement, sentences)``."""

    def score_pairs(self, query: str, sentences: Sequence[str]) -> list[float]: ...


@dataclass(frozen=True, slots=True)
class SelectedEvidence:
    """Result of selecting evidence sentences from an article."""

    text: str
    sentence_indices: list[int]
    num_sentences: int
    word_count: int
    method: Literal["bm25", "reranker"]


@dataclass(frozen=True, slots=True)
class SourcedSentence:
    """A selected evidence sentence retaining its source URL and domain."""

    text: str
    url: str
    domain: str


@dataclass(frozen=True, slots=True)
class SourcedEvidenceResult:
    """Result of selecting evidence across multiple sourced contexts (Brief #5 §4.4 & §12.3.5)."""

    text: str
    items: list[SourcedSentence]
    word_count: int
    method: Literal["bm25", "reranker"]


def tokenize_words(text: str) -> list[str]:
    """Extract lowercased word tokens for lexical matching and 3-gram containment."""
    return _WORD_RE.findall(normalize_text(text).lower())


def word_shingles(text: str, n: int = 3) -> set[str]:
    """Return word n-gram shingles matching ``DATA_AUDIT.md §6``."""
    tokens = tokenize_words(text)
    if not tokens:
        return set()
    if len(tokens) < n:
        return set(tokens)
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def compute_3gram_containment(gold_evidence: str, candidate_text: str) -> float:
    """Compute share of ``gold_evidence`` word 3-grams present in ``candidate_text``."""
    ev_shingles = word_shingles(gold_evidence, n=3)
    if not ev_shingles:
        return 0.0
    cand_shingles = word_shingles(candidate_text, n=3)
    return len(ev_shingles & cand_shingles) / len(ev_shingles)


def _extract_bm25_terms(tokens: Sequence[str]) -> list[str]:
    """Return unigrams + weighted bigrams for sharper phrase matching."""
    terms = list(tokens)
    if len(tokens) >= 2:
        terms.extend(f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1))
    return terms


def score_sentences_bm25(statement: str, sentences: Sequence[str]) -> list[float]:
    """Score each sentence in ``sentences`` against ``statement`` using unigram+bigram BM25."""
    if not sentences:
        return []
    q_tokens = tokenize_words(statement)
    if not q_tokens:
        return [0.0 for _ in sentences]

    q_unigrams = q_tokens
    q_bigrams = (
        [f"{q_tokens[i]}_{q_tokens[i + 1]}" for i in range(len(q_tokens) - 1)]
        if len(q_tokens) >= 2
        else []
    )

    sent_tokens = [tokenize_words(s) for s in sentences]
    sent_terms = [_extract_bm25_terms(toks) for toks in sent_tokens]
    num_sents = len(sentences)
    avgdl = sum(len(toks) for toks in sent_tokens) / max(num_sents, 1)

    df: Counter[str] = Counter()
    for terms in sent_terms:
        for term in set(terms):
            df[term] += 1

    scores: list[float] = []
    for idx, terms in enumerate(sent_terms):
        tf = Counter(terms)
        dl = len(sent_tokens[idx])
        norm_denom = _BM25_K1 * (1.0 - _BM25_B + _BM25_B * dl / max(avgdl, 1.0))
        score = 0.0
        for q_tok in q_unigrams:
            freq = tf.get(q_tok, 0)
            if freq == 0:
                continue
            idf = math.log(1.0 + (num_sents - df[q_tok] + 0.5) / (df[q_tok] + 0.5))
            score += idf * (freq * (_BM25_K1 + 1.0)) / (freq + norm_denom)
        for q_bg in q_bigrams:
            freq = tf.get(q_bg, 0)
            if freq == 0:
                continue
            idf = math.log(1.0 + (num_sents - df[q_bg] + 0.5) / (df[q_bg] + 0.5))
            score += _BIGRAM_WEIGHT * idf * (freq * (_BM25_K1 + 1.0)) / (freq + norm_denom)
        scores.append(score)
    return scores


def select_evidence_from_sentences(
    statement: str,
    sentences: Sequence[str],
    *,
    top_k: int = _DEFAULT_TOP_K_SENTENCES,
    max_tokens: int = _DEFAULT_MAX_EVIDENCE_TOKENS,
    scorer: SentenceScorerProtocol | None = None,
) -> SelectedEvidence:
    """Select top-k relevant sentences from pre-split ``sentences``, preserving original order."""
    clean_sents: list[str] = []
    seen: set[str] = set()
    for s in sentences:
        s_norm = s.strip()
        if not s_norm:
            continue
        key = s_norm.lower()
        if key not in seen:
            seen.add(key)
            clean_sents.append(s_norm)

    if not clean_sents:
        return SelectedEvidence(
            text="",
            sentence_indices=[],
            num_sentences=0,
            word_count=0,
            method="reranker" if scorer is not None else "bm25",
        )

    method: Literal["bm25", "reranker"]
    if scorer is not None:
        scores = scorer.score_pairs(normalize_text(statement), clean_sents)
        method = "reranker"
    else:
        scores = score_sentences_bm25(statement, clean_sents)
        method = "bm25"

    word_lists = [s.split() for s in clean_sents]
    ranked = sorted(range(len(clean_sents)), key=lambda i: (-scores[i], i))

    chosen_indices: list[int] = []
    chosen_texts: dict[int, str] = {}
    total_words = 0

    for idx in ranked:
        if len(chosen_indices) >= top_k:
            break
        words = word_lists[idx]
        w_len = len(words)
        if not chosen_indices and w_len > max_tokens:
            truncated = " ".join(words[:max_tokens])
            chosen_indices.append(idx)
            chosen_texts[idx] = truncated
            total_words += max_tokens
            break
        if total_words + w_len <= max_tokens:
            chosen_indices.append(idx)
            chosen_texts[idx] = clean_sents[idx]
            total_words += w_len

    chosen_indices.sort()
    joined = " ".join(chosen_texts[i] for i in chosen_indices)
    return SelectedEvidence(
        text=joined,
        sentence_indices=chosen_indices,
        num_sentences=len(chosen_indices),
        word_count=total_words,
        method=method,
    )


def select_evidence_from_context(
    statement: str,
    context: str,
    *,
    top_k: int = _DEFAULT_TOP_K_SENTENCES,
    max_tokens: int = _DEFAULT_MAX_EVIDENCE_TOKENS,
    scorer: SentenceScorerProtocol | None = None,
) -> SelectedEvidence:
    """Select top-k relevant sentences from ``context``, preserving original order.

    Args:
        statement: Claim statement.
        context: Full article context (or web page text).
        top_k: Maximum number of sentences to select (default 3).
        max_tokens: Maximum total word count across selected sentences (~150).
        scorer: Optional cross-encoder scorer; defaults to fast BM25.

    Returns:
        :class:`SelectedEvidence` with joined sentences in original article order.
    """
    clean_ctx = normalize_text(context)
    raw_sentences = split_sentences(clean_ctx)
    return select_evidence_from_sentences(
        statement,
        raw_sentences,
        top_k=top_k,
        max_tokens=max_tokens,
        scorer=scorer,
    )


def select_evidence_with_sources(
    statement: str,
    sourced_contexts: Sequence[tuple[str, str, str]],
    *,
    top_k: int = _DEFAULT_TOP_K_SENTENCES,
    max_tokens: int = _DEFAULT_MAX_EVIDENCE_TOKENS,
    scorer: SentenceScorerProtocol | None = None,
) -> SourcedEvidenceResult:
    """Select top-k sentences across multiple ``(page_text, url, domain)`` sources via BM25.

    Splits sentences per source, deduplicates case-insensitively preserving the first
    source metadata, runs the exact Phase 3 selector, and retains per-sentence
    ``{text, url, domain}`` metadata (Brief #5 §4.4 & §12.3.5).
    """
    unique_items: list[SourcedSentence] = []
    seen_keys: set[str] = set()

    for page_text, url, domain in sourced_contexts:
        clean_ctx = normalize_text(page_text)
        if not clean_ctx:
            continue
        for raw_sent in split_sentences(clean_ctx):
            s_norm = raw_sent.strip()
            if not s_norm:
                continue
            key = s_norm.lower()
            if key in seen_keys:
                continue
            seen_keys.add(key)
            unique_items.append(SourcedSentence(text=s_norm, url=url, domain=domain))

    if not unique_items:
        return SourcedEvidenceResult(
            text="",
            items=[],
            word_count=0,
            method="reranker" if scorer is not None else "bm25",
        )

    sel = select_evidence_from_sentences(
        statement,
        [item.text for item in unique_items],
        top_k=top_k,
        max_tokens=max_tokens,
        scorer=scorer,
    )

    chosen_items: list[SourcedSentence] = []
    if len(sel.sentence_indices) == 1:
        # Handle single sentence possibly truncated to max_tokens
        orig = unique_items[sel.sentence_indices[0]]
        chosen_items.append(SourcedSentence(text=sel.text, url=orig.url, domain=orig.domain))
    else:
        for idx in sel.sentence_indices:
            chosen_items.append(unique_items[idx])

    return SourcedEvidenceResult(
        text=sel.text,
        items=chosen_items,
        word_count=sel.word_count,
        method=sel.method,
    )
