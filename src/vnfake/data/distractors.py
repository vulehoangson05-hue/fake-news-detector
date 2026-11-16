"""Distractor / unlinked evidence generation for robustness testing (Phase 4, Brief #5 §2).

Generates two kinds of unlinked evidence from disjoint articles (excluding claim context):
1. ``random``: sentences picked from a randomly chosen different article.
2. ``hard_bm25``: top sentences from another article with highest BM25 similarity to Statement.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from vnfake.data.chunking import split_sentences
from vnfake.data.segmentation import SegmenterProtocol
from vnfake.retrieval.evidence_selector import (
    compute_3gram_containment,
    select_evidence_from_context,
    select_evidence_from_sentences,
    tokenize_words,
)
from vnfake.utils.logging import get_logger

logger = get_logger(__name__)


def _context_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class ArticleInfo:
    """An article context with its pre-split sentences and tokenized terms."""

    article_id: str
    context_text: str
    sentences: list[str]


@dataclass(frozen=True, slots=True)
class DistractorPair:
    """One Statement-Evidence pair with unlinked/distractor evidence."""

    sample_idx: int
    statement: str
    statement_segmented: str
    distractor_evidence: str
    distractor_evidence_segmented: str
    distractor_type: Literal["random", "hard_bm25", "same_article"]
    source_article_id: str
    original_label: int
    target_label: int  # Always 2 (unverifiable)
    containment_with_gold: float = 0.0


class DistractorPool:
    """Indexes articles to efficiently sample random and BM25 hard-negative distractors."""

    def __init__(self, articles: Sequence[ArticleInfo]) -> None:
        self.articles = list(articles)
        self.article_by_id = {a.article_id: a for a in self.articles}
        self.article_ids = list(self.article_by_id.keys())

        # Build sentence inverted index for fast global BM25 hard negative search
        self._doc_lens: list[int] = []
        self._sent_corpus: list[tuple[str, str, int]] = []  # (article_id, sentence_text, sent_idx)
        self._sent_tf: list[dict[str, int]] = []
        self._inv_index: defaultdict[str, list[int]] = defaultdict(list)
        self._df: defaultdict[str, int] = defaultdict(int)

        for a in self.articles:
            for s_idx, s in enumerate(a.sentences):
                tokens = tokenize_words(s)
                d_idx = len(self._sent_corpus)
                self._doc_lens.append(len(tokens))
                self._sent_corpus.append((a.article_id, s, s_idx))
                tf = Counter(tokens)
                self._sent_tf.append(dict(tf))
                for term in tf:
                    self._df[term] += 1
                    self._inv_index[term].append(d_idx)

        self.num_docs = len(self._sent_corpus)
        self.avg_doc_len = float(sum(self._doc_lens) / max(self.num_docs, 1))

    @classmethod
    def from_records(cls, records: Sequence[dict[str, Any]]) -> DistractorPool:
        """Create pool from split records containing ``Context``."""
        seen_articles: dict[str, ArticleInfo] = {}
        for r in records:
            ctx = str(r.get("Context", "")).strip()
            if not ctx:
                continue
            a_id = _context_hash(ctx)
            if a_id not in seen_articles:
                sents = split_sentences(ctx)
                # Keep sentences that have at least 3 words
                clean_sents = [s.strip() for s in sents if len(tokenize_words(s)) >= 3]
                if clean_sents:
                    seen_articles[a_id] = ArticleInfo(
                        article_id=a_id,
                        context_text=ctx,
                        sentences=clean_sents,
                    )
        return cls(list(seen_articles.values()))

    def sample_random_distractor(
        self,
        statement: str,
        exclude_article_id: str,
        rng: random.Random,
    ) -> tuple[str, str]:
        """Return (distractor_evidence_text, source_article_id) from another article."""
        candidates = [aid for aid in self.article_ids if aid != exclude_article_id]
        if not candidates:
            raise ValueError("No available articles to sample distractor from!")
        chosen_id = rng.choice(candidates)
        art = self.article_by_id[chosen_id]

        # Use select_evidence_from_context with BM25 on the chosen article
        res = select_evidence_from_context(statement, art.context_text)
        res_text = res.text if res.text.strip() else " ".join(art.sentences[:3])
        return res_text, chosen_id

    def find_hard_negative_distractor(
        self,
        statement: str,
        exclude_article_id: str,
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> tuple[str, str]:
        """Find the article (not excluded) with highest BM25 match to statement, return its evidence."""
        q_tokens = tokenize_words(statement)
        if not q_tokens:
            candidates = [aid for aid in self.article_ids if aid != exclude_article_id]
            chosen_id = candidates[0] if candidates else exclude_article_id
            return " ".join(self.article_by_id[chosen_id].sentences[:3]), chosen_id

        q_terms = Counter(q_tokens)
        candidate_d_indices: set[int] = set()
        for term in q_terms:
            for d_idx in self._inv_index.get(term, []):
                if self._sent_corpus[d_idx][0] != exclude_article_id:
                    candidate_d_indices.add(d_idx)

        if not candidate_d_indices:
            candidates = [aid for aid in self.article_ids if aid != exclude_article_id]
            chosen_id = candidates[0] if candidates else exclude_article_id
            return " ".join(self.article_by_id[chosen_id].sentences[:3]), chosen_id

        best_score = -1.0
        best_article_id: str | None = None

        for d_idx in candidate_d_indices:
            art_id = self._sent_corpus[d_idx][0]
            d_len = self._doc_lens[d_idx]
            tf_map = self._sent_tf[d_idx]

            score = 0.0
            for term, q_tf in q_terms.items():
                if term in tf_map:
                    tf = tf_map[term]
                    df = self._df[term]
                    idf = max(0.0, (self.num_docs - df + 0.5) / (df + 0.5))
                    num = tf * (k1 + 1.0)
                    denom = tf + k1 * (1.0 - b + b * (d_len / max(self.avg_doc_len, 1e-6)))
                    score += idf * (num / max(denom, 1e-6)) * q_tf

            if score > best_score:
                best_score = score
                best_article_id = art_id

        if best_article_id is None:
            candidates = [aid for aid in self.article_ids if aid != exclude_article_id]
            best_article_id = candidates[0]

        best_art = self.article_by_id[best_article_id]
        res = select_evidence_from_context(statement, best_art.context_text)
        res_text = res.text if res.text.strip() else " ".join(best_art.sentences[:3])
        return res_text, best_article_id


def generate_dev_distractors(
    dev_records: Sequence[dict[str, Any]],
    pool: DistractorPool,
    *,
    seed: int = 42,
    segmenter: SegmenterProtocol | None = None,
) -> list[DistractorPair]:
    """Generate 2 distractor pairs (1 random, 1 hard BM25) for every dev record."""
    rng = random.Random(seed)
    pairs: list[DistractorPair] = []

    for idx, r in enumerate(dev_records):
        s_idx = int(r.get("sample_idx", idx))
        stmt = str(r["Statement"]).strip()
        orig_lbl = int(r["labels"])
        ctx = str(r.get("Context", "")).strip()
        gold_ev = str(r.get("Evidence", "")).strip()
        art_id = _context_hash(ctx)

        stmt_seg = (
            str(r.get("statement_segmented", ""))
            if r.get("statement_segmented")
            else (segmenter.segment(stmt) if segmenter else stmt)
        )

        # 1. Random distractor
        rand_ev, rand_src = pool.sample_random_distractor(stmt, art_id, rng)
        rand_seg = segmenter.segment(rand_ev) if segmenter else rand_ev
        rand_containment = compute_3gram_containment(gold_ev, rand_ev)
        pairs.append(
            DistractorPair(
                sample_idx=s_idx,
                statement=stmt,
                statement_segmented=stmt_seg,
                distractor_evidence=rand_ev,
                distractor_evidence_segmented=rand_seg,
                distractor_type="random",
                source_article_id=rand_src,
                original_label=orig_lbl,
                target_label=2,  # unverifiable
                containment_with_gold=rand_containment,
            )
        )

        # 2. Hard BM25 distractor
        hard_ev, hard_src = pool.find_hard_negative_distractor(stmt, art_id)
        hard_seg = segmenter.segment(hard_ev) if segmenter else hard_ev
        hard_containment = compute_3gram_containment(gold_ev, hard_ev)
        pairs.append(
            DistractorPair(
                sample_idx=s_idx,
                statement=stmt,
                statement_segmented=stmt_seg,
                distractor_evidence=hard_ev,
                distractor_evidence_segmented=hard_seg,
                distractor_type="hard_bm25",
                source_article_id=hard_src,
                original_label=orig_lbl,
                target_label=2,  # unverifiable
                containment_with_gold=hard_containment,
            )
        )

    return pairs


def generate_train_distractors(
    train_records: Sequence[dict[str, Any]],
    pool: DistractorPool,
    target_count: int,
    *,
    seed: int = 42,
    segmenter: SegmenterProtocol | None = None,
) -> list[DistractorPair]:
    """Generate ``target_count`` distractor pairs (50% random, 50% hard BM25) strictly from train."""
    rng = random.Random(seed)
    n_rand = target_count // 2
    n_hard = target_count - n_rand

    # Subsample records for random and hard distractors
    shuffled_records = list(train_records)
    rng.shuffle(shuffled_records)

    pairs: list[DistractorPair] = []

    # Random distractors
    for idx, r in enumerate(shuffled_records[:n_rand]):
        s_idx = int(r.get("sample_idx", idx))
        stmt = str(r["Statement"]).strip()
        orig_lbl = int(r["labels"])
        ctx = str(r.get("Context", "")).strip()
        gold_ev = str(r.get("Evidence", "")).strip()
        art_id = _context_hash(ctx)

        stmt_seg = (
            str(r.get("statement_segmented", ""))
            if r.get("statement_segmented")
            else (segmenter.segment(stmt) if segmenter else stmt)
        )

        rand_ev, rand_src = pool.sample_random_distractor(stmt, art_id, rng)
        rand_seg = segmenter.segment(rand_ev) if segmenter else rand_ev
        rand_containment = compute_3gram_containment(gold_ev, rand_ev)
        pairs.append(
            DistractorPair(
                sample_idx=s_idx,
                statement=stmt,
                statement_segmented=stmt_seg,
                distractor_evidence=rand_ev,
                distractor_evidence_segmented=rand_seg,
                distractor_type="random",
                source_article_id=rand_src,
                original_label=orig_lbl,
                target_label=2,
                containment_with_gold=rand_containment,
            )
        )

    # Hard BM25 distractors
    for idx, r in enumerate(shuffled_records[n_rand : n_rand + n_hard], start=n_rand):
        s_idx = int(r.get("sample_idx", idx))
        stmt = str(r["Statement"]).strip()
        orig_lbl = int(r["labels"])
        ctx = str(r.get("Context", "")).strip()
        gold_ev = str(r.get("Evidence", "")).strip()
        art_id = _context_hash(ctx)

        stmt_seg = (
            str(r.get("statement_segmented", ""))
            if r.get("statement_segmented")
            else (segmenter.segment(stmt) if segmenter else stmt)
        )

        hard_ev, hard_src = pool.find_hard_negative_distractor(stmt, art_id)
        hard_seg = segmenter.segment(hard_ev) if segmenter else hard_ev
        hard_containment = compute_3gram_containment(gold_ev, hard_ev)
        pairs.append(
            DistractorPair(
                sample_idx=s_idx,
                statement=stmt,
                statement_segmented=stmt_seg,
                distractor_evidence=hard_ev,
                distractor_evidence_segmented=hard_seg,
                distractor_type="hard_bm25",
                source_article_id=hard_src,
                original_label=orig_lbl,
                target_label=2,
                containment_with_gold=hard_containment,
            )
        )

    return pairs


def filter_distractors_by_containment(
    pairs: Sequence[DistractorPair],
    gold_map: dict[int, str] | None = None,
    *,
    max_containment: float = 0.3,
) -> list[DistractorPair]:
    """Filter out distractor pairs whose 3-gram containment with gold evidence is >= max_containment."""
    filtered: list[DistractorPair] = []
    for p in pairs:
        c = p.containment_with_gold
        if gold_map is not None and p.sample_idx in gold_map:
            c = compute_3gram_containment(gold_map[p.sample_idx], p.distractor_evidence)
        if c < max_containment:
            filtered.append(p)
    return filtered


def generate_same_article_distractors(
    records: Sequence[dict[str, Any]],
    *,
    segmenter: SegmenterProtocol | None = None,
    max_containment: float = 0.3,
) -> list[DistractorPair]:
    """Generate 'same article, different sentence' distractors (Brief #5 §11.4).

    For each record, filters out sentences from Context that have 3-gram containment >= max_containment
    with gold Evidence, then picks top-3 sentences (preserving original order, <= 150 words) using BM25.
    """
    pairs: list[DistractorPair] = []
    for idx, r in enumerate(records):
        s_idx = int(r.get("sample_idx", idx))
        stmt = str(r["Statement"]).strip()
        orig_lbl = int(r["labels"])
        ctx = str(r.get("Context", "")).strip()
        gold_ev = str(r.get("Evidence", "")).strip()
        art_id = _context_hash(ctx)

        stmt_seg = (
            str(r.get("statement_segmented", ""))
            if r.get("statement_segmented")
            else (segmenter.segment(stmt) if segmenter else stmt)
        )

        sents = split_sentences(ctx)
        # Exclude sentences that contain >= max_containment with gold Evidence
        cand_sents = [s for s in sents if compute_3gram_containment(gold_ev, s) < max_containment]

        if not cand_sents:
            # Fallback to least-overlapping sentences
            scored_cand = sorted(
                sents, key=lambda s: compute_3gram_containment(gold_ev, s)
            )
            cand_sents = scored_cand[:3]

        sel = select_evidence_from_sentences(stmt, cand_sents, top_k=3, max_tokens=150)
        ev_text = sel.text.strip()
        if not ev_text:
            ev_text = " ".join(cand_sents[:3])

        ev_seg = segmenter.segment(ev_text) if segmenter else ev_text
        containment = compute_3gram_containment(gold_ev, ev_text)

        pairs.append(
            DistractorPair(
                sample_idx=s_idx,
                statement=stmt,
                statement_segmented=stmt_seg,
                distractor_evidence=ev_text,
                distractor_evidence_segmented=ev_seg,
                distractor_type="same_article",
                source_article_id=art_id,
                original_label=orig_lbl,
                target_label=2,  # unverifiable
                containment_with_gold=containment,
            )
        )
    return pairs
