"""Phase 0 — measure overlap between ViFactCheck gold Evidence and KB corpus.

For each Evidence in the given split, compute the maximum *containment*
(share of evidence word-n-grams appearing in any single KB document). Report
coverage at several thresholds so we can decide whether retrieval evaluation
can use gold evidence directly.

Containment (not Jaccard) is used because evidence is much shorter than the
KB documents — Jaccard would unfairly penalize short-in-long matches.

Usage:
    uv run python scripts/01b_verify_overlap.py --split dev
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path

from vnfake.config import AppConfig, load_config
from vnfake.utils.logging import get_logger

logger = get_logger(__name__)

_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)


def _iter_jsonl(path: Path) -> Iterator[dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skip malformed %s:%d — %s", path.name, lineno, exc)
                continue
            if isinstance(obj, dict):
                yield obj


def _word_shingles(text: str, n: int) -> set[str]:
    """Return the set of word n-grams (lowercased, punctuation stripped)."""
    tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    if not tokens:
        return set()
    if len(tokens) < n:
        return set(tokens)
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _load_kb_texts(cfg: AppConfig) -> list[tuple[str, str]]:
    """Return ``[(doc_id, 'title\\ntext'), ...]`` for both KB files."""
    docs: list[tuple[str, str]] = []
    for path in (cfg.paths.kb_dantri, cfg.paths.kb_vnexpress):
        for i, rec in enumerate(_iter_jsonl(path)):
            doc_id = str(rec.get("id") or f"{path.stem}-{i}")
            title = str(rec.get("title") or "")
            text = str(rec.get("text") or "")
            docs.append((doc_id, f"{title}\n{text}"))
    return docs


def _build_index(
    shingle_sets: list[set[str]],
) -> dict[str, list[int]]:
    """Inverted index: shingle → list of KB document indices."""
    inv: dict[str, list[int]] = defaultdict(list)
    for idx, shs in enumerate(shingle_sets):
        for sh in shs:
            inv[sh].append(idx)
    return inv


def _max_containment(
    ev_shingles: set[str],
    kb_sets: list[set[str]],
    inv: dict[str, list[int]],
    top_k: int,
) -> float:
    """Best containment of ``ev_shingles`` in any single KB doc (capped to top-k candidates)."""
    if not ev_shingles:
        return 0.0
    shared: Counter[int] = Counter()
    for sh in ev_shingles:
        for doc_idx in inv.get(sh, ()):
            shared[doc_idx] += 1
    if not shared:
        return 0.0
    best = 0.0
    for doc_idx, _ in shared.most_common(top_k):
        inter = len(ev_shingles & kb_sets[doc_idx])
        cont = inter / len(ev_shingles)
        if cont > best:
            best = cont
    return best


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. Restricts to ``dev``/``train`` (test blocked — leakage rule)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        default="dev",
        choices=["train", "dev"],
        help="Which ViFactCheck split to analyze (test is blocked by leakage policy).",
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to base.yaml")
    parser.add_argument("--ngram", type=int, default=3, help="Word n-gram size")
    parser.add_argument(
        "--top-k",
        type=int,
        default=200,
        help="Max candidate KB docs per evidence to verify (speed vs exactness)",
    )
    args = parser.parse_args(argv)

    cfg: AppConfig = load_config(args.config)
    split_path = cfg.paths.vifactcheck_dir / f"{args.split}.jsonl"

    logger.info("Loading KB from %s and %s", cfg.paths.kb_dantri, cfg.paths.kb_vnexpress)
    kb = _load_kb_texts(cfg)
    logger.info("KB docs: %d — building %d-gram sets", len(kb), args.ngram)
    kb_sets = [_word_shingles(text, args.ngram) for _, text in kb]
    inv = _build_index(kb_sets)
    logger.info("Inverted index: %d unique shingles", len(inv))

    evidences: list[str] = []
    for rec in _iter_jsonl(split_path):
        ev = str(rec.get(cfg.vifactcheck.evidence_field) or "").strip()
        evidences.append(ev)
    logger.info("Split=%s — %d evidences", args.split, len(evidences))

    containments: list[float] = []
    for i, ev in enumerate(evidences, start=1):
        containments.append(
            _max_containment(_word_shingles(ev, args.ngram), kb_sets, inv, args.top_k)
        )
        if i % 200 == 0:
            logger.info("  processed %d/%d", i, len(evidences))

    n = len(containments)
    counts = {t: sum(1 for c in containments if c >= t) for t in _THRESHOLDS}

    print()
    print(f"# Overlap report — split={args.split}, ngram={args.ngram}, n={n}")
    print()
    print("| Ngưỡng containment | % Evidence có trong KB |")
    print("|---|---:|")
    for t in _THRESHOLDS:
        pct = 100.0 * counts[t] / n if n else 0.0
        print(f"| ≥ {t:.1f} | {pct:.1f}% |")
    print()
    print(f"Max containment (all): {max(containments):.3f}")
    print(f"Mean containment    : {sum(containments) / n:.3f}")
    print()
    print(
        "Hướng dẫn diễn giải: nếu '% ≥ 0.7' cao (>=30%) thì recall@k đo trực tiếp"
        " được trên gold evidence; nếu thấp, dùng proxy (entity/n-gram) và ghi vào"
        " docs/DATA_AUDIT.md §6 + Decision Log."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
