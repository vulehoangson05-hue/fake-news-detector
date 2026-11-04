"""Phase 0 — check duplicate Statements across ViFactCheck splits.

Mitigates risk R7 (ARCHITECTURE.md §8). Uses exact match on NFC-normalized,
lowercased Statement. Exit code 1 if any cross-split duplicate is found.

Usage:
    uv run python scripts/01c_check_leakage.py [--config configs/base.yaml]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections.abc import Iterator
from pathlib import Path

from vnfake.config import AppConfig, load_config
from vnfake.utils.logging import get_logger

logger = get_logger(__name__)

_SPLITS: tuple[str, ...] = ("train", "dev", "test")


def _iter_jsonl(path: Path) -> Iterator[dict[str, object]]:
    """Yield parsed JSON objects from a JSONL file, skipping malformed lines."""
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


def _hash_statement(text: str) -> str:
    """Return a stable hash of the NFC-normalized, lowercased Statement."""
    norm = unicodedata.normalize("NFC", text).strip().lower()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def _load_statements(path: Path, field: str) -> dict[str, int]:
    """Return ``{hash: count}`` for one split file."""
    out: dict[str, int] = {}
    for rec in _iter_jsonl(path):
        raw = rec.get(field)
        if not raw:
            continue
        h = _hash_statement(str(raw))
        out[h] = out.get(h, 0) + 1
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. Returns 1 if leakage is detected, else 0."""
    parser = argparse.ArgumentParser(description="Phase 0 leakage check")
    parser.add_argument("--config", type=Path, default=None, help="Path to base.yaml")
    args = parser.parse_args(argv)

    cfg: AppConfig = load_config(args.config)
    field = cfg.vifactcheck.statement_field

    per_split: dict[str, dict[str, int]] = {}
    for name in _SPLITS:
        path = cfg.paths.vifactcheck_dir / f"{name}.jsonl"
        if not path.exists():
            logger.error("Missing split file: %s", path)
            return 2
        per_split[name] = _load_statements(path, field)
        logger.info(
            "%s: rows=%d, unique statements=%d",
            name,
            sum(per_split[name].values()),
            len(per_split[name]),
        )

    print()
    print("| Cặp split | Statement trùng (exact) |")
    print("|---|---:|")
    contaminated = False
    for i, a in enumerate(_SPLITS):
        for b in _SPLITS[i + 1 :]:
            n = len(set(per_split[a]) & set(per_split[b]))
            flag = "⚠️" if n > 0 else "✅"
            print(f"| {a} ↔ {b} | {n} {flag} |")
            contaminated = contaminated or n > 0

    print()
    if contaminated:
        print("⚠️  PHÁT HIỆN LEAKAGE. Không mở Phase 3 trước khi loại Statement trùng.")
        return 1
    print("✅ Không phát hiện trùng Statement chéo split (exact match).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
