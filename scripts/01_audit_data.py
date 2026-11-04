"""Phase 0 data audit.

Reads KB JSONL files and ViFactCheck splits, prints summary tables (record
counts, label distribution, text-length stats). Output feeds the open
questions in ``docs/ARCHITECTURE.md`` §10.

Usage:
    uv run python scripts/01_audit_data.py [--config configs/base.yaml]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd

from vnfake.config import AppConfig, load_config
from vnfake.utils.logging import get_logger

logger = get_logger(__name__)


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield parsed objects from a JSONL file, skipping malformed lines."""
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


def _load_jsonl_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        logger.warning("File not found: %s", path)
        return pd.DataFrame()
    return pd.DataFrame(list(_iter_jsonl(path)))


def _length_stats(df: pd.DataFrame, column: str) -> dict[str, float]:
    if df.empty or column not in df.columns:
        return {}
    s = df[column].astype(str).str.len()
    return {
        "count": float(s.count()),
        "min": float(s.min()),
        "mean": round(float(s.mean()), 1),
        "median": float(s.median()),
        "p95": round(float(s.quantile(0.95)), 1),
        "max": float(s.max()),
    }


def audit_kb(name: str, path: Path) -> dict[str, Any]:
    """Summarize one KB JSONL file."""
    df = _load_jsonl_df(path)
    row: dict[str, Any] = {"source": name, "path": str(path), "records": len(df)}
    if df.empty:
        return row
    row["empty_text"] = (
        int((df["text"].astype(str).str.len() == 0).sum()) if "text" in df.columns else "n/a"
    )
    row["dup_urls"] = int(df["url"].duplicated().sum()) if "url" in df.columns else "n/a"
    stats = _length_stats(df, "text")
    row["text_mean_len"] = stats.get("mean", "n/a")
    row["text_p95_len"] = stats.get("p95", "n/a")
    return row


def audit_vifactcheck(name: str, path: Path, label_field: str) -> dict[str, Any]:
    """Summarize one ViFactCheck split file."""
    df = _load_jsonl_df(path)
    row: dict[str, Any] = {"split": name, "path": str(path), "records": len(df)}
    if df.empty:
        return row
    if label_field in df.columns:
        counts = Counter(df[label_field].tolist())
        row["labels"] = {
            str(k): int(v) for k, v in sorted(counts.items(), key=lambda kv: str(kv[0]))
        }
    row["stmt_mean_len"] = _length_stats(df, "Statement").get("mean", "n/a")
    row["evid_mean_len"] = _length_stats(df, "Evidence").get("mean", "n/a")
    return row


def _print_table(title: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        print(f"\n== {title} ==\n(no data)")
        return
    print(f"\n== {title} ==")
    print(pd.DataFrame(rows).to_string(index=False))


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for Phase 0 data audit."""
    parser = argparse.ArgumentParser(description="Phase 0 data audit")
    parser.add_argument("--config", type=Path, default=None, help="Path to base.yaml")
    args = parser.parse_args(argv)

    cfg: AppConfig = load_config(args.config)

    print("Data audit — Vietnamese Fake News Detection")
    print(f"seed={cfg.seed} | label_map={cfg.label_map}  (placeholder — verify before Phase 1)")

    _print_table(
        "KB files",
        [
            audit_kb("dantri", cfg.paths.kb_dantri),
            audit_kb("vnexpress", cfg.paths.kb_vnexpress),
        ],
    )

    _print_table(
        "ViFactCheck splits",
        [
            audit_vifactcheck(
                split, cfg.paths.vifactcheck_dir / f"{split}.jsonl", cfg.vifactcheck.label_field
            )
            for split in ("train", "dev", "test")
        ],
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
