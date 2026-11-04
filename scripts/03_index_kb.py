"""Phase 2 — CLI wrapper around :func:`vnfake.retrieval.index_builder.build_index`.

Usage:
    uv run python scripts/03_index_kb.py
    uv run python scripts/03_index_kb.py --force
    uv run python scripts/03_index_kb.py --no-progress --chunks data/processed/kb_chunks.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vnfake.config import load_config
from vnfake.retrieval.embedder import BgeM3Embedder
from vnfake.retrieval.index_builder import build_index
from vnfake.retrieval.vector_store import QdrantStore


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Phase 2 — build KB index")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--chunks",
        type=Path,
        default=None,
        help="Chunks JSONL (default: <processed_dir>/kb_chunks.jsonl)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete + recreate the collection if it already exists",
    )
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    chunks_path = args.chunks or (cfg.paths.processed_dir / "kb_chunks.jsonl")

    embedder = BgeM3Embedder(
        model_name=cfg.retrieval.embedding.model_name,
        dim=cfg.retrieval.embedding.dim,
        use_fp16=cfg.retrieval.embedding.use_fp16,
    )
    store = QdrantStore.from_env()

    stats = build_index(
        cfg,
        chunks_path=chunks_path,
        embedder=embedder,
        store=store,
        force=args.force,
        show_progress=not args.no_progress,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
