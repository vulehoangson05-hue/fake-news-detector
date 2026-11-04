"""Phase 1 — CLI wrapper around :func:`vnfake.data.prepare.run_prepare`.

Usage:
    uv run python scripts/02_prepare_data.py
    uv run python scripts/02_prepare_data.py --skip-test
    uv run python scripts/02_prepare_data.py --skip-segmentation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vnfake.config import load_config
from vnfake.data.prepare import run_prepare
from vnfake.data.segmentation import get_segmenter


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Phase 1 data preparation")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--skip-segmentation", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    segmenter = get_segmenter(cfg.segmentation.backend, str(cfg.segmentation.vncorenlp_dir))
    stats = run_prepare(
        cfg,
        segmenter=segmenter,
        skip_test=args.skip_test,
        skip_segmentation=args.skip_segmentation,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
