"""CLI entrypoint to train and evaluate Verifier models (Phase 3, Brief #4 §2.4-§2.8).

Examples:
    python scripts/10_train_verifier.py --variant baseline --seed 42
    python scripts/10_train_verifier.py --variant V-A --seed 42
    python scripts/10_train_verifier.py --variant V-B --seed 42
    python scripts/10_train_verifier.py --variant V-C --seed 42
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import cast

os.environ.setdefault("HF_HOME", r"D:\hf-cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from vnfake.config import load_config
from vnfake.training.train import VariantName, train_and_evaluate_verifier
from vnfake.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Verifier on ViFactCheck (Phase 3).")
    parser.add_argument(
        "--variant",
        type=str,
        choices=["V-A", "V-B", "V-C", "V-D", "baseline"],
        default="V-A",
        help="Training variant (V-A=gold, V-B=selected, V-C=50/50 mix, V-D=distractor mix, baseline=claim-only).",
    )
    parser.add_argument("--model", type=str, default=None, help="Override HF model name.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_train_epochs.")
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override per-device batch size."
    )
    parser.add_argument(
        "--grad-accum", type=int, default=None, help="Override gradient accumulation."
    )
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument(
        "--fp16", action="store_true", help="Force fp16 mixed precision (e.g. Colab T4)."
    )
    parser.add_argument(
        "--max-train-samples", type=int, default=None, help="Optional cap for smoke testing."
    )
    parser.add_argument("--run-id", type=str, default=None, help="Custom run identifier.")
    parser.add_argument("--online-hf", action="store_true", help="Allow HF Hub network downloads.")
    parser.add_argument(
        "--train-split", type=Path, default=None, help="Custom path to train selected jsonl."
    )
    parser.add_argument(
        "--val-split", type=Path, default=None, help="Custom path to val selected jsonl."
    )
    parser.add_argument(
        "--holdout-split", type=Path, default=None, help="Custom path to holdout selected jsonl."
    )
    args = parser.parse_args()

    cfg = load_config()
    fp16_override = True if args.fp16 else None

    train_and_evaluate_verifier(
        cfg,
        variant=cast(VariantName, args.variant),
        model_name=args.model,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        fp16=fp16_override,
        max_train_samples=args.max_train_samples,
        run_id=args.run_id,
        local_files_only=not args.online_hf,
        train_split_path=args.train_split,
        val_split_path=args.val_split,
        holdout_split_path=args.holdout_split,
    )


if __name__ == "__main__":
    main()
