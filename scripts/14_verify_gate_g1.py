"""Gate G1 Verification Script (Brief #5 §1).

Verifies seed evaluation integrity:
1. Computes full SHA-256 of checkpoint weights for seeds 42, 43, 44.
2. Loads each checkpoint independently and predicts logits on dev gold & dev selected.
3. Computes pairwise prediction agreement rate:
   agree(s_i, s_j) = sum(y_i == y_j) / N * 100%
4. Reports confusion matrices for all 3 seeds on gold and selected.
5. Saves predictions to ``artifacts/g1_seed_predictions.json``.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HOME", r"D:\hf-cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

from vnfake.config import load_config
from vnfake.data.dataset import ModelType
from vnfake.data.segmentation import get_segmenter
from vnfake.models.classifier import VerifierClassifier
from vnfake.models.tokenization import encode_verifier_dataset
from vnfake.training.metrics import evaluate_verifier_logits
from vnfake.training.train import build_variant_texts, load_selected_split
from vnfake.utils.logging import get_logger

logger = get_logger(__name__)


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    cfg = load_config()
    models_dir = cfg.paths.artifacts_dir / "models"
    dev_records = load_selected_split(cfg.paths.interim_dir / "selected_dev.jsonl")

    # Locate V-C runs for seeds 42, 43, 44
    seeds = (42, 43, 44)
    run_dirs: dict[int, Path] = {}
    for d in sorted(models_dir.glob("phobert_v-c_*")):
        rep_file = d / "eval_report.json"
        if rep_file.exists():
            data = json.loads(rep_file.read_text(encoding="utf-8"))
            s = int(data.get("seed", -1))
            if s in seeds and s not in run_dirs:
                run_dirs[s] = d

    if len(run_dirs) < 3:
        raise RuntimeError(f"Could not locate all 3 runs for seeds 42, 43, 44: found {run_dirs}")

    logger.info("Found runs: %s", {s: p.name for s, p in run_dirs.items()})

    # 1. Print sha256 of checkpoints
    ckpt_hashes: dict[int, dict[str, str]] = {}
    for s in seeds:
        d = run_dirs[s]
        ckpt = d / "checkpoint"
        weight_file = ckpt / "model.safetensors"
        if not weight_file.exists():
            weight_file = ckpt / "pytorch_model.bin"
        h = compute_sha256(weight_file)
        ckpt_hashes[s] = {
            "dir": str(ckpt.as_posix()),
            "file": weight_file.name,
            "sha256": h,
            "size_bytes": str(weight_file.stat().st_size),
        }
        logger.info("Seed %d | File: %s | SHA256: %s", s, weight_file.name, h)

    # 2. Build dev datasets
    model_type: ModelType = "phobert"
    max_length = cfg.classifier.max_len_phobert
    segmenter = get_segmenter(cfg.segmentation.backend, str(cfg.segmentation.vncorenlp_dir))

    # Tokenizer from seed 42 checkpoint
    init_verifier = VerifierClassifier(
        run_dirs[42] / "checkpoint",
        cfg.label_map,
        local_files_only=True,
    )
    tokenizer = init_verifier.load_tokenizer()

    # dev_gold
    gold_s, gold_e, gold_y, _, gold_preseg = build_variant_texts(
        dev_records, "V-C", model_type=model_type, seed=42, eval_mode="gold"
    )
    dev_gold_ds = encode_verifier_dataset(
        gold_s,
        gold_e,
        gold_y,
        tokenizer=tokenizer,
        model_type=model_type,
        max_length=max_length,
        label_map=cfg.label_map,
        segmenter=segmenter,
        statement_only=False,
        presegmented=gold_preseg,
    )

    # dev_selected
    sel_s, sel_e, sel_y, _, sel_preseg = build_variant_texts(
        dev_records, "V-C", model_type=model_type, seed=42, eval_mode="selected"
    )
    dev_sel_ds = encode_verifier_dataset(
        sel_s,
        sel_e,
        sel_y,
        tokenizer=tokenizer,
        model_type=model_type,
        max_length=max_length,
        label_map=cfg.label_map,
        segmenter=segmenter,
        statement_only=False,
        presegmented=sel_preseg,
    )

    del init_verifier
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. Predict with each checkpoint
    preds_gold: dict[int, np.ndarray] = {}
    preds_selected: dict[int, np.ndarray] = {}
    eval_reports: dict[str, dict[int, Any]] = {"gold": {}, "selected": {}}

    for s in seeds:
        ckpt_dir = run_dirs[s] / "checkpoint"
        logger.info("Evaluating seed %d using checkpoint: %s", s, ckpt_dir)
        verifier = VerifierClassifier(
            ckpt_dir,
            cfg.label_map,
            local_files_only=True,
        )

        # Dev gold
        g_logits, g_true = verifier.predict_logits(dev_gold_ds, batch_size=16)
        g_pred = np.argmax(g_logits, axis=-1)
        preds_gold[s] = g_pred
        rep_g = evaluate_verifier_logits(g_logits, g_true, cfg.label_map)
        eval_reports["gold"][s] = rep_g

        # Dev selected
        s_logits, s_true = verifier.predict_logits(dev_sel_ds, batch_size=16)
        s_pred = np.argmax(s_logits, axis=-1)
        preds_selected[s] = s_pred
        rep_s = evaluate_verifier_logits(s_logits, s_true, cfg.label_map)
        eval_reports["selected"][s] = rep_s

        del verifier
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 4. Pairwise prediction agreement
    n_samples = len(dev_records)
    pairs = [(42, 43), (42, 44), (43, 44)]

    agreement_gold: dict[str, float] = {}
    for s1, s2 in pairs:
        match_count = int(np.sum(preds_gold[s1] == preds_gold[s2]))
        pct = round(match_count / n_samples * 100.0, 2)
        agreement_gold[f"{s1}-{s2}"] = pct
        logger.info("Agreement Gold %d vs %d: %d/%d (%.2f%%)", s1, s2, match_count, n_samples, pct)

    agreement_selected: dict[str, float] = {}
    for s1, s2 in pairs:
        match_count = int(np.sum(preds_selected[s1] == preds_selected[s2]))
        pct = round(match_count / n_samples * 100.0, 2)
        agreement_selected[f"{s1}-{s2}"] = pct
        logger.info(
            "Agreement Selected %d vs %d: %d/%d (%.2f%%)", s1, s2, match_count, n_samples, pct
        )

    # 5. Check accuracy and confusion matrix
    logger.info("=== SUMMARY OF ACCURACY & CONFUSION MATRICES ===")
    summary_results: dict[str, Any] = {
        "checkpoint_hashes": ckpt_hashes,
        "agreement_gold_pct": agreement_gold,
        "agreement_selected_pct": agreement_selected,
        "seeds": {},
    }

    for s in seeds:
        rg = eval_reports["gold"][s]
        rs = eval_reports["selected"][s]
        logger.info(
            "Seed %d: Gold Acc=%.4f (F1=%.4f), Selected Acc=%.4f (F1=%.4f)",
            s,
            rg.accuracy,
            rg.macro_f1,
            rs.accuracy,
            rs.macro_f1,
        )
        logger.info("Seed %d Gold CM: %s", s, rg.confusion_matrix)
        logger.info("Seed %d Selected CM: %s", s, rs.confusion_matrix)

        summary_results["seeds"][s] = {
            "gold": {
                "accuracy": rg.accuracy,
                "macro_f1": rg.macro_f1,
                "confusion_matrix": rg.confusion_matrix,
            },
            "selected": {
                "accuracy": rs.accuracy,
                "macro_f1": rs.macro_f1,
                "confusion_matrix": rs.confusion_matrix,
            },
            "predictions_gold": preds_gold[s].tolist(),
            "predictions_selected": preds_selected[s].tolist(),
        }

    # Compare predictions of seed 42 and seed 44 on selected
    diff_42_44_sel = int(np.sum(preds_selected[42] != preds_selected[44]))
    logger.info(
        "Differences between Seed 42 and Seed 44 on Selected: %d / %d samples",
        diff_42_44_sel,
        n_samples,
    )

    out_file = cfg.paths.artifacts_dir / "g1_seed_predictions.json"
    out_file.write_text(json.dumps(summary_results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved G1 verification report to %s", out_file)


if __name__ == "__main__":
    main()
