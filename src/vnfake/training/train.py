"""Verifier training, evaluation, calibration, and error analysis (Phase 3, Brief #4 §2.4-§2.8).

Supports all 4 variants defined in Brief #4 §2.4:
- ``V-A``: train on gold ``Evidence``; evaluate on dev (gold & selected).
- ``V-B``: train on ``selected`` sentences from ``Context``; evaluate on dev (selected & gold).
- ``V-C``: train on 50/50 deterministic mix of gold + selected; evaluate on dev (both).
- ``baseline``: train on ``Statement`` only (no evidence); evaluate on dev.
"""

from __future__ import annotations

import json
import random
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from vnfake.config import AppConfig
from vnfake.data.dataset import ModelType, build_class_mapping
from vnfake.data.segmentation import get_segmenter
from vnfake.models.calibration import CalibrationResult, TemperatureScaler
from vnfake.models.classifier import (
    VerifierClassifier,
    collate_pair_batch,
    infer_model_type,
)
from vnfake.models.tokenization import encode_verifier_dataset
from vnfake.training.metrics import (
    VerifierEvalReport,
    evaluate_verifier_logits,
    softmax_with_temperature,
)
from vnfake.utils.logging import get_logger
from vnfake.utils.seed import set_seed

logger = get_logger(__name__)

VariantName = Literal["V-A", "V-B", "V-C", "V-D", "baseline"]
_VALID_VARIANTS: tuple[VariantName, ...] = ("V-A", "V-B", "V-C", "V-D", "baseline")


@dataclass(frozen=True, slots=True)
class PreparedSplitRecord:
    """One sample loaded from ``data/interim/selected_{split}.jsonl``."""

    sample_idx: int
    statement: str
    gold_evidence: str
    selected_evidence: str
    label: int
    statement_segmented: str
    gold_evidence_segmented: str
    selected_evidence_segmented: str


@dataclass(frozen=True, slots=True)
class ErrorSample:
    """One representative misclassified sample on ``dev``."""

    sample_idx: int
    statement: str
    evidence: str
    true_label: str
    predicted_label: str
    confidence: float


def load_selected_split(path: Path) -> list[PreparedSplitRecord]:
    """Load pre-selected and pre-segmented split from ``data/interim/selected_*.jsonl``."""
    if not path.exists():
        raise FileNotFoundError(
            f"Selected split not found at {path}. Run 'scripts/09_select_evidence.py' first."
        )
    records: list[PreparedSplitRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            records.append(
                PreparedSplitRecord(
                    sample_idx=int(obj["sample_idx"]),
                    statement=str(obj["Statement"]),
                    gold_evidence=str(obj["Evidence"]),
                    selected_evidence=str(obj["selected_evidence"]),
                    label=int(obj["labels"]),
                    statement_segmented=str(obj.get("statement_segmented", "")),
                    gold_evidence_segmented=str(obj.get("gold_evidence_segmented", "")),
                    selected_evidence_segmented=str(obj.get("selected_evidence_segmented", "")),
                )
            )
    return records


def build_variant_texts(
    records: Sequence[PreparedSplitRecord],
    variant: VariantName,
    *,
    model_type: ModelType,
    seed: int,
    eval_mode: Literal["gold", "selected", "statement_only"] | None = None,
) -> tuple[list[str], list[str], list[int], bool, bool]:
    """Return ``(statements, evidences, labels, statement_only, presegmented)`` for a variant."""
    if variant not in _VALID_VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; expected one of {_VALID_VARIANTS}")

    use_seg = model_type == "phobert" and bool(records) and bool(records[0].statement_segmented)
    statements = [(r.statement_segmented if use_seg else r.statement) for r in records]
    labels = [r.label for r in records]

    if eval_mode == "statement_only" or (eval_mode is None and variant == "baseline"):
        return statements, ["" for _ in records], labels, True, use_seg

    if eval_mode == "gold" or (eval_mode is None and variant == "V-A"):
        evidences = [(r.gold_evidence_segmented if use_seg else r.gold_evidence) for r in records]
        return statements, evidences, labels, False, use_seg

    if eval_mode == "selected" or (eval_mode is None and variant == "V-B"):
        evidences = [
            (r.selected_evidence_segmented if use_seg else r.selected_evidence) for r in records
        ]
        return statements, evidences, labels, False, use_seg

    # V-C and V-D training: deterministic 50/50 mix of gold and selected
    rng = random.Random(seed)
    evidences = []
    for r in records:
        choose_gold = rng.random() < 0.5
        if choose_gold:
            evidences.append(r.gold_evidence_segmented if use_seg else r.gold_evidence)
        else:
            evidences.append(r.selected_evidence_segmented if use_seg else r.selected_evidence)
    return statements, evidences, labels, False, use_seg


def extract_top_errors(
    records: Sequence[PreparedSplitRecord],
    logits: np.ndarray,
    label_map: Mapping[int, str],
    *,
    temperature: float = 1.0,
    evidence_source: Literal["gold", "selected", "statement_only"] = "gold",
    top_n: int = 10,
) -> list[ErrorSample]:
    """Return top-N high-confidence misclassifications on ``dev``."""
    class_names, src_to_class = build_class_mapping(label_map)
    probs = softmax_with_temperature(logits, temperature=temperature)
    preds = np.argmax(probs, axis=-1)
    confs = np.max(probs, axis=-1)

    errors: list[ErrorSample] = []
    for i, rec in enumerate(records):
        true_cls = src_to_class[rec.label]
        pred_cls = int(preds[i])
        if pred_cls != true_cls:
            ev_text = (
                ""
                if evidence_source == "statement_only"
                else (rec.gold_evidence if evidence_source == "gold" else rec.selected_evidence)
            )
            errors.append(
                ErrorSample(
                    sample_idx=rec.sample_idx,
                    statement=rec.statement,
                    evidence=ev_text,
                    true_label=class_names[true_cls],
                    predicted_label=class_names[pred_cls],
                    confidence=round(float(confs[i]), 4),
                )
            )

    errors.sort(key=lambda e: -e.confidence)
    return errors[:top_n]


def _get_git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def _report_to_dict(rep: VerifierEvalReport) -> dict[str, Any]:
    return {
        "accuracy": rep.accuracy,
        "macro_f1": rep.macro_f1,
        "ece": rep.ece,
        "num_samples": rep.num_samples,
        "confusion_matrix": rep.confusion_matrix,
        "per_class": {k: asdict(v) for k, v in rep.per_class.items()},
    }


def train_and_evaluate_verifier(
    cfg: AppConfig,
    *,
    variant: VariantName,
    model_name: str | None = None,
    seed: int | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    grad_accum: int | None = None,
    lr: float | None = None,
    fp16: bool | None = None,
    max_train_samples: int | None = None,
    run_id: str | None = None,
    local_files_only: bool = True,
    train_split_path: Path | None = None,
    val_split_path: Path | None = None,
    holdout_split_path: Path | None = None,
) -> dict[str, Any]:
    """Train a Verifier variant on ``train``, select best checkpoint by ``dev`` macro-F1, and calibrate."""
    resolved_seed = seed if seed is not None else cfg.seed
    set_seed(resolved_seed)

    resolved_model = model_name or cfg.classifier.model_name
    model_type = infer_model_type(resolved_model)
    max_length = (
        cfg.classifier.max_len_phobert if model_type == "phobert" else cfg.classifier.max_len_xlmr
    )
    num_epochs = epochs if epochs is not None else cfg.training.num_train_epochs
    per_device_bs = (
        batch_size if batch_size is not None else cfg.training.per_device_train_batch_size
    )
    accum_steps = grad_accum if grad_accum is not None else cfg.training.gradient_accumulation_steps
    learning_rate = lr if lr is not None else cfg.training.learning_rate
    use_fp16 = (fp16 if fp16 is not None else cfg.training.fp16) and torch.cuda.is_available()
    weight_decay = cfg.training.weight_decay
    warmup_ratio = cfg.training.warmup_ratio

    tr_path = train_split_path or (cfg.paths.interim_dir / "selected_train.jsonl")
    v_path = val_split_path or (cfg.paths.interim_dir / "selected_dev.jsonl")
    train_records = load_selected_split(tr_path)
    dev_records = load_selected_split(v_path)
    if max_train_samples is not None and max_train_samples > 0:
        train_records = train_records[:max_train_samples]

    verifier = VerifierClassifier(
        resolved_model,
        cfg.label_map,
        local_files_only=local_files_only,
    )
    tokenizer = verifier.load_tokenizer()
    model = verifier.load_model()
    pad_id = int(getattr(tokenizer, "pad_token_id", 1) or 1)
    segmenter = (
        get_segmenter(cfg.segmentation.backend, str(cfg.segmentation.vncorenlp_dir))
        if model_type == "phobert"
        else None
    )

    tr_stmts, tr_evids, tr_labels, tr_stmt_only, tr_preseg = build_variant_texts(
        train_records,
        variant,
        model_type=model_type,
        seed=resolved_seed,
    )
    if variant == "V-D":
        dist_tr_path = cfg.paths.interim_dir / "distractor_train.jsonl"
        if not dist_tr_path.exists():
            raise FileNotFoundError(f"Distractor train data not found at {dist_tr_path}")
        with dist_tr_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d_obj = json.loads(line)
                st = d_obj["statement_segmented"] if tr_preseg else d_obj["statement"]
                ev = (
                    d_obj["distractor_evidence_segmented"]
                    if tr_preseg
                    else d_obj["distractor_evidence"]
                )
                tr_stmts.append(st)
                tr_evids.append(ev)
                tr_labels.append(int(d_obj.get("target_label", 2)))
        rng_shuffle = random.Random(resolved_seed)
        shuff_idx = list(range(len(tr_stmts)))
        rng_shuffle.shuffle(shuff_idx)
        tr_stmts = [tr_stmts[i] for i in shuff_idx]
        tr_evids = [tr_evids[i] for i in shuff_idx]
        tr_labels = [tr_labels[i] for i in shuff_idx]
        logger.info(
            "V-D: Appended distractor train samples; total training pairs = %d", len(tr_stmts)
        )

    train_ds = encode_verifier_dataset(
        tr_stmts,
        tr_evids,
        tr_labels,
        tokenizer=tokenizer,
        model_type=model_type,
        max_length=max_length,
        label_map=cfg.label_map,
        segmenter=segmenter,
        statement_only=tr_stmt_only,
        presegmented=tr_preseg,
    )

    # Build dev datasets: gold, selected, and (if baseline) statement_only
    dev_gold_s, dev_gold_e, dev_gold_y, _, dev_preseg = build_variant_texts(
        dev_records, variant, model_type=model_type, seed=resolved_seed, eval_mode="gold"
    )
    dev_gold_ds = encode_verifier_dataset(
        dev_gold_s,
        dev_gold_e,
        dev_gold_y,
        tokenizer=tokenizer,
        model_type=model_type,
        max_length=max_length,
        label_map=cfg.label_map,
        segmenter=segmenter,
        statement_only=(variant == "baseline"),
        presegmented=dev_preseg,
    )

    dev_sel_s, dev_sel_e, dev_sel_y, _, _ = build_variant_texts(
        dev_records, variant, model_type=model_type, seed=resolved_seed, eval_mode="selected"
    )
    dev_sel_ds = encode_verifier_dataset(
        dev_sel_s,
        dev_sel_e,
        dev_sel_y,
        tokenizer=tokenizer,
        model_type=model_type,
        max_length=max_length,
        label_map=cfg.label_map,
        segmenter=segmenter,
        statement_only=(variant == "baseline"),
        presegmented=dev_preseg,
    )

    primary_eval_mode: Literal["gold", "selected", "statement_only"] = (
        "statement_only" if variant == "baseline" else ("gold" if variant == "V-A" else "selected")
    )
    primary_dev_ds = dev_gold_ds if primary_eval_mode != "selected" else dev_sel_ds

    dev_dist_ds = None
    dist_dev_path = cfg.paths.interim_dir / "distractor_dev.jsonl"
    if dist_dev_path.exists():
        with dist_dev_path.open("r", encoding="utf-8") as f:
            dist_dev_objs = [json.loads(line) for line in f if line.strip()]
        d_stmts = [
            o["statement_segmented"] if dev_preseg else o["statement"] for o in dist_dev_objs
        ]
        d_evids = [
            o["distractor_evidence_segmented"] if dev_preseg else o["distractor_evidence"]
            for o in dist_dev_objs
        ]
        d_labels = [int(o.get("target_label", 2)) for o in dist_dev_objs]
        dev_dist_ds = encode_verifier_dataset(
            d_stmts,
            d_evids,
            d_labels,
            tokenizer=tokenizer,
            model_type=model_type,
            max_length=max_length,
            label_map=cfg.label_map,
            segmenter=segmenter,
            statement_only=False,
            presegmented=dev_preseg,
        )

    holdout_gold_ds = None
    holdout_sel_ds = None
    if holdout_split_path is not None and holdout_split_path.exists():
        holdout_records = load_selected_split(holdout_split_path)
        h_gold_s, h_gold_e, h_gold_y, _, h_preseg = build_variant_texts(
            holdout_records, variant, model_type=model_type, seed=resolved_seed, eval_mode="gold"
        )
        holdout_gold_ds = encode_verifier_dataset(
            h_gold_s,
            h_gold_e,
            h_gold_y,
            tokenizer=tokenizer,
            model_type=model_type,
            max_length=max_length,
            label_map=cfg.label_map,
            segmenter=segmenter,
            statement_only=(variant == "baseline"),
            presegmented=h_preseg,
        )
        h_sel_s, h_sel_e, h_sel_y, _, _ = build_variant_texts(
            holdout_records,
            variant,
            model_type=model_type,
            seed=resolved_seed,
            eval_mode="selected",
        )
        holdout_sel_ds = encode_verifier_dataset(
            h_sel_s,
            h_sel_e,
            h_sel_y,
            tokenizer=tokenizer,
            model_type=model_type,
            max_length=max_length,
            label_map=cfg.label_map,
            segmenter=segmenter,
            statement_only=(variant == "baseline"),
            presegmented=h_preseg,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=per_device_bs,
        shuffle=True,
        collate_fn=lambda b: collate_pair_batch(b, pad_token_id=pad_id),
    )

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    steps_per_epoch = max(1, (len(train_loader) + accum_steps - 1) // accum_steps)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)

    from transformers import get_linear_schedule_with_warmup

    scheduler = get_linear_schedule_with_warmup(  # type: ignore[no-untyped-call]
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    resolved_run_id = (
        run_id or f"{model_type}_{variant.lower()}_s{resolved_seed}_{int(time.time())}"
    )
    out_dir = cfg.paths.artifacts_dir / "models" / resolved_run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_dir = out_dir / "checkpoint"

    best_macro_f1 = -1.0
    best_epoch = 0
    epoch_history: list[dict[str, Any]] = []
    dev = verifier.device

    logger.info(
        "Starting training run_id=%s variant=%s model=%s seed=%d epochs=%d bs=%d accum=%d fp16=%s device=%s",
        resolved_run_id,
        variant,
        resolved_model,
        resolved_seed,
        num_epochs,
        per_device_bs,
        accum_steps,
        use_fp16,
        dev,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)

    t_start = time.perf_counter()
    for epoch in range(1, num_epochs + 1):
        ep_t0 = time.perf_counter()
        model.train()
        optimizer.zero_grad()
        running_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(dev)
            attention_mask = batch["attention_mask"].to(dev)
            labels_t = batch["labels"].to(dev)
            kwargs: dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels_t,
            }
            if "token_type_ids" in batch:
                kwargs["token_type_ids"] = batch["token_type_ids"].to(dev)

            with torch.amp.autocast("cuda", enabled=use_fp16):
                outputs = model(**kwargs)
                loss = outputs.loss / accum_steps

            scaler.scale(loss).backward()
            running_loss += float(outputs.loss.detach().item())
            num_batches += 1

            if step % accum_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

        avg_loss = running_loss / max(num_batches, 1)
        dev_logits, dev_y = verifier.predict_logits(primary_dev_ds, batch_size=per_device_bs * 2)
        dev_rep = evaluate_verifier_logits(dev_logits, dev_y, cfg.label_map)
        ep_elapsed = time.perf_counter() - ep_t0
        peak_vram_mb = (
            round(torch.cuda.max_memory_allocated(0) / (1024 * 1024), 2)
            if torch.cuda.is_available()
            else 0.0
        )

        logger.info(
            "Epoch %d/%d (%.1fs) | train_loss=%.4f | dev_%s acc=%.4f macro_f1=%.4f ece=%.4f | peak_vram=%.1f MB",
            epoch,
            num_epochs,
            ep_elapsed,
            avg_loss,
            primary_eval_mode,
            dev_rep.accuracy,
            dev_rep.macro_f1,
            dev_rep.ece,
            peak_vram_mb,
        )
        epoch_history.append(
            {
                "epoch": epoch,
                "train_loss": round(avg_loss, 4),
                "dev_accuracy": dev_rep.accuracy,
                "dev_macro_f1": dev_rep.macro_f1,
                "dev_ece_uncalibrated": dev_rep.ece,
                "peak_vram_mb": peak_vram_mb,
                "elapsed_s": round(ep_elapsed, 2),
            }
        )

        if dev_rep.macro_f1 > best_macro_f1:
            best_macro_f1 = dev_rep.macro_f1
            best_epoch = epoch
            best_ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_ckpt_dir)
            tokenizer.save_pretrained(best_ckpt_dir)

    total_train_s = time.perf_counter() - t_start

    # Reload best checkpoint for final evaluation & temperature scaling on dev
    best_verifier = VerifierClassifier(
        best_ckpt_dir,
        cfg.label_map,
        device=str(dev),
        local_files_only=True,
    )

    gold_logits, gold_y = best_verifier.predict_logits(dev_gold_ds, batch_size=per_device_bs * 2)
    sel_logits, sel_y = best_verifier.predict_logits(dev_sel_ds, batch_size=per_device_bs * 2)

    calib_logits = gold_logits if primary_eval_mode != "selected" else sel_logits
    calib_y = gold_y if primary_eval_mode != "selected" else sel_y

    temp_scaler = TemperatureScaler()
    calib_res: CalibrationResult = temp_scaler.fit(calib_logits, calib_y)
    temp_scaler.save(out_dir / "temperature.json", result=calib_res)

    gold_rep_uncal = evaluate_verifier_logits(gold_logits, gold_y, cfg.label_map, temperature=1.0)
    gold_rep_cal = evaluate_verifier_logits(
        gold_logits, gold_y, cfg.label_map, temperature=temp_scaler.temperature
    )
    sel_rep_uncal = evaluate_verifier_logits(sel_logits, sel_y, cfg.label_map, temperature=1.0)
    sel_rep_cal = evaluate_verifier_logits(
        sel_logits, sel_y, cfg.label_map, temperature=temp_scaler.temperature
    )

    distractor_eval: dict[str, Any] | None = None
    if dev_dist_ds is not None:
        dist_logits, _ = best_verifier.predict_logits(dev_dist_ds, batch_size=per_device_bs * 2)
        dist_preds = np.argmax(dist_logits, axis=-1)
        n_dist = len(dist_preds)
        n_unv = int(np.sum(dist_preds == 2))
        n_real = int(np.sum(dist_preds == 0))
        n_fake = int(np.sum(dist_preds == 1))
        distractor_eval = {
            "total": n_dist,
            "unverifiable_pct": round(n_unv / n_dist * 100.0, 2),
            "real_pct": round(n_real / n_dist * 100.0, 2),
            "fake_pct": round(n_fake / n_dist * 100.0, 2),
            "counts": {"real": n_real, "fake": n_fake, "unverifiable": n_unv},
        }
        logger.info(
            "Distractor Dev Evaluation (%d pairs): unverifiable=%.2f%% | real=%.2f%% | fake=%.2f%%",
            n_dist,
            distractor_eval["unverifiable_pct"],
            distractor_eval["real_pct"],
            distractor_eval["fake_pct"],
        )

    holdout_eval: dict[str, Any] | None = None
    if holdout_gold_ds is not None and holdout_sel_ds is not None:
        h_gold_logits, h_gold_targets = best_verifier.predict_logits(
            holdout_gold_ds, batch_size=per_device_bs * 2
        )
        h_sel_logits, h_sel_targets = best_verifier.predict_logits(
            holdout_sel_ds, batch_size=per_device_bs * 2
        )
        h_gold_rep = evaluate_verifier_logits(
            h_gold_logits, h_gold_targets, cfg.label_map, temperature=temp_scaler.temperature
        )
        h_sel_rep = evaluate_verifier_logits(
            h_sel_logits, h_sel_targets, cfg.label_map, temperature=temp_scaler.temperature
        )
        holdout_eval = {
            "num_samples": len(h_gold_targets),
            "gold": _report_to_dict(h_gold_rep),
            "selected": _report_to_dict(h_sel_rep),
        }
        logger.info(
            "Holdout H Evaluation (Article-Disjoint, n=%d): Gold F1=%.4f (Acc=%.4f) | Selected F1=%.4f (Acc=%.4f)",
            len(h_gold_targets),
            h_gold_rep.macro_f1,
            h_gold_rep.accuracy,
            h_sel_rep.macro_f1,
            h_sel_rep.accuracy,
        )

    top_errors = extract_top_errors(
        dev_records,
        calib_logits,
        cfg.label_map,
        temperature=temp_scaler.temperature,
        evidence_source=primary_eval_mode,
        top_n=10,
    )

    # Save config snapshot alongside checkpoint
    with (out_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.model_dump(mode="json"), f, allow_unicode=True, sort_keys=False)

    summary_payload: dict[str, Any] = {
        "run_id": resolved_run_id,
        "git_commit": _get_git_commit(),
        "variant": variant,
        "model_name": resolved_model,
        "model_type": model_type,
        "seed": resolved_seed,
        "max_length": max_length,
        "epochs": num_epochs,
        "best_epoch": best_epoch,
        "per_device_batch_size": per_device_bs,
        "gradient_accumulation_steps": accum_steps,
        "effective_batch_size": per_device_bs * accum_steps,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "fp16": use_fp16,
        "device": str(dev),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "peak_vram_mb": (
            round(torch.cuda.max_memory_allocated(0) / (1024 * 1024), 2)
            if torch.cuda.is_available()
            else 0.0
        ),
        "total_train_time_s": round(total_train_s, 2),
        "primary_eval_mode": primary_eval_mode,
        "calibration": asdict(calib_res),
        "dev_gold_uncalibrated": _report_to_dict(gold_rep_uncal),
        "dev_gold_calibrated": _report_to_dict(gold_rep_cal),
        "dev_selected_uncalibrated": _report_to_dict(sel_rep_uncal),
        "dev_selected_calibrated": _report_to_dict(sel_rep_cal),
        "dev_distractor": distractor_eval,
        "holdout_h": holdout_eval,
        "epoch_history": epoch_history,
        "top_10_errors": [asdict(e) for e in top_errors],
        "checkpoint_dir": str(best_ckpt_dir),
    }

    (out_dir / "eval_report.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(
        "Completed run_id=%s | best_epoch=%d | dev_gold macro_f1=%.4f | dev_selected macro_f1=%.4f | ECE %.4f -> %.4f",
        resolved_run_id,
        best_epoch,
        gold_rep_cal.macro_f1,
        sel_rep_cal.macro_f1,
        calib_res.ece_before,
        calib_res.ece_after,
    )
    return summary_payload
