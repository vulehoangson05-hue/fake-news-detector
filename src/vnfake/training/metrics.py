"""Classification metrics and Expected Calibration Error (Phase 3, Brief #4 §2.5-§2.8).

Computes accuracy, macro-F1, per-class precision/recall/F1, confusion matrix,
and Expected Calibration Error (ECE) over 3 classes (``0=real, 1=fake, 2=unverifiable``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from vnfake.data.dataset import build_class_mapping

_DEFAULT_ECE_BINS = 15


@dataclass(frozen=True, slots=True)
class ClassMetrics:
    """Precision, recall, F1, and support for one class."""

    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True, slots=True)
class VerifierEvalReport:
    """Comprehensive evaluation report for a 3-class verifier."""

    accuracy: float
    macro_f1: float
    per_class: dict[str, ClassMetrics]
    confusion_matrix: list[list[int]]
    ece: float
    num_samples: int


def softmax_with_temperature(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Compute numerically stable softmax probabilities after scaling logits by ``temperature``."""
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if logits.size == 0:
        return np.zeros_like(logits, dtype=np.float64)
    scaled = logits.astype(np.float64) / float(temperature)
    shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
    exp_vals = np.exp(shifted)
    denom = np.sum(exp_vals, axis=-1, keepdims=True)
    return np.asarray(exp_vals / np.maximum(denom, 1e-12), dtype=np.float64)


def compute_ece(
    probs: np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    num_bins: int = _DEFAULT_ECE_BINS,
) -> float:
    """Compute Expected Calibration Error (ECE) with ``num_bins`` equal-width bins in ``[0, 1]``."""
    y_true = np.asarray(labels, dtype=np.int64)
    if y_true.size == 0 or probs.size == 0:
        return 0.0
    confidences = np.max(probs, axis=-1)
    predictions = np.argmax(probs, axis=-1)
    accuracies = (predictions == y_true).astype(np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    n = float(y_true.size)

    for b_idx in range(num_bins):
        lo = bin_boundaries[b_idx]
        hi = bin_boundaries[b_idx + 1]
        if b_idx == 0:
            in_bin = (confidences >= lo) & (confidences <= hi)
        else:
            in_bin = (confidences > lo) & (confidences <= hi)
        bin_count = int(np.sum(in_bin))
        if bin_count > 0:
            avg_conf = float(np.mean(confidences[in_bin]))
            avg_acc = float(np.mean(accuracies[in_bin]))
            ece += (bin_count / n) * abs(avg_acc - avg_conf)

    return float(ece)


def evaluate_verifier_logits(
    logits: np.ndarray,
    labels: Sequence[int] | np.ndarray,
    label_map: Mapping[int, str],
    *,
    temperature: float = 1.0,
    num_ece_bins: int = _DEFAULT_ECE_BINS,
) -> VerifierEvalReport:
    """Evaluate raw logits against ground-truth labels."""
    class_names, _ = build_class_mapping(label_map)
    num_classes = len(class_names)
    y_true = np.asarray(labels, dtype=np.int64)
    n = int(y_true.size)
    if n == 0:
        return VerifierEvalReport(
            accuracy=0.0,
            macro_f1=0.0,
            per_class={
                name: ClassMetrics(precision=0.0, recall=0.0, f1=0.0, support=0)
                for name in class_names
            },
            confusion_matrix=[[0] * num_classes for _ in range(num_classes)],
            ece=0.0,
            num_samples=0,
        )

    probs = softmax_with_temperature(logits, temperature=temperature)
    y_pred = np.argmax(probs, axis=-1)

    accuracy = float(np.mean(y_pred == y_true))
    conf_mat: list[list[int]] = [[0] * num_classes for _ in range(num_classes)]
    for t_val, p_val in zip(y_true.tolist(), y_pred.tolist(), strict=True):
        if 0 <= t_val < num_classes and 0 <= p_val < num_classes:
            conf_mat[t_val][p_val] += 1

    per_class: dict[str, ClassMetrics] = {}
    f1_scores: list[float] = []

    for c_idx, c_name in enumerate(class_names):
        tp = conf_mat[c_idx][c_idx]
        fp = sum(conf_mat[r][c_idx] for r in range(num_classes) if r != c_idx)
        fn = sum(conf_mat[c_idx][c] for c in range(num_classes) if c != c_idx)
        support = sum(conf_mat[c_idx])

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0.0 else 0.0
        f1_scores.append(f1)
        per_class[c_name] = ClassMetrics(
            precision=round(prec, 4),
            recall=round(rec, 4),
            f1=round(f1, 4),
            support=support,
        )

    macro_f1 = float(sum(f1_scores) / max(len(f1_scores), 1))
    ece = compute_ece(probs, y_true, num_bins=num_ece_bins)

    return VerifierEvalReport(
        accuracy=round(accuracy, 4),
        macro_f1=round(macro_f1, 4),
        per_class=per_class,
        confusion_matrix=conf_mat,
        ece=round(ece, 4),
        num_samples=n,
    )
