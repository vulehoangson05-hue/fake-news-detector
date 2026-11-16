"""Decision layer, relevance gating, and ensemble inference (Brief #5 §3 & §11.5).

Implements:
1. Relevance Gate:
   If r < tau_rel -> 'unverifiable' (reason: "không đủ bằng chứng liên quan")
2. Confidence Gate:
   If max(P) < tau_conf -> 'unverifiable' (reason: "độ tin cậy mô hình thấp")
3. Argmax verdict:
   argmax(P) (reason: "mô hình xác nhận")

Where probabilities P are computed from a 3-seed ensemble of temperature-calibrated
softmax predictions:
    P = (1/K) * sum_{k=1}^K softmax(logits_k / T_k)
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from vnfake.config import AppConfig, load_config
from vnfake.data.dataset import ModelType
from vnfake.data.segmentation import SegmenterProtocol, get_segmenter
from vnfake.models.classifier import VerifierClassifier
from vnfake.models.tokenization import encode_verifier_dataset
from vnfake.utils.logging import get_logger

logger = get_logger(__name__)

REASON_LOW_RELEVANCE = "không đủ bằng chứng liên quan"
REASON_LOW_CONFIDENCE = "độ tin cậy mô hình thấp"
REASON_CONFIRMED = "mô hình xác nhận"

VerdictType = Literal["real", "fake", "unverifiable"]


def _sha256_file(path: Path) -> str:
    """Compute lowercase hexadecimal SHA-256 digest of ``path``."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest().lower()


@dataclass(frozen=True, slots=True)
class DecisionResult:
    """Output of the DecisionGate after relevance and confidence checks."""

    verdict: str  # "real", "fake", "unverifiable"
    label_idx: int  # 0, 1, 2
    probs: dict[str, float]  # Class probabilities, e.g. {"real": 0.1, "fake": 0.85, ...}
    relevance_score: float
    passed_relevance: bool
    confidence: float
    passed_confidence: bool
    reason: str
    show_probs: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DecisionGate:
    """Applies relevance gating and confidence thresholding to model probabilities."""

    def __init__(
        self,
        tau_rel: float = 0.7559,
        tau_conf: float = 0.55,
        class_names: Sequence[str] | Mapping[int, str] = ("real", "fake", "unverifiable"),
    ) -> None:
        self.tau_rel = float(tau_rel)
        self.tau_conf = float(tau_conf)
        if isinstance(class_names, Mapping):
            self.class_names = [class_names[k] for k in sorted(class_names.keys())]
        else:
            self.class_names = [str(c) for c in class_names]
        self.unverifiable_idx = (
            self.class_names.index("unverifiable") if "unverifiable" in self.class_names else 2
        )

    def decide(
        self,
        relevance_score: float,
        probs: np.ndarray | Sequence[float],
    ) -> DecisionResult:
        """Make final verdict given relevance score and class probabilities."""
        p_arr = np.asarray(probs, dtype=np.float64)
        if p_arr.ndim != 1 or len(p_arr) != len(self.class_names):
            raise ValueError(
                f"Expected 1D probability array of length {len(self.class_names)}, got shape {p_arr.shape}"
            )

        # Normalize probs if needed
        p_sum = float(np.sum(p_arr))
        if p_sum > 0:
            p_arr = p_arr / p_sum

        prob_dict = {
            c_name: round(float(p_arr[idx]), 4) for idx, c_name in enumerate(self.class_names)
        }
        best_idx = int(np.argmax(p_arr))
        best_conf = float(p_arr[best_idx])

        # Step 1: Relevance Gate
        if relevance_score < self.tau_rel:
            return DecisionResult(
                verdict="unverifiable",
                label_idx=self.unverifiable_idx,
                probs=prob_dict,
                relevance_score=round(float(relevance_score), 4),
                passed_relevance=False,
                confidence=round(best_conf, 4),
                passed_confidence=False,
                reason=REASON_LOW_RELEVANCE,
                show_probs=False,
            )

        # Step 2: Confidence Gate
        if best_conf < self.tau_conf:
            return DecisionResult(
                verdict="unverifiable",
                label_idx=self.unverifiable_idx,
                probs=prob_dict,
                relevance_score=round(float(relevance_score), 4),
                passed_relevance=True,
                confidence=round(best_conf, 4),
                passed_confidence=False,
                reason=REASON_LOW_CONFIDENCE,
                show_probs=False,
            )

        # Step 3: Argmax verdict
        return DecisionResult(
            verdict=self.class_names[best_idx],
            label_idx=best_idx,
            probs=prob_dict,
            relevance_score=round(float(relevance_score), 4),
            passed_relevance=True,
            confidence=round(best_conf, 4),
            passed_confidence=True,
            reason=REASON_CONFIRMED,
            show_probs=True,
        )


class EnsembleVerifier:
    """3-seed ensemble of temperature-calibrated PhoBERT verifiers."""

    def __init__(
        self,
        checkpoint_dirs: Sequence[Path | str],
        temperatures: Sequence[float],
        label_map: Mapping[int, str] | Sequence[str] = (0, 1, 2),  # type: ignore[assignment]
        device: str = "cuda",
        max_length: int = 256,
        segmenter: SegmenterProtocol | None = None,
        use_fp16: bool = False,
    ) -> None:
        if len(checkpoint_dirs) != len(temperatures):
            raise ValueError(
                f"Mismatched checkpoints ({len(checkpoint_dirs)}) and temperatures ({len(temperatures)})"
            )
        self.checkpoint_dirs = [Path(p) for p in checkpoint_dirs]
        self.temperatures = [float(t) for t in temperatures]
        if isinstance(label_map, Mapping):
            self.label_map: dict[int, str] = dict(label_map)
        else:
            self.label_map = {i: str(name) for i, name in enumerate(label_map)}
        self.class_names = [self.label_map[i] for i in sorted(self.label_map.keys())]
        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.max_length = max_length
        self.segmenter = segmenter
        self.use_fp16 = use_fp16

        self._verifiers: list[VerifierClassifier] | None = None
        self._tokenizer: Any = None

    def _ensure_loaded(self) -> None:
        if self._verifiers is None:
            self._verifiers = []
            for ckpt in self.checkpoint_dirs:
                v = VerifierClassifier(
                    ckpt,
                    self.label_map,
                    device=self.device,
                    local_files_only=True,
                    use_fp16=self.use_fp16,
                )
                self._verifiers.append(v)
            self._tokenizer = self._verifiers[0].load_tokenizer()

    def unload(self) -> None:
        """Release loaded models from memory/VRAM."""
        if self._verifiers is not None:
            for v in self._verifiers:
                v._model = None
            self._verifiers = None
        self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict_probs_batch(
        self,
        statements: Sequence[str],
        evidences: Sequence[str],
        *,
        presegmented: bool = False,
        batch_size: int = 16,
    ) -> np.ndarray:
        """Compute mean temperature-calibrated softmax probabilities across seeds.

        Returns:
            np.ndarray of shape (N, 3).
        """
        self._ensure_loaded()
        assert self._verifiers is not None
        assert self._tokenizer is not None

        dummy_labels = [0] * len(statements)
        model_type: ModelType = "phobert"
        ds = encode_verifier_dataset(
            list(statements),
            list(evidences),
            dummy_labels,
            tokenizer=self._tokenizer,
            model_type=model_type,
            max_length=self.max_length,
            label_map=self.label_map,
            segmenter=self.segmenter,
            statement_only=False,
            presegmented=presegmented,
        )

        all_probs: list[np.ndarray] = []
        for verifier, temp in zip(self._verifiers, self.temperatures, strict=True):
            logits, _ = verifier.predict_logits(ds, batch_size=batch_size)
            scaled_logits = logits / max(temp, 1e-4)
            # Softmax
            exp_l = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
            probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
            all_probs.append(probs)

        ensemble_probs: np.ndarray = np.asarray(np.mean(all_probs, axis=0), dtype=np.float64)
        return ensemble_probs

    def predict_single(
        self,
        statement: str,
        evidence: str,
        *,
        presegmented: bool = False,
    ) -> np.ndarray:
        """Compute calibrated probabilities for a single (statement, evidence) pair."""
        probs = self.predict_probs_batch([statement], [evidence], presegmented=presegmented, batch_size=1)
        return np.asarray(probs[0], dtype=np.float64)


def load_vc_ensemble(
    cfg: AppConfig | None = None,
    device: str = "cuda",
    *,
    use_fp16: bool | None = None,
    verify_sha256: bool = True,
) -> EnsembleVerifier:
    """Factory function to load the pinned V-C 3-seed ensemble with SHA-256 verification."""
    if cfg is None:
        cfg = load_config()

    specs = cfg.decision.checkpoints
    if not specs:
        raise ValueError(
            "No pinned checkpoints configured under cfg.decision.checkpoints (Brief #5 §12.3.4)."
        )

    models_dir = cfg.paths.artifacts_dir / "models"
    checkpoints: list[Path] = []
    temperatures: list[float] = []

    for spec in specs:
        chosen_dir = models_dir / spec.run_id
        if not chosen_dir.exists():
            raise FileNotFoundError(
                f"Pinned checkpoint directory not found for seed {spec.seed}: {chosen_dir}"
            )
        ckpt = chosen_dir / "checkpoint"
        weights_file = ckpt / "model.safetensors"
        if not weights_file.exists():
            raise FileNotFoundError(f"Missing model.safetensors at {weights_file}")

        if verify_sha256:
            actual_hash = _sha256_file(weights_file)
            expected_hash = spec.sha256.strip().lower()
            if actual_hash != expected_hash:
                raise ValueError(
                    f"SHA-256 mismatch for pinned V-C seed {spec.seed} ({spec.run_id}): "
                    f"expected {expected_hash}, got {actual_hash}"
                )

        temp_file = chosen_dir / "temperature.json"
        if not temp_file.exists():
            raise FileNotFoundError(f"Missing temperature.json at {temp_file}")
        temp_data = json.loads(temp_file.read_text(encoding="utf-8"))
        temp_val = float(temp_data.get("temperature", 1.0))

        checkpoints.append(ckpt)
        temperatures.append(temp_val)

    segmenter = get_segmenter(cfg.segmentation.backend, str(cfg.segmentation.vncorenlp_dir))
    effective_fp16 = cfg.decision.use_fp16 if use_fp16 is None else use_fp16

    return EnsembleVerifier(
        checkpoint_dirs=checkpoints,
        temperatures=temperatures,
        label_map=cfg.label_map,
        device=device,
        max_length=cfg.classifier.max_len_phobert,
        segmenter=segmenter,
        use_fp16=effective_fp16,
    )
