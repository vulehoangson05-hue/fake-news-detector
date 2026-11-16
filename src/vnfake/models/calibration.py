"""Temperature scaling calibration for Verifier probability outputs (Phase 3, Brief #4 §2.6).

Rules enforced (copilot-instructions §5 & Brief #4 §2.6):
- Temperature scaling MUST be fit strictly on the ``dev`` split.
- Output probabilities represent calibrated model confidence on the provided
  evidence, not objective probability of truth.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from vnfake.training.metrics import compute_ece, softmax_with_temperature
from vnfake.utils.logging import get_logger

logger = get_logger(__name__)

_MIN_TEMPERATURE = 0.05
_MAX_TEMPERATURE = 10.0


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Summary of temperature scaling fitted on ``dev``."""

    temperature: float
    ece_before: float
    ece_after: float
    nll_before: float
    nll_after: float


class TemperatureScaler:
    """Single-parameter temperature scaling post-processor."""

    def __init__(self, temperature: float = 1.0) -> None:
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self._temperature = float(temperature)

    @property
    def temperature(self) -> float:
        return self._temperature

    def fit(
        self,
        dev_logits: np.ndarray,
        dev_labels: Sequence[int] | np.ndarray,
        *,
        max_iter: int = 200,
        lr: float = 0.05,
    ) -> CalibrationResult:
        """Fit temperature ``T > 0`` on ``dev_logits`` and ``dev_labels`` by minimizing NLL."""
        y_true = np.asarray(dev_labels, dtype=np.int64)
        if dev_logits.size == 0 or y_true.size == 0:
            raise ValueError("Cannot fit TemperatureScaler on empty dev logits/labels.")

        probs_before = softmax_with_temperature(dev_logits, temperature=1.0)
        ece_before = compute_ece(probs_before, y_true)

        logits_t = torch.tensor(dev_logits, dtype=torch.float32)
        labels_t = torch.tensor(y_true, dtype=torch.long)
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            nll_before = float(criterion(logits_t, labels_t).item())

        # Optimize log_temperature so T = exp(log_T) is strictly positive
        log_temp = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        optimizer = torch.optim.LBFGS([log_temp], lr=lr, max_iter=max_iter)

        def _closure() -> torch.Tensor:
            optimizer.zero_grad()
            temp = torch.exp(log_temp).clamp(min=_MIN_TEMPERATURE, max=_MAX_TEMPERATURE)
            loss: torch.Tensor = criterion(logits_t / temp, labels_t)
            loss.backward()  # type: ignore[no-untyped-call]
            return loss

        optimizer.step(_closure)  # type: ignore[no-untyped-call]

        fitted_t = float(
            torch.exp(log_temp.detach()).clamp(min=_MIN_TEMPERATURE, max=_MAX_TEMPERATURE).item()
        )
        with torch.no_grad():
            nll_after = float(criterion(logits_t / fitted_t, labels_t).item())

        # Never accept a degenerate temperature that worsens NLL
        if nll_after > nll_before:
            fitted_t = 1.0
            nll_after = nll_before

        self._temperature = round(fitted_t, 4)
        probs_after = softmax_with_temperature(dev_logits, temperature=self._temperature)
        ece_after = compute_ece(probs_after, y_true)

        logger.info(
            "TemperatureScaler fit on dev (n=%d): T=%.4f | NLL %.4f -> %.4f | ECE %.4f -> %.4f",
            y_true.size,
            self._temperature,
            nll_before,
            nll_after,
            ece_before,
            ece_after,
        )
        return CalibrationResult(
            temperature=self._temperature,
            ece_before=round(ece_before, 4),
            ece_after=round(ece_after, 4),
            nll_before=round(nll_before, 4),
            nll_after=round(nll_after, 4),
        )

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """Return calibrated softmax probabilities for ``logits``."""
        return softmax_with_temperature(logits, temperature=self._temperature)

    def save(self, path: Path, result: CalibrationResult | None = None) -> None:
        """Persist fitted temperature and calibration metrics to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, float] = {"temperature": self._temperature}
        if result is not None:
            payload.update(
                {
                    "ece_before": result.ece_before,
                    "ece_after": result.ece_after,
                    "nll_before": result.nll_before,
                    "nll_after": result.nll_after,
                }
            )
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> TemperatureScaler:
        """Load :class:`TemperatureScaler` from JSON."""
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(temperature=float(data["temperature"]))
