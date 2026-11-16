"""Sequence classification wrapper for PhoBERT and XLM-RoBERTa (Phase 3, Brief #4 §2.4-§2.5).

Lazy-loads Hugging Face ``AutoModelForSequenceClassification`` and ``AutoTokenizer``
with label mappings derived strictly from ``cfg.label_map`` (3 classes:
``0=real, 1=fake, 2=unverifiable``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from vnfake.data.dataset import ModelType, PairDataset, build_class_mapping
from vnfake.utils.logging import get_logger

logger = get_logger(__name__)


def infer_model_type(model_name: str) -> ModelType:
    """Infer ``"phobert"`` or ``"xlmr"`` from Hugging Face model identifier."""
    lower = model_name.lower()
    if "phobert" in lower:
        return "phobert"
    if "xlm" in lower or "roberta" in lower:
        return "xlmr"
    raise ValueError(
        f"Cannot infer model_type from {model_name!r}; expected 'phobert' or 'xlmr' model family."
    )


def build_hf_label_maps(
    label_map: Mapping[int, str],
) -> tuple[list[str], dict[int, str], dict[str, int]]:
    """Build ``(class_names, id2label, label2id)`` from ``cfg.label_map``."""
    class_names, _ = build_class_mapping(label_map)
    id2label = dict(enumerate(class_names))
    label2id = {name: i for i, name in enumerate(class_names)}
    return class_names, id2label, label2id


def collate_pair_batch(
    batch: Sequence[dict[str, torch.Tensor]],
    *,
    pad_token_id: int = 1,
) -> dict[str, torch.Tensor]:
    """Dynamic padding collator for :class:`PairDataset` batches."""
    max_len = max(int(item["input_ids"].shape[0]) for item in batch)
    batch_size = len(batch)

    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    labels = torch.zeros(batch_size, dtype=torch.long)
    has_token_type = "token_type_ids" in batch[0]
    token_type_ids = (
        torch.zeros((batch_size, max_len), dtype=torch.long) if has_token_type else None
    )

    for i, item in enumerate(batch):
        seq_len = int(item["input_ids"].shape[0])
        input_ids[i, :seq_len] = item["input_ids"]
        attention_mask[i, :seq_len] = item["attention_mask"]
        labels[i] = item["labels"]
        if token_type_ids is not None and "token_type_ids" in item:
            token_type_ids[i, :seq_len] = item["token_type_ids"]

    out: dict[str, torch.Tensor] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    if token_type_ids is not None:
        out["token_type_ids"] = token_type_ids
    return out


class VerifierClassifier:
    """Lazy-loaded wrapper around ``AutoModelForSequenceClassification``."""

    def __init__(
        self,
        model_name_or_path: str | Path,
        label_map: Mapping[int, str],
        *,
        device: str | None = None,
        local_files_only: bool = False,
        use_fp16: bool = False,
    ) -> None:
        self._model_name_or_path = str(model_name_or_path)
        self._class_names, self._id2label, self._label2id = build_hf_label_maps(label_map)
        self._num_labels = len(self._class_names)
        self._device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._local_files_only = local_files_only
        self._use_fp16 = use_fp16
        self._model: Any = None
        self._tokenizer: Any = None

    @property
    def num_labels(self) -> int:
        return self._num_labels

    @property
    def class_names(self) -> list[str]:
        return list(self._class_names)

    @property
    def device(self) -> torch.device:
        return torch.device(self._device_str)

    def load_tokenizer(self) -> Any:
        """Lazily load and return the Hugging Face tokenizer."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name_or_path,
                local_files_only=self._local_files_only,
            )
        return self._tokenizer

    def load_model(self) -> Any:
        """Lazily load and return the sequence classification model on target device."""
        if self._model is None:
            from transformers import AutoModelForSequenceClassification

            self._model = AutoModelForSequenceClassification.from_pretrained(
                self._model_name_or_path,
                num_labels=self._num_labels,
                id2label=self._id2label,
                label2id=self._label2id,
                local_files_only=self._local_files_only,
            )
            self._model.to(self.device)
            effective_fp16 = self._use_fp16 and self.device.type == "cuda"
            if effective_fp16:
                self._model.half()
            logger.info(
                "Loaded VerifierClassifier model=%s num_labels=%d device=%s fp16=%s",
                self._model_name_or_path,
                self._num_labels,
                self._device_str,
                effective_fp16,
            )
        return self._model

    @torch.no_grad()
    def predict_logits(
        self,
        dataset: PairDataset,
        *,
        batch_size: int = 16,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run inference over ``dataset`` and return ``(logits, true_labels)`` as numpy arrays."""
        if len(dataset) == 0:
            return (
                np.zeros((0, self._num_labels), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )

        tokenizer = self.load_tokenizer()
        pad_id = int(getattr(tokenizer, "pad_token_id", 1) or 1)
        model = self.load_model()
        model.eval()

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_pair_batch(b, pad_token_id=pad_id),
        )

        all_logits: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        dev = self.device

        for batch in loader:
            input_ids = batch["input_ids"].to(dev)
            attention_mask = batch["attention_mask"].to(dev)
            kwargs: dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if "token_type_ids" in batch:
                kwargs["token_type_ids"] = batch["token_type_ids"].to(dev)

            outputs = model(**kwargs)
            logits_np = outputs.logits.detach().cpu().to(torch.float32).numpy()
            labels_np = batch["labels"].detach().cpu().numpy()
            all_logits.append(logits_np)
            all_labels.append(labels_np)

        return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)
