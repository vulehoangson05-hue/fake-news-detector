"""Tokenized ``(Statement, Evidence)`` datasets for PhoBERT and XLM-R.

PhoBERT inputs MUST be word-segmented (copilot-instructions §4); XLM-R uses
raw NFC text. Truncation always cuts Evidence first (``only_second``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import torch
from torch.utils.data import Dataset

from vnfake.data.cleaning import normalize_text
from vnfake.data.segmentation import SegmenterProtocol
from vnfake.schemas.vifactcheck import ViFactCheckSample
from vnfake.utils.logging import get_logger

logger = get_logger(__name__)

ModelType = Literal["phobert", "xlmr"]
_VALID_MODEL_TYPES: tuple[ModelType, ...] = ("phobert", "xlmr")


class TokenizerProtocol(Protocol):
    """Structural interface compatible with HF ``PreTrainedTokenizerBase.__call__``."""

    def __call__(
        self,
        text: str,
        text_pair: str | None = ...,
        *,
        max_length: int = ...,
        truncation: bool | str = ...,
        padding: bool | str = ...,
        return_tensors: str | None = ...,
    ) -> Mapping[str, list[int]]: ...


@dataclass(frozen=True, slots=True)
class PairItem:
    """One tokenized sample."""

    input_ids: list[int]
    attention_mask: list[int]
    token_type_ids: list[int] | None
    label: int


class PairDataset(Dataset[dict[str, torch.Tensor]]):
    """Torch Dataset of tokenized ``(Statement, Evidence)`` pairs."""

    def __init__(self, items: Sequence[PairItem]) -> None:
        self._items: list[PairItem] = list(items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self._items[idx]
        out: dict[str, torch.Tensor] = {
            "input_ids": torch.tensor(item.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(item.attention_mask, dtype=torch.long),
            "labels": torch.tensor(item.label, dtype=torch.long),
        }
        if item.token_type_ids is not None:
            out["token_type_ids"] = torch.tensor(item.token_type_ids, dtype=torch.long)
        return out


def build_class_mapping(
    label_map: Mapping[int, str],
) -> tuple[list[str], dict[int, int]]:
    """Return ``(class_names, source_int_to_class_id)`` from the config label_map.

    The config must expose ``label_map`` with keys ``0..N-1`` (contiguous) and
    unique string values. ``class_names[i]`` is the model's class ``i``.

    Args:
        label_map: Mapping ``{source_label_int -> class_name}``.

    Raises:
        ValueError: If keys are not contiguous ``0..N-1`` or values are not unique.

    Returns:
        ``class_names`` (ordered by key) and ``source_to_class`` mapping.
    """
    sorted_keys = sorted(label_map.keys())
    expected = list(range(len(sorted_keys)))
    if sorted_keys != expected:
        raise ValueError(f"label_map keys must be contiguous 0..N-1; got {sorted_keys}")
    class_names = [label_map[k] for k in sorted_keys]
    if len(set(class_names)) != len(class_names):
        raise ValueError(f"label_map values must be unique; got {class_names}")
    return class_names, {k: i for i, k in enumerate(sorted_keys)}


def _validate_model_type(model_type: str) -> None:
    """Fail-fast on unknown ``model_type`` (config-driven; no implicit default)."""
    if model_type not in _VALID_MODEL_TYPES:
        raise ValueError(
            f"Unknown model_type: {model_type!r}; expected one of {_VALID_MODEL_TYPES}"
        )


def build_pair_dataset(
    samples: Sequence[ViFactCheckSample],
    *,
    tokenizer: TokenizerProtocol,
    model_type: ModelType,
    max_length: int,
    label_map: Mapping[int, str],
    segmenter: SegmenterProtocol | None = None,
) -> PairDataset:
    """Tokenize ``(Statement, Evidence)`` pairs into a :class:`PairDataset`.

    Args:
        samples: ViFactCheck samples.
        tokenizer: HF tokenizer (already loaded by caller).
        model_type: ``"phobert"`` or ``"xlmr"``.
        max_length: Truncation limit (Evidence is truncated first).
        label_map: ``cfg.label_map`` — used to derive class IDs.
        segmenter: Required when ``model_type == "phobert"``.

    Raises:
        ValueError: If ``model_type`` is unknown, if ``"phobert"`` is requested
            without a ``segmenter``, or if ``label_map`` is malformed.

    Returns:
        A ``PairDataset`` ready for a torch ``DataLoader``.
    """
    _validate_model_type(model_type)

    if not samples:
        logger.warning("build_pair_dataset called with empty samples")
        return PairDataset([])

    _, src_to_class = build_class_mapping(label_map)

    if model_type == "phobert":
        if segmenter is None:
            raise ValueError("PhoBERT requires a segmenter (word segmentation).")
        stmt_texts = segmenter.segment_batch([normalize_text(s.statement) for s in samples])
        evid_texts = segmenter.segment_batch([normalize_text(s.evidence) for s in samples])
    else:
        stmt_texts = [normalize_text(s.statement) for s in samples]
        evid_texts = [normalize_text(s.evidence) for s in samples]

    items: list[PairItem] = []
    for i, sample in enumerate(samples):
        enc = tokenizer(
            stmt_texts[i],
            evid_texts[i],
            max_length=max_length,
            truncation="only_second",
            padding=False,
        )
        items.append(
            PairItem(
                input_ids=list(enc["input_ids"]),
                attention_mask=list(enc["attention_mask"]),
                token_type_ids=(list(enc["token_type_ids"]) if "token_type_ids" in enc else None),
                label=src_to_class[sample.labels],
            )
        )

    logger.info(
        "build_pair_dataset: model_type=%s n=%d max_length=%d",
        model_type,
        len(items),
        max_length,
    )
    return PairDataset(items)
