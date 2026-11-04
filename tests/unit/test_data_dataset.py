"""Unit tests for vnfake.data.dataset (no real tokenizer/segmenter)."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch

from vnfake.data.dataset import (
    PairDataset,
    build_class_mapping,
    build_pair_dataset,
)
from vnfake.schemas.vifactcheck import ViFactCheckSample


class _FakeSegmenter:
    def segment(self, text: str) -> str:
        return text.replace(" ", "_")

    def segment_batch(self, texts: list[str]) -> list[str]:
        return [self.segment(t) for t in texts]


class _FakeTokenizer:
    """Minimal HF-compatible tokenizer returning fixed-shape encodings."""

    def __call__(
        self,
        text: str,
        text_pair: str | None = None,
        *,
        max_length: int = 512,
        truncation: bool | str = False,
        padding: bool | str = False,
        return_tensors: str | None = None,
    ) -> Mapping[str, list[int]]:
        a = len(text.split()) if text else 0
        b = len(text_pair.split()) if text_pair else 0
        total = min(a + b + 2, max_length)  # [CLS] a [SEP] b [SEP]
        ids = list(range(1, total + 1))
        return {
            "input_ids": ids,
            "attention_mask": [1] * total,
            "token_type_ids": [0] * (min(a + 2, total)) + [1] * max(0, total - min(a + 2, total)),
        }


def _sample(stmt: str = "A B", evid: str = "C D", label: int = 0) -> ViFactCheckSample:
    return ViFactCheckSample(statement=stmt, evidence=evid, labels=label)  # type: ignore[arg-type]


def test_build_class_mapping_ok() -> None:
    names, src_to_class = build_class_mapping({0: "real", 1: "fake", 2: "unverifiable"})
    assert names == ["real", "fake", "unverifiable"]
    assert src_to_class == {0: 0, 1: 1, 2: 2}


def test_build_class_mapping_requires_contiguous() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        build_class_mapping({1: "a", 2: "b"})


def test_build_class_mapping_unique_values() -> None:
    with pytest.raises(ValueError, match="unique"):
        build_class_mapping({0: "real", 1: "real"})


def test_build_pair_dataset_phobert_uses_segmenter() -> None:
    ds = build_pair_dataset(
        [_sample()],
        tokenizer=_FakeTokenizer(),
        model_type="phobert",
        max_length=32,
        label_map={0: "real", 1: "fake", 2: "unverifiable"},
        segmenter=_FakeSegmenter(),
    )
    item = ds[0]
    assert set(item.keys()) == {"input_ids", "attention_mask", "token_type_ids", "labels"}
    assert int(item["labels"].item()) == 0


def test_build_pair_dataset_phobert_requires_segmenter() -> None:
    with pytest.raises(ValueError, match="segmenter"):
        build_pair_dataset(
            [_sample()],
            tokenizer=_FakeTokenizer(),
            model_type="phobert",
            max_length=32,
            label_map={0: "real"},
            segmenter=None,
        )


def test_build_pair_dataset_xlmr_no_segmenter_ok() -> None:
    ds = build_pair_dataset(
        [_sample()],
        tokenizer=_FakeTokenizer(),
        model_type="xlmr",
        max_length=32,
        label_map={0: "real", 1: "fake", 2: "unverifiable"},
        segmenter=None,
    )
    assert len(ds) == 1


def test_build_pair_dataset_label_mapping_respected() -> None:
    ds = build_pair_dataset(
        [_sample(label=2)],
        tokenizer=_FakeTokenizer(),
        model_type="xlmr",
        max_length=32,
        label_map={0: "real", 1: "fake", 2: "unverifiable"},
    )
    assert int(ds[0]["labels"].item()) == 2


def test_build_pair_dataset_rejects_bad_model_type() -> None:
    with pytest.raises(ValueError, match="model_type"):
        build_pair_dataset(
            [_sample()],
            tokenizer=_FakeTokenizer(),
            model_type="bogus",  # type: ignore[arg-type]
            max_length=32,
            label_map={0: "real"},
        )


def test_build_pair_dataset_empty() -> None:
    ds = build_pair_dataset(
        [],
        tokenizer=_FakeTokenizer(),
        model_type="xlmr",
        max_length=32,
        label_map={0: "real"},
    )
    assert len(ds) == 0


def test_pair_dataset_len_and_getitem() -> None:
    ds = build_pair_dataset(
        [_sample("A", "B", 0), _sample("C", "D", 1)],
        tokenizer=_FakeTokenizer(),
        model_type="xlmr",
        max_length=32,
        label_map={0: "real", 1: "fake"},
    )
    assert len(ds) == 2
    assert isinstance(ds, PairDataset)
    assert ds[1]["labels"].dtype == torch.int64
