"""Dense + sparse embedding wrapper for BAAI/bge-m3.

Model loads lazily (copilot-instructions §5). Batch encoding, GPU + fp16 when
available.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from vnfake.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SparseVectorData:
    """Qdrant-ready sparse vector (indices are token ids, values are weights)."""

    indices: list[int]
    values: list[float]


@runtime_checkable
class EmbedderProtocol(Protocol):
    """Structural interface so tests can inject a dummy embedder."""

    @property
    def dim(self) -> int: ...

    def embed_dense(self, texts: Sequence[str], batch_size: int) -> list[list[float]]: ...

    def embed_sparse(self, texts: Sequence[str], batch_size: int) -> list[SparseVectorData]: ...


class BgeM3Embedder:
    """BAAI/bge-m3 dense + sparse embedder (lazy, batch, GPU-aware)."""

    def __init__(
        self,
        model_name: str,
        dim: int,
        *,
        use_fp16: bool = True,
        device: str | None = None,
        weight_threshold: float = 0.0,
    ) -> None:
        self._model_name = model_name
        self._dim = dim
        self._use_fp16 = use_fp16
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._weight_threshold = weight_threshold
        self._model: object | None = None

    @property
    def dim(self) -> int:
        return self._dim

    def unload(self) -> None:
        """Release the underlying embedding model from GPU/CPU memory."""
        if self._model is not None:
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Unloaded embedding model %s from memory", self._model_name)

    def _load(self) -> object:
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel

            effective_fp16 = self._use_fp16 and self._device == "cuda"
            logger.info(
                "Loading embedding model: %s on %s (fp16=%s)",
                self._model_name,
                self._device,
                effective_fp16,
            )
            self._model = BGEM3FlagModel(
                self._model_name, use_fp16=effective_fp16, device=self._device
            )
        return self._model

    def embed_dense(self, texts: Sequence[str], batch_size: int) -> list[list[float]]:
        """Embed ``texts`` into dense vectors (shape ``(n, dim)``)."""
        if not texts:
            return []
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        model = self._load()
        result = model.encode(  # type: ignore[attr-defined]
            list(texts),
            batch_size=batch_size,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense = result["dense_vecs"]
        return [[float(x) for x in row] for row in dense]

    def embed_sparse(self, texts: Sequence[str], batch_size: int) -> list[SparseVectorData]:
        """Embed ``texts`` into sparse lexical-weight vectors.

        ``lexical_weights`` from bge-m3 uses **string** token ids; Qdrant's
        ``SparseVector`` requires **int** indices. We convert here.
        """
        if not texts:
            return []
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        model = self._load()
        result = model.encode(  # type: ignore[attr-defined]
            list(texts),
            batch_size=batch_size,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        lexical_weights: list[dict[str, float]] = result["lexical_weights"]
        out: list[SparseVectorData] = []
        for row in lexical_weights:
            indices: list[int] = []
            values: list[float] = []
            for token_id, weight in row.items():
                w = float(weight)
                if w <= self._weight_threshold:
                    continue
                indices.append(int(token_id))
                values.append(w)
            out.append(SparseVectorData(indices=indices, values=values))
        return out
