"""Dense embedding wrapper for BAAI/bge-m3.

Model loads lazily (copilot-instructions §5). Batch encoding, GPU + fp16 when
available. Phase 2 Part A provides dense only; sparse + colbert will be added
when hybrid search is wired up.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import torch

from vnfake.utils.logging import get_logger

logger = get_logger(__name__)


@runtime_checkable
class EmbedderProtocol(Protocol):
    """Structural interface so tests can inject a dummy embedder."""

    @property
    def dim(self) -> int:
        """Embedding dimensionality (must match the Qdrant collection)."""
        ...

    def embed_dense(self, texts: Sequence[str], batch_size: int) -> list[list[float]]:
        """Embed ``texts`` into dense vectors of shape ``(len(texts), dim)``."""
        ...


class BgeM3Embedder:
    """BAAI/bge-m3 dense embedder (lazy, batch, GPU-aware).

    Args:
        model_name: Hugging Face model id (from ``cfg.retrieval.embedding.model_name``).
        dim: Embedding dimensionality (from config; used for Qdrant collection).
        use_fp16: Use half precision on CUDA (ignored on CPU).
        device: ``"cuda"`` / ``"cpu"`` / ``None`` → auto-detect.
    """

    def __init__(
        self,
        model_name: str,
        dim: int,
        *,
        use_fp16: bool = True,
        device: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._dim = dim
        self._use_fp16 = use_fp16
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: object | None = None  # lazy

    @property
    def dim(self) -> int:
        return self._dim

    def _load(self) -> object:
        """Load the underlying BGE-M3 model on first use."""
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel  # heavy import, lazy per §5

            effective_fp16 = self._use_fp16 and self._device == "cuda"
            logger.info(
                "Loading embedding model: %s on %s (fp16=%s)",
                self._model_name,
                self._device,
                effective_fp16,
            )
            self._model = BGEM3FlagModel(
                self._model_name,
                use_fp16=effective_fp16,
                device=self._device,
            )
        return self._model

    def embed_dense(self, texts: Sequence[str], batch_size: int) -> list[list[float]]:
        """Embed ``texts`` into dense vectors.

        Args:
            texts: Input strings (already NFC-normalized by caller).
            batch_size: Mini-batch size forwarded to the model encoder.

        Returns:
            List of length ``len(texts)`` of float lists of length ``dim``.
        """
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
