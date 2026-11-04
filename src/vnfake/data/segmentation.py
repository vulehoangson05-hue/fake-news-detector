"""Vietnamese word segmentation.

Only this module may call word-segmentation APIs (copilot-instructions §2).
Segmenters load lazily; callers obtain a cached instance via :func:`get_segmenter`.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from vnfake.utils.logging import get_logger

logger = get_logger(__name__)

Backend = Literal["underthesea", "vncorenlp"]


@runtime_checkable
class SegmenterProtocol(Protocol):
    """Structural interface for word segmenters."""

    def segment(self, text: str) -> str:
        """Return the input with multi-syllable words joined by ``_``."""
        ...

    def segment_batch(self, texts: Sequence[str]) -> list[str]:
        """Return segmented versions of ``texts`` in the same order."""
        ...


class UndertheseaSegmenter:
    """Pure-Python segmenter (fast enough, no Java required)."""

    def __init__(self) -> None:
        self._fn = None  # lazy

    def _get(self) -> object:
        if self._fn is None:
            from underthesea import word_tokenize  # heavy import, do lazily

            self._fn = word_tokenize
        return self._fn

    def segment(self, text: str) -> str:
        if not text:
            return ""
        fn = self._get()
        return str(fn(text, format="text"))  # type: ignore[operator]

    def segment_batch(self, texts: Sequence[str]) -> list[str]:
        return [self.segment(t) for t in texts]


class VnCoreNLPSegmenter:
    """VnCoreNLP segmenter (needs Java; standard for PhoBERT)."""

    def __init__(self, vncorenlp_dir: Path) -> None:
        self._dir = vncorenlp_dir
        self._seg = None  # lazy

    def _get(self) -> object:
        if self._seg is None:
            from py_vncorenlp import VnCoreNLP  # heavy import, do lazily

            self._dir.mkdir(parents=True, exist_ok=True)
            self._seg = VnCoreNLP(str(self._dir), annotators=["wseg"])
        return self._seg

    def segment(self, text: str) -> str:
        if not text:
            return ""
        seg = self._get()
        result = seg.word_segment(text)  # type: ignore[attr-defined]
        if isinstance(result, list):
            return " ".join(result)
        return str(result)

    def segment_batch(self, texts: Sequence[str]) -> list[str]:
        return [self.segment(t) for t in texts]


@lru_cache(maxsize=4)
def get_segmenter(backend: Backend, vncorenlp_dir: str | None = None) -> SegmenterProtocol:
    """Return a cached segmenter for the given backend.

    Args:
        backend: ``"underthesea"`` or ``"vncorenlp"``.
        vncorenlp_dir: Required when ``backend == "vncorenlp"``.

    Raises:
        ValueError: Unknown backend or missing directory for VnCoreNLP.
    """
    if backend == "underthesea":
        return UndertheseaSegmenter()
    if backend == "vncorenlp":
        if not vncorenlp_dir:
            raise ValueError("vncorenlp_dir is required for the 'vncorenlp' backend")
        return VnCoreNLPSegmenter(Path(vncorenlp_dir))
    raise ValueError(f"Unknown segmentation backend: {backend!r}")
