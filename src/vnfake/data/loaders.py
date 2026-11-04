"""Streaming JSONL loaders with per-line Pydantic validation.

Invalid lines are written to a ``rejected`` file and never crash the pipeline
(copilot-instructions §4). KB loading is streaming; ViFactCheck splits are
small enough to return as lists.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Literal, TextIO

from pydantic import ValidationError

from vnfake.schemas.kb import KBRecord
from vnfake.schemas.vifactcheck import ViFactCheckSample
from vnfake.utils.logging import get_logger

logger = get_logger(__name__)

SplitName = Literal["train", "dev", "test"]


def _iter_jsonl_lines(path: Path) -> Iterator[tuple[int, str]]:
    """Yield ``(lineno, raw_line)`` for non-blank lines of a UTF-8 JSONL file."""
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            stripped = line.strip()
            if stripped:
                yield lineno, stripped


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_rejected(fh: TextIO, lineno: int, error: str, raw: str) -> None:
    """Append one structured rejected record (truncated raw for safety)."""
    payload = {"line_no": lineno, "error": error, "raw": raw[:500]}
    fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _fill_missing_kb_id(obj: dict[str, object], stem: str, lineno: int) -> None:
    """Assign a deterministic fallback ID when the raw line lacks one.

    Prefers a URL hash (stable across runs); falls back to ``<file>-<lineno>``.
    """
    if obj.get("id"):
        return
    url = obj.get("url")
    if isinstance(url, str) and url:
        obj["id"] = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    else:
        obj["id"] = f"{stem}-{lineno}"


def iter_kb_records(path: Path, rejected_path: Path) -> Iterator[KBRecord]:
    """Stream and validate KB records from a JSONL file.

    Args:
        path: Input JSONL file (one KB record per line).
        rejected_path: Where invalid lines are written as JSONL. Overwritten.

    Yields:
        Validated :class:`KBRecord` objects in file order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"KB file not found: {path}")
    _ensure_parent(rejected_path)
    return _iter_kb_impl(path, rejected_path)


def _iter_kb_impl(path: Path, rejected_path: Path) -> Iterator[KBRecord]:
    stem = path.stem
    n_valid = n_rejected = 0
    with rejected_path.open("w", encoding="utf-8") as rej:
        for lineno, raw in _iter_jsonl_lines(path):
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                _write_rejected(rej, lineno, f"json_error: {exc}", raw)
                n_rejected += 1
                continue
            if not isinstance(obj, dict):
                _write_rejected(rej, lineno, "not_a_dict", raw)
                n_rejected += 1
                continue
            _fill_missing_kb_id(obj, stem, lineno)
            try:
                rec = KBRecord.model_validate(obj)
            except ValidationError as exc:
                _write_rejected(rej, lineno, str(exc), raw)
                n_rejected += 1
                continue
            n_valid += 1
            yield rec
    logger.info("KB loaded: %s — valid=%d rejected=%d", path.name, n_valid, n_rejected)


def load_split(
    name: SplitName,
    vifactcheck_dir: Path,
    *,
    allow_test: bool = False,
    rejected_path: Path | None = None,
) -> list[ViFactCheckSample]:
    """Load one ViFactCheck split from ``<vifactcheck_dir>/<name>.jsonl``.

    Args:
        name: Split name.
        vifactcheck_dir: Directory containing ``train.jsonl`` etc.
        allow_test: Must be True to load the ``test`` split (leakage policy;
            only ``evaluation/`` code should pass this via an explicit ``--final``).
        rejected_path: If given, invalid lines are written here as JSONL.

    Raises:
        PermissionError: If ``name == "test"`` and ``allow_test`` is False.
        FileNotFoundError: If the split file does not exist.
    """
    if name == "test" and not allow_test:
        raise PermissionError(
            "Loading the 'test' split requires allow_test=True "
            "(evaluation code only, behind an explicit --final flag)."
        )
    path = vifactcheck_dir / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"ViFactCheck split not found: {path}")

    samples: list[ViFactCheckSample] = []
    rejected: list[dict[str, object]] = []
    for lineno, raw in _iter_jsonl_lines(path):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            rejected.append({"line_no": lineno, "error": f"json_error: {exc}", "raw": raw[:500]})
            continue
        if not isinstance(obj, dict):
            rejected.append({"line_no": lineno, "error": "not_a_dict", "raw": raw[:500]})
            continue
        try:
            samples.append(ViFactCheckSample.model_validate(obj))
        except ValidationError as exc:
            rejected.append({"line_no": lineno, "error": str(exc), "raw": raw[:500]})

    if rejected_path is not None:
        _ensure_parent(rejected_path)
        with rejected_path.open("w", encoding="utf-8") as f:
            for rec in rejected:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    logger.info("ViFactCheck[%s]: valid=%d rejected=%d", name, len(samples), len(rejected))
    return samples
