"""Package source code and processed/selected ViFactCheck splits into a Colab bundle (Brief #4 §2.7).

Creates ``artifacts/colab_verifier_bundle.zip`` containing:
- ``src/vnfake/``
- ``configs/``
- ``scripts/09_select_evidence.py``, ``scripts/10_train_verifier.py``
- ``pyproject.toml``
- ``data/processed/{train,dev}_clean.jsonl`` (excludes ``test_clean.jsonl`` per leakage policy)
- ``data/interim/selected_{train,dev}.jsonl``
- ``notebooks/01_colab_train_verifier.ipynb``
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from vnfake.config import load_config
from vnfake.utils.logging import get_logger

logger = get_logger(__name__)


def _build_colab_notebook() -> dict[str, object]:
    """Return a minimal Jupyter notebook dictionary for Google Colab T4 training."""
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Phase 3 Verifier Training on Google Colab (T4 fp16)\n",
                "1. Upload `colab_verifier_bundle.zip` to `/content/`.\n",
                "2. Run all cells below to unzip, install dependencies, train Baseline & V-A/V-B/V-C, and zip `artifacts/models/` for download.",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!unzip -q -o /content/colab_verifier_bundle.zip -d /content/vnfake_work\n",
                "%cd /content/vnfake_work\n",
                "!pip install -q -e .",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!python scripts/10_train_verifier.py --variant baseline --seed 42 --fp16 --online-hf --run-id phobert_baseline_s42\n",
                "!python scripts/10_train_verifier.py --variant V-A --seed 42 --fp16 --online-hf --run-id phobert_va_s42\n",
                "!python scripts/10_train_verifier.py --variant V-B --seed 42 --fp16 --online-hf --run-id phobert_vb_s42\n",
                "!python scripts/10_train_verifier.py --variant V-C --seed 42 --fp16 --online-hf --run-id phobert_vc_s42",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["!zip -r /content/verifier_checkpoints.zip artifacts/models/"],
        },
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def pack_for_colab(output_zip: Path | None = None) -> Path:
    """Create the zip bundle for Colab training and return its path."""
    cfg = load_config()
    target_zip = output_zip or (cfg.paths.artifacts_dir / "colab_verifier_bundle.zip")
    target_zip.parent.mkdir(parents=True, exist_ok=True)

    nb_path = Path("notebooks/01_colab_train_verifier.ipynb")
    nb_path.parent.mkdir(parents=True, exist_ok=True)
    nb_path.write_text(
        json.dumps(_build_colab_notebook(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    files_to_pack: list[Path] = [Path("pyproject.toml"), nb_path]
    for folder in (Path("src/vnfake"), Path("configs")):
        for p in folder.rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts:
                files_to_pack.append(p)

    for script_name in ("09_select_evidence.py", "10_train_verifier.py"):
        sp = Path("scripts") / script_name
        if sp.exists():
            files_to_pack.append(sp)

    for split_file in (
        cfg.paths.processed_dir / "train_clean.jsonl",
        cfg.paths.processed_dir / "dev_clean.jsonl",
        cfg.paths.interim_dir / "selected_train.jsonl",
        cfg.paths.interim_dir / "selected_dev.jsonl",
    ):
        if split_file.exists():
            files_to_pack.append(split_file)

    with zipfile.ZipFile(target_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in files_to_pack:
            zf.write(file_path, arcname=str(file_path).replace("\\", "/"))

    logger.info("Packed %d files into Colab bundle: %s", len(files_to_pack), target_zip)
    return target_zip


if __name__ == "__main__":
    pack_for_colab()
