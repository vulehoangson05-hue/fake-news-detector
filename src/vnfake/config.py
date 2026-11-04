"""Typed application configuration loaded from YAML + environment.

All modules must obtain config via :func:`load_config`. Do not instantiate
``AppConfig`` directly outside tests.

Design notes:
    - YAML provides defaults.
    - Environment variables prefixed with ``VNFAKE_`` override the loader
      (e.g. ``VNFAKE_CONFIG_PATH``).
    - Result is cached per path with :func:`functools.lru_cache`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from vnfake.utils.logging import get_logger

logger = get_logger(__name__)


class PathsConfig(BaseModel):
    """Filesystem paths used across pipelines."""

    kb_dantri: Path
    kb_vnexpress: Path
    vifactcheck_dir: Path
    interim_dir: Path
    processed_dir: Path
    artifacts_dir: Path


class ViFactCheckConfig(BaseModel):
    """Field names inside ViFactCheck JSONL files."""

    statement_field: str = "Statement"
    evidence_field: str = "Evidence"
    context_field: str = "Context"
    label_field: str = "labels"


class RetrievalConfig(BaseModel):
    """Retrieval / rerank settings."""

    retrieve_k: int = 50
    rerank_k: int = 5
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    qdrant_collection_prefix: str = "kb_chunks"


class ClassifierConfig(BaseModel):
    """Classifier model selection and token limits."""

    model_name: str = "vinai/phobert-base-v2"
    max_len_phobert: int = 256
    max_len_xlmr: int = 512


class TrainingConfig(BaseModel):
    """Training hyperparameters (must fit 6GB VRAM)."""

    fp16: bool = True
    gradient_accumulation_steps: int = 4
    per_device_train_batch_size: int = 8
    learning_rate: float = 2.0e-5
    num_train_epochs: int = 4
    warmup_ratio: float = 0.1


class DecisionConfig(BaseModel):
    """Thresholds for the abstain rule in Pipeline C."""

    tau_retrieval: float = 0.35
    tau_conf: float = 0.60


class AppConfig(BaseModel):
    """Root configuration."""

    seed: int = 42
    log_level: str = "INFO"
    paths: PathsConfig
    vifactcheck: ViFactCheckConfig = ViFactCheckConfig()
    label_map: dict[int, str]
    retrieval: RetrievalConfig = RetrievalConfig()
    classifier: ClassifierConfig = ClassifierConfig()
    training: TrainingConfig = TrainingConfig()
    decision: DecisionConfig = DecisionConfig()


class _Settings(BaseSettings):
    """Environment-variable overrides (prefix ``VNFAKE_``)."""

    model_config = SettingsConfigDict(
        env_prefix="VNFAKE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    config_path: Path = Path("configs/base.yaml")


def _read_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and assert it contains a mapping."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at top of {path}, got {type(data).__name__}")
    return data


@lru_cache(maxsize=4)
def load_config(path: Path | None = None) -> AppConfig:
    """Load and validate configuration.

    Args:
        path: Optional override; defaults to ``VNFAKE_CONFIG_PATH`` or
            ``configs/base.yaml``.

    Returns:
        Validated :class:`AppConfig` (cached per path).
    """
    target = path or _Settings().config_path
    raw = _read_yaml(Path(target))
    cfg = AppConfig.model_validate(raw)
    logger.info("Loaded config from %s (seed=%d)", target, cfg.seed)
    return cfg
