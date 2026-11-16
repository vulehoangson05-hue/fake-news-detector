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

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from vnfake.utils.logging import get_logger

load_dotenv()
logger = get_logger(__name__)

_GEMINI_FLASH_MODEL_RE = re.compile(r"^gemini-(?!1\.)[a-z0-9.-]*flash[a-z0-9.-]*$")
_DEFAULT_FORBIDDEN_PHRASES: tuple[str, ...] = (
    "xin chào",
    "chào bạn",
    "dạ ",
    "dạ,",
    "vâng",
    "tất nhiên",
    "dưới đây là",
    "giải thích",
    "tôi là",
    "không thể",
    "lưu ý:",
    "chú ý:",
    "bài thơ",
    "mùa thu hà nội",
    "lá vàng rơi",
)
_DEFAULT_DROP_TERMS: tuple[str, ...] = (
    "mấy bữa nay",
    "hôm nay",
    "ngày mai",
    "hôm qua",
    "năm nay",
    "mới đây",
    "sắp tới",
    "tháng sau",
    "chiều nay",
    "sáng nay",
    "tối nay",
    "thật hông",
    "thật không",
    "phải hông",
    "phải không",
    "đúng không",
    "đúng ko",
    "hả",
    "vậy trời",
    "mọi người",
    "bà con",
    "dân tình",
)


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


class EmbeddingConfig(BaseModel):
    """Dense embedding model parameters (Phase 2 Part A)."""

    model_name: str = "BAAI/bge-m3"
    dim: int = 1024
    batch_size: int = 16  # fits bge-m3 fp16 in ~1.5GB VRAM on 4GB cards
    use_fp16: bool = True  # effective only on CUDA


class SparseEmbeddingConfig(BaseModel):
    """Sparse embedding parameters (lexical_weights from bge-m3)."""

    # bge-m3 emits ~30K-dim lexical weights; threshold drops near-zero noise.
    weight_threshold: float = 0.0
    batch_size: int = 16


class HybridConfig(BaseModel):
    """Hybrid (dense + sparse) retrieval parameters."""

    dense_prefetch_k: int = 50
    sparse_prefetch_k: int = 50
    rrf_k: int = 60  # RRF constant; typical 60 per Cormack et al.


class RerankerConfig(BaseModel):
    """Cross-encoder reranker parameters."""

    model_name: str = "BAAI/bge-reranker-v2-m3"
    batch_size: int = 4
    dtype: Literal["fp32", "fp16", "bf16"] = "fp32"
    use_fp16: bool = False
    max_length: int = 512  # truncation for the cross-encoder


class RetrievalConfig(BaseModel):
    """Retrieval / rerank settings."""

    retrieve_k: int = 30
    rerank_k: int = 5
    embedding: EmbeddingConfig = EmbeddingConfig()
    sparse: SparseEmbeddingConfig = SparseEmbeddingConfig()
    hybrid: HybridConfig = HybridConfig()
    reranker: RerankerConfig = RerankerConfig()
    qdrant_collection_prefix: str = "kb_chunks"
    dense_vector_name: str = "dense"
    sparse_vector_name: str = "sparse"


class ClassifierConfig(BaseModel):
    """Classifier model selection and token limits."""

    model_name: str = "vinai/phobert-base-v2"
    max_len_phobert: int = 256
    max_len_xlmr: int = 512


class TrainingConfig(BaseModel):
    """Training hyperparameters (must fit 6GB VRAM on GTX 1660 in fp32)."""

    fp16: bool = False
    gradient_accumulation_steps: int = 4
    per_device_train_batch_size: int = 8
    learning_rate: float = 2.0e-5
    num_train_epochs: int = 4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01


class ChunkConfig(BaseModel):
    """Sliding-window chunking parameters."""

    window: int = 3
    overlap: int = 1


class DedupConfig(BaseModel):
    """MinHash near-duplicate parameters."""

    minhash_threshold: float = 0.8
    minhash_num_perm: int = 128
    minhash_shingle_n: int = 5


class PreprocessingConfig(BaseModel):
    """Text preprocessing knobs."""

    min_chars: int = 100
    chunk: ChunkConfig = ChunkConfig()
    dedup: DedupConfig = DedupConfig()


class SegmentationConfig(BaseModel):
    """Word-segmentation backend selection."""

    backend: Literal["underthesea", "vncorenlp"] = "underthesea"
    vncorenlp_dir: Path = Path("vncorenlp")


class VerifierCheckpointSpec(BaseModel):
    """Pinned verifier checkpoint specification (Brief #5 §12.3.4)."""

    seed: int
    run_id: str
    sha256: str


class DecisionConfig(BaseModel):
    """Thresholds for the relevance gate and decision layer in Phase 4 (Brief #5 §3 & §12.3.4)."""

    tau_rel: float = 0.7559
    tau_conf: float = 0.55
    tau_retrieval: float = 0.35
    ensemble_seeds: list[int] = Field(default_factory=lambda: [42, 43, 44])
    checkpoints: list[VerifierCheckpointSpec] = Field(default_factory=list)
    live_deadline_s: float = 20.0
    use_fp16: bool = False
    demo_mode: bool = False


class QueryRewriteConfig(BaseModel):
    """LLM Query Rewrite settings (ARCHITECTURE §5.3.1, copilot-instructions §6.2)."""

    enabled: bool = False
    model: str = "gemini-3.1-flash-lite"
    timeout_s: float = 10.0
    total_deadline_s: float = 12.0
    max_retries: int = 2
    max_retries_batch: int = 2
    max_retries_online: int = 0
    rpm_limit: int = 12
    daily_request_budget: int = 80
    quota_reset_tz: str = "US/Pacific"
    temperature: float = 0.0
    max_output_tokens: int = 64
    max_keywords: int = 15
    max_chars: int = 256
    fusion: Literal["rewritten_only", "original+rewritten"] = "rewritten_only"
    cache_dir: Path = Path("artifacts/cache/query_rewrite")
    prompt_path: Path = Path("configs/prompts/query_rewrite.txt")
    prompt_version: str = "v3"
    required: bool = False
    forbidden_phrases: list[str] = list(_DEFAULT_FORBIDDEN_PHRASES)
    drop_terms: list[str] = list(_DEFAULT_DROP_TERMS)

    @field_validator("model")
    @classmethod
    def _validate_flash_model(cls, value: str) -> str:
        cleaned = value.strip()
        if not _GEMINI_FLASH_MODEL_RE.match(cleaned):
            raise ValueError(
                f"Invalid query_rewrite.model {value!r}: must match Gemini Flash family "
                f"regex {_GEMINI_FLASH_MODEL_RE.pattern!r} (gemini-1.5-* and non-Flash models are banned)."
            )
        return cleaned


class SearchConfig(BaseModel):
    """Web search configuration loaded from ``configs/search.yaml`` (Brief #4 §3.1)."""

    provider: Literal["tavily"] = "tavily"
    search_depth: Literal["basic", "advanced"] = "basic"
    max_results: int = 5
    topic: Literal["general", "news"] = "general"
    timeout_s: float = 15.0
    max_credits_spike: int = 150
    cache_dir: Path = Path("artifacts/cache/web_search")
    rate_limit_s: float = 1.0
    user_agent: str = "vnfake-research-bot/0.1 (contact: vulehoangson05@gmail.com)"
    include_domains: list[str] = [
        "vnexpress.net",
        "dantri.com.vn",
        "tuoitre.vn",
        "thanhnien.vn",
        "vietnamnet.vn",
        "vtv.vn",
        "vov.vn",
        "nhandan.vn",
        "baochinhphu.vn",
        "chinhphu.vn",
        "plo.vn",
        "laodong.vn",
        "tienphong.vn",
        "nld.com.vn",
        "vietnamplus.vn",
        "znews.vn",
    ]


class AppConfig(BaseModel):
    """Root configuration."""

    seed: int = 42
    log_level: str = "INFO"
    paths: PathsConfig
    vifactcheck: ViFactCheckConfig = ViFactCheckConfig()
    label_map: dict[int, str]
    retrieval: RetrievalConfig = RetrievalConfig()
    query_rewrite: QueryRewriteConfig = QueryRewriteConfig()
    classifier: ClassifierConfig = ClassifierConfig()
    training: TrainingConfig = TrainingConfig()
    decision: DecisionConfig = DecisionConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    segmentation: SegmentationConfig = SegmentationConfig()


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


@lru_cache(maxsize=4)
def load_search_config(path: Path = Path("configs/search.yaml")) -> SearchConfig:
    """Load and validate web search configuration from ``configs/search.yaml``."""
    raw = _read_yaml(Path(path))
    cfg = SearchConfig.model_validate(raw)
    logger.info(
        "Loaded search config from %s (provider=%s, domains=%d)",
        path,
        cfg.provider,
        len(cfg.include_domains),
    )
    return cfg
