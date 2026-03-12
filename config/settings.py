"""Centralised environment configuration for Churro.

This module ensures `.env` loading happens in one place and exposes a
typed snapshot of provider credentials, runtime ports, and tuning knobs.
Downstream modules call `get_settings()` instead of touching `os.environ`
directly, making it easier to validate values and override behaviour in
tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path

from dotenv import load_dotenv


_DEFAULT_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


def _coerce_int(value: str | None) -> int | None:
    """Parse optional integer environment values safely."""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _coerce_float(value: str | None) -> float | None:
    """Parse optional float environment values safely."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _coerce_bool(value: str | None) -> bool:
    """Treat 1/true/yes (case-insensitive) as True; anything else as False."""
    if value is None or value == "":
        return False
    return value.strip().lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class AzureOpenAISettings:
    api_base: str | None
    api_version: str | None
    api_key: str | None


@dataclass(frozen=True)
class AzureDocumentIntelligenceSettings:
    endpoint: str | None
    api_key: str | None


@dataclass(frozen=True)
class VertexAISettings:
    project_id: str | None
    location: str
    document_ai_location: str
    ocr_processor_id: str | None
    ocr_processor_version: str | None


@dataclass(frozen=True)
class LocalRuntimeSettings:
    vllm_port: int | None
    huggingface_token: str | None
    # When True, orchestration code should reuse an already-running local vLLM
    # endpoint instead of launching a Docker container for vLLM.
    use_existing_vllm: bool
    # Optional override for expected served model id exposed by /v1/models.
    # If unset, engine key (e.g. "churro") is treated as the expected model id.
    vllm_model_name: str | None


@dataclass(frozen=True)
class TokenSettings:
    openai: str | None
    mistral: str | None


@dataclass(frozen=True)
class ChurroSettings:
    """Top-level snapshot of configuration values."""

    env_file: Path
    azure_openai: AzureOpenAISettings
    azure_document_intelligence: AzureDocumentIntelligenceSettings
    vertex_ai: VertexAISettings
    google_cloud_project: str | None
    local: LocalRuntimeSettings
    tokens: TokenSettings


def _resolve_env_path(env_file: os.PathLike[str] | str | None) -> Path:
    if env_file is None:
        return _DEFAULT_ENV_PATH
    return Path(env_file).resolve()


@lru_cache(maxsize=4)
def _load_settings(env_path: Path) -> ChurroSettings:
    # Load the environment file once per unique path. We avoid override=True so
    # that existing environment variables take precedence over `.env` defaults.
    load_dotenv(dotenv_path=env_path, override=False)

    azure_openai = AzureOpenAISettings(
        api_base=os.getenv("AZURE_API_BASE"),
        api_version=os.getenv("AZURE_API_VERSION"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"),
    )
    azure_di = AzureDocumentIntelligenceSettings(
        endpoint=os.getenv("AZURE_DI_ENDPOINT"),
        api_key=os.getenv("AZURE_DOC_KEY"),
    )

    # Support both variable names for project id to preserve compatibility with
    # existing environments/scripts.
    combined_project_id = os.getenv("VERTEX_AI_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
    vertex_ai = VertexAISettings(
        project_id=combined_project_id,
        location=os.getenv("VERTEX_AI_LOCATION", "us-east5"),
        document_ai_location=os.getenv("DOCUMENT_VERTEX_AI_LOCATION", "us"),
        ocr_processor_id=os.getenv("VERTEX_AI_OCR_PROCESSOR_ID"),
        ocr_processor_version=os.getenv("VERTEX_AI_OCR_PROCESSOR_VERSION"),
    )

    # Reuse mode is intentionally opt-in. It is required for external vLLM
    # deployments where benchmark/infer should not attempt Docker startup.
    use_existing_vllm = _coerce_bool(os.getenv("USE_EXISTING_VLLM"))
    # In reuse mode, default to port 8000 since that is the common vLLM serve port.
    # Otherwise keep repository default of 9000 for Docker-managed vLLM.
    default_local_port = "8000" if use_existing_vllm else "9000"
    local_runtime = LocalRuntimeSettings(
        # LOCAL_VLLM_PORT is optional in env: we inject a mode-aware default.
        vllm_port=_coerce_int(os.getenv("LOCAL_VLLM_PORT") or default_local_port),
        huggingface_token=os.getenv("HF_TOKEN"),
        use_existing_vllm=use_existing_vllm,
        # Empty strings are normalized to None to simplify downstream checks.
        vllm_model_name=(os.getenv("LOCAL_VLLM_MODEL_NAME") or "").strip() or None,
    )

    tokens = TokenSettings(
        openai=os.getenv("OPENAI_API_KEY"),
        mistral=os.getenv("MISTRAL_API_KEY"),
    )

    return ChurroSettings(
        env_file=env_path,
        azure_openai=azure_openai,
        azure_document_intelligence=azure_di,
        vertex_ai=vertex_ai,
        google_cloud_project=combined_project_id,
        local=local_runtime,
        tokens=tokens,
    )


def get_settings(
    env_file: os.PathLike[str] | str | None = None,
    *,
    reload: bool = False,
) -> ChurroSettings:
    """Return the cached settings snapshot.

    Args:
        env_file: Optional explicit path to a `.env` file. When omitted the repo
            root `.env` file is used.
        reload: When True the cached snapshot is cleared before loading.
    """
    env_path = _resolve_env_path(env_file)
    if reload:
        _load_settings.cache_clear()
    return _load_settings(env_path)
