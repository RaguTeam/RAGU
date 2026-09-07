"""
Service configuration.
"""

from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ragu.api.models import CAPABILITIES


class ServiceSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAGU_API_", env_file=".env", extra="ignore"
    )

    backend: Literal["ragu", "stub"] = Field(
        default="ragu",
        description="'ragu' loads a real knowledge graph, 'stub' serves canned answers for local development",
    )

    host: str = Field(
        default="127.0.0.1",
        description="Bind address. Loopback by default; the container image passes "
        "--host 0.0.0.0 explicitly, so exposing the service is a deliberate act.",
    )
    port: int = Field(default=8020, description="Bind port")

    storage_folder: str = Field(
        default="ragu_working_dir",
        description="Folder with the built graph. Set explicitly: the default Settings.storage_folder is timestamped "
        "per run and would point at an empty directory.",
    )
    language: str = Field(
        default="russian", description="Graph language passed to Settings.language"
    )
    settings_file: str | None = Field(
        default=None,
        description="Optional Settings JSON saved at build time (Settings.save) to load instead of the defaults",
    )
    embedder_dim: int | None = Field(
        default=None,
        description="Embedding dimension. None auto-detects it with a probe request on the first call.",
    )
    rate_min_delay: float | None = Field(
        default=None, description="Minimum delay between LLM calls, seconds"
    )
    rate_max_simultaneous: int | None = Field(
        default=None, description="Maximum simultaneous LLM calls"
    )
    llm_cache: str | None = Field(
        default=None,
        description="Path to the LLM response cache; None disables caching",
    )

    max_top_k: int = Field(
        default=100,
        gt=0,
        description="Upper bound applied to a client-supplied top_k / rerank_top_k; "
        "requests above it are clamped",
    )
    engine_cache_size: int = Field(
        default=32,
        gt=0,
        description="How many (mode, language) engines to keep built. Clients choose "
        "the language, so the cache is bounded.",
    )
    max_batch_size: int = Field(
        default=50,
        gt=0,
        description="Maximum number of queries accepted by a /batch route",
    )
    min_cluster_size_floor: int = Field(
        default=1,
        gt=0,
        description="Lower bound applied to a client-supplied global min_cluster_size. "
        "Global search rates every surviving community with its own LLM call, so this is "
        "the knob that caps the cost of one request; raise it on large graphs.",
    )

    stub_missing_capabilities: str = Field(
        default="",
        description="Stub backend only: comma-separated capabilities to report as missing "
        "(entity_graph, community_summaries, vector_index)",
    )

    @field_validator("stub_missing_capabilities")
    @classmethod
    def _reject_unknown_capabilities(cls, value: str) -> str:
        """
        Fail on a misspelled capability instead of silently simulating nothing.
        """
        unknown = sorted(cls._split(value) - CAPABILITIES)
        if unknown:
            raise ValueError(
                f"unknown capabilities {unknown}; expected any of {sorted(CAPABILITIES)}"
            )
        return value

    @staticmethod
    def _split(value: str) -> set[str]:
        return {item.strip() for item in value.split(",") if item.strip()}

    def missing_capabilities(self) -> set[str]:
        """
        Capabilities the stub backend should report as absent.
        """
        return self._split(self.stub_missing_capabilities)
