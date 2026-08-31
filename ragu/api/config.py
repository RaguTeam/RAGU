"""
Service configuration.
"""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServiceSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAGU_API_", env_file=".env", extra="ignore"
    )

    backend: Literal["ragu", "stub"] = Field(
        default="ragu",
        description="'ragu' loads a real knowledge graph, 'stub' serves canned answers for local development",
    )

    host: str = Field(default="0.0.0.0", description="Bind address")
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
        description="Upper bound applied to a client-supplied top_k; requests above it are clamped",
    )

    stub_missing_capabilities: str = Field(
        default="",
        description="Stub backend only: comma-separated capabilities to report as missing "
        "(entity_graph, community_summaries, vector_index)",
    )

    def missing_capabilities(self) -> set[str]:
        return {
            item.strip()
            for item in self.stub_missing_capabilities.split(",")
            if item.strip()
        }
