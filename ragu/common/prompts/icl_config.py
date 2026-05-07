"""
In-context learning configuration for RAGU extractors.

This module provides configuration for managing few-shot examples
that stabilize LLM-based entity and relation extraction.

Classes
-------
ICLConfig - Configuration for in-context learning behavior.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class ICLConfig:
    """
    Configuration for in-context learning.

    Controls how few-shot examples are selected and used
    to stabilize LLM-based artifact extraction.

    :param enabled: Enable or disable in-context learning entirely.
    :param num_examples: Number of examples to include per query (1-3 recommended).
    :param examples_base_path: Base directory path for JSON example files.
    :param selection_strategy: Strategy for selecting relevant examples.
    :param similarity_threshold: Minimum cosine similarity for example inclusion.
    :param cache_embeddings: Cache example embeddings in memory after initialization.
    :param language: Target language for example selection.
    """

    enabled: bool = True
    num_examples: int = 2
    examples_base_path: str = "ragu/common/prompts/icl_examples"
    selection_strategy: Literal["semantic", "hybrid"] = "semantic"
    similarity_threshold: float = 0.3
    cache_embeddings: bool = True
    language: str = "english"
