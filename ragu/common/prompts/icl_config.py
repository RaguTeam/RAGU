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
        When ``None`` (default), the built-in examples shipped with the package
        are used.  Pass an absolute or relative path to use custom examples.
    :param selection_strategy: Strategy for selecting relevant examples.

        - ``"semantic"``: cosine similarity on dense embeddings (default).
          Requires an ``Embedder``.
        - ``"bm25"``: lexical matching via BM25 sparse embeddings (FastEmbed).
          No embedder needed, fast and terminology-focused.
        - ``"hybrid"``: Reciprocal Rank Fusion of semantic and BM25 rankings.
          Requires an ``Embedder``.
        - ``"random"``: uniform random sampling from the candidate pool.
          Useful as a baseline for evaluating other strategies.

    :param low_match_warning_threshold: Fraction of queries that received no
        examples at which a WARNING is logged (0.3 = warn when 30%+ queries
        are unmatched).  Set to ``0.0`` to disable the warning or ``1.0``
        to warn only when every query is unmatched.
    """

    enabled: bool = True
    num_examples: int = 2
    examples_base_path: str | None = None
    selection_strategy: Literal["semantic", "bm25", "hybrid", "random"] = "semantic"
    low_match_warning_threshold: float = 0.3
