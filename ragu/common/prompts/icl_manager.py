"""
In-context learning manager for RAGU extractors.

This module manages few-shot examples and provides semantic
selection of relevant examples for LLM-based extraction.

Key design decisions:
- Embeddings are computed at initialization using provided Embedder
- Example storage is independent of specific embedding model
- Portable examples can be used with any embedder
- Supports semantic selection strategies

Classes
-------
InContextLearningManager - Manages example loading, embedding, and selection.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
from uuid import uuid4

import numpy as np

from ragu.common.logger import logger
from ragu.models.embedder import Embedder
from ragu.common.prompts.icl_config import ICLConfig

_BUILTIN_EXAMPLES_DIR = Path(__file__).parent / "icl_examples"


def resolve_example_path(base_path: str | None, filename: str) -> str:
    """
    Resolve the full path to an ICL example file.

    :param base_path: Custom base directory, or ``None`` to use the
        built-in examples shipped with the package.
    :param filename: JSON file name (e.g. ``"artifact_extraction_examples.json"``).
    :return: Absolute path as a string.
    """
    if base_path is None:
        return str(_BUILTIN_EXAMPLES_DIR / filename)
    p = Path(base_path)
    if not p.is_absolute():
        p = Path.cwd() / p
    return str(p / filename)


@dataclass(frozen=True, slots=True)
class Example:
    """
    Single in-context learning example.

    :param id: Unique identifier for the example.
    :param input_text: Input text for the example.
    :param output: Structured output (entities, relations, etc.).
    :param metadata: Additional metadata (domain, difficulty, etc.).
    :param language: Language of the example.
    :param quality_rating: Quality rating from judge (1-10).
    """

    id: str
    input_text: str
    output: Dict[str, Any]
    metadata: Dict[str, Any]
    language: str
    quality_rating: int | None


class InContextLearningManager:
    """
    Manages in-context learning examples with on-the-fly embedding computation.

    This manager loads examples from JSON files, computes embeddings
    at initialization using the provided Embedder, and selects
    the most relevant examples for queries using semantic similarity.

    Key features:
    - Embeddings computed on-the-fly at initialization
    - Example storage independent of embedding model
    - Semantic selection by cosine similarity
    - Multi-language support
    - Portable examples for any embedder

    Example usage:
    ```python
    from ragu.common.prompts.icl_manager import resolve_example_path

    embedder = EmbedderOpenAI(client=client, model_name="text-embedding-3-small", dim=1536)
    config = ICLConfig(num_examples=2, language="english")
    manager = InContextLearningManager(
        embedder=embedder,
        example_file=resolve_example_path(None, "artifact_extraction_examples.json"),
        config=config,
        language="english"
    )

    # Select relevant examples for a query
    examples = await manager.select_examples(
        query_text="Tim Cook announced Apple Vision Pro...",
        num_examples=2
    )
    ```

    :param embedder: Embedder instance for computing embeddings.
    :param example_file: Path to JSON file containing examples.
    :param config: ICL configuration.
    :param language: Target language for example selection.
    """

    def __init__(
        self,
        embedder: Embedder,
        example_file: str,
        config: ICLConfig,
        language: str = "english",
    ):
        """
        Initialize ICL manager.

        :param embedder: Embedder for computing embeddings on-the-fly.
        :param example_file: Path to JSON file with examples.
        :param config: ICL configuration.
        :param language: Target language for example selection.
        """
        self.embedder = embedder
        self.example_file = example_file
        self.config = config
        self.language = language
        self.examples: List[Example] = []
        self._embeddings: Dict[str, np.ndarray] = {}
        self._embeddings_computed = False

    async def initialize(self) -> None:
        """
        Load examples and compute embeddings.

        This should be called after initialization to load examples
        and compute embeddings for all example texts.

        For 20-50 examples, this typically takes < 1 second.
        """
        await self._load_examples()
        await self._compute_embeddings()
        self._embeddings_computed = True
        logger.info(
            f"Initialized InContextLearningManager with "
            f"{len(self.examples)} examples for language '{self.language}'"
        )

    async def _load_examples(self) -> None:
        """
        Load examples from JSON file.

        Loads examples and filters by language if specified.
        """
        if not os.path.exists(self.example_file):
            logger.warning(f"Example file not found: {self.example_file}")
            return

        with open(self.example_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        examples_data = data.get("examples", [])
        self.examples = []

        for ex_data in examples_data:
            example_language = ex_data.get("language", "english")
            if example_language != self.language:
                continue

            example = Example(
                id=ex_data.get("id", str(uuid4())),
                input_text=ex_data["input_text"],
                output=ex_data["output"],
                metadata=ex_data.get("metadata", {}),
                language=example_language,
                quality_rating=ex_data.get("quality_rating"),
            )
            self.examples.append(example)

        logger.debug(
            f"Loaded {len(self.examples)} examples for language '{self.language}' "
            f"from {self.example_file}"
        )

    async def _compute_embeddings(self) -> None:
        """
        Compute embeddings for all example texts.

        Embeddings are stored in a separate dict keyed by example id,
        keeping Example objects immutable and free of derived data.
        """
        if not self.examples:
            return

        texts = [ex.input_text for ex in self.examples]

        logger.debug("Computing embeddings for examples...")
        embeddings = await self.embedder.batch_embed_text(
            texts=texts,
            desc="Computing example embeddings",
        )

        if self.config.cache_embeddings:
            for example, emb in zip(self.examples, embeddings):
                self._embeddings[example.id] = np.array(emb, dtype=np.float32)
        else:
            self._embeddings.clear()

        logger.debug("Computed embeddings for all examples")

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.

        :param a: First vector.
        :param b: Second vector.
        :return: Cosine similarity in [-1, 1], or 0.0 for zero-norm vectors.
        """
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    async def select_examples(
        self,
        query_text: str,
        num_examples: int | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Select most relevant examples for a given query.

        Process:
        1. Compute query embedding
        2. Compute cosine similarity with example embeddings
        3. Return top-k examples by similarity

        :param query_text: Input text for which to select examples.
        :param num_examples: Number of examples to return (uses config default if None).
        :return: List of example dictionaries with input_text and output.
        """
        if not self.examples:
            return []

        if not self._embeddings_computed:
            logger.warning(
                "Embeddings not computed. Call initialize() first."
            )
            return []

        if num_examples is None:
            num_examples = self.config.num_examples

        query_embedding = np.array(
            await self.embedder.embed_text(query_text),
            dtype=np.float32,
        )

        similarities: List[float] = []
        for example in self.examples:
            emb = self._embeddings.get(example.id)
            if emb is None:
                similarities.append(0.0)
                continue
            similarities.append(self._cosine_similarity(query_embedding, emb))

        similarities_array = np.array(similarities)
        indices = np.where(
            similarities_array >= self.config.similarity_threshold
        )[0]

        if len(indices) == 0:
            logger.debug(
                f"No examples passed similarity threshold "
                f"({self.config.similarity_threshold})"
            )
            return []

        top_indices = indices[np.argsort(similarities_array[indices])[-num_examples:]][::-1]

        selected_examples = []
        for idx in top_indices:
            example = self.examples[idx]
            selected_examples.append({
                "id": example.id,
                "input_text": example.input_text,
                "output": example.output,
                "metadata": example.metadata,
                "language": example.language,
                "quality_rating": example.quality_rating,
            })

        logger.debug(
            f"Selected {len(selected_examples)} examples for query "
            f"(similarities: {[similarities[i] for i in top_indices]})"
        )

        return selected_examples
