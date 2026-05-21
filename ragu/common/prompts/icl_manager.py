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

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
from uuid import uuid4

import numpy as np

from ragu.common.global_parameters import Settings
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
    :param task: Task name this example belongs to (e.g. ``"entity_extraction"``).
    """

    id: str
    input_text: str
    output: Dict[str, Any]
    metadata: Dict[str, Any]
    language: str
    quality_rating: int | None
    task: str = ""


class InContextLearningManager:
    """
    Manages in-context learning examples with on-the-fly embedding computation.

    This manager loads examples from multiple JSON files (one per task),
    computes embeddings at initialization using the provided Embedder,
    and selects the most relevant examples for queries using semantic
    similarity with optional per-task filtering.

    Key features:
    - Loads examples from multiple task-specific JSON files
    - Embeddings computed on-the-fly at initialization (single pass)
    - Task-based filtering during example selection
    - Semantic selection by cosine similarity
    - Multi-language support
    - Portable examples for any embedder
    - Query embedding caching to avoid redundant API calls

    Example usage:
    ```python
    from ragu.common.prompts.icl_manager import resolve_example_path

    embedder = EmbedderOpenAI(client=client, model_name="text-embedding-3-small", dim=1536)
    config = ICLConfig(num_examples=2)
    manager = InContextLearningManager(
        embedder=embedder,
        example_files={
            "entity_extraction": resolve_example_path(None, "entity_extraction_examples.json"),
            "relation_extraction": resolve_example_path(None, "relation_extraction_examples.json"),
        },
        config=config,
    )

    # Select relevant entity examples for multiple queries (batch)
    examples_per_query = await manager.batch_select_examples(
        query_texts=["Tim Cook announced Apple Vision Pro...", "Another text..."],
        task="entity_extraction",
        num_examples=2
    )
    ```

    :param embedder: Embedder instance for computing embeddings.
    :param example_files: Mapping from task name to path of JSON file with examples.
    :param config: ICL configuration.
    :param language: Target language for example selection.
        Defaults to ``Settings.language`` when ``None``.
    """

    def __init__(
        self,
        embedder: Embedder,
        example_files: Dict[str, str],
        config: ICLConfig,
        language: str | None = None,
    ):
        """
        Initialize ICL manager.

        :param embedder: Embedder for computing embeddings on-the-fly.
        :param example_files: Mapping from task name to JSON file path with examples.
        :param config: ICL configuration.
        :param language: Target language for example selection.
            Defaults to ``Settings.language`` when ``None``.
        """
        self.embedder = embedder
        self.example_files = example_files
        self.config = config
        self.language = language if language else Settings.language
        self.examples: List[Example] = []
        self._task_indices: Dict[str, List[int]] = {}
        self._embeddings_computed = False
        self._example_matrix: np.ndarray | None = None
        self._example_norms: np.ndarray | None = None
        self._cached_query_key: int | None = None
        self._cached_query_matrix: np.ndarray | None = None

    async def initialize(self) -> None:
        """
        Load examples and compute embeddings.

        This should be called after initialization to load examples
        and compute embeddings for all example texts.  Subsequent
        calls are no-ops — examples and embeddings are reused.

        For 20-50 examples, this typically takes < 1 second.
        """
        if self._embeddings_computed:
            return
        await self._load_examples()
        await self._compute_embeddings()
        self._embeddings_computed = True

        task_summary = ", ".join(
            f"{task}={len(indices)}" for task, indices in self._task_indices.items()
        )
        logger.info(
            f"Initialized InContextLearningManager with "
            f"{len(self.examples)} examples for language '{self.language}' "
            f"({task_summary})"
        )

    async def _load_examples(self) -> None:
        """
        Load examples from JSON files.

        Iterates over ``example_files``, loads each JSON, filters by
        language, and tags each example with its task name.
        Builds ``_task_indices`` for efficient task-based filtering.
        """
        self.examples = []
        self._task_indices = {}

        for task_name, file_path in self.example_files.items():
            if not os.path.exists(file_path):
                logger.warning(f"Example file not found: {file_path}")
                continue

            def _read_sync(path: str = file_path) -> list[dict]:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f).get("examples", [])

            examples_data = await asyncio.to_thread(_read_sync)
            task_start = len(self.examples)

            for ex_data in examples_data:
                example_language = ex_data.get("metadata", {}).get("language", "english")
                if example_language != self.language:
                    continue

                example = Example(
                    id=ex_data.get("id", str(uuid4())),
                    input_text=ex_data["input_text"],
                    output=ex_data["output"],
                    metadata=ex_data.get("metadata", {}),
                    language=example_language,
                    quality_rating=ex_data.get("quality_rating"),
                    task=task_name,
                )
                self.examples.append(example)

            task_indices = list(range(task_start, len(self.examples)))
            self._task_indices[task_name] = task_indices

            logger.debug(
                f"Loaded {len(task_indices)} examples for task '{task_name}', "
                f"language '{self.language}' from {file_path}"
            )

    async def _compute_embeddings(self) -> None:
        """
        Compute embeddings for all example texts.

        Embeddings are stored as a precomputed matrix for fast
        vectorized similarity search.
        """
        if not self.examples:
            return

        texts = [ex.input_text for ex in self.examples]

        logger.debug("Computing embeddings for examples...")
        embeddings = await self.embedder.batch_embed_text(
            texts=texts,
            desc="Computing example embeddings",
        )

        self._example_matrix = np.array(embeddings, dtype=np.float32)
        self._example_norms = np.linalg.norm(self._example_matrix, axis=1)

        logger.debug("Computed embeddings for all examples")

    async def _get_query_matrix(self, query_texts: List[str]) -> np.ndarray:
        """
        Get query embeddings matrix, using cached result if texts unchanged.

        :param query_texts: Input texts to embed.
        :return: Query embeddings as (Q, D) matrix.
        """
        query_key = hash(tuple(query_texts))
        if self._cached_query_key == query_key and self._cached_query_matrix is not None:
            return self._cached_query_matrix

        query_embeddings = await self.embedder.batch_embed_text(
            texts=query_texts,
        )
        matrix = np.array(query_embeddings, dtype=np.float32)
        self._cached_query_key = query_key
        self._cached_query_matrix = matrix
        return matrix

    async def batch_select_examples(
        self,
        query_texts: List[str],
        task: str | None = None,
        num_examples: int | None = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        Select most relevant examples for a batch of queries.

        Computes all query embeddings in a single batch API call, then
        selects top-k examples per query using cosine similarity.

        When ``task`` is provided, only examples tagged with that task
        name are considered during selection.

        Query embeddings are cached: if the same ``query_texts`` list is
        passed on a subsequent call (e.g. extraction then validation),
        the embeddings are reused without an additional API call.

        :param query_texts: Input texts for which to select examples.
        :param task: Task name to filter examples by (e.g. ``"entity_extraction"``).
            When ``None``, all loaded examples are considered.
        :param num_examples: Number of examples to return per query
            (uses config default if None).
        :return: Per-query lists of example dictionaries.
        """
        if not self._embeddings_computed:
            logger.warning(
                "Embeddings not computed. Call initialize() first."
            )
            return [[] for _ in query_texts]

        if num_examples is None:
            num_examples = self.config.num_examples

        if task is not None:
            candidate_indices = self._task_indices.get(task, [])
        else:
            candidate_indices = list(range(len(self.examples)))

        if not candidate_indices or self._example_matrix is None:
            return [[] for _ in query_texts]

        query_matrix = await self._get_query_matrix(query_texts)

        idx_arr = np.array(candidate_indices, dtype=np.intp)
        ex_matrix = self._example_matrix[idx_arr]
        ex_norms = self._example_norms[idx_arr]

        query_norms = np.linalg.norm(query_matrix, axis=1)
        valid_examples = ex_norms > 0.0
        ex_norms_safe = np.where(valid_examples, ex_norms, 1.0)

        sim_matrix = (query_matrix @ ex_matrix.T) / (
            query_norms[:, np.newaxis] * ex_norms_safe[np.newaxis, :]
        )
        sim_matrix[:, ~valid_examples] = 0.0

        results: List[List[Dict[str, Any]]] = []
        for i in range(len(query_texts)):
            if query_norms[i] == 0.0:
                results.append([])
                continue

            similarities = sim_matrix[i]
            indices = np.where(similarities >= self.config.similarity_threshold)[0]

            if len(indices) == 0:
                logger.debug(
                    f"No examples passed similarity threshold "
                    f"({self.config.similarity_threshold}) for query {i}"
                )
                results.append([])
                continue

            top_local = indices[np.argsort(similarities[indices])[-num_examples:]][::-1]
            top_global = idx_arr[top_local]

            selected = []
            for idx in top_global:
                example = self.examples[idx]
                selected.append({
                    "id": example.id,
                    "input_text": example.input_text,
                    "output": example.output,
                    "metadata": example.metadata,
                    "language": example.language,
                    "quality_rating": example.quality_rating,
                })

            logger.debug(
                f"Selected {len(selected)} examples for query {i} "
                f"(task='{task}', similarities: {[float(similarities[j]) for j in top_local]})"
            )

            results.append(selected)

        total = len(results)
        empty_count = sum(1 for r in results if not r)
        if total > 0 and self.config.low_match_warning_threshold > 0.0 and empty_count / total >= self.config.low_match_warning_threshold:
            logger.warning(
                f"ICL low match rate for task='{task}': "
                f"{empty_count}/{total} queries ({empty_count / total:.0%}) "
                f"received no examples (similarity_threshold={self.config.similarity_threshold}, "
                f"available_examples={len(candidate_indices)}). "
                f"Consider lowering similarity_threshold or adding more examples."
            )

        return results
