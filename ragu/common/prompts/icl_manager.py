"""
In-context learning manager for RAGU extractors.

This module manages few-shot examples and provides multiple
selection strategies (semantic, BM25, hybrid, random) for
LLM-based extraction.

Key design decisions:
- Embeddings are computed at initialization using provided Embedder
  (required for ``semantic`` and ``hybrid`` strategies)
- BM25 sparse embeddings are built at initialization via FastEmbed BM25
  (used by ``bm25`` and ``hybrid`` strategies)
- ``random`` strategy needs neither embedder nor BM25
- Example storage is independent of specific embedding model
- Portable examples can be used with any embedder

Classes
-------
InContextLearningManager - Manages example loading, embedding, and selection.
"""

from __future__ import annotations

import asyncio
import json
import os
import random as _random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

import numpy as np

from ragu.common.global_parameters import Settings
from ragu.common.logger import logger
from ragu.models.embedder import Embedder
from ragu.models.sparse_embedder import BM25 as BM25SparseEmbedder
from ragu.common.prompts.icl_config import ICLConfig
from ragu.storage.types import SparseEmbedding

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
    Manages in-context learning examples with multiple selection strategies.

    This manager loads examples from multiple JSON files (one per task),
    computes embeddings and/or builds a BM25 index at initialization,
    and selects the most relevant examples for queries using one of
    four strategies:

    - ``"semantic"``: cosine similarity on dense embeddings.
    - ``"bm25"``: lexical matching via BM25 sparse embeddings (FastEmbed).
    - ``"hybrid"``: Reciprocal Rank Fusion of semantic and BM25 rankings.
    - ``"random"``: uniform random sampling.

    Example usage:
    ```python
    from ragu.common.prompts.icl_manager import resolve_example_path

    embedder = EmbedderOpenAI(client=client, model_name="text-embedding-3-small", dim=1536)
    config = ICLConfig(num_examples=2, selection_strategy="hybrid")
    manager = InContextLearningManager(
        example_files={
            "entity_extraction": resolve_example_path(None, "entity_extraction_examples.json"),
            "relation_extraction": resolve_example_path(None, "relation_extraction_examples.json"),
        },
        config=config,
        embedder=embedder,
    )

    # Select relevant entity examples for multiple queries (batch)
    examples_per_query = await manager.batch_select_examples(
        query_texts=["Tim Cook announced Apple Vision Pro...", "Another text..."],
        task="entity_extraction",
        num_examples=2
    )
    ```

    :param example_files: Mapping from task name to path of JSON file with examples.
    :param config: ICL configuration.
    :param embedder: Embedder instance for computing embeddings.
        Required for ``"semantic"`` and ``"hybrid"`` strategies,
        not needed for ``"bm25"`` and ``"random"``.
    :param language: Target language for example selection.
        Defaults to ``Settings.language`` when ``None``.
    """

    def __init__(
        self,
        example_files: Dict[str, str],
        config: ICLConfig,
        embedder: Embedder | None = None,
        language: str | None = None,
    ):
        self.example_files = example_files
        self.config = config
        self.embedder = embedder
        self.language = language if language else Settings.language
        self.examples: List[Example] = []
        self._task_indices: Dict[str, List[int]] = {}
        self._initialized = False
        self._example_matrix: np.ndarray | None = None
        self._example_norms: np.ndarray | None = None
        self._cached_query_key: int | None = None
        self._cached_query_matrix: np.ndarray | None = None
        self._bm25_embedder: BM25SparseEmbedder | None = None
        self._bm25_doc_embeddings: list[SparseEmbedding] | None = None
        self._bm25_task_embeddings: Dict[str, tuple[list[SparseEmbedding], List[int]]] | None = None

    async def initialize(self) -> None:
        """
        Load examples and build indices according to the selected strategy.

        Subsequent calls are no-ops — examples and indices are reused.
        """
        if self._initialized:
            return

        await self._load_examples()

        strategy = self.config.selection_strategy
        if strategy in ("semantic", "hybrid"):
            self._validate_embedder()
            await self._compute_embeddings()
        if strategy in ("bm25", "hybrid"):
            self._build_bm25_index()

        self._initialized = True

        task_summary = ", ".join(
            f"{task}={len(indices)}" for task, indices in self._task_indices.items()
        )
        logger.info(
            f"Initialized InContextLearningManager with "
            f"{len(self.examples)} examples for language '{self.language}' "
            f"(strategy='{strategy}', {task_summary})"
        )

    def _validate_embedder(self) -> None:
        if self.embedder is None:
            raise ValueError(
                f"Embedder is required for selection_strategy="
                f"'{self.config.selection_strategy}'"
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

    def _build_bm25_index(self) -> None:
        """
        Build BM25 sparse embeddings over example texts using FastEmbed BM25.

        Constructs a global embedding list (for ``task=None`` fallback) and
        per-task subsets (one for each key in ``_task_indices``).  Per-task
        subsets avoid post-hoc filtering.
        """
        if not self.examples:
            return

        self._bm25_embedder = BM25SparseEmbedder(language=self.language)

        texts = [ex.input_text for ex in self.examples]
        self._bm25_doc_embeddings = self._bm25_embedder.embed_document(texts)

        self._bm25_task_embeddings = {}
        for task_name, indices in self._task_indices.items():
            if not indices:
                continue
            task_embeddings = [self._bm25_doc_embeddings[i] for i in indices]
            self._bm25_task_embeddings[task_name] = (task_embeddings, indices)

        task_summary = ", ".join(
            f"{t}={len(idx)}" for t, idx in self._task_indices.items()
        )
        logger.debug(
            f"Built BM25 sparse embeddings: global ({len(self.examples)} docs), "
            f"per-task ({task_summary})"
        )

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

    def _get_candidate_indices(self, task: str | None) -> List[int]:
        if task is not None:
            return self._task_indices.get(task, [])
        return list(range(len(self.examples)))

    @staticmethod
    def _example_to_dict(example: Example) -> Dict[str, Any]:
        return {
            "id": example.id,
            "input_text": example.input_text,
            "output": example.output,
            "metadata": example.metadata,
            "language": example.language,
            "quality_rating": example.quality_rating,
        }

    @staticmethod
    def _sparse_dot_product(a: SparseEmbedding, b: SparseEmbedding) -> float:
        b_map = dict(zip(b.indices, b.values))
        return sum(
            val * b_map.get(idx, 0.0)
            for idx, val in zip(a.indices, a.values)
        )

    def _check_low_match_rate(
        self,
        results: List[List[Dict[str, Any]]],
        task: str | None,
        candidate_count: int,
    ) -> None:
        total = len(results)
        empty_count = sum(1 for r in results if not r)
        if (
            total > 0
            and self.config.low_match_warning_threshold > 0.0
            and empty_count / total >= self.config.low_match_warning_threshold
        ):
            logger.warning(
                f"ICL low match rate for task='{task}': "
                f"{empty_count}/{total} queries ({empty_count / total:.0%}) "
                f"received no examples "
                f"(available_examples={candidate_count}). "
                f"Consider adding more examples."
            )

    async def batch_select_examples(
        self,
        query_texts: List[str],
        task: str | None = None,
        num_examples: int | None = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        Select most relevant examples for a batch of queries.

        Routing is based on ``config.selection_strategy``:

        - ``"semantic"``: cosine similarity on dense embeddings.
        - ``"bm25"``: lexical matching via BM25.
        - ``"hybrid"``: Reciprocal Rank Fusion of semantic and BM25.
        - ``"random"``: uniform random sampling.

        :param query_texts: Input texts for which to select examples.
        :param task: Task name to filter examples by (e.g. ``"entity_extraction"``).
            When ``None``, all loaded examples are considered.
        :param num_examples: Number of examples to return per query
            (uses config default if None).
        :return: Per-query lists of example dictionaries.
        """
        if not self._initialized:
            logger.warning("Not initialized. Call initialize() first.")
            return [[] for _ in query_texts]

        if num_examples is None:
            num_examples = self.config.num_examples

        strategy = self.config.selection_strategy
        if strategy == "semantic":
            results = await self._select_semantic(query_texts, task, num_examples)
        elif strategy == "bm25":
            results = self._select_bm25(query_texts, task, num_examples)
        elif strategy == "hybrid":
            results = await self._select_hybrid(query_texts, task, num_examples)
        elif strategy == "random":
            results = self._select_random(query_texts, task, num_examples)
        else:
            logger.warning(f"Unknown selection strategy: {strategy}")
            return [[] for _ in query_texts]

        candidate_indices = self._get_candidate_indices(task)
        self._check_low_match_rate(results, task, len(candidate_indices))

        return results

    async def _select_semantic(
        self,
        query_texts: List[str],
        task: str | None,
        num_examples: int,
    ) -> List[List[Dict[str, Any]]]:
        candidate_indices = self._get_candidate_indices(task)
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
            k = min(num_examples, len(candidate_indices))
            top_local = np.argsort(similarities)[-k:][::-1]
            top_global = idx_arr[top_local]

            selected = [
                self._example_to_dict(self.examples[idx])
                for idx in top_global
            ]

            logger.debug(
                f"Selected {len(selected)} examples for query {i} "
                f"(task='{task}', strategy='semantic', "
                f"similarities: {[float(similarities[j]) for j in top_local]})"
            )

            results.append(selected)

        return results

    def _select_bm25(
        self,
        query_texts: List[str],
        task: str | None,
        num_examples: int,
    ) -> List[List[Dict[str, Any]]]:
        if self._bm25_embedder is None or self._bm25_doc_embeddings is None:
            return [[] for _ in query_texts]

        query_embeddings = self._bm25_embedder.embed_query(query_texts)

        if (
            task is not None
            and self._bm25_task_embeddings
            and task in self._bm25_task_embeddings
        ):
            task_embs, task_indices = self._bm25_task_embeddings[task]
            k = min(num_examples, len(task_indices))
            if k == 0:
                return [[] for _ in query_texts]

            output: List[List[Dict[str, Any]]] = []
            for i, query_emb in enumerate(query_embeddings):
                scored = sorted(
                    (
                        (self._sparse_dot_product(query_emb, doc_emb), local_idx)
                        for local_idx, doc_emb in enumerate(task_embs)
                    ),
                    key=lambda x: (-x[0], x[1]),
                )

                selected = []
                for _score, local_idx in scored[:num_examples]:
                    global_idx = task_indices[local_idx]
                    selected.append(self._example_to_dict(self.examples[global_idx]))

                logger.debug(
                    f"Selected {len(selected)} examples for query {i} "
                    f"(task='{task}', strategy='bm25', index='per-task')"
                )

                output.append(selected)

            return output

        candidate_indices = self._get_candidate_indices(task)
        if not candidate_indices:
            return [[] for _ in query_texts]

        output: List[List[Dict[str, Any]]] = []
        for i, query_emb in enumerate(query_embeddings):
            scored = sorted(
                (
                    (self._sparse_dot_product(query_emb, self._bm25_doc_embeddings[j]), j)
                    for j in candidate_indices
                ),
                key=lambda x: (-x[0], x[1]),
            )

            selected = []
            for _score, idx in scored[:num_examples]:
                selected.append(self._example_to_dict(self.examples[idx]))

            logger.debug(
                f"Selected {len(selected)} examples for query {i} "
                f"(task='{task}', strategy='bm25', index='global')"
            )

            output.append(selected)

        return output

    async def _select_hybrid(
        self,
        query_texts: List[str],
        task: str | None,
        num_examples: int,
    ) -> List[List[Dict[str, Any]]]:
        assert self.embedder is not None

        if (self._bm25_embedder is None or self._bm25_doc_embeddings is None
                or self._example_matrix is None):
            return [[] for _ in query_texts]

        candidate_indices = self._get_candidate_indices(task)
        if not candidate_indices:
            return [[] for _ in query_texts]

        candidate_set = set(candidate_indices)
        idx_arr = np.array(candidate_indices, dtype=np.intp)

        query_matrix = await self._get_query_matrix(query_texts)
        ex_matrix = self._example_matrix[idx_arr]
        ex_norms = self._example_norms[idx_arr]

        query_norms = np.linalg.norm(query_matrix, axis=1)
        valid_examples = ex_norms > 0.0
        ex_norms_safe = np.where(valid_examples, ex_norms, 1.0)

        sim_matrix = (query_matrix @ ex_matrix.T) / (
            query_norms[:, np.newaxis] * ex_norms_safe[np.newaxis, :]
        )
        sim_matrix[:, ~valid_examples] = 0.0

        query_embeddings = self._bm25_embedder.embed_query(query_texts)

        use_per_task = (
            task is not None
            and self._bm25_task_embeddings
            and task in self._bm25_task_embeddings
        )

        rrf_k = max(len(candidate_indices) // 2, 1)

        output: List[List[Dict[str, Any]]] = []
        for i, query_emb in enumerate(query_embeddings):
            rrf_scores: Dict[int, float] = {}

            sims = sim_matrix[i]
            semantic_order = np.argsort(-sims)
            for rank, local_idx in enumerate(semantic_order):
                global_idx = int(idx_arr[local_idx])
                rrf_scores[global_idx] = 1.0 / (rrf_k + rank + 1)

            if use_per_task:
                task_embs, task_bm25_indices = self._bm25_task_embeddings[task]
                bm25_scored = sorted(
                    (
                        (self._sparse_dot_product(query_emb, doc_emb), local_idx)
                        for local_idx, doc_emb in enumerate(task_embs)
                    ),
                    key=lambda x: (-x[0], x[1]),
                )
                bm25_global_results = [
                    task_bm25_indices[local_idx] for _, local_idx in bm25_scored
                ]
            else:
                bm25_scored = sorted(
                    (
                        (self._sparse_dot_product(query_emb, self._bm25_doc_embeddings[j]), j)
                        for j in candidate_indices
                    ),
                    key=lambda x: (-x[0], x[1]),
                )
                bm25_global_results = [idx for _, idx in bm25_scored]

            for rank, doc_idx in enumerate(bm25_global_results):
                doc_idx_int = int(doc_idx)
                if not use_per_task and doc_idx_int not in candidate_set:
                    continue
                rrf_scores[doc_idx_int] = (
                    rrf_scores.get(doc_idx_int, 0.0) + 1.0 / (rrf_k + rank + 1)
                )

            sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
            selected = []
            for global_idx, score in sorted_items[:num_examples]:
                example = self.examples[global_idx]
                selected.append(self._example_to_dict(example))

            index_type = "per-task" if use_per_task else "global"
            logger.debug(
                f"Selected {len(selected)} examples for query {i} "
                f"(task='{task}', strategy='hybrid', rrf_k={rrf_k}, "
                f"bm25_index='{index_type}')"
            )

            output.append(selected)

        return output

    def _select_random(
        self,
        query_texts: List[str],
        task: str | None,
        num_examples: int,
    ) -> List[List[Dict[str, Any]]]:
        candidate_indices = self._get_candidate_indices(task)
        if not candidate_indices:
            return [[] for _ in query_texts]

        k = min(num_examples, len(candidate_indices))
        output: List[List[Dict[str, Any]]] = []
        for i in range(len(query_texts)):
            chosen = _random.sample(candidate_indices, k)
            selected = [self._example_to_dict(self.examples[idx]) for idx in chosen]

            logger.debug(
                f"Selected {len(selected)} examples for query {i} "
                f"(task='{task}', strategy='random')"
            )

            output.append(selected)

        return output
