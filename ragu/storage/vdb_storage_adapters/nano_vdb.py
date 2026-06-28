# Dense vector storage backed by the in-repo :class:`DenseVectorDB` engine
# (see ``dense_vdb_core.py``), which replaces the former external
# ``nano-vectordb`` dependency while preserving its on-disk format.

import os
from typing import Any, List, Dict
from typing_extensions import override

import numpy as np

from ragu.common.global_parameters import Settings
from ragu.common.logger import logger
from ragu.storage.base_storage import BaseVectorStorage
from ragu.storage.types import Point, EmbeddingHit
from ragu.storage.vdb_storage_adapters.dense_vdb_core import DenseVectorDB, F_ID, F_VECTOR


class NanoVectorDBStorage(BaseVectorStorage):
    """
    Vector storage implementation backed by :class:`DenseVectorDB`.

    Provides a simple file-backed dense vector database for storing and
    retrieving embeddings and performing batched nearest-neighbor search.
    """

    def __init__(
        self,
        embedding_dim: int,
        cosine_threshold: float = 0.2,
        storage_folder: str = Settings.storage_folder,
        filename: str = "data.json",
        **kwargs: Any,
    ):
        """
        Initialize the dense vector storage.

        :param embedding_dim: Embedding dimensionality.
        :param cosine_threshold: Minimum cosine similarity threshold for query filtering.
        :param storage_folder: Folder where the vector storage file is located.
        :param filename: Name of the JSON file containing the stored vectors.
        :param kwargs: Additional keyword arguments passed to the base class.
        """
        super().__init__(**kwargs)

        self.filename = os.path.join(storage_folder, filename)
        self.embedding_dim = embedding_dim
        self.cosine_threshold = cosine_threshold
        self._client = DenseVectorDB(embedding_dim, storage_file=self.filename)

    @override
    async def upsert(self, data: List[Point], **kwargs) -> None:
        """
        Insert or update a batch of embeddings in the database.

        :param data: Embedding records with vectors and metadata.
        """
        if not data:
            logger.warning("Attempted to insert empty data into vector DB.")
            return

        if any(item.sparse_embedding is not None for item in data):
            logger.warning("NanoVDB does not support sparse embeddings. Ignoring.")

        items: List[Dict[str, Any]] = [
            {
                F_ID: embedding.id,
                F_VECTOR: np.array(embedding.dense_embedding),
                **embedding.metadata,
            }
            for embedding in data
        ]

        if not items:
            return

        self._client.upsert(items)

    @override
    async def query(self, points: List[Point], **kwargs: Any) -> List[List[EmbeddingHit]]:
        """
        Search for the most similar records for each query embedding.

        Performs a single batched cosine similarity search against all stored
        vectors, returning per-query hits that exceed the similarity threshold.

        :param points: Query embedding payloads, one per search.
        :param kwargs:
            top_k: Number of nearest neighbors to return per query.
        :return: Per-query lists of matched records, index-aligned with ``points``.
        """
        top_k: int = kwargs.pop("top_k", 20)
        if not points:
            return []

        if any(point.sparse_embedding is not None for point in points):
            logger.warning("NanoVDB does not support sparse embeddings. Ignoring sparse query payload.")

        for point in points:
            if point.dense_embedding is None:
                raise ValueError("Empty dense embedding payload.")

        query_matrix = np.array([point.dense_embedding for point in points])
        batched = self._client.query(
            query_matrix,
            top_k=top_k,
            threshold=self.cosine_threshold,
        )

        results: List[List[EmbeddingHit]] = []
        for hits in batched:
            results.append([
                EmbeddingHit(
                    id=record_id,
                    distance=score,
                    metadata=dict(metadata),
                )
                for record_id, score, metadata in hits
            ])
        return results

    @override
    async def delete(self, ids: List[str], **kwargs: Any) -> None:
        """
        Delete embeddings by their IDs from the vector database.

        :param ids: List of IDs to remove from the vector storage.
        :type ids: List[str]
        """
        if not ids:
            return
        self._client.delete(ids)

    @override
    async def get_all_ids(self) -> List[str]:
        """
        Return all record IDs currently stored in the vector database.
        """
        return self._client.all_ids()

    @override
    async def get_points_by_ids(self, ids: List[str]) -> List[Point | None]:
        """
        Retrieve stored points by ID, preserving input order.

        :param ids: Record identifiers to fetch.
        :return: Points aligned with ``ids``; missing IDs mapped to ``None``.
        """
        points: List[Point | None] = []
        for entry in self._client.get_points(ids):
            if entry is None:
                points.append(None)
                continue
            record_id, vector, metadata = entry
            points.append(
                Point(
                    id=record_id,
                    dense_embedding=vector,
                    metadata=metadata,
                )
            )
        return points

    @override
    async def get_payloads_by_ids(self, ids: List[str]) -> List[Dict | None]:
        """
        Retrieve stored payloads by ID, preserving input order.

        :param ids: Record identifiers to fetch.
        :return: Payloads aligned with ``ids``; missing IDs mapped to ``None``.
        """
        return [None if row is None else dict(row) for row in self._client.get_rows(ids)]

    async def index_start_callback(self):
        """
        Pre-index hook for interface compatibility.
        """
        pass

    async def query_done_callback(self):
        """
        Post-query hook for interface compatibility.
        """
        pass

    async def index_done_callback(self) -> None:
        """
        Save the current state of the vector database to disk.

        This method ensures that any newly inserted or updated vectors
        are persisted in the storage file.
        """
        self._client.save()
