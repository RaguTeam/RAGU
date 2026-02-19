# Based on https://github.com/gusye1234/nano-graphrag/blob/main/nano_graphrag/_storage/vdb_nanovectordb.py

import os
from typing import Any, List

import numpy as np
from nano_vectordb import NanoVectorDB

from ragu.common.global_parameters import Settings
from ragu.common.logger import logger
from ragu.storage.base_storage import BaseVectorStorage
from ragu.storage.types import Embedding, EmbeddingHit


class NanoVectorDBStorage(BaseVectorStorage):
    """
    Vector storage implementation using NanoVectorDB as the backend.

    This class provides a simple vector database for storing and retrieving
    embeddings, enabling similarity search operations such as nearest
    neighbor queries.
    """

    def __init__(
        self,
        embedding_dim: int,
        cosine_threshold: float = 0.2,
        storage_folder: str = Settings.storage_folder,
        filename: str = "data.json",
        **kwargs
    ):
        """
        Initialize the NanoVectorDB-based vector storage.

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
        self._client = NanoVectorDB(
            embedding_dim,
            storage_file=self.filename
        )

    async def upsert(self, data: List[Embedding]) -> List[Any]:
        """
        Insert or update a batch of embeddings in the database.

        :param data: Embedding records with vectors and metadata.
        :return: List of records successfully inserted or updated.
        """
        if not data:
            logger.warning("Attempted to insert empty data into vector DB.")
            return []

        valid_data: List[dict[str, Any]] = []
        skipped = 0

        for embedding in data:
            if embedding.vector is None:
                skipped += 1
                continue

            item = {
                "__id__": embedding.id,
                "__vector__": np.array(embedding.vector),
                **embedding.metadata,
            }
            valid_data.append(item)

        if skipped:
            logger.warning(f"Skipped {skipped} items with missing embeddings.")

        if not valid_data:
            return []

        return self._client.upsert(datas=valid_data)  # type: ignore

    async def query(self, vector: Embedding, top_k: int = 5) -> List[EmbeddingHit]:
        """
        Search for the most similar documents in the vector database.

        Performs a cosine similarity search against all stored vectors,
        returning the top ``k`` results exceeding the similarity threshold.

        :param vector: Query embedding payload.
        :param top_k: Number of nearest neighbors to return.
        :return: List of matched records and their distances.
        """
        results = self._client.query(
            query=np.array(vector.vector),
            top_k=top_k,
            better_than_threshold=self.cosine_threshold
        )
        hits: List[EmbeddingHit] = []
        for result in results:
            metadata = {
                key: value
                for key, value in result.items()
                if key not in {"__id__", "__metrics__", "__vector__"}
            }
            hits.append(
                EmbeddingHit(
                    id=result["__id__"],
                    distance=float(result["__metrics__"]),
                    metadata=metadata,
                )
            )
        return hits

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

    async def delete(self, ids: List[str]) -> None:
        """
        Delete embeddings by their IDs from the vector database.

        :param ids: List of IDs to remove from the vector storage.
        :type ids: List[str]
        """
        if not ids:
            return
        self._client.delete(ids)

    async def index_done_callback(self) -> None:
        """
        Save the current state of the NanoVectorDB to disk.

        This method ensures that any newly inserted or updated vectors
        are persisted in the storage file.
        """
        self._client.save()
