import asyncio
import uuid
from pathlib import Path
from typing import Any, List, Dict, Literal
from typing_extensions import override

from qdrant_client.conversions.common_types import CollectionInfo, QueryResponse
from qdrant_client.http.models import FusionQuery, Fusion

from pydantic import BaseModel
from ragu.common.global_parameters import Settings
from ragu.storage.base_storage import BaseVectorStorage
from ragu.storage.types import EmbeddingHit, Point, SparseEmbedding
from ragu.common.logger import logger

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import (
    Distance,
    Modifier,
    PointIdsList,
    PointStruct,
    Prefetch,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from ragu.utils.ragu_utils import split_on_batches_by_size


class QdrantVectorDBStorage(BaseVectorStorage):
    """
    Qdrant-backed vector storage for dense-only and hybrid retrieval.

    This adapter creates or validates one Qdrant collection containing:

    - one dense vector field named ``"dense"``
    - optionally one sparse vector field named after ``sparse_type``
      such as ``"bm25"``, ``"bm42"``, or ``"splade"``

    Storage modes
    -------------
    The constructor supports three common Qdrant deployment styles:

    1. Local on-disk mode
       If no remote connection arguments are provided, the adapter starts
       Qdrant in local mode. The directory containing ``filename`` is used
       as the Qdrant storage path. This is the default mode used by RAGU.

    2. In-memory mode
       Pass ``location=":memory:"`` to use Qdrant's in-memory local backend.
       This is useful for tests, examples, and short-lived indexing sessions.

    3. Remote server mode
       Pass connection parameters such as ``url``, ``host``, ``port``,
       ``grpc_port``, or ``api_key`` to connect to an external Qdrant server.
       In this case no local path is used.

    Sparse retrieval modes
    ----------------------
    ``sparse_type`` controls whether the collection stores sparse vectors and
    which Qdrant sparse vector name is used:

    - ``None``: dense-only retrieval
    - ``"bm25"``: sparse vector named ``"bm25"``, with Qdrant ``IDF`` modifier
    - ``"bm42"``: sparse vector named ``"bm42"``, with Qdrant ``IDF`` modifier
    - ``"splade"``: sparse vector named ``"splade"``
    - ``"custom"``: sparse vector named ``"custom"``

    For dense+sparse queries, the adapter uses Qdrant reciprocal-rank fusion
    by default.

    Examples
    --------
    Local on-disk dense-only collection:

    ```python
    from ragu.storage.vdb_storage_adapters.qdrant_vdb import QdrantVectorDBStorage

    vdb = QdrantVectorDBStorage(
        embedding_dim=768,
        storage_folder="./storage",
        filename="entity_vectors.json",
    )
    ```

    In-memory collection:

    ```python
    vdb = QdrantVectorDBStorage(
        embedding_dim=768,
        location=":memory:",
        collection_name="temp_vectors",
    )
    ```

    Remote Qdrant server:

    ```python
    vdb = QdrantVectorDBStorage(
        embedding_dim=768,
        url="http://localhost:6333",
        api_key="secret",
        collection_name="prod_chunks",
    )
    ```

    Hybrid dense + BM25 sparse retrieval:

    ```python
    vdb = QdrantVectorDBStorage(
        embedding_dim=768,
        filename="chunk_vectors.json",
        sparse_type="bm25",
    )
    ```

    Hybrid dense + BM42 sparse retrieval:

    ```python
    vdb = QdrantVectorDBStorage(
        embedding_dim=768,
        location=":memory:",
        sparse_type="bm42",
        collection_name="hybrid_demo",
    )
    ```
    """

    DENSE_VECTOR_NAME = "dense"
    DEFAULT_SPARSE_VECTOR_NAME = "sparse"

    def __init__(
        self,
        embedding_dim: int,
        storage_folder: str | None = None,
        filename: str = "data.json",
        collection_name: str | None = None,
        path: str | None = None,
        url: str | None = None,
        host: str | None = None,
        port: int | None = None,
        grpc_port: int | None = None,
        api_key: str | None = None,
        location: str | None = None,
        sparse_type: Literal["bm25", "bm42", "splade", "custom"] | None = None,
        non_default_dense_config: VectorParams | None = None,
        non_default_sparse_config: SparseVectorParams | None = None,
        max_payload_size_in_mb: int = 16,
        **kwargs: Any,
    ):
        """
        Initialize a Qdrant-backed vector store.

        Collection naming defaults to the stem of ``filename`` under
        ``storage_folder``. For example, ``filename="vdb_entity.json"``
        becomes the collection name ``"..._vdb_entity"``.

        Remote connection arguments are passed directly into
        :class:`qdrant_client.AsyncQdrantClient`. If any of them are provided,
        the adapter does not open a local on-disk path.

        :param embedding_dim: Dense vector dimensionality.
        :param storage_folder: Base folder for local Qdrant storage.
        :param filename: Storage filename used to derive collection name and local path.
        :param collection_name: Explicit Qdrant collection name override.
        :param path: Explicit local Qdrant path override for on-disk local mode.
        :param url: Remote Qdrant URL.
        :param host: Remote Qdrant host.
        :param port: Remote Qdrant HTTP port.
        :param grpc_port: Remote Qdrant gRPC port.
        :param api_key: Remote Qdrant API key.
        :param location: Qdrant location identifier, including ``":memory:"`` for in-memory mode.
        :param sparse_type: Optional sparse vector mode. Supported values are
            ``"bm25"``, ``"bm42"``, ``"splade"``, and ``"custom"``.
        :param non_default_dense_config: Optional custom dense vector config.
        :param non_default_sparse_config: Optional custom sparse vector config.
        :param max_payload_size_in_mb: Maximum upsert batch payload size before splitting.
        :param kwargs: Additional client options forwarded to ``AsyncQdrantClient``.
        """
        super().__init__()
        resolved_storage_folder = storage_folder if storage_folder else Settings.storage_folder
        resolved_filename = Path(resolved_storage_folder) / filename

        self.filename = str(resolved_filename)
        self.embedding_dim = embedding_dim

        self._dense_config = non_default_dense_config or VectorParams(
            size=self.embedding_dim,
            distance=Distance.COSINE,
        )
        self._sparse_config = non_default_sparse_config or SparseVectorParams(
            modifier=Modifier.IDF if sparse_type in {"bm25", "bm42"} else None
        )

        self._sparse_mode = sparse_type
        self._max_payload_size_in_bytes = max_payload_size_in_mb * 1024 * 1024

        remote_kwargs = {
            "url": url,
            "host": host,
            "port": port,
            "grpc_port": grpc_port,
            "api_key": api_key,
            "location": location,
        }
        remote_kwargs = {key: value for key, value in remote_kwargs.items() if value is not None}
        client_kwargs = dict(kwargs)

        filename_path = Path(self.filename)
        self.collection_name = (collection_name or
                                str(Path(resolved_storage_folder) / filename_path.stem).replace("/", "_").replace("\\", "_"))
        self._client_path = None if remote_kwargs else str(Path(path) if path is not None else filename_path.parent)
        self._client_kwargs = {**remote_kwargs, **client_kwargs}

        self._client: AsyncQdrantClient | None = None
        self._collection_ready = False
        self._collection_lock = asyncio.Lock()

    def _get_client(self) -> AsyncQdrantClient:
        """
        Return the lazily initialized Qdrant client.

        :returns: Async Qdrant client instance.
        """
        if self._client is None:
            self._client = AsyncQdrantClient(path=self._client_path, **self._client_kwargs)
        assert self._client is not None
        return self._client

    def _to_qdrant_point_id(self, record_id: str) -> str:
        """
        Convert arbitrary RAGU record IDs into a Qdrant-compatible point ID.
        """
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{self.collection_name}:{record_id}"))

    async def _ensure_collection(self) -> None:
        """
        Create or validate the Qdrant collection schema.

        :raises ValueError: If an existing collection schema does not match the adapter configuration.
        """
        async with self._collection_lock:
            if self._collection_ready:
                return

            client = self._get_client()

            expected_sparse_name = self._sparse_mode
            expected_sparse_modifier = Modifier.IDF if self._sparse_mode in {"bm25", "bm42"} else None

            if await client.collection_exists(self.collection_name):
                collection_info: CollectionInfo = await client.get_collection(self.collection_name)
                params = collection_info.config.params

                vectors_config = params.vectors
                if vectors_config is None:
                    raise ValueError(
                        f"Qdrant collection '{self.collection_name}' does not define any dense vectors"
                    )

                dense_config: VectorParams | None
                if isinstance(vectors_config, dict):
                    dense_config = vectors_config.get(self.DENSE_VECTOR_NAME)
                else:
                    dense_config = vectors_config

                if dense_config is None:
                    raise ValueError(
                        f"Qdrant collection '{self.collection_name}' does not define "
                        f"the required dense vector '{self.DENSE_VECTOR_NAME}'"
                    )

                if dense_config.size != self.embedding_dim:
                    raise ValueError(
                        f"Qdrant collection '{self.collection_name}' expects dense dimension "
                        f"{dense_config.size}, got {self.embedding_dim}"
                    )

                if dense_config.distance != Distance.COSINE:
                    raise ValueError(
                        f"Qdrant collection '{self.collection_name}' uses dense distance "
                        f"{dense_config.distance}, expected {Distance.COSINE}"
                    )

                sparse_vectors_config = getattr(params, "sparse_vectors", None) or {}

                if self._sparse_mode:
                    sparse_config = sparse_vectors_config.get(expected_sparse_name)
                    if sparse_config is None:
                        raise ValueError(
                            f"Qdrant collection '{self.collection_name}' does not define "
                            f"the required sparse vector '{expected_sparse_name}'"
                        )

                    existing_modifier = getattr(sparse_config, "modifier", None)
                    if existing_modifier != expected_sparse_modifier:
                        raise ValueError(
                            f"Qdrant collection '{self.collection_name}' has sparse modifier "
                            f"{existing_modifier} for '{expected_sparse_name}', "
                            f"expected {expected_sparse_modifier}"
                        )

            else:
                create_kwargs: dict[str, Any] = {
                    "collection_name": self.collection_name,
                    "vectors_config": {
                        self.DENSE_VECTOR_NAME: self._dense_config,
                    },
                }

                if self._sparse_mode:
                    create_kwargs["sparse_vectors_config"] = {
                        expected_sparse_name: self._sparse_config,
                    }

                await client.create_collection(**create_kwargs)

            self._collection_ready = True

    @override
    async def upsert(self, data: List[Point], **kwargs: Any) -> None:
        """
        Insert or update dense records with optional sparse side channels.

        :param data: Embedding records to upsert.
        """

        if not data:
            return

        await self._ensure_collection()

        has_sparse = [item.sparse_embedding is not None for item in data]
        if any(has_sparse) and not all(has_sparse):
            raise ValueError("All points in this batch must either have sparse embeddings or not have them.")

        if not self._sparse_mode and any(has_sparse):
            raise ValueError(f"Try to insert sparse embeddings, but `sparse_type` parameter is set to None. "
                             f"Please, set `sparse_type` parameter. Possible values: bm25, bm42, splade, custom")

        points: list[PointStruct] = []
        for point in data:
            dense_embedding = point.dense_embedding
            sparse_embedding = point.sparse_embedding

            payload = dict(point.metadata)
            payload["__ragu_id__"] = point.id

            vector_payload: Dict[str, list[float] | models.SparseVector] = {}
            if dense_embedding is not None:
                vector_payload[self.DENSE_VECTOR_NAME] = dense_embedding.tolist()

            if sparse_embedding is not None and self._sparse_mode:
                vector_payload[self._sparse_mode] = models.SparseVector(
                    indices=sparse_embedding.indices,
                    values=sparse_embedding.values
                )

            points.append(
                PointStruct(
                    id=self._to_qdrant_point_id(point.id),
                    vector=vector_payload,
                    payload=payload,
                )
            )

        if not points:
            return

        for batch in split_on_batches_by_size(points, self._max_payload_size_in_bytes):
            await self._get_client().upsert(
                collection_name=self.collection_name,
                points=batch,
                wait=True,
            )

    @override
    async def query(self, point: Point, **kwargs: Any) -> List[EmbeddingHit]:
        """
        Query Qdrant using dense-only or dense+sparse reciprocal-rank fusion.
        :param point: Qdrant point
        :param kwargs:
            top_k: int Maximum number of results to return.
            hybrid_search_query_type.
        :returns: Ranked embedding hits.
        """

        # Parameters from kwargs
        hybrid_query_type: BaseModel = kwargs.pop("hybrid_query_type", FusionQuery(fusion=Fusion.RRF))
        top_k: int = kwargs.pop("top_k", 20)

        await self._ensure_collection()

        prefetch: list[Prefetch] | None = None
        query: Any
        using: str | None = None

        if point.dense_embedding is not None and point.sparse_embedding is None:
            query = point.dense_embedding
            using = self.DENSE_VECTOR_NAME

        elif point.dense_embedding is not None and point.sparse_embedding is not None:
            if not self._sparse_mode:
                logger.warning(f"Try to use sparse embeddings, but `sparse_type` parameter is set to None. "
                               f"This can lead to unexpected results.")
            prefetch = [
                Prefetch(
                    query=point.dense_embedding.tolist(),
                    using=self.DENSE_VECTOR_NAME,
                    limit=top_k,
                ),
                Prefetch(
                    query=SparseVector(
                        values=point.sparse_embedding.values,
                        indices=point.sparse_embedding.indices,
                    ),
                    using=self._sparse_mode,
                    limit=top_k,
                ),
            ]
            query = hybrid_query_type

        else:
            raise NotImplementedError("Only dense and dense+sparse queries are supported")

        query_response: QueryResponse = await self._get_client().query_points(
            collection_name=self.collection_name,
            query=query,
            prefetch=prefetch,
            using=using,
            with_payload=True,
            limit=top_k,
        )

        hits: list[EmbeddingHit] = []
        for retrieved_point in query_response.points:
            payload = dict(retrieved_point.payload or {})
            record_id = str(payload.pop("__ragu_id__", point.id))
            hits.append(
                EmbeddingHit(
                    id=record_id,
                    distance=float(retrieved_point.score),
                    metadata=payload,
                )
            )
        return hits

    async def delete(self, ids: List[str], **kwargs: Any) -> None:
        """
        Delete records from the collection by RAGU IDs.

        :param ids: Record identifiers to remove.
        """
        if not ids:
            return

        await self._ensure_collection()
        await self._get_client().delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=[self._to_qdrant_point_id(record_id) for record_id in ids]),
            wait=True,
        )

    @override
    async def get_all_ids(self) -> List[str]:
        """
        Return all RAGU record IDs stored in the Qdrant collection.
        """
        ids: List[str] = []
        await self._ensure_collection()

        scroll_offset = None
        while True:
            points, scroll_offset = await self._get_client().scroll(
                collection_name=self.collection_name,
                limit=1000,
                offset=scroll_offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in points:
                payload = dict(getattr(point, "payload", None) or {})
                record_id = payload.get("__ragu_id__")
                if record_id is None:
                    raise ValueError("Qdrant point payload is missing '__ragu_id__'")
                ids.append(str(record_id))

            if scroll_offset is None:
                break

        return ids

    @override
    async def get_payloads_by_ids(self, ids: List[str]) -> List[Dict | None]:
        """
        Retrieve stored payloads by RAGU IDs, preserving input order.

        :param ids: Record identifiers to fetch.
        :return: Payloads aligned with ``ids``; missing IDs mapped to ``None``.
        """
        if not ids:
            return []

        await self._ensure_collection()
        qdrant_ids = [self._to_qdrant_point_id(record_id) for record_id in ids]
        records = await self._get_client().retrieve(
            collection_name=self.collection_name,
            ids=qdrant_ids,
            with_vectors=False,
            with_payload=True,
        )
        payloads_by_id = {}
        for record in records:
            payload = dict(getattr(record, "payload", None) or {})
            record_id = payload.pop("__ragu_id__", None)
            if record_id is None:
                raise ValueError("Qdrant point payload is missing '__ragu_id__'")
            payloads_by_id[str(record_id)] = {
                "__id__": str(record_id),
                **payload,
            }
        return [payloads_by_id.get(record_id) for record_id in ids]

    @override
    async def get_points_by_ids(self, ids: List[str]) -> List[Point | None]:
        """
        Retrieve stored points by RAGU IDs, preserving input order.

        :param ids: Record identifiers to fetch.
        :return: Points aligned with ``ids``; missing IDs mapped to ``None``.
        """
        if not ids:
            return []

        await self._ensure_collection()
        qdrant_ids = [self._to_qdrant_point_id(record_id) for record_id in ids]
        records = await self._get_client().retrieve(
            collection_name=self.collection_name,
            ids=qdrant_ids,
            with_vectors=True,
            with_payload=True,
        )
        points_by_id = {}
        for record in records:
            payload = dict(getattr(record, "payload", None) or {})
            record_id = payload.pop("__ragu_id__", None)
            if record_id is None:
                raise ValueError("Qdrant point payload is missing '__ragu_id__'")

            vector = getattr(record, "vector", None)
            if vector is None:
                dense_vector = None
            elif isinstance(vector, dict):
                dense_vector = vector.get(self.DENSE_VECTOR_NAME)
            else:
                dense_vector = vector

            if dense_vector is None:
                continue

            sparse_embedding = None
            if self._sparse_mode is not None and isinstance(vector, dict):
                sparse_vector = vector.get(self._sparse_mode)
                if sparse_vector is not None:
                    if isinstance(sparse_vector, dict):
                        indices = sparse_vector.get("indices")
                        values = sparse_vector.get("values")
                    else:
                        indices = getattr(sparse_vector, "indices", None)
                        values = getattr(sparse_vector, "values", None)

                    if indices is not None and values is not None:
                        sparse_embedding = SparseEmbedding(
                            indices=list(indices),
                            values=list(values),
                        )

            points_by_id[record_id] = Point(
                id=record_id,
                dense_embedding=dense_vector,
                sparse_embedding=sparse_embedding,
                metadata=payload,
            )
        return [points_by_id.get(record_id) for record_id in ids]

    async def index_start_callback(self):
        """
        Ensure collection availability before indexing starts.
        """
        await self._ensure_collection()

    async def index_done_callback(self):
        """
        Finalize indexing.
        """
        pass

    async def query_done_callback(self):
        """
        Finalize query processing.
        """
        pass
