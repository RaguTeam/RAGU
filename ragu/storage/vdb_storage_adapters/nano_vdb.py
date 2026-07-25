# Vendored and refactored from NanoVectorDB (MIT,
# https://github.com/gusye1234/nano-vectordb), originally adapted in RAGU from
# nano-graphrag (https://github.com/gusye1234/nano-graphrag).
#
# Differences from the upstream library:
#   * native batch query (a single matrix multiply for many queries);
#   * cleaner public surface (no name-mangled internal storage access);
#   * a configurable distance metric: upstream always normalized on write,
#     which silently discards vector magnitude for dot-product scoring.

import base64
import json
import os
import tempfile
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from typing_extensions import override

from ragu.common.global_parameters import Settings
from ragu.common.logger import logger
from ragu.storage.base_storage import BaseVectorStorage
from ragu.storage.types import EmbeddingHit, Point

_FLOAT = np.float32

#: Reserved record keys used internally by the store.
F_ID = "__id__"
F_VECTOR = "__vector__"
F_METRICS = "__metrics__"

#: Default query score thresholds per metric. Cosine scores live in ``[-1, 1]``,
#: so a fixed cut-off is meaningful; dot products are unbounded and depend on
#: the embedding model, so no default filtering is applied.
DEFAULT_SCORE_THRESHOLDS: Dict[str, Optional[float]] = {
    "cosine": 0.2,
    "dot": None,
}


def _coerce_metric(metric: str) -> Literal["cosine", "dot"]:
    """
    Validate and normalize a distance metric name.

    :param metric: Metric name, case-insensitive.
    :type metric: str
    :returns: Lower-cased metric name.
    :rtype: Literal["cosine", "dot"]
    :raises ValueError: If the metric is not supported.
    """
    normalized = str(metric).lower()
    if normalized not in DEFAULT_SCORE_THRESHOLDS:
        supported = ", ".join(DEFAULT_SCORE_THRESHOLDS)
        raise ValueError(
            f"Unsupported distance metric: {metric!r}. Supported: {supported}"
        )
    return normalized  # type: ignore[return-value]


def _encode_matrix(matrix: np.ndarray) -> str:
    """
    Encode a float32 matrix into the NanoVectorDB base64 buffer string.

    :param matrix: Dense matrix to encode.
    :type matrix: numpy.ndarray
    :returns: Base64-encoded little-endian float32 buffer.
    :rtype: str
    """
    return base64.b64encode(matrix.astype(_FLOAT).tobytes()).decode() # type: ignore


def _decode_matrix(buffer: str, embedding_dim: int) -> np.ndarray:
    """
    Decode a NanoVectorDB base64 buffer string into a float32 matrix.

    :param buffer: Base64-encoded float32 buffer.
    :type buffer: str
    :param embedding_dim: Vector dimensionality used to reshape the flat buffer.
    :type embedding_dim: int
    :returns: Decoded matrix of shape ``(-1, embedding_dim)``.
    :rtype: numpy.ndarray
    """
    return np.frombuffer(
        base64.b64decode(buffer), dtype=_FLOAT
    ).reshape(-1, embedding_dim).copy()


def _normalize(matrix: np.ndarray) -> np.ndarray:
    """
    L2-normalize matrix rows, leaving zero vectors untouched.

    :param matrix: Matrix whose rows are normalized in place of a copy.
    :type matrix: numpy.ndarray
    :returns: Row-normalized matrix.
    :rtype: numpy.ndarray
    """
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.where(norms == 0.0, 1.0, norms)


def _coerce_vector(vector: Any, embedding_dim: int, label: str) -> np.ndarray:
    """
    Convert one vector to a 1-D float32 array and validate its dimension.
    """
    array = np.asarray(vector, dtype=_FLOAT).reshape(-1)
    if array.shape[0] != embedding_dim:
        raise ValueError(
            f"{label} embedding dim mismatch, expected: {embedding_dim}, "
            f"but got: {array.shape[0]}"
        )
    return array


def _coerce_query_matrix(queries: np.ndarray, embedding_dim: int) -> np.ndarray:
    """
    Convert query input to a 2-D float32 matrix and validate its dimension.
    """
    matrix = np.asarray(queries, dtype=_FLOAT)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    elif matrix.ndim != 2:
        raise ValueError("Query embeddings must be a 1-D vector or a 2-D matrix")

    if matrix.shape[1] != embedding_dim:
        raise ValueError(
            f"Query embedding dim mismatch, expected: {embedding_dim}, "
            f"but got: {matrix.shape[1]}"
        )
    return matrix


class DenseVectorDB:
    """
    Brute-force vector store backed by a single matrix.

    How vectors are stored depends on :attr:`metric`:

    * ``"cosine"`` — vectors are L2-normalized on insertion, so cosine
      similarity reduces to a dot product. The original magnitude is not
      recoverable afterwards.
    * ``"dot"`` — vectors are stored verbatim and scored by raw dot product,
      which is magnitude-sensitive.

    In both cases a higher score means a better match. Queries are evaluated as
    one matrix multiply over the whole batch, which is markedly faster than
    scoring queries one at a time.

    :param embedding_dim: Vector dimensionality.
    :type embedding_dim: int
    :param storage_file: Path of the JSON file used for persistence.
    :type storage_file: str
    :param metric: Similarity metric determining storage and scoring.
    :type metric: Literal["cosine", "dot"]
    :raises ValueError: If ``metric`` is not supported.
    """

    def __init__(
        self,
        embedding_dim: int,
        storage_file: str,
        metric: Literal["cosine", "dot"] = "cosine",
    ) -> None:
        self.embedding_dim = embedding_dim
        self.storage_file = storage_file
        self.metric = _coerce_metric(metric)
        self._matrix: np.ndarray = np.zeros((0, embedding_dim), dtype=_FLOAT)
        self._rows: List[Dict[str, Any]] = []
        self._id_to_index: Dict[str, int] = {}
        self._load()

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def normalizes_vectors(self) -> bool:
        """
        Whether vectors are L2-normalized before being stored.

        :returns: ``True`` for cosine similarity, ``False`` otherwise.
        :rtype: bool
        """
        return self.metric == "cosine"

    def _load(self) -> None:
        """
        Load persisted state from :attr:`storage_file` if it exists.

        Files written before metrics were configurable carry no ``metric`` key;
        those always held cosine-normalized vectors and are read as such.

        :raises ValueError: If the stored embedding dimension or metric does not
            match this instance, or if rows and vectors are out of sync.
        """
        if not os.path.exists(self.storage_file):
            logger.debug(f"No vector store file at '{self.storage_file}', starting empty.")
            return
        with open(self.storage_file, "r", encoding="utf-8") as handle:
            storage = json.load(handle)

        loaded_dim = storage.get("embedding_dim", self.embedding_dim)
        if loaded_dim != self.embedding_dim:
            raise ValueError(
                f"Embedding dim mismatch, expected: {self.embedding_dim}, "
                f"but loaded: {loaded_dim}"
            )

        loaded_metric = _coerce_metric(storage.get("metric", "cosine"))
        if loaded_metric != self.metric:
            raise ValueError(
                f"Distance metric mismatch for '{self.storage_file}', expected: "
                f"{self.metric}, but loaded: {loaded_metric}. Stored vectors are kept "
                f"differently per metric, so the index must be rebuilt to switch."
            )

        self._rows = storage.get("data", [])
        matrix_buffer = storage.get("matrix")
        if matrix_buffer:
            self._matrix = _decode_matrix(matrix_buffer, self.embedding_dim)
        else:
            self._matrix = np.zeros((0, self.embedding_dim), dtype=_FLOAT)

        if len(self._rows) != self._matrix.shape[0]:
            raise ValueError(
                f"Vector store row count mismatch, rows: {len(self._rows)}, "
                f"vectors: {self._matrix.shape[0]}"
            )
        self._rebuild_index()
        logger.info(f"Loaded {len(self._rows)} points from vector store '{self.storage_file}'.")

    def save(self) -> None:
        """
        Persist the current state to :attr:`storage_file`.

        The written JSON is compatible with the historical NanoVectorDB layout.
        A ``sparse`` section is intentionally reserved (currently unused) so a
        future hybrid mode does not require an on-disk migration.
        """
        storage = {
            "embedding_dim": self.embedding_dim,
            "metric": self.metric,
            "data": self._rows,
            "matrix": _encode_matrix(self._matrix),
        }
        storage_dir = os.path.dirname(self.storage_file) or "."
        os.makedirs(storage_dir, exist_ok=True)

        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=storage_dir,
            prefix=os.path.basename(self.storage_file) + ".",
            suffix=".tmp",
            delete=False,
        )
        try:
            with handle as file:
                json.dump(storage, file, ensure_ascii=False)
                file.flush()
                os.fsync(file.fileno())
            os.replace(handle.name, self.storage_file)
        except BaseException:
            if os.path.exists(handle.name):
                os.unlink(handle.name)
            raise

        logger.info(f"Saved {len(self._rows)} points to vector store '{self.storage_file}'.")

    def _rebuild_index(self) -> None:
        """Rebuild the id-to-row-index lookup from :attr:`_rows`."""
        self._id_to_index = {row[F_ID]: i for i, row in enumerate(self._rows)}

    def upsert(self, datas: List[Dict[str, Any]]) -> None:
        """
        Insert or update records.

        :param datas: Records, each containing ``__id__``, ``__vector__`` and
            optional metadata keys. Duplicate ids within the batch keep the
            last occurrence.
        :type datas: list[dict]
        """
        if not datas:
            return

        # Collapse duplicate ids within the batch, keeping the last write.
        deduped: Dict[str, Dict[str, Any]] = {data[F_ID]: data for data in datas}

        record_ids = list(deduped)
        vectors = np.vstack([
            _coerce_vector(deduped[record_id][F_VECTOR], self.embedding_dim, f"Record '{record_id}'")
            for record_id in record_ids
        ])
        if self.normalizes_vectors:
            vectors = _normalize(vectors)

        new_vectors: List[np.ndarray] = []
        new_rows: List[Dict[str, Any]] = []
        for record_id, vector in zip(record_ids, vectors):
            row = {
                key: value
                for key, value in deduped[record_id].items()
                if key != F_VECTOR
            }
            row[F_ID] = record_id

            existing = self._id_to_index.get(record_id)
            if existing is not None:
                self._matrix[existing] = vector
                self._rows[existing] = row
            else:
                self._id_to_index[record_id] = len(self._rows) + len(new_rows)
                new_rows.append(row)
                new_vectors.append(vector)

        if new_rows:
            stacked = np.vstack(new_vectors)
            self._rows.extend(new_rows)
            self._matrix = np.vstack([self._matrix, stacked]) if self._matrix.size else stacked

    def delete(self, ids: List[str]) -> None:
        """
        Remove records by id. Unknown ids are ignored.

        :param ids: Record identifiers to delete.
        :type ids: list[str]
        """
        if not ids:
            return
        drop = {record_id for record_id in ids if record_id in self._id_to_index}
        if not drop:
            return
        keep_index = [i for i, row in enumerate(self._rows) if row[F_ID] not in drop]
        self._rows = [self._rows[i] for i in keep_index]
        self._matrix = (
            self._matrix[keep_index]
            if keep_index
            else np.zeros((0, self.embedding_dim), dtype=_FLOAT)
        )
        self._rebuild_index()

    def all_ids(self) -> List[str]:
        """
        Return all stored record ids in insertion order.

        :returns: Record identifiers.
        :rtype: list[str]
        """
        return [row[F_ID] for row in self._rows]

    def get_rows(self, ids: List[str]) -> List[Optional[Dict[str, Any]]]:
        """
        Fetch stored rows (id + metadata) by id, preserving input order.

        :param ids: Record identifiers to fetch.
        :type ids: list[str]
        :returns: Rows aligned with ``ids``; missing ids mapped to ``None``.
        :rtype: list[dict | None]
        """
        return [
            self._rows[self._id_to_index[record_id]] if record_id in self._id_to_index else None
            for record_id in ids
        ]

    def get_points(self, ids: List[str]) -> List[Optional[Tuple[str, np.ndarray, Dict[str, Any]]]]:
        """
        Fetch stored vectors and metadata by id, preserving input order.

        Returns vectors **as stored**: under the ``cosine`` metric these are
        L2-normalized and therefore differ in magnitude from what was written.

        :param ids: Record identifiers to fetch.
        :type ids: list[str]
        :returns: Tuples of ``(id, vector, metadata)`` aligned with ``ids``;
            missing ids mapped to ``None``.
        :rtype: list[tuple[str, numpy.ndarray, dict] | None]
        """
        points: List[Optional[Tuple[str, np.ndarray, Dict[str, Any]]]] = []
        for record_id in ids:
            index = self._id_to_index.get(record_id)
            if index is None:
                points.append(None)
                continue
            metadata = {key: value for key, value in self._rows[index].items() if key != F_ID}
            points.append((record_id, self._matrix[index], metadata))
        return points

    def query(
        self,
        queries: np.ndarray,
        top_k: int,
        threshold: Optional[float] = None,
    ) -> List[List[Tuple[str, float, Dict[str, Any]]]]:
        """
        Score a batch of query vectors against all stored vectors.

        Queries are normalized only under the ``cosine`` metric; a dot-product
        store scores raw magnitudes on both sides. Higher scores are better for
        every supported metric.

        :param queries: Query matrix of shape ``(n_queries, embedding_dim)``.
            A single 1-D vector is also accepted.
        :type queries: numpy.ndarray
        :param top_k: Maximum number of hits per query.
        :type top_k: int
        :param threshold: Optional minimum score; hits scoring below it are
            dropped.
        :type threshold: float | None
        :returns: Per-query lists of ``(id, score, metadata)`` ordered by
            descending score, index-aligned with the input rows.
        :rtype: list[list[tuple[str, float, dict]]]
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got: {top_k}")

        matrix = _coerce_query_matrix(queries, self.embedding_dim)
        if self._matrix.shape[0] == 0:
            return [[] for _ in range(matrix.shape[0])]

        if self.normalizes_vectors:
            matrix = _normalize(matrix)
        scores = matrix @ self._matrix.T  # (n_queries, n_records)
        limit = min(top_k, self._matrix.shape[0])

        results: List[List[Tuple[str, float, Dict[str, Any]]]] = []
        for row_scores in scores:
            top_index = np.argpartition(row_scores, -limit)[-limit:]
            top_index = top_index[np.argsort(row_scores[top_index])[::-1]]

            hits: List[Tuple[str, float, Dict[str, Any]]] = []
            for index in top_index:
                score = float(row_scores[index])
                if threshold is not None and score < threshold:
                    break
                row = self._rows[index]
                metadata = {key: value for key, value in row.items() if key != F_ID}
                hits.append((row[F_ID], score, metadata))
            results.append(hits)
        return results


class NanoVectorDBStorage(BaseVectorStorage):
    """
    Vector storage implementation backed by :class:`DenseVectorDB`.

    Provides a simple file-backed dense vector database for storing and
    retrieving embeddings and performing batched nearest-neighbor search.
    """

    def __init__(
        self,
        embedding_dim: int,
        score_threshold: Optional[float] = None,
        storage_folder: str = Settings.storage_folder,
        filename: str = "data.json",
        metric: Literal["cosine", "dot"] = "cosine",
        cosine_threshold: Optional[float] = None,
        **kwargs: Any,
    ):
        """
        Initialize the dense vector storage.

        :param embedding_dim: Embedding dimensionality.
        :param score_threshold: Minimum similarity score for query results. When
            omitted, a metric-specific default is used (``0.2`` for cosine, no
            filtering for dot product).
        :param storage_folder: Folder where the vector storage file is located.
        :param filename: Name of the JSON file containing the stored vectors.
        :param metric: Similarity metric. ``cosine`` stores L2-normalized vectors;
            ``dot`` stores them verbatim.
        :param cosine_threshold: Deprecated alias for ``score_threshold``.
        :param kwargs: Additional keyword arguments passed to the base class.
        :raises TypeError: If both ``score_threshold`` and ``cosine_threshold``
            are provided.
        """
        super().__init__(**kwargs)

        if cosine_threshold is not None:
            if score_threshold is not None:
                raise TypeError(
                    "Pass only 'score_threshold'; 'cosine_threshold' is a deprecated alias."
                )
            logger.warning(
                "'cosine_threshold' is deprecated and will be removed; use 'score_threshold'."
            )
            score_threshold = cosine_threshold

        self.filename = os.path.join(storage_folder, filename)
        self.embedding_dim = embedding_dim
        self.metric = _coerce_metric(metric)
        self.score_threshold = (
            score_threshold
            if score_threshold is not None
            else DEFAULT_SCORE_THRESHOLDS[self.metric]
        )
        self._client = DenseVectorDB(
            embedding_dim,
            storage_file=self.filename,
            metric=self.metric,
        )

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
            threshold=self.score_threshold,
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

        Vectors are returned as stored: under the ``cosine`` metric they are
        L2-normalized, so their magnitude differs from the written vector.

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
