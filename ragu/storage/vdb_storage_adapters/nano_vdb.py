# Vendored and refactored from NanoVectorDB (MIT,
# https://github.com/gusye1234/nano-vectordb), originally adapted in RAGU from
# nano-graphrag (https://github.com/gusye1234/nano-graphrag).
#
# Differences from the upstream library:
#   * native batch query (a single matrix multiply for many queries);
#   * cleaner public surface (no name-mangled internal storage access);
#   * a configurable distance metric: upstream always normalized on write,
#     which silently discards vector magnitude for dot-product scoring;
#   * optional sparse vectors and hybrid (dense + sparse) retrieval, scored and
#     fused the way the Qdrant adapter does it.

import base64
import json
import math
import os
import tempfile
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, get_args

import numpy as np
from typing_extensions import override

from ragu.common.global_parameters import Settings
from ragu.common.logger import logger
from ragu.storage.base_storage import BaseVectorStorage
from ragu.storage.types import EmbeddingHit, Point, SparseEmbedding

_FLOAT = np.float32

SparseType = Literal["bm25", "bm42", "splade", "custom"]
FusionType = Literal["rrf", "dbsf"]

#: Sparse modes whose query weights are rescaled by inverse document frequency,
#: matching the ``Modifier.IDF`` the Qdrant adapter configures for them.
IDF_SPARSE_TYPES = frozenset({"bm25", "bm42"})

#: Ranking constant of reciprocal-rank fusion. Qdrant's default: the best hit of
#: a ranking contributes ``1 / 2``, the next ``1 / 3``, and so on.
RRF_RANKING_CONSTANT = 2

#: Raw scores of one dense or sparse search: ``(row index, score)``, best first.
_Ranking = List[Tuple[int, float]]

#: Reserved record keys used internally by the store.
F_ID = "__id__"
F_VECTOR = "__vector__"
F_METRICS = "__metrics__"
F_SPARSE = "__sparse__"

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


def _coerce_sparse_type(sparse_type: Optional[str]) -> Optional[SparseType]:
    """
    Validate a sparse mode name.

    :param sparse_type: Sparse mode, or ``None`` for a dense-only store.
    :type sparse_type: str | None
    :returns: The validated sparse mode.
    :rtype: SparseType | None
    :raises ValueError: If the sparse mode is not supported.
    """
    if sparse_type is None:
        return None
    supported = get_args(SparseType)
    if sparse_type not in supported:
        raise ValueError(
            f"Unsupported sparse type: {sparse_type!r}. Supported: {', '.join(supported)}"
        )
    return sparse_type  # type: ignore[return-value]


def _canonical_sparse(sparse: SparseEmbedding, label: str) -> SparseEmbedding:
    """
    Return ``sparse`` with indices sorted and as plain Python numbers.

    :param sparse: Sparse vector to normalize.
    :type sparse: SparseEmbedding
    :param label: Record description used in error messages.
    :type label: str
    :returns: A new sparse vector sorted by index.
    :rtype: SparseEmbedding
    :raises ValueError: If an index occurs more than once.
    """
    pairs = sorted((int(index), float(value)) for index, value in zip(sparse.indices, sparse.values))
    indices = [index for index, _ in pairs]
    if len(set(indices)) != len(indices):
        raise ValueError(f"{label} sparse embedding has duplicate indices")
    return SparseEmbedding(indices=indices, values=[value for _, value in pairs])


def _sparse_to_json(sparse: Optional[SparseEmbedding]) -> Optional[Dict[str, List[Any]]]:
    """Serialize one optional sparse vector for the JSON store."""
    if sparse is None:
        return None
    return {"indices": list(sparse.indices), "values": list(sparse.values)}


def _sparse_from_json(raw: Optional[Dict[str, List[Any]]]) -> Optional[SparseEmbedding]:
    """Deserialize one optional sparse vector from the JSON store."""
    if raw is None:
        return None
    return SparseEmbedding(
        indices=[int(index) for index in raw["indices"]],
        values=[float(value) for value in raw["values"]],
    )


def _idf(document_frequency: int, num_documents: int) -> float:
    """
    Inverse document frequency, as Qdrant computes it for ``Modifier.IDF``.

    :param document_frequency: Number of stored sparse vectors holding the index.
    :type document_frequency: int
    :param num_documents: Number of stored records that have a sparse vector.
    :type num_documents: int
    :returns: ``ln((N - df + 0.5) / (df + 0.5) + 1)``.
    :rtype: float
    """
    return math.log((num_documents - document_frequency + 0.5) / (document_frequency + 0.5) + 1)


def _reciprocal_rank_fusion(rankings: Sequence[_Ranking], limit: int) -> _Ranking:
    """
    Fuse rankings by reciprocal rank, the default hybrid fusion of Qdrant.

    A record at 0-based position ``i`` of a ranking contributes
    ``1 / (i + RRF_RANKING_CONSTANT)``; contributions are summed across
    rankings. Ties keep first-seen order.

    :param rankings: Rankings to fuse, each ordered best first.
    :type rankings: Sequence[list[tuple[int, float]]]
    :param limit: Maximum number of fused results.
    :type limit: int
    :returns: Fused ``(row index, score)`` pairs, best first.
    :rtype: list[tuple[int, float]]
    """
    scores: Dict[int, float] = {}
    for ranking in rankings:
        for position, (row, _) in enumerate(ranking):
            scores[row] = scores.get(row, 0.0) + 1.0 / (position + RRF_RANKING_CONSTANT)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]


def _distribution_based_score_fusion(rankings: Sequence[_Ranking], limit: int) -> _Ranking:
    """
    Fuse rankings by distribution-based score fusion (Qdrant ``Fusion.DBSF``).

    Each ranking's scores are rescaled from ``[mean - 3 sd, mean + 3 sd]`` to
    ``[0, 1]`` (a single hit, or zero variance, maps to ``0.5``) and summed per
    record.

    :param rankings: Rankings to fuse, each ordered best first.
    :type rankings: Sequence[list[tuple[int, float]]]
    :param limit: Maximum number of fused results.
    :type limit: int
    :returns: Fused ``(row index, score)`` pairs, best first.
    :rtype: list[tuple[int, float]]
    """
    scores: Dict[int, float] = {}
    for ranking in rankings:
        if not ranking:
            continue
        raw = [score for _, score in ranking]
        if len(raw) == 1:
            normalized = [0.5]
        else:
            mean = sum(raw) / len(raw)
            variance = sum((score - mean) ** 2 for score in raw) / (len(raw) - 1)
            if variance == 0:
                normalized = [0.5] * len(raw)
            else:
                low = mean - 3 * math.sqrt(variance)
                high = mean + 3 * math.sqrt(variance)
                normalized = [(score - low) / (high - low) for score in raw]
        for (row, _), score in zip(ranking, normalized):
            scores[row] = scores.get(row, 0.0) + score
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]


_FUSIONS = {
    "rrf": _reciprocal_rank_fusion,
    "dbsf": _distribution_based_score_fusion,
}


def _coerce_fusion(fusion: Any) -> FusionType:
    """
    Resolve the requested hybrid fusion strategy.

    Accepts ``None`` (reciprocal-rank fusion), the strings ``"rrf"`` / ``"dbsf"``,
    or any object with a ``fusion`` attribute holding one of them. The last form
    covers ``qdrant_client.models.FusionQuery``, so the ``hybrid_query_type``
    a caller passes to the Qdrant adapter works here unchanged.

    :param fusion: Requested fusion.
    :type fusion: Any
    :returns: Normalized fusion name.
    :rtype: FusionType
    :raises ValueError: If the fusion strategy is not supported.
    """
    if fusion is None:
        return "rrf"
    value = getattr(fusion, "fusion", fusion)
    name = str(getattr(value, "value", value)).lower()
    if name not in _FUSIONS:
        raise ValueError(
            f"Unsupported hybrid fusion: {fusion!r}. Supported: {', '.join(_FUSIONS)}"
        )
    return name  # type: ignore[return-value]


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

    Each record may also carry a sparse vector (``__sparse__`` key on upsert),
    kept alongside the matrix and scored by :meth:`query_sparse`. Sparse
    vectors are persisted only when :attr:`sparse_type` is set, so a dense-only
    store writes exactly the layout it always did.

    :param embedding_dim: Vector dimensionality.
    :type embedding_dim: int
    :param storage_file: Path of the JSON file used for persistence.
    :type storage_file: str
    :param metric: Similarity metric determining storage and scoring.
    :type metric: Literal["cosine", "dot"]
    :param sparse_type: Sparse mode, or ``None`` for a dense-only store.
        ``"bm25"`` and ``"bm42"`` weight sparse queries by inverse document
        frequency.
    :type sparse_type: SparseType | None
    :raises ValueError: If ``metric`` or ``sparse_type`` is not supported.
    """

    def __init__(
        self,
        embedding_dim: int,
        storage_file: str,
        metric: Literal["cosine", "dot"] = "cosine",
        sparse_type: Optional[SparseType] = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.storage_file = storage_file
        self.metric = _coerce_metric(metric)
        self.sparse_type = _coerce_sparse_type(sparse_type)
        self._matrix: np.ndarray = np.zeros((0, embedding_dim), dtype=_FLOAT)
        self._rows: List[Dict[str, Any]] = []
        self._sparse: List[Optional[SparseEmbedding]] = []
        self._postings: Optional[Dict[int, List[Tuple[int, float]]]] = None
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
        those always held cosine-normalized vectors and are read as such. Files
        written before sparse vectors were supported carry no ``sparse`` section;
        their records load without sparse vectors and keep answering dense
        queries as before.

        :raises ValueError: If the stored embedding dimension, metric or sparse
            mode does not match this instance, or if rows and vectors are out of
            sync.
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

        sparse_section = storage.get("sparse")
        if sparse_section is None:
            self._sparse = [None] * len(self._rows)
        else:
            loaded_sparse_type = _coerce_sparse_type(sparse_section.get("type"))
            if self.sparse_type is None:
                # Opening a hybrid index as dense-only: the sparse vectors are
                # kept, and written back on save, but not used for scoring.
                self.sparse_type = loaded_sparse_type
                logger.info(
                    f"Vector store '{self.storage_file}' holds '{loaded_sparse_type}' "
                    f"sparse vectors; they are kept but not used for queries."
                )
            elif loaded_sparse_type != self.sparse_type:
                raise ValueError(
                    f"Sparse type mismatch for '{self.storage_file}', expected: "
                    f"{self.sparse_type}, but loaded: {loaded_sparse_type}. Sparse vectors "
                    f"from different models are not comparable, so the index must be rebuilt."
                )
            self._sparse = [_sparse_from_json(raw) for raw in sparse_section.get("vectors", [])]
            if len(self._sparse) != len(self._rows):
                raise ValueError(
                    f"Vector store sparse count mismatch, rows: {len(self._rows)}, "
                    f"sparse vectors: {len(self._sparse)}"
                )
        self._rebuild_index()
        logger.info(f"Loaded {len(self._rows)} points from vector store '{self.storage_file}'.")

    def save(self) -> None:
        """
        Persist the current state to :attr:`storage_file`.

        The written JSON is compatible with the historical NanoVectorDB layout.
        Stores with a sparse mode add a ``sparse`` section holding the mode and
        one entry per row (``{"indices": [...], "values": [...]}`` or ``null``);
        readers that predate it ignore the extra key.
        """
        storage: Dict[str, Any] = {
            "embedding_dim": self.embedding_dim,
            "metric": self.metric,
            "data": self._rows,
            "matrix": _encode_matrix(self._matrix),
        }
        if self.sparse_type is not None:
            storage["sparse"] = {
                "type": self.sparse_type,
                "vectors": [_sparse_to_json(sparse) for sparse in self._sparse],
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
        """Rebuild the id-to-row-index lookup and drop the sparse postings."""
        self._id_to_index = {row[F_ID]: i for i, row in enumerate(self._rows)}
        self._postings = None

    def upsert(self, datas: List[Dict[str, Any]]) -> None:
        """
        Insert or update records.

        :param datas: Records, each containing ``__id__``, ``__vector__``, an
            optional ``__sparse__`` :class:`SparseEmbedding` and optional
            metadata keys. Duplicate ids within the batch keep the last
            occurrence. A record written without ``__sparse__`` loses any sparse
            vector it had, as a whole-point replace does in Qdrant.
        :type datas: list[dict]
        :raises ValueError: If a sparse vector has duplicate indices.
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
        new_sparse: List[Optional[SparseEmbedding]] = []
        for record_id, vector in zip(record_ids, vectors):
            row = {
                key: value
                for key, value in deduped[record_id].items()
                if key not in (F_VECTOR, F_SPARSE)
            }
            row[F_ID] = record_id
            raw_sparse = deduped[record_id].get(F_SPARSE)
            sparse = (
                None
                if raw_sparse is None
                else _canonical_sparse(raw_sparse, f"Record '{record_id}'")
            )

            existing = self._id_to_index.get(record_id)
            if existing is not None:
                self._matrix[existing] = vector
                self._rows[existing] = row
                self._sparse[existing] = sparse
            else:
                self._id_to_index[record_id] = len(self._rows) + len(new_rows)
                new_rows.append(row)
                new_vectors.append(vector)
                new_sparse.append(sparse)

        if new_rows:
            stacked = np.vstack(new_vectors)
            self._rows.extend(new_rows)
            self._sparse.extend(new_sparse)
            self._matrix = np.vstack([self._matrix, stacked]) if self._matrix.size else stacked
        self._postings = None

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
        self._sparse = [self._sparse[i] for i in keep_index]
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

    def get_sparse(self, ids: List[str]) -> List[Optional[SparseEmbedding]]:
        """
        Fetch stored sparse vectors by id, preserving input order.

        :param ids: Record identifiers to fetch.
        :type ids: list[str]
        :returns: Sparse vectors aligned with ``ids``; missing ids and records
            stored without a sparse vector are mapped to ``None``.
        :rtype: list[SparseEmbedding | None]
        """
        return [
            self._sparse[self._id_to_index[record_id]] if record_id in self._id_to_index else None
            for record_id in ids
        ]

    def _get_postings(self) -> Dict[int, List[Tuple[int, float]]]:
        """
        Return the inverted index ``{sparse index: [(row, value), ...]}``.

        Built on first use after a write and cached until the next one.
        """
        if self._postings is None:
            postings: Dict[int, List[Tuple[int, float]]] = {}
            for row, sparse in enumerate(self._sparse):
                if sparse is None:
                    continue
                for index, value in zip(sparse.indices, sparse.values):
                    postings.setdefault(index, []).append((row, value))
            self._postings = postings
        return self._postings

    def _rank_sparse(self, query: SparseEmbedding, top_k: int) -> _Ranking:
        """
        Score one sparse query against every stored sparse vector.

        The score is the dot product over shared indices; records sharing no
        index with the query are not returned, as in Qdrant. Under ``bm25`` and
        ``bm42`` each query weight is first multiplied by its IDF over the
        records that have a sparse vector.

        :param query: Sparse query vector.
        :type query: SparseEmbedding
        :param top_k: Maximum number of hits.
        :type top_k: int
        :returns: ``(row index, score)`` pairs, best first.
        :rtype: list[tuple[int, float]]
        """
        postings = self._get_postings()
        use_idf = self.sparse_type in IDF_SPARSE_TYPES
        num_documents = sum(sparse is not None for sparse in self._sparse) if use_idf else 0

        scores: Dict[int, float] = {}
        for index, value in zip(query.indices, query.values):
            entries = postings.get(int(index))
            if not entries:
                continue
            weight = float(value)
            if use_idf:
                weight *= _idf(len(entries), num_documents)
            for row, stored_value in entries:
                scores[row] = scores.get(row, 0.0) + weight * stored_value

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:top_k]

    def query_sparse(
        self,
        queries: List[SparseEmbedding],
        top_k: int,
    ) -> List[List[Tuple[str, float, Dict[str, Any]]]]:
        """
        Score a batch of sparse queries against the stored sparse vectors.

        :param queries: Sparse query vectors.
        :type queries: list[SparseEmbedding]
        :param top_k: Maximum number of hits per query.
        :type top_k: int
        :returns: Per-query lists of ``(id, score, metadata)`` ordered by
            descending score, index-aligned with ``queries``.
        :rtype: list[list[tuple[str, float, dict]]]
        :raises ValueError: If ``top_k`` is not positive.
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got: {top_k}")
        return [
            [self._hit(row, score) for row, score in self._rank_sparse(query, top_k)]
            for query in queries
        ]

    def query_hybrid(
        self,
        dense_queries: np.ndarray,
        sparse_queries: List[Optional[SparseEmbedding]],
        top_k: int,
        threshold: Optional[float] = None,
        fusion: FusionType = "rrf",
    ) -> List[List[Tuple[str, float, Dict[str, Any]]]]:
        """
        Score a batch of queries by fusing dense and sparse rankings.

        Mirrors a Qdrant hybrid query: per query, the dense and the sparse
        search each fetch up to ``top_k`` candidates, the two rankings are fused
        (reciprocal rank or distribution-based score fusion), and the best
        ``top_k`` fused records are returned with the fused score. A query whose
        sparse vector is ``None`` is answered by the dense search alone, with
        dense scores.

        :param dense_queries: Dense query matrix ``(n_queries, embedding_dim)``.
        :type dense_queries: numpy.ndarray
        :param sparse_queries: Sparse query per dense row, or ``None``.
        :type sparse_queries: list[SparseEmbedding | None]
        :param top_k: Maximum number of hits per query and per branch.
        :type top_k: int
        :param threshold: Optional minimum dense score; applied to the dense
            branch only, as in a dense-only query.
        :type threshold: float | None
        :param fusion: ``"rrf"`` or ``"dbsf"``.
        :type fusion: FusionType
        :returns: Per-query lists of ``(id, score, metadata)`` ordered by
            descending score, index-aligned with the input rows.
        :rtype: list[list[tuple[str, float, dict]]]
        :raises ValueError: If inputs are misaligned or ``top_k`` is not positive.
        """
        dense_rankings = self._rank_dense(dense_queries, top_k, threshold)
        if len(sparse_queries) != len(dense_rankings):
            raise ValueError(
                f"Got {len(dense_rankings)} dense and {len(sparse_queries)} sparse queries"
            )
        fuse = _FUSIONS[fusion]

        results: List[List[Tuple[str, float, Dict[str, Any]]]] = []
        for dense_ranking, sparse_query in zip(dense_rankings, sparse_queries):
            if sparse_query is None:
                ranking = dense_ranking
            else:
                ranking = fuse([dense_ranking, self._rank_sparse(sparse_query, top_k)], top_k)
            results.append([self._hit(row, score) for row, score in ranking])
        return results

    def _hit(self, row: int, score: float) -> Tuple[str, float, Dict[str, Any]]:
        """Build an ``(id, score, metadata)`` hit for a stored row."""
        record = self._rows[row]
        metadata = {key: value for key, value in record.items() if key != F_ID}
        return record[F_ID], float(score), metadata

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
        return [
            [self._hit(row, score) for row, score in ranking]
            for ranking in self._rank_dense(queries, top_k, threshold)
        ]

    def _rank_dense(
        self,
        queries: np.ndarray,
        top_k: int,
        threshold: Optional[float] = None,
    ) -> List[_Ranking]:
        """
        Score a batch of dense queries; see :meth:`query` for the semantics.

        :returns: Per-query ``(row index, score)`` pairs, best first.
        :rtype: list[list[tuple[int, float]]]
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

        rankings: List[_Ranking] = []
        for row_scores in scores:
            top_index = np.argpartition(row_scores, -limit)[-limit:]
            top_index = top_index[np.argsort(row_scores[top_index])[::-1]]

            ranking: _Ranking = []
            for index in top_index:
                score = float(row_scores[index])
                if threshold is not None and score < threshold:
                    break
                ranking.append((int(index), score))
            rankings.append(ranking)
        return rankings


class NanoVectorDBStorage(BaseVectorStorage):
    """
    Vector storage implementation backed by :class:`DenseVectorDB`.

    Provides a simple file-backed vector database for storing and retrieving
    embeddings and performing batched nearest-neighbor search.

    With ``sparse_type`` set it also stores one sparse vector per point and
    answers dense + sparse queries with hybrid retrieval, following the Qdrant
    adapter: both searches fetch ``top_k`` candidates, the rankings are fused by
    reciprocal rank (or distribution-based score fusion, via
    ``hybrid_query_type``), and hits carry the fused score. Indexes written
    without sparse vectors keep loading; their points simply take part in the
    dense branch only.
    """

    def __init__(
        self,
        embedding_dim: int,
        score_threshold: Optional[float] = None,
        storage_folder: str = Settings.storage_folder,
        filename: str = "data.json",
        metric: Literal["cosine", "dot"] = "cosine",
        cosine_threshold: Optional[float] = None,
        sparse_type: Optional[SparseType] = None,
        **kwargs: Any,
    ):
        """
        Initialize the vector storage.

        :param embedding_dim: Embedding dimensionality.
        :param score_threshold: Minimum similarity score for query results. When
            omitted, a metric-specific default is used (``0.2`` for cosine, no
            filtering for dot product).
        :param storage_folder: Folder where the vector storage file is located.
        :param filename: Name of the JSON file containing the stored vectors.
        :param metric: Similarity metric. ``cosine`` stores L2-normalized vectors;
            ``dot`` stores them verbatim.
        :param cosine_threshold: Deprecated alias for ``score_threshold``.
        :param sparse_type: Optional sparse mode, as in the Qdrant adapter:
            ``"bm25"`` / ``"bm42"`` (query weights rescaled by IDF),
            ``"splade"`` or ``"custom"`` (plain dot product). ``None`` keeps the
            store dense-only, ignoring sparse payloads with a warning.
        :param kwargs: Additional keyword arguments passed to the base class.
        :raises TypeError: If both ``score_threshold`` and ``cosine_threshold``
            are provided.
        :raises ValueError: If ``metric`` or ``sparse_type`` is not supported,
            or the persisted index was built with a different one.
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
        self.sparse_type = _coerce_sparse_type(sparse_type)
        self._client = DenseVectorDB(
            embedding_dim,
            storage_file=self.filename,
            metric=self.metric,
            sparse_type=self.sparse_type,
        )

    @override
    async def upsert(self, data: List[Point], **kwargs: Any) -> None:
        """
        Insert or update a batch of embeddings in the database.

        :param data: Embedding records with vectors and metadata.
        :raises ValueError: If the batch mixes points with and without sparse
            embeddings while ``sparse_type`` is set.
        """
        if not data:
            logger.warning("Attempted to insert empty data into vector DB.")
            return

        has_sparse = [item.sparse_embedding is not None for item in data]
        store_sparse = self.sparse_type is not None
        if any(has_sparse):
            if not store_sparse:
                logger.warning(
                    "NanoVDB was created without `sparse_type`; ignoring sparse embeddings. "
                    "Set `sparse_type` (bm25, bm42, splade, custom) to store them."
                )
            elif not all(has_sparse):
                raise ValueError(
                    "All points in this batch must either have sparse embeddings or not have them."
                )

        items: List[Dict[str, Any]] = []
        for embedding in data:
            item: Dict[str, Any] = {
                F_ID: embedding.id,
                F_VECTOR: np.array(embedding.dense_embedding),
                **embedding.metadata,
            }
            if store_sparse and embedding.sparse_embedding is not None:
                item[F_SPARSE] = embedding.sparse_embedding
            items.append(item)

        if not items:
            return

        self._client.upsert(items)

    @override
    async def query(self, points: List[Point], **kwargs: Any) -> List[List[EmbeddingHit]]:
        """
        Search for the most similar records for each query embedding.

        Dense-only points run one batched similarity search against all stored
        vectors, returning hits that reach the score threshold. When
        ``sparse_type`` is set, points that also carry a sparse embedding are
        answered by hybrid retrieval instead: dense and sparse candidates are
        fused and returned with the fused score, as in the Qdrant adapter.

        :param points: Query embedding payloads, one per search.
        :param kwargs:
            top_k: Number of nearest neighbors to return per query.
            hybrid_query_type: Fusion for dense+sparse queries: ``"rrf"``
                (default) or ``"dbsf"``, or a Qdrant ``FusionQuery``.
        :return: Per-query lists of matched records, index-aligned with ``points``.
        """
        top_k: int = kwargs.pop("top_k", 20)
        fusion = _coerce_fusion(kwargs.pop("hybrid_query_type", None))
        if not points:
            return []

        for point in points:
            if point.dense_embedding is None:
                raise ValueError("Empty dense embedding payload.")

        query_matrix = np.array([point.dense_embedding for point in points])
        has_sparse = any(point.sparse_embedding is not None for point in points)
        if has_sparse and self.sparse_type is None:
            logger.warning(
                "NanoVDB was created without `sparse_type`; ignoring sparse query payload."
            )

        if has_sparse and self.sparse_type is not None:
            batched = self._client.query_hybrid(
                query_matrix,
                [point.sparse_embedding for point in points],
                top_k=top_k,
                threshold=self.score_threshold,
                fusion=fusion,
            )
        else:
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
        for entry, sparse in zip(self._client.get_points(ids), self._client.get_sparse(ids)):
            if entry is None:
                points.append(None)
                continue
            record_id, vector, metadata = entry
            points.append(
                Point(
                    id=record_id,
                    dense_embedding=vector,
                    sparse_embedding=sparse,
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

    async def index_start_callback(self) -> None:
        """
        Pre-index hook for interface compatibility.
        """
        pass

    async def query_done_callback(self) -> None:
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
