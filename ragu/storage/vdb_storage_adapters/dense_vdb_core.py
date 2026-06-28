# Vendored and refactored from NanoVectorDB (MIT,
# https://github.com/gusye1234/nano-vectordb), originally adapted in RAGU from
# nano-graphrag (https://github.com/gusye1234/nano-graphrag).
#
# Differences from the upstream library:
#   * native batch query (a single matrix multiply for many queries);
#   * no dependency on the external ``nano-vectordb`` package;
#   * cleaner public surface (no name-mangled internal storage access).
#
# The on-disk JSON format is kept byte-compatible with NanoVectorDB so that
# existing storage files load without any migration.

import base64
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ragu.common.logger import logger

_FLOAT = np.float32

#: Reserved record keys used internally by the store.
F_ID = "__id__"
F_VECTOR = "__vector__"
F_METRICS = "__metrics__"


def _encode_matrix(matrix: np.ndarray) -> str:
    """
    Encode a float32 matrix into the NanoVectorDB base64 buffer string.

    :param matrix: Dense matrix to encode.
    :type matrix: numpy.ndarray
    :returns: Base64-encoded little-endian float32 buffer.
    :rtype: str
    """
    return base64.b64encode(matrix.astype(_FLOAT).tobytes()).decode()


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
    return np.frombuffer(base64.b64decode(buffer), dtype=_FLOAT).reshape(-1, embedding_dim)


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


class DenseVectorDB:
    """
    Brute-force cosine vector store backed by a single normalized matrix.

    Vectors are L2-normalized on insertion, so cosine similarity reduces to a
    dot product. Queries are evaluated as one matrix multiply over the whole
    batch, which is markedly faster than scoring queries one at a time.

    :param embedding_dim: Vector dimensionality.
    :type embedding_dim: int
    :param storage_file: Path of the JSON file used for persistence.
    :type storage_file: str
    """

    def __init__(self, embedding_dim: int, storage_file: str) -> None:
        self.embedding_dim = embedding_dim
        self.storage_file = storage_file
        self._matrix: np.ndarray = np.zeros((0, embedding_dim), dtype=_FLOAT)
        self._rows: List[Dict[str, Any]] = []
        self._id_to_index: Dict[str, int] = {}
        self._load()

    def __len__(self) -> int:
        return len(self._rows)

    def _load(self) -> None:
        """
        Load persisted state from :attr:`storage_file` if it exists.

        :raises ValueError: If the stored embedding dimension does not match.
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

        self._rows = storage.get("data", [])
        matrix_buffer = storage.get("matrix")
        if matrix_buffer:
            self._matrix = _decode_matrix(matrix_buffer, self.embedding_dim)
        else:
            self._matrix = np.zeros((0, self.embedding_dim), dtype=_FLOAT)
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
            "data": self._rows,
            "matrix": _encode_matrix(self._matrix),
        }
        with open(self.storage_file, "w", encoding="utf-8") as handle:
            json.dump(storage, handle, ensure_ascii=False)
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

        new_vectors: List[np.ndarray] = []
        new_rows: List[Dict[str, Any]] = []
        for record_id, data in deduped.items():
            vector = _normalize(np.asarray(data[F_VECTOR], dtype=_FLOAT).reshape(1, -1))[0]
            row = {key: value for key, value in data.items() if key != F_VECTOR}
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

        :param queries: Query matrix of shape ``(n_queries, embedding_dim)``.
            A single 1-D vector is also accepted.
        :type queries: numpy.ndarray
        :param top_k: Maximum number of hits per query.
        :type top_k: int
        :param threshold: Optional minimum cosine score; hits scoring below it
            are dropped.
        :type threshold: float | None
        :returns: Per-query lists of ``(id, score, metadata)`` ordered by
            descending score, index-aligned with the input rows.
        :rtype: list[list[tuple[str, float, dict]]]
        """
        matrix = np.atleast_2d(np.asarray(queries, dtype=_FLOAT))
        if self._matrix.shape[0] == 0:
            return [[] for _ in range(matrix.shape[0])]

        normalized = _normalize(matrix)
        scores = normalized @ self._matrix.T  # (n_queries, n_records)
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
