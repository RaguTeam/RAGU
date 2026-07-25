from __future__ import annotations

import importlib
import math
import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from typing import Any


@dataclass
class FakeVectorParams:
    size: int
    distance: str


@dataclass
class FakeSparseVectorParams:
    modifier: str | None = None


@dataclass
class FakeSparseVector:
    indices: list[int]
    values: list[float]


@dataclass
class FakePointStruct:
    id: str
    vector: list[float] | dict[str, object] | None
    payload: dict


@dataclass
class FakeBatch:
    """Column-oriented upsert payload, mirroring ``qdrant_client.models.Batch``."""

    ids: list[str]
    vectors: dict[str, list] | list
    payloads: list[dict] | None = None

    def to_point_structs(self) -> list["FakePointStruct"]:
        """Expand columns back into per-point records."""
        points: list[FakePointStruct] = []
        for index, point_id in enumerate(self.ids):
            if isinstance(self.vectors, dict):
                vector = {name: column[index] for name, column in self.vectors.items()}
            else:
                vector = self.vectors[index]
            payload = self.payloads[index] if self.payloads is not None else {}
            points.append(FakePointStruct(id=point_id, vector=vector, payload=payload))
        return points


@dataclass
class FakePointIdsList:
    points: list[str]


@dataclass
class FakeScoredPoint:
    id: str
    score: float
    payload: dict


@dataclass
class FakeQueryResponse:
    points: list[FakeScoredPoint]


@dataclass
class FakePrefetch:
    query: list[float] | FakeSparseVector
    using: str
    limit: int


@dataclass
class FakeFusionQuery:
    fusion: str


@dataclass
class FakeQueryRequest:
    query: Any
    prefetch: FakePrefetch | list[FakePrefetch] | None = None
    using: str | None = None
    limit: int = 10
    with_payload: bool = True


class FakeAsyncQdrantClient:
    registries: dict[str, dict[str, dict[str, object]]] = {}
    instances: list["FakeAsyncQdrantClient"] = []

    def __init__(self, path: str | None = None, **kwargs: Any):
        self.path = None if path is None else str(path)
        self.kwargs = kwargs
        self.location = self.path or "__memory__"
        self.registry = self.registries.setdefault(self.location, {})
        self.instances.append(self)

    @classmethod
    def reset(cls) -> None:
        cls.registries = {}
        cls.instances = []

    async def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.registry

    async def create_collection(
        self,
        collection_name: str,
        vectors_config,
        sparse_vectors_config=None,
        **kwargs: Any,
    ) -> None:
        self.registry.setdefault(
            collection_name,
            {
                "vectors_config": vectors_config,
                "sparse_vectors_config": sparse_vectors_config or {},
                "points": {},
            },
        )

    async def get_collection(self, collection_name: str):
        collection = self.registry[collection_name]
        vectors = collection["vectors_config"]
        sparse_vectors = collection.get("sparse_vectors_config", {})
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors=vectors if isinstance(vectors, dict) else SimpleNamespace(
                        size=vectors.size,
                        distance=vectors.distance,
                    ),
                    sparse_vectors=sparse_vectors,
                )
            )
        )

    async def upsert(
        self,
        collection_name: str,
        points: "list[FakePointStruct] | FakeBatch",
        **kwargs: Any,
    ) -> None:
        collection = self.registry[collection_name]
        stored_points = collection["points"]
        if isinstance(points, FakeBatch):
            points = points.to_point_structs()
        for point in points:
            stored_points[point.id] = self._normalize_cosine_vectors(collection, point)

    @staticmethod
    def _normalize_cosine_vectors(
        collection: dict[str, object],
        point: FakePointStruct,
    ) -> FakePointStruct:
        """
        L2-normalize dense vectors stored under a cosine-distance config.

        Real Qdrant normalizes on upload for ``Distance.COSINE`` and returns the
        normalized vector on retrieve, discarding the original magnitude. Storing
        vectors verbatim here would let the fake accept behaviour the real engine
        rejects, which is exactly the divergence this testkit must not hide.
        """
        vectors_config = collection["vectors_config"]
        if not isinstance(point.vector, dict) or not isinstance(vectors_config, dict):
            return point

        normalized = dict(point.vector)
        for name, value in point.vector.items():
            params = vectors_config.get(name)
            if getattr(params, "distance", None) != "cosine":
                continue
            if not isinstance(value, list):
                continue
            norm = math.sqrt(sum(component * component for component in value))
            if norm:
                normalized[name] = [component / norm for component in value]

        return FakePointStruct(id=point.id, vector=normalized, payload=point.payload)

    async def query_points(
        self,
        collection_name: str,
        query,
        limit: int,
        with_payload: bool = True,
        using: str | None = None,
        prefetch: FakePrefetch | list[FakePrefetch] | None = None,
        score_threshold: float | None = None,
        **kwargs: Any,
    ) -> FakeQueryResponse:
        collection = self.registry[collection_name]
        stored_points = collection["points"]
        if prefetch is not None and isinstance(query, FakeFusionQuery):
            prefetches = prefetch if isinstance(prefetch, list) else [prefetch]
            ranked_lists = [
                self._rank_points(
                    stored_points,
                    query=entry.query,
                    using=entry.using,
                    limit=entry.limit,
                    with_payload=with_payload,
                )
                for entry in prefetches
            ]
            fused = self._fuse_rrf(ranked_lists)
            return FakeQueryResponse(points=fused[:limit])

        scored_points = self._rank_points(
            stored_points,
            query=query,
            using=using,
            limit=limit,
            with_payload=with_payload,
            score_threshold=score_threshold,
        )
        return FakeQueryResponse(points=scored_points)

    async def query_batch_points(
        self,
        collection_name: str,
        requests: list[FakeQueryRequest],
        **kwargs: Any,
    ) -> list[FakeQueryResponse]:
        responses: list[FakeQueryResponse] = []
        for request in requests:
            responses.append(
                await self.query_points(
                    collection_name=collection_name,
                    query=request.query,
                    limit=request.limit,
                    with_payload=request.with_payload,
                    using=request.using,
                    prefetch=request.prefetch,
                )
            )
        return responses

    async def delete(self, collection_name: str, points_selector, **kwargs: Any) -> None:
        collection = self.registry[collection_name]
        stored_points = collection["points"]

        if isinstance(points_selector, FakePointIdsList):
            ids = points_selector.points
        else:
            ids = list(points_selector)

        for point_id in ids:
            stored_points.pop(point_id, None)

    async def scroll(
        self,
        collection_name: str,
        limit: int,
        offset=None,
        with_payload: bool = False,
        with_vectors: bool = False,
        **kwargs: Any,
    ):
        points = list(self.registry[collection_name]["points"].values())
        start_index = 0
        if offset is not None:
            for index, point in enumerate(points):
                if point.id == offset:
                    start_index = index + 1
                    break

        batch = points[start_index:start_index + limit]
        next_offset = None
        if start_index + limit < len(points):
            next_offset = batch[-1].id

        return [
            FakePointStruct(
                id=point.id,
                vector=point.vector if with_vectors else None,
                payload=point.payload if with_payload else {},
            )
            for point in batch
        ], next_offset

    async def retrieve(
        self,
        collection_name: str,
        ids: list[str],
        with_vectors: bool = False,
        with_payload: bool = True,
        **kwargs: Any,
    ):
        stored_points = self.registry[collection_name]["points"]
        results = []
        for point_id in ids:
            point = stored_points.get(point_id)
            if point is None:
                continue
            results.append(
                FakePointStruct(
                    id=point.id,
                    vector=point.vector if with_vectors else None,
                    payload=point.payload if with_payload else {},
                )
            )
        return results

    async def close(self, **kwargs: Any) -> None:
        return None

    @staticmethod
    def _vector_for_using(point: FakePointStruct, using: str | None):
        if isinstance(point.vector, dict):
            if using is None:
                return point.vector.get("dense")
            return point.vector.get(using)
        return point.vector

    def _rank_points(
        self,
        stored_points: dict[str, FakePointStruct],
        query,
        using: str | None,
        limit: int,
        with_payload: bool,
        score_threshold: float | None = None,
    ) -> list[FakeScoredPoint]:
        scored_points: list[FakeScoredPoint] = []
        for point in stored_points.values():
            target_vector = self._vector_for_using(point, using)
            if target_vector is None:
                continue
            if isinstance(query, FakeSparseVector):
                assert isinstance(target_vector, FakeSparseVector)
                score = self._sparse_similarity(query, target_vector)
            else:
                assert isinstance(target_vector, list)
                score = self._cosine_similarity(query, target_vector)
            if score_threshold is not None and score < score_threshold:
                continue
            scored_points.append(
                FakeScoredPoint(
                    id=point.id,
                    score=score,
                    payload=point.payload if with_payload else {},
                )
            )
        scored_points.sort(key=lambda item: item.score, reverse=True)
        return scored_points[:limit]

    @staticmethod
    def _fuse_rrf(ranked_lists: list[list[FakeScoredPoint]]) -> list[FakeScoredPoint]:
        payload_by_id: dict[str, dict] = {}
        score_by_id: dict[str, float] = {}
        for ranked in ranked_lists:
            for rank, point in enumerate(ranked, start=1):
                payload_by_id.setdefault(point.id, point.payload)
                score_by_id[point.id] = score_by_id.get(point.id, 0.0) + 1.0 / (60 + rank)
        fused = [
            FakeScoredPoint(id=point_id, score=score, payload=payload_by_id[point_id])
            for point_id, score in score_by_id.items()
        ]
        fused.sort(key=lambda item: item.score, reverse=True)
        return fused

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        numerator = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return numerator / (left_norm * right_norm)

    @staticmethod
    def _sparse_similarity(left: FakeSparseVector, right: FakeSparseVector) -> float:
        right_lookup = dict(zip(right.indices, right.values))
        return sum(value * right_lookup.get(index, 0.0) for index, value in zip(left.indices, left.values))


def install_fake_qdrant(monkeypatch) -> None:
    FakeAsyncQdrantClient.reset()

    models_module = ModuleType("qdrant_client.models")
    models_module.Distance = SimpleNamespace(COSINE="cosine")
    models_module.Modifier = SimpleNamespace(IDF="idf")
    models_module.Batch = FakeBatch
    models_module.PointIdsList = FakePointIdsList
    models_module.PointStruct = FakePointStruct
    models_module.Prefetch = FakePrefetch
    models_module.QueryRequest = FakeQueryRequest
    models_module.SparseVector = FakeSparseVector
    models_module.SparseVectorParams = FakeSparseVectorParams
    models_module.VectorParams = FakeVectorParams

    http_models_models_module = ModuleType("qdrant_client.http.models.models")
    http_models_models_module.Distance = models_module.Distance
    http_models_models_module.Fusion = SimpleNamespace(RRF="rrf")
    http_models_models_module.FusionQuery = FakeFusionQuery
    http_models_models_module.Modifier = models_module.Modifier
    http_models_models_module.Batch = FakeBatch
    http_models_models_module.PointIdsList = FakePointIdsList
    http_models_models_module.Prefetch = FakePrefetch
    http_models_models_module.PointStruct = FakePointStruct
    http_models_models_module.SparseVector = FakeSparseVector
    http_models_models_module.SparseVectorParams = FakeSparseVectorParams
    http_models_models_module.VectorParams = FakeVectorParams

    http_models_module = ModuleType("qdrant_client.http.models")
    http_models_module.Fusion = http_models_models_module.Fusion
    http_models_module.FusionQuery = FakeFusionQuery
    http_models_module.models = http_models_models_module

    http_module = ModuleType("qdrant_client.http")
    http_module.models = http_models_module

    conversions_common_types_module = ModuleType("qdrant_client.conversions.common_types")
    conversions_common_types_module.CollectionInfo = object
    conversions_common_types_module.QueryResponse = FakeQueryResponse

    conversions_module = ModuleType("qdrant_client.conversions")
    conversions_module.common_types = conversions_common_types_module

    qdrant_module = ModuleType("qdrant_client")
    qdrant_module.AsyncQdrantClient = FakeAsyncQdrantClient
    qdrant_module.conversions = conversions_module
    qdrant_module.models = models_module
    qdrant_module.http = http_module

    monkeypatch.setitem(sys.modules, "qdrant_client", qdrant_module)
    monkeypatch.setitem(sys.modules, "qdrant_client.conversions", conversions_module)
    monkeypatch.setitem(sys.modules, "qdrant_client.conversions.common_types", conversions_common_types_module)
    monkeypatch.setitem(sys.modules, "qdrant_client.models", models_module)
    monkeypatch.setitem(sys.modules, "qdrant_client.http", http_module)
    monkeypatch.setitem(sys.modules, "qdrant_client.http.models", http_models_module)
    monkeypatch.setitem(sys.modules, "qdrant_client.http.models.models", http_models_models_module)


def load_qdrant_storage(monkeypatch):
    install_fake_qdrant(monkeypatch)
    sys.modules.pop("ragu.storage.vdb_storage_adapters.qdrant_vdb", None)
    module = importlib.import_module("ragu.storage.vdb_storage_adapters.qdrant_vdb")
    return module.QdrantVectorDBStorage
