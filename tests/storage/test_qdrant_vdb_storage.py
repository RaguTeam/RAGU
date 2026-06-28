from __future__ import annotations

import importlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from ragu.storage.types import Point


@dataclass
class _FakeVectorParams:
    size: int
    distance: str


@dataclass
class _FakePointStruct:
    id: str
    vector: list[float] | dict[str, object]
    payload: dict


@dataclass
class _FakePointIdsList:
    points: list[str]


@dataclass
class _FakeScoredPoint:
    id: str
    score: float
    payload: dict


@dataclass
class _FakeQueryResponse:
    points: list[_FakeScoredPoint]


@dataclass
class _FakeSparseVector:
    indices: list[int]
    values: list[float]


@dataclass
class _FakeSparseVectorParams:
    modifier: str | None = None


@dataclass
class _FakePrefetch:
    query: list[float] | _FakeSparseVector
    using: str
    limit: int


@dataclass
class _FakeFusionQuery:
    fusion: str


@dataclass
class _FakeQueryRequest:
    query: object
    prefetch: object = None
    using: str | None = None
    limit: int = 10
    with_payload: bool = True


class FakeAsyncQdrantClient:
    registries: dict[str, dict[str, dict[str, object]]] = {}
    instances: list["FakeAsyncQdrantClient"] = []

    def __init__(self, path: str | None = None, **kwargs):
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
        **kwargs,
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

    async def upsert(self, collection_name: str, points: list[_FakePointStruct], **kwargs) -> None:
        collection = self.registry[collection_name]
        stored_points = collection["points"]
        for point in points:
            stored_points[point.id] = point

    async def query_points(
        self,
        collection_name: str,
        query,
        limit: int,
        with_payload: bool = True,
        using: str | None = None,
        prefetch: _FakePrefetch | list[_FakePrefetch] | None = None,
        score_threshold: float | None = None,
        **kwargs,
    ) -> _FakeQueryResponse:
        collection = self.registry[collection_name]
        stored_points = collection["points"]
        if prefetch is not None and isinstance(query, _FakeFusionQuery):
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
            return _FakeQueryResponse(points=fused[:limit])

        scored_points = self._rank_points(
            stored_points,
            query=query,
            using=using,
            limit=limit,
            with_payload=with_payload,
            score_threshold=score_threshold,
        )
        return _FakeQueryResponse(points=scored_points)

    async def query_batch_points(self, collection_name: str, requests, **kwargs):
        return [
            await self.query_points(
                collection_name=collection_name,
                query=request.query,
                limit=request.limit,
                with_payload=request.with_payload,
                using=request.using,
                prefetch=request.prefetch,
            )
            for request in requests
        ]

    async def delete(self, collection_name: str, points_selector, **kwargs) -> None:
        collection = self.registry[collection_name]
        stored_points = collection["points"]

        if isinstance(points_selector, _FakePointIdsList):
            ids = points_selector.points
        else:
            ids = list(points_selector)

        for point_id in ids:
            stored_points.pop(point_id, None)

    async def close(self, **kwargs) -> None:
        return None

    @staticmethod
    def _vector_for_using(point: _FakePointStruct, using: str | None):
        if isinstance(point.vector, dict):
            if using is None:
                return point.vector.get("dense")
            return point.vector.get(using)
        return point.vector

    def _rank_points(
        self,
        stored_points: dict[str, _FakePointStruct],
        query,
        using: str | None,
        limit: int,
        with_payload: bool,
        score_threshold: float | None = None,
    ) -> list[_FakeScoredPoint]:
        scored_points: list[_FakeScoredPoint] = []
        for point in stored_points.values():
            target_vector = self._vector_for_using(point, using)
            if target_vector is None:
                continue
            if isinstance(query, _FakeSparseVector):
                assert isinstance(target_vector, _FakeSparseVector)
                score = _sparse_similarity(query, target_vector)
            else:
                assert isinstance(target_vector, list)
                score = _cosine_similarity(query, target_vector)
            if score_threshold is not None and score < score_threshold:
                continue
            scored_points.append(
                _FakeScoredPoint(
                    id=point.id,
                    score=score,
                    payload=point.payload if with_payload else {},
                )
            )
        scored_points.sort(key=lambda item: item.score, reverse=True)
        return scored_points[:limit]

    @staticmethod
    def _fuse_rrf(ranked_lists: list[list[_FakeScoredPoint]]) -> list[_FakeScoredPoint]:
        payload_by_id: dict[str, dict] = {}
        score_by_id: dict[str, float] = {}
        for ranked in ranked_lists:
            for rank, point in enumerate(ranked, start=1):
                payload_by_id.setdefault(point.id, point.payload)
                score_by_id[point.id] = score_by_id.get(point.id, 0.0) + 1.0 / (60 + rank)
        fused = [
            _FakeScoredPoint(id=point_id, score=score, payload=payload_by_id[point_id])
            for point_id, score in score_by_id.items()
        ]
        fused.sort(key=lambda item: item.score, reverse=True)
        return fused


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _sparse_similarity(left: _FakeSparseVector, right: _FakeSparseVector) -> float:
    right_lookup = dict(zip(right.indices, right.values))
    return sum(value * right_lookup.get(index, 0.0) for index, value in zip(left.indices, left.values))


def _install_fake_qdrant(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeAsyncQdrantClient.reset()

    models_module = ModuleType("qdrant_client.models")
    models_module.Distance = SimpleNamespace(COSINE="cosine")
    models_module.Modifier = SimpleNamespace(IDF="idf")
    models_module.PointIdsList = _FakePointIdsList
    models_module.PointStruct = _FakePointStruct
    models_module.Prefetch = _FakePrefetch
    models_module.QueryRequest = _FakeQueryRequest
    models_module.SparseVector = _FakeSparseVector
    models_module.SparseVectorParams = _FakeSparseVectorParams
    models_module.VectorParams = _FakeVectorParams

    http_models_models_module = ModuleType("qdrant_client.http.models.models")
    http_models_models_module.Distance = models_module.Distance
    http_models_models_module.Fusion = SimpleNamespace(RRF="rrf")
    http_models_models_module.FusionQuery = _FakeFusionQuery
    http_models_models_module.Modifier = models_module.Modifier
    http_models_models_module.PointIdsList = _FakePointIdsList
    http_models_models_module.Prefetch = _FakePrefetch
    http_models_models_module.PointStruct = _FakePointStruct
    http_models_models_module.SparseVector = _FakeSparseVector
    http_models_models_module.SparseVectorParams = _FakeSparseVectorParams
    http_models_models_module.VectorParams = _FakeVectorParams

    http_models_module = ModuleType("qdrant_client.http.models")
    http_models_module.Fusion = http_models_models_module.Fusion
    http_models_module.FusionQuery = _FakeFusionQuery
    http_models_module.models = http_models_models_module

    http_module = ModuleType("qdrant_client.http")
    http_module.models = http_models_module

    conversions_common_types_module = ModuleType("qdrant_client.conversions.common_types")
    conversions_common_types_module.CollectionInfo = object
    conversions_common_types_module.QueryResponse = _FakeQueryResponse

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


def _load_qdrant_storage(monkeypatch: pytest.MonkeyPatch):
    _install_fake_qdrant(monkeypatch)
    sys.modules.pop("ragu.storage.vdb_storage_adapters.qdrant_vdb", None)
    module = importlib.import_module("ragu.storage.vdb_storage_adapters.qdrant_vdb")
    return module.QdrantVectorDBStorage


@pytest.mark.asyncio
async def test_upsert_accepts_empty_sparse_list(monkeypatch, tmp_path):
    QdrantVectorDBStorage = _load_qdrant_storage(monkeypatch)
    storage_file = tmp_path / "vdb.json"
    vdb = QdrantVectorDBStorage(embedding_dim=3, filename=str(storage_file))

    await vdb.upsert(
        [Point(id="doc-1", dense_embedding=np.array([1.0, 0.0, 0.0]), metadata={"tag": "dense-only"})],
        sparse_data=[],
    )

    collection = FakeAsyncQdrantClient.registries[str(tmp_path)][vdb.collection_name]
    stored_point = next(iter(collection["points"].values()))
    assert isinstance(stored_point.vector, dict)
    assert stored_point.vector["dense"] == [1.0, 0.0, 0.0]
    assert "sparse" not in stored_point.vector


@pytest.mark.asyncio
async def test_filename_drives_local_path_and_collection_name(monkeypatch, tmp_path):
    QdrantVectorDBStorage = _load_qdrant_storage(monkeypatch)
    storage_file = tmp_path / "vdb_entity.json"
    vdb = QdrantVectorDBStorage(embedding_dim=3, filename=str(storage_file), cosine_threshold=0.0)

    await vdb.upsert([Point(id="ent-1", dense_embedding=np.array([1.0, 0.0, 0.0]), metadata={"kind": "entity"})])

    assert len(FakeAsyncQdrantClient.instances) == 1
    assert Path(FakeAsyncQdrantClient.instances[0].path) == tmp_path
    assert vdb.collection_name in FakeAsyncQdrantClient.registries[str(tmp_path)]
    assert vdb.collection_name.endswith("_vdb_entity")
    created_collection = FakeAsyncQdrantClient.registries[str(tmp_path)][vdb.collection_name]
    assert "dense" in created_collection["vectors_config"]
    assert created_collection["sparse_vectors_config"] == {}
