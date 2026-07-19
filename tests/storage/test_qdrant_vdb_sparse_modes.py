from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from ragu.storage.types import Point, SparseEmbedding


@dataclass
class _FakeVectorParams:
    size: int
    distance: str


@dataclass
class _FakeSparseVectorParams:
    modifier: str | None = None


@dataclass
class _FakeSparseVector:
    indices: list[int]
    values: list[float]


@dataclass
class _FakePointStruct:
    id: str
    vector: dict[str, object]
    payload: dict


@dataclass
class _FakeBatch:
    """Column-oriented upsert payload, mirroring ``qdrant_client.models.Batch``."""

    ids: list[str]
    vectors: dict[str, list] | list
    payloads: list[dict] | None = None

    def to_point_structs(self) -> list["_FakePointStruct"]:
        """Expand columns back into per-point records."""
        points: list[_FakePointStruct] = []
        for index, point_id in enumerate(self.ids):
            if isinstance(self.vectors, dict):
                vector = {name: column[index] for name, column in self.vectors.items()}
            else:
                vector = self.vectors[index]
            payload = self.payloads[index] if self.payloads is not None else {}
            points.append(_FakePointStruct(id=point_id, vector=vector, payload=payload))
        return points


@dataclass
class _FakePointIdsList:
    points: list[str]


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
        self.registry[collection_name] = {
            "vectors_config": vectors_config,
            "sparse_vectors_config": sparse_vectors_config or {},
            "points": {},
        }

    async def get_collection(self, collection_name: str):
        collection = self.registry[collection_name]
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors=collection["vectors_config"],
                    sparse_vectors=collection["sparse_vectors_config"],
                )
            )
        )

    async def upsert(self, collection_name: str, points, **kwargs) -> None:
        if isinstance(points, _FakeBatch):
            points = points.to_point_structs()
        self.registry[collection_name]["points"].update({point.id: point for point in points})

    async def query_points(self, *args, **kwargs):
        raise NotImplementedError

    async def scroll(
        self,
        collection_name: str,
        limit: int,
        offset=None,
        with_payload: bool = False,
        with_vectors: bool = False,
        **kwargs,
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
            _FakePointStruct(
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
        **kwargs,
    ):
        stored_points = self.registry[collection_name]["points"]
        results = []
        for point_id in ids:
            point = stored_points.get(point_id)
            if point is None:
                continue
            results.append(
                _FakePointStruct(
                    id=point.id,
                    vector=point.vector if with_vectors else None,
                    payload=point.payload if with_payload else {},
                )
            )
        return results

    async def delete(self, *args, **kwargs):
        return None


def _install_fake_qdrant(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeAsyncQdrantClient.reset()

    models_module = ModuleType("qdrant_client.models")
    models_module.Distance = SimpleNamespace(COSINE="cosine")
    models_module.Modifier = SimpleNamespace(IDF="idf")
    models_module.Batch = _FakeBatch
    models_module.PointIdsList = _FakePointIdsList
    models_module.PointStruct = _FakePointStruct
    models_module.Prefetch = object
    models_module.QueryRequest = object
    models_module.SparseVector = _FakeSparseVector
    models_module.SparseVectorParams = _FakeSparseVectorParams
    models_module.VectorParams = _FakeVectorParams

    http_models_module = ModuleType("qdrant_client.http.models")
    http_models_module.Fusion = SimpleNamespace(RRF="rrf")
    http_models_module.FusionQuery = object

    http_module = ModuleType("qdrant_client.http")
    http_module.models = http_models_module

    conversions_common_types_module = ModuleType("qdrant_client.conversions.common_types")
    conversions_common_types_module.CollectionInfo = object
    conversions_common_types_module.QueryResponse = object

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


def _load_qdrant_storage(monkeypatch: pytest.MonkeyPatch):
    _install_fake_qdrant(monkeypatch)
    sys.modules.pop("ragu.storage.vdb_storage_adapters.qdrant_vdb", None)
    module = importlib.import_module("ragu.storage.vdb_storage_adapters.qdrant_vdb")
    return module.QdrantVectorDBStorage


@pytest.mark.asyncio
async def test_bm42_sparse_mode_creates_idf_index_and_uses_bm42_vector_name(monkeypatch, tmp_path):
    QdrantVectorDBStorage = _load_qdrant_storage(monkeypatch)
    storage = QdrantVectorDBStorage(
        embedding_dim=3,
        filename=str(tmp_path / "vdb.json"),
        sparse_type="bm42",
    )

    await storage.upsert(
        [
            Point(
                id="doc-1",
                dense_embedding=np.array([1.0, 0.0, 0.0]),
                sparse_embedding=SparseEmbedding(indices=[7, 9], values=[1.0, 0.5]),
                metadata={"tag": "hybrid"},
            )
        ]
    )

    collection = FakeAsyncQdrantClient.registries[str(tmp_path)][storage.collection_name]
    sparse_config = collection["sparse_vectors_config"]["bm42"]
    stored_point = next(iter(collection["points"].values()))

    assert sparse_config.modifier == "idf"
    assert "bm42" in stored_point.vector
    assert stored_point.vector["bm42"] == _FakeSparseVector(indices=[7, 9], values=[1.0, 0.5])


@pytest.mark.asyncio
async def test_bm42_sparse_mode_validates_existing_collection_modifier(monkeypatch, tmp_path):
    QdrantVectorDBStorage = _load_qdrant_storage(monkeypatch)
    storage = QdrantVectorDBStorage(
        embedding_dim=3,
        filename=str(tmp_path / "vdb.json"),
        sparse_type="bm42",
    )

    FakeAsyncQdrantClient.registries[str(tmp_path)] = {
        storage.collection_name: {
            "vectors_config": {
                "dense": _FakeVectorParams(size=3, distance="cosine"),
            },
            "sparse_vectors_config": {
                "bm42": _FakeSparseVectorParams(modifier="idf"),
            },
            "points": {},
        }
    }

    await storage.index_start_callback()


@pytest.mark.asyncio
async def test_remote_qdrant_args_are_explicit_constructor_parameters(monkeypatch, tmp_path):
    QdrantVectorDBStorage = _load_qdrant_storage(monkeypatch)
    storage = QdrantVectorDBStorage(
        embedding_dim=3,
        filename=str(tmp_path / "vdb.json"),
        url="http://qdrant.example",
        host="qdrant.example",
        port=6333,
        grpc_port=6334,
        api_key="secret",
        location="us-east",
        timeout=10,
    )

    await storage.index_start_callback()

    client = FakeAsyncQdrantClient.instances[0]
    assert client.path is None
    assert client.kwargs == {
        "url": "http://qdrant.example",
        "host": "qdrant.example",
        "port": 6333,
        "grpc_port": 6334,
        "api_key": "secret",
        "location": "us-east",
        "timeout": 10,
    }

