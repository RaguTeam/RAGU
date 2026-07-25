from __future__ import annotations

import numpy as np
import pytest

from ragu.storage.types import Point, SparseEmbedding

from tests.storage.qdrant_testkit import (
    FakeAsyncQdrantClient,
    FakeSparseVector,
    FakeSparseVectorParams,
    FakeVectorParams,
    load_qdrant_storage,
)


@pytest.mark.asyncio
async def test_bm42_sparse_mode_creates_idf_index_and_uses_bm42_vector_name(monkeypatch, tmp_path):
    QdrantVectorDBStorage = load_qdrant_storage(monkeypatch)
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
    assert stored_point.vector["bm42"] == FakeSparseVector(indices=[7, 9], values=[1.0, 0.5])


@pytest.mark.asyncio
async def test_bm42_sparse_mode_validates_existing_collection_modifier(monkeypatch, tmp_path):
    QdrantVectorDBStorage = load_qdrant_storage(monkeypatch)
    storage = QdrantVectorDBStorage(
        embedding_dim=3,
        filename=str(tmp_path / "vdb.json"),
        sparse_type="bm42",
    )

    FakeAsyncQdrantClient.registries[str(tmp_path)] = {
        storage.collection_name: {
            "vectors_config": {
                "dense": FakeVectorParams(size=3, distance="cosine"),
            },
            "sparse_vectors_config": {
                "bm42": FakeSparseVectorParams(modifier="idf"),
            },
            "points": {},
        }
    }

    await storage.index_start_callback()


@pytest.mark.asyncio
async def test_remote_qdrant_args_are_explicit_constructor_parameters(monkeypatch, tmp_path):
    QdrantVectorDBStorage = load_qdrant_storage(monkeypatch)
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

