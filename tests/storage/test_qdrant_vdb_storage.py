from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ragu.storage.types import Point

from tests.storage.qdrant_testkit import FakeAsyncQdrantClient, load_qdrant_storage


@pytest.mark.asyncio
async def test_upsert_accepts_empty_sparse_list(monkeypatch, tmp_path):
    QdrantVectorDBStorage = load_qdrant_storage(monkeypatch)
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
    QdrantVectorDBStorage = load_qdrant_storage(monkeypatch)
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


@pytest.mark.asyncio
async def test_cosine_collection_normalizes_stored_vectors(monkeypatch, tmp_path):
    """
    Real Qdrant normalizes dense vectors on upload for cosine collections and
    returns them normalized, so the fake must do the same. Verified against a
    live Qdrant 1.17.1: writing [3, 4, 0] reads back as [0.6, 0.8, 0].
    """
    QdrantVectorDBStorage = load_qdrant_storage(monkeypatch)
    vdb = QdrantVectorDBStorage(embedding_dim=3, filename=str(tmp_path / "vdb.json"))

    await vdb.upsert([Point(id="ent-1", dense_embedding=np.array([3.0, 4.0, 0.0]), metadata={})])

    point = (await vdb.get_points_by_ids(["ent-1"]))[0]
    assert point is not None
    assert np.allclose(np.asarray(point.dense_embedding, dtype=float), [0.6, 0.8, 0.0])
