from __future__ import annotations

import numpy as np
import pytest

from ragu.storage.types import EmbeddingHit, Point, SparseEmbedding


def _point(record_id: str, dense: list[float], **metadata) -> Point:
    return Point(
        id=record_id,
        dense_embedding=np.array(dense),
        metadata=metadata,
    )


@pytest.mark.asyncio
async def test_vdb_contract_upsert_and_query_round_trip(vdb_storage):
    await vdb_storage.upsert([
        _point("id-alpha", [1.0, 0.0, 0.0], tag="A"),
        _point("id-beta", [0.0, 1.0, 0.0], tag="B"),
    ])

    results = (await vdb_storage.query([Point(dense_embedding=np.array([1.0, 0.0, 0.0]))], top_k=10))[0]

    assert len(results) >= 1
    assert isinstance(results[0], EmbeddingHit)
    assert results[0].id == "id-alpha"
    assert results[0].metadata["tag"] == "A"


@pytest.mark.asyncio
async def test_vdb_contract_get_all_ids(vdb_storage):
    await vdb_storage.upsert([
        _point("id-alpha", [1.0, 0.0, 0.0], tag="A"),
        _point("id-beta", [0.0, 1.0, 0.0], tag="B"),
    ])

    ids = await vdb_storage.get_all_ids()

    assert set(ids) == {"id-alpha", "id-beta"}


@pytest.mark.asyncio
async def test_vdb_contract_get_payloads_by_ids_preserves_input_order(vdb_storage):
    await vdb_storage.upsert([
        _point("id-alpha", [1.0, 0.0, 0.0], tag="A"),
        _point("id-beta", [0.0, 1.0, 0.0], tag="B"),
    ])

    payloads = await vdb_storage.get_payloads_by_ids(["id-beta", "missing", "id-alpha"])

    assert payloads[0] is not None
    assert payloads[1] is None
    assert payloads[2] is not None
    assert payloads[0]["__id__"] == "id-beta"
    assert payloads[2]["__id__"] == "id-alpha"


@pytest.mark.asyncio
async def test_vdb_contract_get_points_by_ids_preserves_input_order(vdb_storage):
    await vdb_storage.upsert([
        _point("id-alpha", [1.0, 0.0, 0.0], tag="A"),
        _point("id-beta", [0.0, 1.0, 0.0], tag="B"),
    ])

    points = await vdb_storage.get_points_by_ids(["id-beta", "missing", "id-alpha"])

    assert points[0] is not None
    assert points[1] is None
    assert points[2] is not None
    assert points[0].id == "id-beta"
    assert list(points[0].dense_embedding) == [0.0, 1.0, 0.0]
    assert points[2].id == "id-alpha"
    assert points[2].metadata["tag"] == "A"


@pytest.mark.asyncio
async def test_vdb_contract_get_points_by_ids_handles_strict_subset(vdb_storage):
    """Fetching a subset that does not start at the first stored record."""
    await vdb_storage.upsert([
        _point("id-alpha", [1.0, 0.0, 0.0], tag="A"),
        _point("id-beta", [0.0, 1.0, 0.0], tag="B"),
        _point("id-gamma", [0.0, 0.0, 1.0], tag="C"),
    ])

    points = await vdb_storage.get_points_by_ids(["id-gamma"])

    assert points[0] is not None
    assert points[0].id == "id-gamma"
    assert points[0].metadata["tag"] == "C"
    assert list(points[0].dense_embedding) == [0.0, 0.0, 1.0]


@pytest.mark.asyncio
async def test_vdb_contract_returned_vector_matches_written_direction(vdb_storage):
    """
    Backends may normalize on write, so vectors round-trip only up to a
    positive scale factor. Direction, however, must be preserved exactly.
    """
    await vdb_storage.upsert([_point("id-scaled", [3.0, 4.0, 0.0], tag="S")])

    point = (await vdb_storage.get_points_by_ids(["id-scaled"]))[0]

    assert point is not None
    returned = np.asarray(point.dense_embedding, dtype=float)
    written = np.array([3.0, 4.0, 0.0])
    scale = np.linalg.norm(returned) / np.linalg.norm(written)
    assert scale > 0
    assert np.allclose(returned, written * scale)


@pytest.mark.asyncio
async def test_vdb_contract_delete_existing_and_missing_ids(vdb_storage):
    await vdb_storage.upsert([
        _point("id-alpha", [1.0, 0.0, 0.0], tag="A"),
        _point("id-beta", [0.0, 1.0, 0.0], tag="B"),
    ])

    await vdb_storage.delete([])
    await vdb_storage.delete(["id-alpha", "missing"])

    ids = await vdb_storage.get_all_ids()

    assert "id-alpha" not in ids
    assert "id-beta" in ids


@pytest.mark.asyncio
async def test_vdb_contract_persistence_round_trip(vdb_backend_case, vdb_storage, tmp_path, monkeypatch):
    if not vdb_backend_case.supports_persistence:
        pytest.skip("Backend does not support persistence round-trip")

    await vdb_storage.upsert([
        _point("id-persist", [1.0, 0.0, 0.0], tag="persist"),
    ])
    await vdb_storage.index_done_callback()

    if vdb_backend_case.name.startswith("qdrant"):
        reload_kwargs = {
            "embedding_dim": 3,
            "filename": str(tmp_path / "vdb.json"),
        }
        if vdb_backend_case.supports_sparse:
            reload_kwargs["sparse_type"] = "bm42"
        reloaded = type(vdb_storage)(**reload_kwargs)
    else:
        reloaded = vdb_backend_case.factory(tmp_path, monkeypatch)
    results = (await reloaded.query([Point(dense_embedding=np.array([1.0, 0.0, 0.0]))], top_k=10))[0]

    assert any(result.id == "id-persist" for result in results)


@pytest.mark.asyncio
async def test_vdb_contract_sparse_round_trip(vdb_backend_case, vdb_storage):
    if not vdb_backend_case.supports_sparse:
        pytest.skip("Backend does not support sparse vectors")

    await vdb_storage.upsert([
        Point(
            id="id-hybrid",
            dense_embedding=np.array([1.0, 0.0, 0.0]),
            sparse_embedding=SparseEmbedding(indices=[7, 9], values=[1.0, 0.5]),
            metadata={"tag": "hybrid"},
        ),
        Point(
            id="id-other",
            dense_embedding=np.array([0.7, 0.7, 0.0]),
            sparse_embedding=SparseEmbedding(indices=[3], values=[0.1]),
            metadata={"tag": "other"},
        ),
    ])

    results = (await vdb_storage.query(
        [Point(
            dense_embedding=np.array([1.0, 0.0, 0.0]),
            sparse_embedding=SparseEmbedding(indices=[7], values=[1.0]),
        )],
        top_k=10,
    ))[0]
    points = await vdb_storage.get_points_by_ids(["id-hybrid"])

    assert results[0].id == "id-hybrid"
    assert points[0] is not None
    assert points[0].sparse_embedding is not None
    assert points[0].sparse_embedding.indices == [7, 9]
    assert points[0].sparse_embedding.values == [1.0, 0.5]
