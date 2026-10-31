"""
Sparse vectors and hybrid retrieval in the NanoVectorDB adapter.

Retrieval is meant to behave like :class:`QdrantVectorDBStorage`, so besides
hand-computed cases the suite checks rankings and fused scores against
qdrant-client's own in-memory engine.
"""

import base64
import json
import math
import random

import numpy as np
import pytest

from ragu.storage.types import Point, SparseEmbedding
from ragu.storage.vdb_storage_adapters.nano_vdb import DenseVectorDB, NanoVectorDBStorage


def _storage(tmp_path, sparse_type="splade", **kwargs) -> NanoVectorDBStorage:
    kwargs.setdefault("score_threshold", -1.0)
    return NanoVectorDBStorage(
        embedding_dim=3,
        storage_folder=str(tmp_path),
        filename="vdb.json",
        sparse_type=sparse_type,
        **kwargs,
    )


def _point(record_id, dense, sparse=None, **metadata) -> Point:
    return Point(
        id=record_id,
        dense_embedding=np.array(dense, dtype=float),
        sparse_embedding=None if sparse is None else SparseEmbedding(
            indices=list(sparse), values=list(sparse.values())
        ),
        metadata=metadata,
    )


def _legacy_file(path, ids):
    """Write a dense-only index in the layout used before sparse support."""
    matrix = np.eye(3, dtype=np.float32)[: len(ids)]
    path.write_text(
        json.dumps({
            "embedding_dim": 3,
            "metric": "cosine",
            "data": [{"__id__": record_id, "tag": record_id} for record_id in ids],
            "matrix": base64.b64encode(matrix.tobytes()).decode(),
        }),
        encoding="utf-8",
    )


# Dense ranking for query [1, 0, 0]: dense-best > both > sparse-best.
# Sparse ranking for query {7: 1.0}: sparse-best > both (dense-best shares no index).
HYBRID_POINTS = [
    _point("dense-best", [1.0, 0.0, 0.0], {1: 1.0}),
    _point("both", [0.8, 0.6, 0.0], {7: 1.0}),
    _point("sparse-best", [0.0, 1.0, 0.0], {7: 2.0}),
]
HYBRID_QUERY = _point("q", [1.0, 0.0, 0.0], {7: 1.0})


# --------------------------------------------------------------------------- #
# Backward compatibility
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_legacy_index_without_sparse_section_loads(tmp_path):
    _legacy_file(tmp_path / "vdb.json", ["a", "b"])

    storage = _storage(tmp_path, sparse_type="bm25")

    assert await storage.get_all_ids() == ["a", "b"]
    points = await storage.get_points_by_ids(["a", "b"])
    assert all(point is not None and point.sparse_embedding is None for point in points)

    hits = (await storage.query([_point("q", [0.0, 1.0, 0.0])], top_k=1))[0]
    assert [hit.id for hit in hits] == ["b"]


@pytest.mark.asyncio
async def test_legacy_index_answers_hybrid_queries_with_dense_branch(tmp_path):
    """Points stored without sparse vectors still take part through the dense ranking."""
    _legacy_file(tmp_path / "vdb.json", ["a", "b"])
    storage = _storage(tmp_path, sparse_type="splade")

    hits = (await storage.query([_point("q", [1.0, 0.0, 0.0], {5: 1.0})], top_k=2))[0]

    assert [hit.id for hit in hits] == ["a", "b"]
    assert [hit.distance for hit in hits] == pytest.approx([1 / 2, 1 / 3])


@pytest.mark.asyncio
async def test_legacy_index_gains_sparse_vectors_after_upsert(tmp_path):
    _legacy_file(tmp_path / "vdb.json", ["a", "b"])
    storage = _storage(tmp_path, sparse_type="splade")

    await storage.upsert([_point("c", [0.0, 0.0, 1.0], {4: 0.5})])
    await storage.index_done_callback()
    reloaded = _storage(tmp_path, sparse_type="splade")

    a, c = await reloaded.get_points_by_ids(["a", "c"])
    assert a is not None and a.sparse_embedding is None
    assert c is not None and c.sparse_embedding == SparseEmbedding(indices=[4], values=[0.5])


def test_dense_only_store_writes_the_historical_layout(tmp_path):
    db = DenseVectorDB(embedding_dim=3, storage_file=str(tmp_path / "dense.json"))
    db.upsert([{"__id__": "a", "__vector__": [1.0, 0.0, 0.0]}])
    db.save()

    stored = json.loads((tmp_path / "dense.json").read_text(encoding="utf-8"))

    assert set(stored) == {"embedding_dim", "metric", "data", "matrix"}


@pytest.mark.asyncio
async def test_sparse_embeddings_are_ignored_without_sparse_type(tmp_path):
    """Dense-only stores keep their previous behaviour: warn and drop sparse payloads."""
    storage = _storage(tmp_path, sparse_type=None)

    await storage.upsert([_point("a", [1.0, 0.0, 0.0], {1: 1.0})])
    hits = (await storage.query([_point("q", [1.0, 0.0, 0.0], {1: 1.0})], top_k=1))[0]
    point = (await storage.get_points_by_ids(["a"]))[0]

    assert hits[0].id == "a"
    assert hits[0].distance == pytest.approx(1.0)  # dense score, not a fused one
    assert point is not None and point.sparse_embedding is None


@pytest.mark.asyncio
async def test_hybrid_index_opened_without_sparse_type_keeps_sparse_vectors(tmp_path):
    storage = _storage(tmp_path, sparse_type="bm42")
    await storage.upsert(HYBRID_POINTS)
    await storage.index_done_callback()

    dense_only = _storage(tmp_path, sparse_type=None)
    hits = (await dense_only.query([HYBRID_QUERY], top_k=3))[0]
    await dense_only.index_done_callback()

    assert [hit.id for hit in hits] == ["dense-best", "both", "sparse-best"]
    reopened = _storage(tmp_path, sparse_type="bm42")
    point = (await reopened.get_points_by_ids(["both"]))[0]
    assert point is not None and point.sparse_embedding == SparseEmbedding(indices=[7], values=[1.0])


@pytest.mark.asyncio
async def test_sparse_type_mismatch_with_persisted_index_raises(tmp_path):
    storage = _storage(tmp_path, sparse_type="bm25")
    await storage.upsert(HYBRID_POINTS)
    await storage.index_done_callback()

    with pytest.raises(ValueError, match="Sparse type mismatch"):
        _storage(tmp_path, sparse_type="splade")


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_sparse_round_trip_through_save_and_load(tmp_path):
    storage = _storage(tmp_path)
    await storage.upsert([
        _point("a", [1.0, 0.0, 0.0], {9: 0.5, 7: 1.0}, tag="A"),
        _point("b", [0.0, 1.0, 0.0], {3: 0.25}, tag="B"),
    ])
    await storage.index_done_callback()

    reloaded = _storage(tmp_path)
    a, b = await reloaded.get_points_by_ids(["a", "b"])

    # Indices come back sorted, with their values kept alongside.
    assert a is not None and a.sparse_embedding == SparseEmbedding(indices=[7, 9], values=[1.0, 0.5])
    assert b is not None and b.sparse_embedding == SparseEmbedding(indices=[3], values=[0.25])
    assert a.metadata == {"tag": "A"}
    assert await reloaded.get_payloads_by_ids(["a"]) == [{"__id__": "a", "tag": "A"}]


@pytest.mark.asyncio
async def test_upsert_replaces_and_delete_keeps_sparse_vectors_aligned(tmp_path):
    storage = _storage(tmp_path)
    await storage.upsert([
        _point("a", [1.0, 0.0, 0.0], {1: 1.0}),
        _point("b", [0.0, 1.0, 0.0], {2: 1.0}),
        _point("c", [0.0, 0.0, 1.0], {3: 1.0}),
    ])
    await storage.upsert([_point("b", [0.0, 1.0, 0.0], {5: 1.0})])
    await storage.delete(["a"])

    b, c = await storage.get_points_by_ids(["b", "c"])
    assert b is not None and b.sparse_embedding == SparseEmbedding(indices=[5], values=[1.0])
    assert c is not None and c.sparse_embedding == SparseEmbedding(indices=[3], values=[1.0])

    hits = (await storage.query([_point("q", [0.0, 0.0, 1.0], {5: 1.0})], top_k=3))[0]
    assert {hit.id for hit in hits} == {"b", "c"}


@pytest.mark.asyncio
async def test_mixed_sparse_batch_is_rejected(tmp_path):
    storage = _storage(tmp_path)

    with pytest.raises(ValueError, match="must either have sparse embeddings"):
        await storage.upsert([
            _point("a", [1.0, 0.0, 0.0], {1: 1.0}),
            _point("b", [0.0, 1.0, 0.0]),
        ])


@pytest.mark.asyncio
async def test_duplicate_sparse_indices_are_rejected(tmp_path):
    storage = _storage(tmp_path)

    with pytest.raises(ValueError, match="duplicate indices"):
        await storage.upsert([
            Point(
                id="a",
                dense_embedding=np.array([1.0, 0.0, 0.0]),
                sparse_embedding=SparseEmbedding(indices=[1, 1], values=[0.5, 0.5]),
            )
        ])


def test_unsupported_sparse_type_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unsupported sparse type"):
        _storage(tmp_path, sparse_type="tfidf")


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_hybrid_query_fuses_rankings_with_rrf_by_default(tmp_path):
    storage = _storage(tmp_path)
    await storage.upsert(HYBRID_POINTS)

    hits = (await storage.query([HYBRID_QUERY], top_k=3))[0]

    # dense ranks: dense-best 0, both 1, sparse-best 2; sparse ranks: sparse-best 0, both 1.
    # RRF contributes 1 / (rank + 2) per ranking.
    assert [hit.id for hit in hits] == ["sparse-best", "both", "dense-best"]
    assert [hit.distance for hit in hits] == pytest.approx([1 / 4 + 1 / 2, 1 / 3 + 1 / 3, 1 / 2])
    assert hits[0].metadata == {}


@pytest.mark.asyncio
async def test_dense_only_query_on_hybrid_store_uses_dense_scores(tmp_path):
    storage = _storage(tmp_path)
    await storage.upsert(HYBRID_POINTS)

    hits = (await storage.query([_point("q", [1.0, 0.0, 0.0])], top_k=3))[0]

    assert [hit.id for hit in hits] == ["dense-best", "both", "sparse-best"]
    assert [hit.distance for hit in hits] == pytest.approx([1.0, 0.8, 0.0], abs=1e-6)


@pytest.mark.asyncio
async def test_batch_mixes_hybrid_and_dense_only_queries(tmp_path):
    storage = _storage(tmp_path)
    await storage.upsert(HYBRID_POINTS)

    hybrid, dense = await storage.query([HYBRID_QUERY, _point("q", [1.0, 0.0, 0.0])], top_k=3)

    assert [hit.id for hit in hybrid] == ["sparse-best", "both", "dense-best"]
    assert [hit.id for hit in dense] == ["dense-best", "both", "sparse-best"]


def _dbsf_expected():
    def normalize(scores):
        mean = sum(scores) / len(scores)
        std = math.sqrt(sum((s - mean) ** 2 for s in scores) / (len(scores) - 1))
        low, high = mean - 3 * std, mean + 3 * std
        return [(s - low) / (high - low) for s in scores]

    dense = dict(zip(["dense-best", "both", "sparse-best"], normalize([1.0, 0.8, 0.0])))
    sparse = dict(zip(["sparse-best", "both"], normalize([2.0, 1.0])))
    fused = {key: dense[key] + sparse.get(key, 0.0) for key in dense}
    return sorted(fused.items(), key=lambda item: item[1], reverse=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("fusion", ["dbsf", "DBSF"])
async def test_hybrid_query_supports_dbsf(tmp_path, fusion):
    storage = _storage(tmp_path)
    await storage.upsert(HYBRID_POINTS)

    hits = (await storage.query([HYBRID_QUERY], top_k=3, hybrid_query_type=fusion))[0]

    expected = _dbsf_expected()
    assert [hit.id for hit in hits] == [key for key, _ in expected] == ["both", "sparse-best", "dense-best"]
    assert [hit.distance for hit in hits] == pytest.approx([score for _, score in expected], abs=1e-6)


@pytest.mark.asyncio
async def test_hybrid_query_accepts_qdrant_fusion_query(tmp_path):
    """The ``hybrid_query_type`` object used with the Qdrant adapter works unchanged."""
    models = pytest.importorskip("qdrant_client.models")
    storage = _storage(tmp_path)
    await storage.upsert(HYBRID_POINTS)

    rrf = (await storage.query([HYBRID_QUERY], top_k=3,
                               hybrid_query_type=models.FusionQuery(fusion=models.Fusion.RRF)))[0]
    dbsf = (await storage.query([HYBRID_QUERY], top_k=3,
                                hybrid_query_type=models.FusionQuery(fusion=models.Fusion.DBSF)))[0]

    assert [hit.id for hit in rrf] == ["sparse-best", "both", "dense-best"]
    assert [hit.id for hit in dbsf] == ["both", "sparse-best", "dense-best"]


@pytest.mark.asyncio
async def test_unsupported_fusion_is_rejected(tmp_path):
    storage = _storage(tmp_path)

    with pytest.raises(ValueError, match="Unsupported hybrid fusion"):
        await storage.query([HYBRID_QUERY], top_k=3, hybrid_query_type="max")


@pytest.mark.asyncio
async def test_hybrid_query_fetches_top_k_candidates_per_branch(tmp_path):
    """As with Qdrant prefetch, each branch contributes only its own top ``top_k``."""
    storage = _storage(tmp_path)
    await storage.upsert(HYBRID_POINTS)

    hits = (await storage.query([HYBRID_QUERY], top_k=2))[0]

    # dense candidates: dense-best, both; sparse candidates: sparse-best, both.
    # sparse-best loses its dense contribution, so "both" now leads; the 1/2 tie
    # between the two single-branch hits keeps the dense one first.
    assert [hit.id for hit in hits] == ["both", "dense-best"]
    assert [hit.distance for hit in hits] == pytest.approx([1 / 3 + 1 / 3, 1 / 2])


@pytest.mark.asyncio
async def test_score_threshold_applies_to_dense_branch_only(tmp_path):
    storage = _storage(tmp_path, score_threshold=0.5)
    await storage.upsert(HYBRID_POINTS)

    hits = (await storage.query([HYBRID_QUERY], top_k=3))[0]

    # sparse-best falls below the dense threshold but is still found by the sparse branch.
    assert [hit.id for hit in hits] == ["both", "dense-best", "sparse-best"]
    assert [hit.distance for hit in hits] == pytest.approx([1 / 3 + 1 / 3, 1 / 2, 1 / 2])


def _idf_db(tmp_path, sparse_type):
    db = DenseVectorDB(embedding_dim=3, storage_file=str(tmp_path / f"{sparse_type}.json"),
                       sparse_type=sparse_type)
    common = SparseEmbedding(indices=[1], values=[1.0])
    db.upsert([
        {"__id__": "common-1", "__vector__": [1.0, 0.0, 0.0], "__sparse__": common},
        {"__id__": "common-2", "__vector__": [1.0, 0.0, 0.0], "__sparse__": common},
        {"__id__": "common-3", "__vector__": [1.0, 0.0, 0.0], "__sparse__": common},
        {"__id__": "rare", "__vector__": [1.0, 0.0, 0.0],
         "__sparse__": SparseEmbedding(indices=[2], values=[0.6])},
    ])
    return db


def test_bm25_weights_sparse_queries_by_idf(tmp_path):
    query = SparseEmbedding(indices=[1, 2], values=[1.0, 1.0])

    plain = _idf_db(tmp_path, "splade").query_sparse([query], top_k=4)[0]
    bm25 = _idf_db(tmp_path, "bm25").query_sparse([query], top_k=4)[0]

    assert plain[-1][0] == "rare"
    assert plain[-1][1] == pytest.approx(0.6)

    idf = lambda df: math.log((4 - df + 0.5) / (df + 0.5) + 1)  # noqa: E731
    assert bm25[0][0] == "rare"
    assert bm25[0][1] == pytest.approx(0.6 * idf(1))
    assert bm25[1][1] == pytest.approx(idf(3))


def test_sparse_query_skips_records_without_shared_indices(tmp_path):
    db = _idf_db(tmp_path, "splade")

    hits = db.query_sparse([SparseEmbedding(indices=[2], values=[1.0])], top_k=4)[0]

    assert [hit[0] for hit in hits] == ["rare"]


# --------------------------------------------------------------------------- #
# Parity with Qdrant
# --------------------------------------------------------------------------- #


def _random_sparse(rng, vocab=30, max_terms=5):
    indices = sorted(rng.sample(range(vocab), rng.randint(1, max_terms)))
    return SparseEmbedding(indices=indices, values=[round(rng.uniform(0.1, 2.0), 3) for _ in indices])


@pytest.mark.asyncio
@pytest.mark.parametrize("sparse_type", ["bm25", "splade"])
@pytest.mark.parametrize("fusion", ["rrf", "dbsf"])
async def test_hybrid_results_match_qdrant(tmp_path, sparse_type, fusion):
    """Same ids, order and fused scores as a Qdrant hybrid query over the same data."""
    qdrant_client = pytest.importorskip("qdrant_client")
    models = qdrant_client.models

    rng = random.Random(7)
    dim, top_k = 8, 10
    points = [
        Point(
            id=f"p{i}",
            dense_embedding=np.array([rng.uniform(-1, 1) for _ in range(dim)]),
            sparse_embedding=_random_sparse(rng),
        )
        for i in range(40)
    ]

    nano = NanoVectorDBStorage(
        embedding_dim=dim,
        storage_folder=str(tmp_path),
        filename="vdb.json",
        score_threshold=-1.0,  # Qdrant applies no score threshold
        sparse_type=sparse_type,
    )
    await nano.upsert(points)

    client = qdrant_client.AsyncQdrantClient(location=":memory:")
    await client.create_collection(
        "parity",
        vectors_config={"dense": models.VectorParams(size=dim, distance=models.Distance.COSINE)},
        sparse_vectors_config={
            sparse_type: models.SparseVectorParams(
                modifier=models.Modifier.IDF if sparse_type == "bm25" else None
            )
        },
    )
    await client.upsert(
        "parity",
        points=[
            models.PointStruct(
                id=i,
                vector={
                    "dense": point.dense_embedding.tolist(),
                    sparse_type: models.SparseVector(
                        indices=point.sparse_embedding.indices,
                        values=point.sparse_embedding.values,
                    ),
                },
                payload={"ragu_id": point.id},
            )
            for i, point in enumerate(points)
        ],
    )

    try:
        for _ in range(5):
            dense = np.array([rng.uniform(-1, 1) for _ in range(dim)])
            sparse = _random_sparse(rng)

            ours = (await nano.query(
                [Point(dense_embedding=dense, sparse_embedding=sparse)],
                top_k=top_k,
                hybrid_query_type=fusion,
            ))[0]
            theirs = (await client.query_points(
                "parity",
                prefetch=[
                    models.Prefetch(query=dense.tolist(), using="dense", limit=top_k),
                    models.Prefetch(
                        query=models.SparseVector(indices=sparse.indices, values=sparse.values),
                        using=sparse_type,
                        limit=top_k,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion(fusion)),
                limit=top_k,
                with_payload=True,
            )).points

            assert [hit.distance for hit in ours] == pytest.approx([p.score for p in theirs], abs=1e-5)
            assert sorted(
                zip([round(h.distance, 5) for h in ours], [h.id for h in ours])
            ) == sorted(
                zip([round(p.score, 5) for p in theirs], [p.payload["ragu_id"] for p in theirs])
            )
    finally:
        await client.close()
