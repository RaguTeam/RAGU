import base64
import json

import numpy as np
import pytest

from ragu.storage.types import Point
from ragu.storage.vdb_storage_adapters.nano_vdb import DenseVectorDB, NanoVectorDBStorage


def _make_db(tmp_path, dim=3, metric="cosine"):
    return DenseVectorDB(
        embedding_dim=dim,
        storage_file=str(tmp_path / "dense.json"),
        metric=metric,
    )


def test_upsert_and_batch_query_alignment(tmp_path):
    db = _make_db(tmp_path)
    db.upsert([
        {"__id__": "a", "__vector__": np.array([1.0, 0.0, 0.0]), "tag": "A"},
        {"__id__": "b", "__vector__": np.array([0.0, 1.0, 0.0]), "tag": "B"},
    ])

    batched = db.query(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), top_k=1)

    assert [hits[0][0] for hits in batched] == ["a", "b"]
    # Metadata is returned without internal keys.
    assert batched[0][0][2] == {"tag": "A"}


def test_batch_query_matches_per_query(tmp_path):
    db = _make_db(tmp_path)
    db.upsert([
        {"__id__": str(i), "__vector__": np.random.rand(3)} for i in range(20)
    ])
    queries = np.random.rand(5, 3)

    batched = db.query(queries, top_k=4)
    per_query = [db.query(q, top_k=4)[0] for q in queries]

    assert [[h[0] for h in hits] for hits in batched] == [[h[0] for h in hits] for hits in per_query]


def test_threshold_filters_low_scores(tmp_path):
    db = _make_db(tmp_path)
    db.upsert([
        {"__id__": "same", "__vector__": np.array([1.0, 0.0, 0.0])},
        {"__id__": "orthogonal", "__vector__": np.array([0.0, 1.0, 0.0])},
    ])

    hits = db.query(np.array([[1.0, 0.0, 0.0]]), top_k=10, threshold=0.5)[0]

    assert [h[0] for h in hits] == ["same"]


def test_delete_removes_rows(tmp_path):
    db = _make_db(tmp_path)
    db.upsert([
        {"__id__": "a", "__vector__": np.array([1.0, 0.0, 0.0])},
        {"__id__": "b", "__vector__": np.array([0.0, 1.0, 0.0])},
    ])
    db.delete(["a"])

    assert db.all_ids() == ["b"]
    assert len(db) == 1


def test_upsert_updates_existing(tmp_path):
    db = _make_db(tmp_path)
    db.upsert([{"__id__": "a", "__vector__": np.array([1.0, 0.0, 0.0]), "tag": "old"}])
    db.upsert([{"__id__": "a", "__vector__": np.array([0.0, 1.0, 0.0]), "tag": "new"}])

    assert len(db) == 1
    rows = db.get_rows(["a"])
    assert rows[0]["tag"] == "new"


def test_empty_db_returns_empty_per_query(tmp_path):
    db = _make_db(tmp_path)
    assert db.query(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), top_k=5) == [[], []]


def test_query_rejects_non_positive_top_k(tmp_path):
    db = _make_db(tmp_path)
    db.upsert([{"__id__": "a", "__vector__": np.array([1.0, 0.0, 0.0])}])

    with pytest.raises(ValueError, match="top_k must be positive"):
        db.query(np.array([[1.0, 0.0, 0.0]]), top_k=0)


def test_upsert_rejects_wrong_embedding_dimension(tmp_path):
    db = _make_db(tmp_path, dim=3)

    with pytest.raises(ValueError, match="Record 'wrong' embedding dim mismatch"):
        db.upsert([{"__id__": "wrong", "__vector__": np.array([1.0, 0.0])}])


def test_query_rejects_wrong_embedding_dimension(tmp_path):
    db = _make_db(tmp_path, dim=3)
    db.upsert([{"__id__": "a", "__vector__": np.array([1.0, 0.0, 0.0])}])

    with pytest.raises(ValueError, match="Query embedding dim mismatch"):
        db.query(np.array([[1.0, 0.0]]), top_k=1)


def test_cosine_store_normalizes_written_vectors(tmp_path):
    db = _make_db(tmp_path)
    db.upsert([{"__id__": "a", "__vector__": np.array([2.0, 0.0, 0.0]), "tag": "A"}])

    point = db.get_points(["a"])[0]

    assert point is not None
    assert point[0] == "a"
    assert point[1].tolist() == [1.0, 0.0, 0.0]
    assert point[2] == {"tag": "A"}
    assert db.query(np.array([[1.0, 0.0, 0.0]]), top_k=1)[0][0][0] == "a"


def test_dot_store_keeps_vectors_verbatim(tmp_path):
    db = _make_db(tmp_path, metric="dot")
    db.upsert([{"__id__": "a", "__vector__": np.array([2.0, 0.0, 0.0]), "tag": "A"}])

    point = db.get_points(["a"])[0]

    assert point is not None
    assert point[1].tolist() == [2.0, 0.0, 0.0]


def test_dot_scores_are_magnitude_sensitive(tmp_path):
    """Under dot product a longer vector outranks a shorter collinear one."""
    db = _make_db(tmp_path, metric="dot")
    db.upsert([
        {"__id__": "short", "__vector__": np.array([1.0, 0.0, 0.0])},
        {"__id__": "long", "__vector__": np.array([5.0, 0.0, 0.0])},
    ])

    hits = db.query(np.array([[1.0, 0.0, 0.0]]), top_k=2)[0]

    assert [hit[0] for hit in hits] == ["long", "short"]
    assert hits[0][1] == pytest.approx(5.0)
    assert hits[1][1] == pytest.approx(1.0)


def test_cosine_scores_ignore_magnitude(tmp_path):
    """The same vectors under cosine are indistinguishable by direction alone."""
    db = _make_db(tmp_path)
    db.upsert([
        {"__id__": "short", "__vector__": np.array([1.0, 0.0, 0.0])},
        {"__id__": "long", "__vector__": np.array([5.0, 0.0, 0.0])},
    ])

    hits = db.query(np.array([[1.0, 0.0, 0.0]]), top_k=2)[0]

    assert {hit[0] for hit in hits} == {"short", "long"}
    assert all(hit[1] == pytest.approx(1.0) for hit in hits)


def test_unsupported_metric_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unsupported distance metric"):
        _make_db(tmp_path, metric="euclid")


def test_metric_mismatch_with_persisted_file_raises(tmp_path):
    """Switching metric on an existing index must fail loudly, not rescore silently."""
    db = _make_db(tmp_path, metric="cosine")
    db.upsert([{"__id__": "a", "__vector__": np.array([2.0, 0.0, 0.0])}])
    db.save()

    with pytest.raises(ValueError, match="Distance metric mismatch"):
        _make_db(tmp_path, metric="dot")


def test_dot_metric_survives_save_load_round_trip(tmp_path):
    db = _make_db(tmp_path, metric="dot")
    db.upsert([{"__id__": "a", "__vector__": np.array([3.0, 0.0, 0.0]), "tag": "A"}])
    db.save()

    reloaded = _make_db(tmp_path, metric="dot")

    point = reloaded.get_points(["a"])[0]
    assert point is not None
    assert point[1].tolist() == [3.0, 0.0, 0.0]


def test_existing_record_can_be_updated_after_load(tmp_path):
    """Regression: decoded matrices were read-only, breaking incremental indexing."""
    db = _make_db(tmp_path)
    db.upsert([{"__id__": "a", "__vector__": np.array([1.0, 0.0, 0.0]), "tag": "v1"}])
    db.save()

    reloaded = _make_db(tmp_path)
    reloaded.upsert([{"__id__": "a", "__vector__": np.array([0.0, 1.0, 0.0]), "tag": "v2"}])

    point = reloaded.get_points(["a"])[0]
    assert point is not None
    assert point[1].tolist() == [0.0, 1.0, 0.0]
    assert point[2] == {"tag": "v2"}
    assert len(reloaded) == 1


def test_save_load_round_trip(tmp_path):
    db = _make_db(tmp_path)
    db.upsert([
        {"__id__": "a", "__vector__": np.array([1.0, 0.0, 0.0]), "tag": "A"},
        {"__id__": "b", "__vector__": np.array([0.0, 1.0, 0.0]), "tag": "B"},
    ])
    db.save()

    reloaded = _make_db(tmp_path)
    assert reloaded.all_ids() == ["a", "b"]
    hits = reloaded.query(np.array([[1.0, 0.0, 0.0]]), top_k=1)[0]
    assert hits[0][0] == "a"


def test_save_creates_parent_directory(tmp_path):
    path = tmp_path / "missing" / "nested" / "dense.json"
    db = DenseVectorDB(embedding_dim=3, storage_file=str(path))
    db.upsert([{"__id__": "a", "__vector__": np.array([1.0, 0.0, 0.0])}])

    db.save()

    assert path.exists()


def test_loads_nano_vectordb_on_disk_format(tmp_path):
    """A file written in the historical NanoVectorDB layout must load as-is."""
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    storage = {
        "embedding_dim": 2,
        "data": [{"__id__": "a", "tag": "A"}, {"__id__": "b", "tag": "B"}],
        "matrix": base64.b64encode(matrix.tobytes()).decode(),
    }
    path = tmp_path / "dense.json"
    path.write_text(json.dumps(storage), encoding="utf-8")

    db = DenseVectorDB(embedding_dim=2, storage_file=str(path))

    assert db.all_ids() == ["a", "b"]
    hits = db.query(np.array([[0.0, 1.0]]), top_k=1)[0]
    assert hits[0][0] == "b"
    assert hits[0][2] == {"tag": "B"}


def test_file_without_metric_key_loads_as_cosine(tmp_path):
    """Indices written before metrics were configurable always held cosine vectors."""
    matrix = np.array([[1.0, 0.0]], dtype=np.float32)
    storage = {
        "embedding_dim": 2,
        "data": [{"__id__": "a", "tag": "A"}],
        "matrix": base64.b64encode(matrix.tobytes()).decode(),
    }
    path = tmp_path / "dense.json"
    path.write_text(json.dumps(storage), encoding="utf-8")

    db = DenseVectorDB(embedding_dim=2, storage_file=str(path))

    assert db.metric == "cosine"
    assert db.all_ids() == ["a"]

    with pytest.raises(ValueError, match="Distance metric mismatch"):
        DenseVectorDB(embedding_dim=2, storage_file=str(path), metric="dot")


def test_metric_is_persisted(tmp_path):
    db = _make_db(tmp_path, dim=2, metric="dot")
    db.upsert([{"__id__": "a", "__vector__": np.array([1.0, 0.0])}])
    db.save()

    stored = json.loads((tmp_path / "dense.json").read_text(encoding="utf-8"))
    assert stored["metric"] == "dot"


def test_embedding_dim_mismatch_raises(tmp_path):
    path = tmp_path / "dense.json"
    path.write_text(json.dumps({"embedding_dim": 4, "data": [], "matrix": ""}), encoding="utf-8")

    with pytest.raises(ValueError):
        DenseVectorDB(embedding_dim=3, storage_file=str(path))


class TestNanoVectorDBStorageMetric:
    """Metric wiring and threshold defaults on the storage adapter."""

    def _storage(self, tmp_path, **kwargs):
        return NanoVectorDBStorage(
            embedding_dim=3,
            storage_folder=str(tmp_path),
            filename="vdb.json",
            **kwargs,
        )

    def test_defaults_to_cosine_with_cosine_threshold(self, tmp_path):
        storage = self._storage(tmp_path)

        assert storage.metric == "cosine"
        assert storage.score_threshold == 0.2

    def test_dot_metric_applies_no_default_threshold(self, tmp_path):
        """Dot products are unbounded, so a fixed cut-off would be arbitrary."""
        storage = self._storage(tmp_path, metric="dot")

        assert storage.metric == "dot"
        assert storage.score_threshold is None
        assert storage._client.normalizes_vectors is False

    def test_explicit_threshold_overrides_metric_default(self, tmp_path):
        storage = self._storage(tmp_path, metric="dot", score_threshold=1.5)

        assert storage.score_threshold == 1.5

    def test_deprecated_cosine_threshold_still_accepted(self, tmp_path):
        storage = self._storage(tmp_path, cosine_threshold=0.0)

        assert storage.score_threshold == 0.0

    def test_conflicting_threshold_arguments_rejected(self, tmp_path):
        with pytest.raises(TypeError, match="deprecated alias"):
            self._storage(tmp_path, score_threshold=0.1, cosine_threshold=0.2)

    @pytest.mark.asyncio
    async def test_dot_metric_ranks_by_magnitude_end_to_end(self, tmp_path):
        storage = self._storage(tmp_path, metric="dot")
        await storage.upsert([
            Point(id="short", dense_embedding=np.array([1.0, 0.0, 0.0]), metadata={}),
            Point(id="long", dense_embedding=np.array([5.0, 0.0, 0.0]), metadata={}),
        ])

        hits = (await storage.query(
            [Point(dense_embedding=np.array([1.0, 0.0, 0.0]))], top_k=2
        ))[0]

        assert [hit.id for hit in hits] == ["long", "short"]
        assert hits[0].distance == pytest.approx(5.0)


def test_interrupted_save_leaves_the_previous_index_intact(tmp_path):
    """
    A vector index cannot be repaired, only rebuilt, so a truncated file is a
    total loss. The write goes through a temporary neighbour for that reason.
    """
    from unittest.mock import patch

    db = _make_db(tmp_path)
    db.upsert([{"__id__": f"id-{i}", "__vector__": np.eye(3, dtype=np.float32)[i % 3]}
               for i in range(3)])
    db.save()
    path = tmp_path / "dense.json"
    before = path.read_bytes()

    db.upsert([{"__id__": "id-later", "__vector__": np.array([1.0, 1.0, 1.0])}])
    with patch("json.dump", side_effect=OSError("disk full")):
        with pytest.raises(OSError):
            db.save()

    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))
    assert len(_make_db(tmp_path)) == 3
