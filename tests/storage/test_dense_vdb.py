import base64
import json

import numpy as np
import pytest

from ragu.storage.vdb_storage_adapters.dense_vdb_core import DenseVectorDB


def _make_db(tmp_path, dim=3):
    return DenseVectorDB(embedding_dim=dim, storage_file=str(tmp_path / "dense.json"))


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


def test_embedding_dim_mismatch_raises(tmp_path):
    path = tmp_path / "dense.json"
    path.write_text(json.dumps({"embedding_dim": 4, "data": [], "matrix": ""}), encoding="utf-8")

    with pytest.raises(ValueError):
        DenseVectorDB(embedding_dim=3, storage_file=str(path))
