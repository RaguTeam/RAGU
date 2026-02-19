from __future__ import annotations

from typing import Dict, List

import pytest

from ragu.embedder.base_embedder import BaseEmbedder
from ragu.storage.vdb_storage_adapters.nano_vdb import NanoVectorDBStorage
from ragu.utils.ragu_utils import FLOATS


class DummyEmbedder(BaseEmbedder):
    def __init__(self, dim: int, vector_by_text: Dict[str, List[float]]):
        super().__init__(dim=dim)
        self.vector_by_text = vector_by_text

    async def embed(self, texts: list[str]) -> list[list[float]] | FLOATS:
        if isinstance(texts, str):
            texts = [texts]
        return [self.vector_by_text[text] for text in texts]


@pytest.mark.asyncio
async def test_upsert_and_query_returns_expected_fields(tmp_path):
    embedder = DummyEmbedder(
        dim=3,
        vector_by_text={
            "alpha": [1.0, 0.0, 0.0],
            "beta": [0.0, 1.0, 0.0],
            "query-alpha": [1.0, 0.0, 0.0],
        },
    )
    storage_file = tmp_path / "vdb.json"
    vdb = NanoVectorDBStorage(
        embedder=embedder,
        filename=str(storage_file),
        cosine_threshold=0.2,
    )

    await vdb.upsert(
        {
            "id-alpha": {"content": "alpha", "tag": "A"},
            "id-beta": {"content": "beta", "tag": "B"},
        }
    )

    results = await vdb.query("query-alpha", top_k=10)

    assert len(results) == 1
    assert results[0]["id"] == "id-alpha"
    assert results[0]["__id__"] == "id-alpha"
    assert "distance" in results[0]
    assert results[0]["tag"] == "A"


@pytest.mark.asyncio
async def test_upsert_empty_returns_empty_list(tmp_path):
    embedder = DummyEmbedder(dim=3, vector_by_text={"query": [1.0, 0.0, 0.0]})
    storage_file = tmp_path / "vdb.json"
    vdb = NanoVectorDBStorage(embedder=embedder, filename=str(storage_file))

    inserted = await vdb.upsert({})

    assert inserted == []


@pytest.mark.asyncio
async def test_upsert_skips_none_embeddings(tmp_path):
    embedder = DummyEmbedder(
        dim=3,
        vector_by_text={
            "keep": [1.0, 0.0, 0.0],
            "drop": None,
            "query-keep": [1.0, 0.0, 0.0],
        },
    )
    storage_file = tmp_path / "vdb.json"
    vdb = NanoVectorDBStorage(embedder=embedder, filename=str(storage_file), cosine_threshold=0.0)

    await vdb.upsert(
        {
            "id-keep": {"content": "keep"},
            "id-drop": {"content": "drop"},
        }
    )

    results = await vdb.query("query-keep", top_k=10)
    ids = [r["id"] for r in results]

    assert "id-keep" in ids
    assert "id-drop" not in ids


@pytest.mark.asyncio
async def test_delete_and_persistence_round_trip(tmp_path):
    embedder = DummyEmbedder(
        dim=3,
        vector_by_text={
            "persist": [1.0, 0.0, 0.0],
            "query": [1.0, 0.0, 0.0],
        },
    )
    storage_file = tmp_path / "vdb.json"
    vdb = NanoVectorDBStorage(embedder=embedder, filename=str(storage_file), cosine_threshold=0.0)

    await vdb.upsert({"id-persist": {"content": "persist"}})
    await vdb.index_done_callback()

    reloaded = NanoVectorDBStorage(embedder=embedder, filename=str(storage_file), cosine_threshold=0.0)
    loaded_results = await reloaded.query("query", top_k=10)
    assert any(r["id"] == "id-persist" for r in loaded_results)

    await reloaded.delete([])
    await reloaded.delete(["id-persist"])
    post_delete_results = await reloaded.query("query", top_k=10)
    assert all(r["id"] != "id-persist" for r in post_delete_results)
