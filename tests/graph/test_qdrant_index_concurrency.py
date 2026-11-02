"""Exercise Index concurrency with the installed Qdrant local backend."""

import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock

import numpy as np
import pytest
from qdrant_client import AsyncQdrantClient
from typing_extensions import override

from ragu.chunker.types import Chunk
from ragu.common.global_parameters import Settings
from ragu.graph.index import Index, StorageArguments
from ragu.graph.types import Entity, Relation
from ragu.models.embedder import Embedder
from ragu.models.sparse_embedder import SparseEmbedder
from ragu.storage.types import Point, SparseEmbedding
from ragu.storage.vdb_storage_adapters.qdrant_vdb import QdrantVectorDBStorage


class _FixedSparseEmbedder(SparseEmbedder):
    @override
    def embed_document(self, texts: list[str]) -> list[SparseEmbedding]:
        return [SparseEmbedding(indices=[1, 2], values=[0.75, 0.25]) for _ in texts]

    @override
    def embed_query(self, texts: list[str]) -> list[SparseEmbedding]:
        return self.embed_document(texts)


@pytest.fixture(params=[None, "custom"], ids=["dense", "hybrid"])
async def qdrant_index(request, tmp_path, monkeypatch):
    monkeypatch.setattr(Settings, "storage_folder", str(tmp_path / "storage"))
    embedder = AsyncMock(spec=Embedder)
    embedder.dim = 3
    embedder.batch_embed_text.side_effect = lambda texts, **kwargs: [
        [1.0, 0.0, 0.0] for _ in texts
    ]
    index = Index(
        StorageArguments(
            vdb_storage_type=QdrantVectorDBStorage,
            vdb_storage_kwargs={"location": ":memory:", "sparse_type": request.param},
        ),
        embedder=embedder,
        sparse_embedder=_FixedSparseEmbedder() if request.param else None,
    )
    try:
        yield index
    finally:
        await index.close()


def _node(item_id: str, chunk: Chunk) -> Entity:
    return Entity(
        id=item_id, entity_name=item_id, entity_type="PERSON",
        description=f"Original {item_id}.", source_chunk_id=[chunk.id],
    )


def _edge(subject: Entity, object_node: Entity, chunk: Chunk) -> Relation:
    return Relation(
        id=f"{subject.id}-{object_node.id}", subject_id=subject.id,
        object_id=object_node.id, subject_name=subject.entity_name,
        object_name=object_node.entity_name, relation_type="RELATED_TO",
        description=f"{subject.id} knows {object_node.id}.", source_chunk_id=[chunk.id],
    )


async def _assert_waiting(task: asyncio.Task) -> None:
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(task), timeout=0.05)
    assert not task.done()


async def test_real_qdrant_read_scopes_and_cascade_deletion(qdrant_index):
    index = qdrant_index
    chunks = [Chunk(f"Evidence for {name}.", position, "doc")
              for position, name in enumerate(("a", "b", "c"))]
    nodes = [_node(name, chunk) for name, chunk in zip(("a", "b", "c"), chunks)]
    edges = [_edge(nodes[0], nodes[1], chunks[0]), _edge(nodes[1], nodes[2], chunks[1])]
    await index.upsert_chunks(chunks)
    await index.upsert_nodes(nodes)
    await index.upsert_edges(edges)

    # These clients must come from the installed package, not qdrant_testkit.
    vector_stores = (index.nodes_vector_db, index.edges_vector_db, index.chunks_vector_db)
    assert all(isinstance(store._client, AsyncQdrantClient) for store in vector_stores)
    query = Point(dense_embedding=np.array([1.0, 0.0, 0.0]))
    if index.sparse_embedder is not None:
        query.sparse_embedding = index.sparse_embedder.embed_query(["Evidence"])[0]
    hits = (await index.nodes_vector_db.query([query], top_k=3))[0]
    assert {hit.id for hit in hits} == {"a", "b", "c"}
    points = await index.nodes_vector_db.get_points_by_ids(["a"])
    assert points[0] is not None
    assert (points[0].sparse_embedding is not None) == (index.sparse_embedder is not None)
    assert (await index.check_consistency()).is_consistent

    writer = None
    try:
        async with index.a_read_lock(node_ids=["a", "b"]):
            shared_reader = asyncio.create_task(index.get_nodes(["a", "b"]))
            assert all(await asyncio.wait_for(shared_reader, timeout=2))
            writer = asyncio.create_task(index.update_nodes([
                replace(nodes[0], description="Updated a."),
            ]))
            await _assert_waiting(writer)
            independent = asyncio.create_task(index.update_nodes([
                replace(nodes[2], description="Updated c."),
            ]))
            await asyncio.wait_for(independent, timeout=2)
            assert (await index.get_nodes(["a"]))[0].description == "Original a."
    finally:
        if writer is not None:
            await asyncio.wait_for(writer, timeout=2)
    assert (await index.get_nodes(["a"]))[0].description == "Updated a."
    payload = (await index.nodes_vector_db.get_payloads_by_ids(["a"]))[0]
    assert payload["description"] == "Updated a."

    deleting = None
    try:
        async with index.a_read_lock(edge_specs=[("a", "b", edges[0].id)]):
            deleting = asyncio.create_task(index.delete_nodes(["a"]))
            await _assert_waiting(deleting)
            assert set(await index.edges_vector_db.get_all_ids()) == {edge.id for edge in edges}
    finally:
        if deleting is not None:
            await asyncio.wait_for(deleting, timeout=2)

    assert await index.get_nodes(["a"]) == [None]
    assert set(await index.nodes_vector_db.get_all_ids()) == {"b", "c"}
    assert await index.edges_vector_db.get_all_ids() == [edges[1].id]
    assert (await index.check_consistency()).is_consistent

    await index.delete_chunks([chunks[1].id])
    assert await index.get_nodes(["b"]) == [None]
    assert await index.nodes_vector_db.get_all_ids() == ["c"]
    assert await index.edges_vector_db.get_all_ids() == []
    assert set(await index.chunks_vector_db.get_all_ids()) == {chunks[0].id, chunks[2].id}
    assert (await index.check_consistency()).is_consistent

    clients = [store._client for store in vector_stores]
    await index.close()
    assert all(store._client is None for store in vector_stores)
    assert all(client._client.closed for client in clients)
