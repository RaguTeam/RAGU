"""Cascade footprints account for graph changes made while a writer waits."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from ragu.chunker.types import Chunk
from ragu.common.global_parameters import Settings
from ragu.graph.index import Index, StorageArguments
from ragu.graph.types import Entity, Relation
from ragu.models.embedder import Embedder


@pytest.fixture
def index(tmp_path, monkeypatch):
    monkeypatch.setattr(Settings, "storage_folder", str(tmp_path / "storage"))
    embedder = AsyncMock(spec=Embedder)
    embedder.dim = 4
    embedder.batch_embed_text.side_effect = lambda texts, **kwargs: [
        [1.0, 0.0, 0.0, 0.0] for _ in texts
    ]
    return Index(StorageArguments(), embedder=embedder)


def _entity(node_id: str, description: str = "Original.") -> Entity:
    return Entity(
        id=node_id,
        entity_name=node_id,
        entity_type="PERSON",
        description=description,
        source_chunk_id=[],
    )


def _relation(
    subject: str,
    object_id: str,
    edge_id: str,
    description: str = "Original relation.",
    sources=(),
) -> Relation:
    return Relation(
        id=edge_id,
        subject_id=subject,
        object_id=object_id,
        subject_name=subject,
        object_name=object_id,
        relation_type="RELATED_TO",
        description=description,
        source_chunk_id=list(sources),
    )


def _pause_new_relation_embedding(index, monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()
    original = index.embedder.batch_embed_text

    async def embedding(texts, **kwargs):
        if "Delayed relation." in texts:
            entered.set()
            await release.wait()
        return await original(texts, **kwargs)

    monkeypatch.setattr(index.embedder, "batch_embed_text", embedding)
    return entered, release


async def _assert_waiting(task):
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(task), timeout=0.05)
    assert not task.done()


async def test_node_delete_rediscovers_edges_inserted_while_waiting(index, monkeypatch):
    await index.upsert_nodes([_entity(node_id) for node_id in "abcd"])
    embedding_entered, embedding_release = _pause_new_relation_embedding(index, monkeypatch)
    discovered_empty = asyncio.Event()
    cleanup_entered, cleanup_release = asyncio.Event(), asyncio.Event()
    delete_task_name = "delete-after-pending-edge"
    original_incident = index.graph_backend.get_all_edges_for_nodes
    original_flush = index.edges_vector_db.index_done_callback

    async def observe_incident(node_ids):
        groups = await original_incident(node_ids)
        if asyncio.current_task().get_name() == delete_task_name and not discovered_empty.is_set():
            assert groups == [[]]
            discovered_empty.set()
        return groups

    async def pause_delete_flush():
        if asyncio.current_task().get_name() == delete_task_name:
            cleanup_entered.set()
            await cleanup_release.wait()
        await original_flush()

    monkeypatch.setattr(index.graph_backend, "get_all_edges_for_nodes", observe_incident)
    monkeypatch.setattr(index.edges_vector_db, "index_done_callback", pause_delete_flush)
    inserting = asyncio.create_task(index.upsert_edges([
        _relation("a", "b", "new-ab", "Delayed relation."),
    ]))
    await asyncio.wait_for(embedding_entered.wait(), timeout=2)
    deleting = asyncio.create_task(index.delete_nodes(["a"]), name=delete_task_name)
    readers = []
    try:
        await asyncio.wait_for(discovered_empty.wait(), timeout=2)
        await _assert_waiting(deleting)
        await asyncio.wait_for(index.update_nodes([
            _entity("c", "Independent C."), _entity("d", "Independent D."),
        ]), timeout=2)
        assert not inserting.done()
        assert not deleting.done()

        embedding_release.set()
        await asyncio.wait_for(cleanup_entered.wait(), timeout=2)
        assert await index.graph_backend.get_nodes(["a"]) == [None]
        assert await index.nodes_vector_db.get_all_ids() == ["b", "c", "d"]
        assert await index.edges_vector_db.get_all_ids() == []

        # ID-only readers need the newly discovered edge grant: endpoint grants
        # alone would let them cross the unfinished cascade persistence step.
        readers = [
            asyncio.create_task(index.get_edge_specs_by_ids({"new-ab"})),
            asyncio.create_task(index.get_edges([("a", "b", "new-ab")])),
        ]
        for reader in readers:
            await _assert_waiting(reader)
        assert all(await asyncio.wait_for(index.get_nodes(["c", "d"]), timeout=2))
    finally:
        embedding_release.set()
        cleanup_release.set()
        await asyncio.wait_for(asyncio.gather(inserting, deleting, *readers), timeout=3)

    assert await index.get_nodes(["a"]) == [None]
    assert await index.get_edges([("a", "b", "new-ab")]) == [[]]
    assert await index.edges_vector_db.get_all_ids() == []
    assert (await index.check_consistency()).is_consistent


async def test_pair_wide_edge_delete_includes_edge_inserted_while_waiting(index, monkeypatch):
    await index.upsert_nodes([_entity(node_id) for node_id in "abcd"])
    await index.upsert_edges([
        _relation("a", "b", "old-ab"), _relation("b", "a", "reverse-ba"),
    ])
    embedding_entered, embedding_release = _pause_new_relation_embedding(index, monkeypatch)
    discovered_old = asyncio.Event()
    observations = []
    delete_task_name = "delete-pair-after-pending-edge"
    original_get_edges = index.graph_backend.get_edges

    async def observe_pair(edge_specs):
        groups = await original_get_edges(edge_specs)
        if asyncio.current_task().get_name() == delete_task_name:
            observations.append({edge.id for group in groups for edge in group})
            discovered_old.set()
        return groups

    monkeypatch.setattr(index.graph_backend, "get_edges", observe_pair)
    inserting = asyncio.create_task(index.upsert_edges([
        _relation("a", "b", "new-ab", "Delayed relation."),
    ]))
    await asyncio.wait_for(embedding_entered.wait(), timeout=2)
    deleting = asyncio.create_task(
        index.delete_edges([("a", "b", None)]), name=delete_task_name,
    )
    try:
        await asyncio.wait_for(discovered_old.wait(), timeout=2)
        assert observations[0] == {"old-ab"}
        await _assert_waiting(deleting)
        await asyncio.wait_for(index.update_nodes([
            _entity("c", "Independent C."), _entity("d", "Independent D."),
        ]), timeout=2)
        assert not inserting.done()
        assert not deleting.done()
    finally:
        embedding_release.set()
        await asyncio.wait_for(asyncio.gather(inserting, deleting), timeout=3)

    assert {"old-ab", "new-ab"} in observations
    assert await index.get_edges([("a", "b", None)]) == [[]]
    assert [edge.id for edge in (await index.get_edges([("b", "a", None)]))[0]] == ["reverse-ba"]
    assert await index.edges_vector_db.get_all_ids() == ["reverse-ba"]
    assert (await index.check_consistency()).is_consistent


async def test_chunk_deletion_persists_edge_only_cascade_when_index_reopens(index):
    deleted_source = Chunk("Evidence for the deleted relation.", 0, "doc")
    kept_source = Chunk("Evidence for the kept relation.", 1, "doc")
    await index.upsert_chunks([deleted_source, kept_source])
    await index.upsert_nodes([_entity(node_id) for node_id in "abcd"])
    await index.upsert_edges([
        _relation("a", "b", "deleted-ab", sources=[deleted_source.id]),
        _relation("c", "d", "kept-cd", sources=[kept_source.id]),
    ])

    await index.delete_chunks([deleted_source.id])
    assert all(await index.get_nodes(list("abcd")))
    assert await index.get_edges([("a", "b", "deleted-ab")]) == [[]]
    await index.close()

    reopened = Index(StorageArguments(), embedder=index.embedder)
    try:
        assert all(await reopened.get_nodes(list("abcd")))
        assert await reopened.get_edges([("a", "b", "deleted-ab")]) == [[]]
        assert [edge.id for edge in (await reopened.get_edges([("c", "d", None)]))[0]] == ["kept-cd"]
        assert await reopened.edges_vector_db.get_all_ids() == ["kept-cd"]
        assert await reopened.get_chunks([deleted_source.id]) == [None]
        assert (await reopened.get_chunks([kept_source.id]))[0].content == kept_source.content
        assert (await reopened.check_consistency()).is_consistent
    finally:
        await reopened.close()
