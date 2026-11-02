"""Concurrent facade operations preserve graph/vector/KV consistency."""

import asyncio
import threading
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

from ragu.chunker.types import Chunk
from ragu.common.global_parameters import Settings
from ragu.graph.graph_builder_pipeline import BuilderArguments
from ragu.graph.knowledge_graph import KnowledgeGraph
from ragu.graph.types import Community, CommunitySummary, Entity, Relation
from ragu.models.embedder import Embedder


@pytest.fixture
def kg(tmp_path, monkeypatch):
    monkeypatch.setattr(Settings, "storage_folder", str(tmp_path / "storage"))
    embedder = AsyncMock(spec=Embedder)
    embedder.dim = 4
    embedder.batch_embed_text.side_effect = lambda texts, **kwargs: [
        [1.0, 0.0, 0.0, 0.0] for _ in texts
    ]
    return KnowledgeGraph(
        llm=None,
        embedder=embedder,
        builder_settings=BuilderArguments(
            use_llm_summarization=False,
            make_community_summary=False,
            remove_isolated_nodes=False,
        ),
    )


def _entity(entity_id: str, description: str = "Original.", sources=()) -> Entity:
    return Entity(
        id=entity_id,
        entity_name=entity_id,
        entity_type="PERSON",
        description=description,
        source_chunk_id=list(sources),
    )


def _relation(subject: str, object_id: str, relation_id: str = "ab") -> Relation:
    return Relation(
        id=relation_id,
        subject_id=subject,
        object_id=object_id,
        subject_name=subject,
        object_name=object_id,
        relation_type="RELATED_TO",
        description=f"{subject} knows {object_id}.",
    )


async def _assert_waiting(task: asyncio.Task) -> None:
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(task), timeout=0.05)
    assert not task.done()


async def test_read_scope_allows_shared_reads_and_unrelated_updates(kg):
    index = kg.index
    await index.upsert_nodes([_entity("a"), _entity("b"), _entity("c")])

    async with kg.a_read_lock(node_ids=["a", "b"]):
        parallel_reader = asyncio.create_task(index.get_nodes(["a", "b"]))
        assert all(await asyncio.wait_for(parallel_reader, timeout=2))
        writer = asyncio.create_task(index.update_nodes([_entity("a", "Changed.")]))
        await _assert_waiting(writer)
        await asyncio.wait_for(
            asyncio.create_task(index.update_nodes([_entity("c", "Independent.")])), timeout=2,
        )
        assert (await index.get_nodes(["a"]))[0].description == "Original."

    await asyncio.wait_for(writer, timeout=2)
    assert (await index.get_nodes(["a"]))[0].description == "Changed."
    assert (await index.get_nodes(["c"]))[0].description == "Independent."


async def test_nested_read_scope_rejects_expansion_and_write_upgrade(kg):
    index = kg.index
    await index.upsert_nodes([_entity("a"), _entity("b")])

    async with kg.a_read_lock(node_ids=["a"]):
        assert (await index.get_nodes(["a"]))[0].description == "Original."
        with pytest.raises(ValueError, match="Nested lock scopes"):
            await index.get_nodes(["b"])
        with pytest.raises(ValueError, match="Nested lock scopes"):
            await kg.update_entities([_entity("a", "Changed.")])

    assert (await index.get_nodes(["a"]))[0].description == "Original."


@pytest.mark.parametrize("stage", ["embedding", "flush"])
async def test_write_scope_spans_embedding_and_persistence(kg, monkeypatch, stage):
    index = kg.index
    await index.upsert_nodes([_entity("a"), _entity("b")])
    entered = asyncio.Event()
    release = asyncio.Event()

    if stage == "embedding":
        original = kg.embedder.batch_embed_text

        async def paused_embedding(texts, **kwargs):
            if any("Changed." in text for text in texts):
                entered.set()
                await release.wait()
            return await original(texts, **kwargs)

        monkeypatch.setattr(kg.embedder, "batch_embed_text", paused_embedding)
    else:
        original = index.nodes_vector_db.index_done_callback
        pause_next = True

        async def paused_flush():
            nonlocal pause_next
            if pause_next:
                pause_next = False
                entered.set()
                await release.wait()
            await original()

        monkeypatch.setattr(index.nodes_vector_db, "index_done_callback", paused_flush)

    writer = asyncio.create_task(index.update_nodes([_entity("a", "Changed.")]))
    await asyncio.wait_for(entered.wait(), timeout=2)
    reader = asyncio.create_task(index.get_nodes(["a"]))
    try:
        await _assert_waiting(reader)
        await asyncio.wait_for(
            index.update_nodes([_entity("b", "Independent.")]), timeout=2,
        )
        assert not reader.done()
    finally:
        release.set()
        await asyncio.wait_for(writer, timeout=2)
    assert (await asyncio.wait_for(reader, timeout=2))[0].description == "Changed."


async def test_relation_reader_delays_cascade_node_deletion(kg):
    index = kg.index
    await index.upsert_nodes([_entity("a"), _entity("b"), _entity("c")])
    await index.upsert_edges([_relation("a", "b")])

    async with index.a_read_lock(edge_specs=[("a", "b", "ab")]):
        deleting = asyncio.create_task(kg.delete_entities(["a"]))
        await _assert_waiting(deleting)
        await asyncio.wait_for(
            asyncio.create_task(index.update_nodes([_entity("c", "Independent.")])), timeout=2,
        )
        assert await index.edges_vector_db.get_all_ids() == ["ab"]

    await asyncio.wait_for(deleting, timeout=2)
    assert await index.get_nodes(["a"]) == [None]
    assert await index.get_edges([("a", "b", "ab")]) == [[]]
    assert await index.edges_vector_db.get_all_ids() == []
    assert set(await index.nodes_vector_db.get_all_ids()) == {"b", "c"}


async def test_cascade_keeps_relation_locked_until_vector_cleanup_finishes(kg, monkeypatch):
    index = kg.index
    await index.upsert_nodes([_entity("a"), _entity("b"), _entity("c")])
    await index.upsert_edges([_relation("a", "b")])
    entered = asyncio.Event()
    release = asyncio.Event()
    original = index.edges_vector_db.delete

    async def paused_delete(ids, **kwargs):
        entered.set()
        await release.wait()
        await original(ids, **kwargs)

    monkeypatch.setattr(index.edges_vector_db, "delete", paused_delete)
    deleting = asyncio.create_task(index.delete_nodes(["a"]))
    await asyncio.wait_for(entered.wait(), timeout=2)
    reader = asyncio.create_task(index.get_edges([("a", "b", "ab")]))
    try:
        await _assert_waiting(reader)
        await asyncio.wait_for(index.get_nodes(["c"]), timeout=2)
    finally:
        release.set()
        await asyncio.wait_for(deleting, timeout=2)
    assert await asyncio.wait_for(reader, timeout=2) == [[]]
    assert await index.edges_vector_db.get_all_ids() == []


async def test_concurrent_entity_upserts_keep_both_merge_results(kg, monkeypatch):
    await kg.upsert_entities([_entity("a", "Initial.", ["initial-source"])])
    entered = asyncio.Event()
    release = asyncio.Event()
    original = kg.embedder.batch_embed_text

    async def paused_embedding(texts, **kwargs):
        if any("First addition." in text for text in texts) and not entered.is_set():
            entered.set()
            await release.wait()
        return await original(texts, **kwargs)

    monkeypatch.setattr(kg.embedder, "batch_embed_text", paused_embedding)
    first = asyncio.create_task(kg.upsert_entities([
        _entity("a", "First addition.", ["first-source"]),
    ]))
    await asyncio.wait_for(entered.wait(), timeout=2)
    second = asyncio.create_task(kg.upsert_entities([
        _entity("a", "Second addition.", ["second-source"]),
    ]))
    try:
        await _assert_waiting(second)
    finally:
        release.set()
        await asyncio.wait_for(asyncio.gather(first, second), timeout=2)

    stored = (await kg.get_entities(["a"]))[0]
    assert all(fragment in stored.description for fragment in (
        "Initial.", "First addition.", "Second addition.",
    ))
    assert set(stored.source_chunk_id) == {"initial-source", "first-source", "second-source"}


async def test_deleting_endpoint_then_inserting_edge_does_not_leave_orphan(kg, monkeypatch):
    index = kg.index
    await index.upsert_nodes([_entity("a"), _entity("b")])
    entered = asyncio.Event()
    release = asyncio.Event()
    original = index.graph_backend.delete_nodes

    async def paused_delete(node_ids):
        entered.set()
        await release.wait()
        await original(node_ids)

    monkeypatch.setattr(index.graph_backend, "delete_nodes", paused_delete)
    deleting = asyncio.create_task(index.delete_nodes(["a"]))
    await asyncio.wait_for(entered.wait(), timeout=2)
    inserting = asyncio.create_task(index.upsert_edges([_relation("a", "b")]))
    try:
        await _assert_waiting(inserting)
    finally:
        release.set()
        await asyncio.wait_for(deleting, timeout=2)
    with pytest.raises(ValueError, match="non-existent"):
        await asyncio.wait_for(inserting, timeout=2)
    assert await index.graph_backend.get_all_edges() == []
    assert await index.edges_vector_db.get_all_ids() == []


@pytest.mark.parametrize("delete_source", ["old", "new"])
async def test_chunk_delete_revalidates_references_after_concurrent_update(
    kg, monkeypatch, delete_source,
):
    index = kg.index
    old = Chunk("Old evidence.", 0, "doc")
    new = Chunk("New evidence.", 1, "doc")
    await index.upsert_chunks([old, new])
    await index.upsert_nodes([_entity("a", sources=[old.id]), _entity("b")])
    await index.upsert_edges([_relation("a", "b")])
    entered = asyncio.Event()
    release = asyncio.Event()
    original = kg.embedder.batch_embed_text

    async def paused_embedding(texts, **kwargs):
        if any("Changed." in text for text in texts):
            entered.set()
            await release.wait()
        return await original(texts, **kwargs)

    monkeypatch.setattr(kg.embedder, "batch_embed_text", paused_embedding)
    updating = asyncio.create_task(index.update_nodes([
        _entity("a", "Changed.", [new.id]),
    ]))
    await asyncio.wait_for(entered.wait(), timeout=2)
    deleting_id = old.id if delete_source == "old" else new.id
    deleting = asyncio.create_task(index.delete_chunks([deleting_id]))
    try:
        await _assert_waiting(deleting)
    finally:
        release.set()
        await asyncio.wait_for(asyncio.gather(updating, deleting), timeout=2)

    stored = (await index.get_nodes(["a"]))[0]
    if delete_source == "old":
        assert stored is not None
        assert stored.source_chunk_id == [new.id]
        assert await index.edges_vector_db.get_all_ids() == ["ab"]
    else:
        assert stored is None
        assert await index.edges_vector_db.get_all_ids() == []
    assert await index.get_chunks([deleting_id]) == [None]
    assert (await index.check_consistency()).is_consistent


async def test_reverse_order_multi_entity_upserts_finish_without_lost_updates(kg, monkeypatch):
    await kg.upsert_entities([_entity("a"), _entity("b")])
    original = kg.embedder.batch_embed_text

    async def yielding_embedding(texts, **kwargs):
        await asyncio.sleep(0)
        return await original(texts, **kwargs)

    monkeypatch.setattr(kg.embedder, "batch_embed_text", yielding_embedding)
    operations = []
    for iteration in range(8):
        node_ids = ("a", "b") if iteration % 2 else ("b", "a")
        operations.append(kg.upsert_entities([
            _entity(node_id, f"Addition {iteration}.") for node_id in node_ids
        ]))
    await asyncio.wait_for(asyncio.gather(*operations), timeout=5)

    for stored in await kg.get_entities(["a", "b"]):
        assert all(f"Addition {iteration}." in stored.description for iteration in range(8))


async def test_neighborhood_scope_protects_neighbor_chunks_and_summaries(kg):
    index = kg.index
    chunk = Chunk("Neighbor evidence.", 0, "doc")
    await index.upsert_chunks([chunk])
    community = Community(level=0, cluster_id=0, entities=[], relations=[])
    a = _entity("a")
    b = replace(_entity("b", sources=[chunk.id]), clusters=[{"level": 0, "cluster_id": 0}])
    await index.upsert_nodes([a, b, _entity("c")])
    await index.upsert_edges([_relation("a", "b")])
    await kg.upsert_summaries([CommunitySummary(id=community.id, summary="Original summary.")])

    async with index.a_read_lock(node_ids=["a"], include_neighbors=True):
        neighbor_update = asyncio.create_task(index.update_nodes([replace(b, description="Changed.")]))
        chunk_delete = asyncio.create_task(index.delete_chunks([chunk.id]))
        summary_update = asyncio.create_task(kg.update_summaries([
            CommunitySummary(id=community.id, summary="Changed summary."),
        ]))
        for task in (neighbor_update, chunk_delete, summary_update):
            await _assert_waiting(task)
        await asyncio.wait_for(
            asyncio.create_task(index.update_nodes([_entity("c", "Independent.")])), timeout=2,
        )

    await asyncio.wait_for(asyncio.gather(neighbor_update, chunk_delete, summary_update), timeout=2)
    assert await index.get_nodes(["b"]) == [None]
    assert (await kg.get_summaries([community.id]))[0].summary == "Changed summary."


async def test_read_results_and_written_inputs_do_not_share_nested_storage_state(kg):
    index = kg.index
    entity = replace(
        _entity("a", sources=["source"]),
        documents_id=["document"],
        clusters=[{"level": 0, "cluster_id": 1}],
    )
    await index.upsert_nodes([entity])
    entity.source_chunk_id.append("changed-input")
    entity.clusters[0]["cluster_id"] = 99

    snapshot = (await index.get_nodes(["a"]))[0]
    assert snapshot.source_chunk_id == ["source"]
    assert snapshot.clusters == [{"level": 0, "cluster_id": 1}]
    snapshot.source_chunk_id.clear()
    snapshot.documents_id.append("changed-snapshot")
    snapshot.clusters[0]["cluster_id"] = 42

    stored = (await index.get_nodes(["a"]))[0]
    assert stored.source_chunk_id == ["source"]
    assert stored.documents_id == ["document"]
    assert stored.clusters == [{"level": 0, "cluster_id": 1}]


async def test_one_index_coordinates_readers_and_writers_across_threads(kg):
    index = kg.index
    await index.upsert_nodes([_entity("a"), _entity("b")])
    entered = threading.Event()
    release = threading.Event()

    async def read_in_other_loop():
        async with index.a_read_lock(node_ids=["a"]):
            assert (await index.get_nodes(["a"]))[0].description == "Original."
            entered.set()
            assert await asyncio.to_thread(release.wait, 5)
            assert (await index.get_nodes(["a"]))[0].description == "Original."

    reader = asyncio.create_task(asyncio.to_thread(lambda: asyncio.run(read_in_other_loop())))
    assert await asyncio.wait_for(asyncio.to_thread(entered.wait, 3), timeout=4)
    writer = asyncio.create_task(index.update_nodes([_entity("a", "Changed.")]))
    try:
        await _assert_waiting(writer)
        await asyncio.wait_for(index.update_nodes([_entity("b", "Independent.")]), timeout=2)
    finally:
        release.set()
        await asyncio.wait_for(asyncio.gather(reader, writer), timeout=4)
    assert (await index.get_nodes(["a"]))[0].description == "Changed."
