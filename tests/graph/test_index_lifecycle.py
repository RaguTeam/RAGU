"""Storage lifecycle hooks follow actual writes and finish before unlock."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ragu.chunker.types import Chunk
from ragu.common.global_parameters import Settings
from ragu.graph.graph_builder_pipeline import BuilderArguments
from ragu.graph.knowledge_graph import KnowledgeGraph
from ragu.graph.types import Community, CommunitySummary, Entity, Relation
from ragu.models.embedder import Embedder


def _node(item_id: str) -> Entity:
    return Entity(item_id, "PERSON", "Description", [], id=item_id)


def _edge() -> Relation:
    return Relation("a", "b", "a", "b", "RELATED_TO", "Description", id="edge")


@pytest.fixture
def kg(tmp_path, monkeypatch):
    monkeypatch.setattr(Settings, "storage_folder", str(tmp_path / "storage"))
    embedder = AsyncMock(spec=Embedder)
    embedder.dim = 4
    embedder.batch_embed_text.side_effect = lambda texts, **kwargs: [[1.0, 0.0, 0.0, 0.0] for _ in texts]
    graph = KnowledgeGraph(
        llm=None,
        embedder=embedder,
        builder_settings=BuilderArguments(
            use_llm_summarization=False,
            make_community_summary=False,
            remove_isolated_nodes=False,
        ),
    )
    for storage in graph.index._storages():
        _record_lifecycle(storage, monkeypatch)
    return graph


def _record_lifecycle(storage, monkeypatch) -> None:
    storage._lifecycle_events = []
    storage._lifecycle_active = False
    original_start = storage.index_start_callback
    original_done = storage.index_done_callback

    async def start():
        if storage._lifecycle_active:
            raise ValueError("A storage lifecycle is already active")
        await original_start()
        storage._lifecycle_active = True
        storage._lifecycle_events.append("start")

    async def done():
        assert storage._lifecycle_active
        storage._lifecycle_events.append("done")
        storage._lifecycle_active = False
        await original_done()

    monkeypatch.setattr(storage, "index_start_callback", start)
    monkeypatch.setattr(storage, "index_done_callback", done)


def _clear_events(index) -> None:
    for storage in index._storages():
        storage._lifecycle_events.clear()


def _assert_no_callbacks(index) -> None:
    assert all(storage._lifecycle_events == [] for storage in index._storages())


@pytest.mark.parametrize("method", [
    "upsert_nodes", "update_nodes", "upsert_edges", "update_edges",
    "upsert_chunks", "upsert_communities", "update_communities", "upsert_summaries",
    "update_summaries", "delete_nodes", "delete_edges", "delete_chunks",
    "delete_communities", "delete_summaries",
])
async def test_empty_mutations_do_not_start_a_storage_lifecycle(kg, method):
    await getattr(kg.index, method)([])
    _assert_no_callbacks(kg.index)


@pytest.mark.parametrize("method,items", [
    ("upsert_nodes", [_node("a"), _node("a")]),
    ("update_nodes", [_node("missing")]),
    ("upsert_edges", [_edge(), _edge()]),
    ("update_edges", [_edge()]),
    ("update_communities", [Community(0, 99, [], [])]),
    ("update_summaries", [CommunitySummary(id="missing", summary="Summary")]),
])
async def test_validation_errors_do_not_start_a_storage_lifecycle(kg, method, items):
    await kg.index.upsert_nodes([_node("a"), _node("b")])
    _clear_events(kg.index)
    with pytest.raises(ValueError):
        await getattr(kg.index, method)(items)
    _assert_no_callbacks(kg.index)


async def test_missing_embedder_does_not_start_a_storage_lifecycle(kg, monkeypatch):
    monkeypatch.setattr(kg.index, "embedder", None)
    with pytest.raises(RuntimeError, match="without an embedder"):
        await kg.index.upsert_nodes([_node("a")])
    _assert_no_callbacks(kg.index)


@pytest.mark.parametrize("kind", ["nodes", "edges", "chunks"])
async def test_failed_vectorization_does_not_start_or_change_stores(kg, kind):
    index = kg.index
    if kind == "edges":
        await index.upsert_nodes([_node("a"), _node("b")])
        _clear_events(index)
    chunk = Chunk("Content", 0, "doc")
    items = {"nodes": [_node("a")], "edges": [_edge()], "chunks": [chunk]}[kind]
    kg.embedder.batch_embed_text.side_effect = ValueError("Embedding failed")

    with pytest.raises(ValueError, match="Embedding failed"):
        await getattr(index, f"upsert_{kind}")(items)

    _assert_no_callbacks(index)
    if kind == "chunks":
        assert await index.get_chunks([chunk.id]) == [None]


@pytest.mark.parametrize("edge_only", [False, True])
async def test_chunk_cascade_only_starts_the_stores_it_changes(kg, edge_only):
    index = kg.index
    chunk = Chunk("Evidence", 0, "doc")
    await index.upsert_chunks([chunk])
    if edge_only:
        await index.upsert_nodes([_node("a"), _node("b")])
        edge = _edge()
        edge.source_chunk_id = [chunk.id]
        await index.upsert_edges([edge])
    _clear_events(index)

    await index.delete_chunks([chunk.id])

    assert index.chunks_kv_storage._lifecycle_events == ["start", "done"]
    assert index.chunks_vector_db._lifecycle_events == ["start", "done"]
    assert index.nodes_vector_db._lifecycle_events == []
    expected = ["start", "done"] if edge_only else []
    assert index.graph_backend._lifecycle_events == expected
    assert index.edges_vector_db._lifecycle_events == expected


@pytest.mark.parametrize("with_communities", [False, True])
async def test_reindex_balances_drop_and_replacement_batches(kg, with_communities):
    node = _node("a")
    await kg.index.upsert_nodes([node])
    community = Community(0, 0, [node], [])
    communities = [community] if with_communities else []
    summaries = [CommunitySummary(id=community.id, summary="Summary")] if with_communities else []
    kg.pipeline.cluster_graph = AsyncMock(return_value=communities)
    kg.pipeline.community_summarizer = SimpleNamespace(summarize=AsyncMock(return_value=summaries))
    _clear_events(kg.index)

    await kg.reindex_community()

    expected = ["start", "done"] * (2 if with_communities else 1)
    assert kg.index.community_kv_storage._lifecycle_events == expected
    assert kg.index.community_summary_kv_storage._lifecycle_events == expected
    assert not any(storage._lifecycle_active for storage in kg.index._storages())


async def test_failed_storage_write_finalizes_started_stores(kg, monkeypatch):
    index = kg.index
    original = index.nodes_vector_db.upsert
    monkeypatch.setattr(index.nodes_vector_db, "upsert", AsyncMock(side_effect=ValueError("Write failed")))

    with pytest.raises(ValueError, match="Write failed"):
        await index.upsert_nodes([_node("a")])

    assert index.graph_backend._lifecycle_events == ["start", "done"]
    assert index.nodes_vector_db._lifecycle_events == ["start", "done"]
    monkeypatch.setattr(index.nodes_vector_db, "upsert", original)
    await index.upsert_nodes([_node("a")])
    assert index.graph_backend._lifecycle_events == ["start", "done"] * 2


async def test_one_failed_finalizer_does_not_skip_other_stores(kg, monkeypatch):
    index = kg.index
    original = index.graph_backend.index_done_callback

    async def failed_done():
        await original()
        raise ValueError("Flush failed")

    monkeypatch.setattr(index.graph_backend, "index_done_callback", failed_done)
    with pytest.raises(ValueError, match="Flush failed"):
        await index.upsert_nodes([_node("a")])
    assert index.graph_backend._lifecycle_events == ["start", "done"]
    assert index.nodes_vector_db._lifecycle_events == ["start", "done"]


async def test_partial_setup_finalizes_only_successfully_started_stores(kg, monkeypatch):
    index = kg.index
    monkeypatch.setattr(
        index.nodes_vector_db, "index_start_callback",
        AsyncMock(side_effect=ValueError("Setup failed")),
    )
    with pytest.raises(ValueError, match="Setup failed"):
        await index.upsert_nodes([_node("a")])
    assert index.graph_backend._lifecycle_events == ["start", "done"]
    assert index.nodes_vector_db._lifecycle_events == []
    assert await index.get_nodes(["a"]) == [None]


async def test_cancellation_keeps_element_locked_until_all_cleanup_finishes(kg, monkeypatch):
    index = kg.index
    write_entered = asyncio.Event()
    cleanup_entered = asyncio.Event()
    cleanup_release = asyncio.Event()
    original_done = index.graph_backend.index_done_callback

    async def paused_write(nodes):
        write_entered.set()
        await asyncio.Event().wait()

    async def paused_done():
        cleanup_entered.set()
        await cleanup_release.wait()
        await original_done()

    monkeypatch.setattr(index.graph_backend, "upsert_nodes", paused_write)
    monkeypatch.setattr(index.graph_backend, "index_done_callback", paused_done)
    writing = asyncio.create_task(index.upsert_nodes([_node("a")]))
    await asyncio.wait_for(write_entered.wait(), 2)
    writing.cancel()
    await asyncio.wait_for(cleanup_entered.wait(), 2)
    writing.cancel()
    reading = asyncio.create_task(index.get_nodes(["a"]))
    try:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(reading), 0.05)
        assert not writing.done()
    finally:
        cleanup_release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(writing, 2)
    assert await asyncio.wait_for(reading, 2) == [None]
    assert index.graph_backend._lifecycle_events == ["start", "done"]
    assert index.nodes_vector_db._lifecycle_events == ["start", "done"]


@pytest.mark.parametrize("mode", ["success", "write_error", "write_cancel", "done_cancel"])
async def test_lifecycle_and_mutations_share_one_task(kg, monkeypatch, mode):
    index = kg.index
    storage = index.community_summary_kv_storage
    original_start = storage.index_start_callback
    original_upsert = storage.upsert
    original_done = storage.index_done_callback
    batches = {}
    observed_tasks = []
    write_entered = asyncio.Event()
    done_entered = asyncio.Event()
    done_release = asyncio.Event()

    async def start():
        task = asyncio.current_task()
        assert task not in batches
        batches[task] = []
        observed_tasks.append(task)
        await original_start()

    async def write(data):
        task = asyncio.current_task()
        batches[task].append("write")
        observed_tasks.append(task)
        write_entered.set()
        if mode == "write_error":
            raise ValueError("Task-local write failed")
        if mode == "write_cancel":
            await asyncio.Event().wait()
        await original_upsert(data)

    async def done():
        task = asyncio.current_task()
        assert batches[task] == ["write"]
        observed_tasks.append(task)
        done_entered.set()
        if mode.endswith("cancel"):
            await done_release.wait()
        await original_done()
        batches.pop(task)

    monkeypatch.setattr(storage, "index_start_callback", start)
    monkeypatch.setattr(storage, "upsert", write)
    monkeypatch.setattr(storage, "index_done_callback", done)
    summary = CommunitySummary(id="summary", summary="Content")
    writing = asyncio.create_task(index.upsert_summaries([summary]))
    if mode in {"success", "write_error"}:
        if mode == "write_error":
            with pytest.raises(ValueError, match="Task-local write failed"):
                await asyncio.wait_for(writing, 2)
        else:
            await asyncio.wait_for(writing, 2)
    else:
        await asyncio.wait_for(write_entered.wait(), 2)
        if mode == "write_cancel":
            writing.cancel()
        await asyncio.wait_for(done_entered.wait(), 2)
        writing.cancel()
        await asyncio.sleep(0)
        writing.cancel()
        reading = asyncio.create_task(index.get_summaries([summary.id]))
        try:
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(reading), 0.05)
            assert not writing.done()
        finally:
            done_release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(writing, 2)
        await asyncio.wait_for(reading, 2)

    assert len(observed_tasks) == 3
    assert len(set(observed_tasks)) == 1
    assert batches == {}
    assert not storage._lifecycle_active
