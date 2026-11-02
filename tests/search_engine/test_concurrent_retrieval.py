"""Deterministic retrieval and online graph-update concurrency scenarios."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ragu.chunker.types import Chunk
from ragu.common.global_parameters import Settings
from ragu.graph.graph_builder_pipeline import BuilderArguments
from ragu.graph.graph_retrieve_backend import GraphRetriever
from ragu.graph.knowledge_graph import KnowledgeGraph
from ragu.graph.types import Community, CommunitySummary, Entity, Relation
from ragu.search_engine.global_search import GlobalSearchEngine
from ragu.search_engine.local_search import LocalSearchEngine
from ragu.storage.types import EmbeddingHit


@pytest.fixture
def online_graph(tmp_path, monkeypatch):
    monkeypatch.setattr(Settings, "storage_folder", str(tmp_path / "storage"))
    embedder = SimpleNamespace(
        dim=4,
        batch_embed_text=AsyncMock(
            side_effect=lambda texts, **kwargs: [[0.1] * 4 for _ in texts]
        ),
    )
    return KnowledgeGraph(
        llm=None,
        embedder=embedder,
        builder_settings=BuilderArguments(build_only_vector_context=True),
    )


async def _populate_neighborhood(graph):
    chunks = [Chunk(f"Source {name}", i, f"doc-{name}") for i, name in enumerate("abc")]
    entities = [
        Entity(name, "PERSON", f"Description {name}", [chunk.id], id=f"node-{name}")
        for name, chunk in zip("abc", chunks)
    ]
    a, b, _ = entities
    relation = Relation(
        a.id, b.id, a.entity_name, b.entity_name,
        "RELATED_TO", "Original relation", source_chunk_id=[chunks[0].id], id="edge-ab",
    )
    community = Community(level=0, cluster_id=1, entities=[a, b], relations=[relation])
    a.clusters = [{"level": 0, "cluster_id": 1}]
    await graph.index.upsert_chunks(chunks)
    await graph.upsert_entities(entities)
    await graph.upsert_relations([relation])
    await graph.upsert_communities([community])
    await graph.upsert_summaries([CommunitySummary(id=community.id, summary="Original summary")])
    return entities, chunks, relation, community


def _paused_reranker():
    entered = asyncio.Event()
    release = asyncio.Event()

    async def score(query, contents):
        if not entered.is_set():
            entered.set()
            await release.wait()
        return [(i, 1.0) for i in range(len(contents))]

    return SimpleNamespace(score=score), entered, release


@pytest.mark.parametrize("mutation", ["node", "edge", "chunk", "summary"])
async def test_local_retrieval_holds_neighborhood_while_disjoint_updates_proceed(online_graph, mutation):
    graph = online_graph
    (a, b, c), chunks, relation, community = await _populate_neighborhood(graph)
    reranker, entered, release = _paused_reranker()
    engine = LocalSearchEngine(SimpleNamespace(), graph, graph.embedder, reranker=reranker)
    engine.retriever.query_entities = AsyncMock(
        return_value=[([a], [EmbeddingHit(id=a.id, distance=0.9)])]
    )
    searching = asyncio.create_task(engine.search("query"))
    await asyncio.wait_for(entered.wait(), 2)

    attempted = asyncio.Event()

    async def mutate():
        attempted.set()
        if mutation == "node":
            await graph.delete_entities([b.id])
        elif mutation == "edge":
            await graph.update_relations([replace(relation, description="New relation")])
        elif mutation == "chunk":
            updated_chunk = Chunk("New chunk", 0, chunks[0].doc_id)
            updated_chunk.id = chunks[0].id
            await graph.index.upsert_chunks([updated_chunk])
        else:
            await graph.upsert_summaries([CommunitySummary(id=community.id, summary="New summary")])

    updating = asyncio.create_task(mutate())
    try:
        await asyncio.wait_for(attempted.wait(), 2)
        assert (await asyncio.wait_for(graph.get_entities([a.id]), 2))[0] is not None
        await asyncio.wait_for(graph.update_entities([replace(c, description="Independent update")]), 2)
        assert not updating.done()
    finally:
        release.set()
        results = await asyncio.wait_for(asyncio.gather(searching, updating), 3)

    result = results[0].result
    assert [entity.id for entity in result.entities] == [a.id]
    assert [(edge.id, edge.description) for edge in result.relations] == [(relation.id, "Original relation")]
    assert [chunk.content for chunk in result.chunks] == [chunks[0].content]
    assert [summary.summary for summary in result.summaries] == ["Original summary"]


async def test_local_retrieval_refetches_candidates_deleted_before_scope(online_graph):
    graph = online_graph
    (a, _, _), _, _, _ = await _populate_neighborhood(graph)
    engine = LocalSearchEngine(SimpleNamespace(), graph, graph.embedder)

    async def candidates(*args, **kwargs):
        await graph.delete_entities([a.id])
        return [([a], [EmbeddingHit(id=a.id, distance=0.9)])]

    engine.retriever.query_entities = candidates
    result = await engine.search("query")

    assert result.result.entities == []
    assert result.result.relations == []
    assert result.result.chunks == []
    assert result.metrics["entities"] == []


async def test_cancelling_local_retrieval_releases_neighborhood(online_graph):
    graph = online_graph
    (a, b, _), _, _, _ = await _populate_neighborhood(graph)
    reranker, entered, _ = _paused_reranker()
    engine = LocalSearchEngine(SimpleNamespace(), graph, graph.embedder, reranker=reranker)
    engine.retriever.query_entities = AsyncMock(
        return_value=[([a], [EmbeddingHit(id=a.id, distance=0.9)])]
    )
    searching = asyncio.create_task(engine.search("query"))
    await asyncio.wait_for(entered.wait(), 2)
    searching.cancel()
    with pytest.raises(asyncio.CancelledError):
        await searching

    await asyncio.wait_for(graph.delete_entities([b.id]), 2)
    assert await graph.get_entities([b.id]) == [None]


async def test_failed_local_retrieval_waits_for_child_cleanup_before_unlocking(online_graph, monkeypatch):
    from ragu.search_engine import local_search

    graph = online_graph
    (a, _, _), _, _, _ = await _populate_neighborhood(graph)
    engine = LocalSearchEngine(SimpleNamespace(), graph, graph.embedder)
    engine.retriever.query_entities = AsyncMock(
        return_value=[([a], [EmbeddingHit(id=a.id, distance=0.9)])]
    )
    sibling_entered = asyncio.Event()
    cleanup_entered = asyncio.Event()
    cleanup_release = asyncio.Event()

    async def fail(*args):
        await sibling_entered.wait()
        raise ValueError("Retrieval failed")

    async def slow_read(*args):
        sibling_entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_entered.set()
            await cleanup_release.wait()

    monkeypatch.setattr(local_search, "_find_most_related_edges_from_entities", fail)
    monkeypatch.setattr(local_search, "_find_most_related_text_unit_from_entities", slow_read)
    searching = asyncio.create_task(engine.search("query"))
    await asyncio.wait_for(cleanup_entered.wait(), 2)
    attempted = asyncio.Event()

    async def delete_anchor():
        attempted.set()
        return await graph.delete_entities([a.id])

    deleting = asyncio.create_task(delete_anchor())
    try:
        await asyncio.wait_for(attempted.wait(), 2)
        assert not searching.done()
        assert not deleting.done()
    finally:
        cleanup_release.set()
        results = await asyncio.wait_for(asyncio.gather(searching, deleting, return_exceptions=True), 3)

    assert isinstance(results[0], ValueError)
    assert str(results[0]) == "Retrieval failed"
    assert results[1] is graph


async def test_retriever_filters_vector_candidates_deleted_before_resolution(online_graph):
    graph = online_graph
    (a, _, _), _, _, _ = await _populate_neighborhood(graph)
    retriever = GraphRetriever(graph, graph.embedder)

    async def stale_hits(*args, **kwargs):
        await graph.delete_entities([a.id])
        return [[EmbeddingHit(id=a.id, distance=0.9)]]

    graph.index.nodes_vector_db.query = stale_hits
    entities, hits = (await retriever.query_entities(["query"]))[0]

    assert entities == []
    assert hits == []


async def test_global_summary_scope_ends_before_llm_processing(online_graph):
    graph = online_graph
    _, _, _, community = await _populate_neighborhood(graph)
    engine = GlobalSearchEngine(SimpleNamespace(), graph)
    contexts = []

    async def process_context(queries, context):
        contexts.extend(context)
        await graph.upsert_summaries([CommunitySummary(id=community.id, summary="Updated during LLM")])
        return [[] for _ in queries]

    engine.get_meta_responses = process_context
    await asyncio.wait_for(engine.search("query"), 2)

    assert contexts == ["Original summary"]
    assert (await graph.get_summaries([community.id]))[0].summary == "Updated during LLM"


@pytest.mark.parametrize("mutation", ["summary", "community"])
async def test_global_retrieval_locks_summaries_and_metadata_together(online_graph, mutation):
    graph = online_graph
    (_, _, c), _, _, community = await _populate_neighborhood(graph)
    engine = GlobalSearchEngine(SimpleNamespace(), graph)
    entered = asyncio.Event()
    release = asyncio.Event()
    read_summaries = engine._read_community_summaries

    async def paused_read(min_cluster_size):
        entered.set()
        await release.wait()
        return await read_summaries(min_cluster_size)

    engine._read_community_summaries = paused_read
    reading = asyncio.create_task(engine._get_community_summaries(2))
    await asyncio.wait_for(entered.wait(), 2)
    attempted = asyncio.Event()

    async def mutate():
        attempted.set()
        if mutation == "summary":
            await graph.upsert_summaries([CommunitySummary(id=community.id, summary="New summary")])
        else:
            await graph.delete_communities([community.id])

    writing = asyncio.create_task(mutate())
    try:
        await asyncio.wait_for(attempted.wait(), 2)
        await asyncio.wait_for(graph.update_entities([replace(c, description="Independent update")]), 2)
        assert not writing.done()
    finally:
        release.set()
        contexts, _ = await asyncio.wait_for(asyncio.gather(reading, writing), 3)

    assert contexts == ["Original summary"]
