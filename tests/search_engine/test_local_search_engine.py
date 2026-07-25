from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ragu.chunker.types import Chunk
from ragu.graph.types import Entity, Relation, CommunitySummary
from ragu.search_engine.base_engine import SearchEngineResponse
from ragu.search_engine.local_search import (
    LocalSearchEngine,
    LocalParams,
    LocalSearchResult,
    LocalSearchRetrieve,
)
from ragu.storage.types import EmbeddingHit


def _make_embedder_mock():
    return SimpleNamespace(embed_text=AsyncMock(return_value=[0.0] * 3))


@pytest.mark.asyncio
async def test_local_search_collects_entities_relations_chunks_and_summaries(real_kg, kg_fixture_ids):
    entity_ids = kg_fixture_ids["entity_ids"]
    existing_entities = await real_kg.index.get_nodes(entity_ids[:2])
    engine = LocalSearchEngine(
        llm=SimpleNamespace(chat_completion=AsyncMock()),
        knowledge_graph=real_kg,
        embedder=_make_embedder_mock(),
    )
    entities = [e for e in existing_entities if e is not None]
    engine.retriever.query_entities = AsyncMock(
        return_value=[(
            entities,
            [EmbeddingHit(id=entity.id, distance=score) for entity, score in zip(entities, [0.9, 0.8])],
        )]
    )

    result = await engine.search("query", LocalParams(top_k=3))

    assert isinstance(result, LocalSearchRetrieve)
    assert [e.id for e in result.result.entities] == entity_ids[:2]
    assert isinstance(result.result.relations, list)
    assert isinstance(result.result.chunks, list)
    assert isinstance(result.result.summaries, list)
    assert isinstance(result.result.documents_id, list)
    assert result.metrics["entities"] == [
        {"id": entities[0].id, "name": entities[0].entity_name, "rank": 0, "relevance_score": 0.9},
        {"id": entities[1].id, "name": entities[1].entity_name, "rank": 1, "relevance_score": 0.8},
    ]


@pytest.mark.asyncio
async def test_local_search_reranks_entities_relations_summaries_and_chunks(monkeypatch, real_kg):
    llm = SimpleNamespace(chat_completion=AsyncMock())
    entity_a = Entity(entity_name="Alpha", entity_type="Person", description="First entity", source_chunk_id=["chunk-a"], documents_id=["doc-a"])
    entity_b = Entity(entity_name="Beta", entity_type="Place", description="Second entity", source_chunk_id=["chunk-b"], documents_id=["doc-b"])
    relation_a = Relation(
        subject_id=entity_a.id,
        object_id=entity_b.id,
        subject_name=entity_a.entity_name,
        object_name=entity_b.entity_name,
        relation_type="knows",
        description="Alpha knows Beta",
    )
    relation_b = Relation(
        subject_id=entity_b.id,
        object_id=entity_a.id,
        subject_name=entity_b.entity_name,
        object_name=entity_a.entity_name,
        relation_type="visited",
        description="Beta visited Alpha",
    )
    chunk_a = Chunk(content="chunk alpha", chunk_order_idx=0, doc_id="doc-a")
    chunk_b = Chunk(content="chunk beta", chunk_order_idx=1, doc_id="doc-b")
    summaries = [CommunitySummary(summary="summary alpha", id="123"), CommunitySummary(summary="summary beta", id="345")]

    # Reranking now happens concurrently for entities/relations/summaries/chunks,
    # so the mock must be order-independent: reverse each list (swap the two items)
    # regardless of which list it is invoked for.
    reranker = SimpleNamespace(
        score=AsyncMock(
            side_effect=lambda query, contents: [
                (idx, float(len(contents) - idx)) for idx in range(len(contents) - 1, -1, -1)
            ]
        )
    )

    from ragu.search_engine import local_search as local_module
    monkeypatch.setattr(local_module, "_find_most_related_edges_from_entities", AsyncMock(return_value=[relation_a, relation_b]))
    monkeypatch.setattr(local_module, "_find_most_related_text_unit_from_entities", AsyncMock(return_value=[chunk_a, chunk_b]))
    monkeypatch.setattr(local_module, "_find_most_related_community_from_entities", AsyncMock(return_value=summaries))
    monkeypatch.setattr(local_module, "_find_documents_id", AsyncMock(return_value=["doc-b", "doc-a"]))

    engine = LocalSearchEngine(
        llm=llm,
        knowledge_graph=real_kg,
        embedder=_make_embedder_mock(),
        reranker=reranker,
    )
    engine.retriever.query_entities = AsyncMock(
        return_value=[(
            [entity_a, entity_b],
            [
                EmbeddingHit(id=entity_a.id, distance=0.11),
                EmbeddingHit(id=entity_b.id, distance=0.95),
            ],
        )]
    )

    result = await engine.search("query", LocalParams(top_k=2))

    assert [entity.id for entity in result.result.entities] == [entity_b.id, entity_a.id]
    assert [relation.id for relation in result.result.relations] == [relation_b.id, relation_a.id]
    assert [c.summary for c in result.result.summaries] == ["summary beta", "summary alpha"]
    assert [chunk.id for chunk in result.result.chunks] == [chunk_b.id, chunk_a.id]
    assert result.result.documents_id == ["doc-b", "doc-a"]
    assert result.metrics["entities"] == [
        {"id": entity_b.id, "name": "Beta", "rank": 0, "relevance_score": 0.95},
        {"id": entity_a.id, "name": "Alpha", "rank": 1, "relevance_score": 0.11},
    ]


@pytest.mark.asyncio
async def test_local_query_returns_raw_result_when_no_response_attr(monkeypatch, real_kg):
    llm = SimpleNamespace(batch_chat_completion=AsyncMock(return_value=["raw-result"]))
    engine = LocalSearchEngine(llm=llm, knowledge_graph=real_kg, embedder=_make_embedder_mock())
    engine.truncation = lambda s: s
    engine.batch_search = AsyncMock(return_value=[LocalSearchRetrieve(query="question", result=LocalSearchResult())])

    from ragu.search_engine import local_search as local_module
    monkeypatch.setattr(
        local_module,
        "render",
        lambda messages, **kwargs: [SimpleNamespace(to_openai=lambda: [{"role": "user", "content": "prompt"}])],
    )
    monkeypatch.setattr(
        engine,
        "get_prompt",
        lambda _: SimpleNamespace(messages=[{"role": "user", "content": "{{query}}"}], pydantic_model=None),
    )

    result = await engine.query("question")
    assert isinstance(result, SearchEngineResponse)
    assert result.response == "raw-result"


@pytest.mark.asyncio
async def test_local_search_rerank_top_k_truncates_entities_before_derivation(monkeypatch, real_kg):
    entity_a = Entity(entity_name="Alpha", entity_type="Person", description="d", source_chunk_id=["c"], documents_id=["doc-a"])
    entity_b = Entity(entity_name="Beta", entity_type="Place", description="d", source_chunk_id=["c"], documents_id=["doc-b"])

    # Reranker reverses the order -> [entity_b, entity_a]; rerank_top_k=1 keeps [entity_b].
    reranker = SimpleNamespace(
        score=AsyncMock(
            side_effect=lambda query, contents: [
                (idx, float(len(contents) - idx)) for idx in range(len(contents) - 1, -1, -1)
            ]
        )
    )

    seen = {}

    async def fake_edges(entities, edges_by_entity):
        seen["edges"] = list(entities)
        return []

    from ragu.search_engine import local_search as local_module
    monkeypatch.setattr(local_module, "_find_most_related_edges_from_entities", fake_edges)
    monkeypatch.setattr(local_module, "_find_most_related_text_unit_from_entities", AsyncMock(return_value=[]))
    monkeypatch.setattr(local_module, "_find_most_related_community_from_entities", AsyncMock(return_value=[]))
    monkeypatch.setattr(local_module, "_find_documents_id", AsyncMock(return_value=[]))

    engine = LocalSearchEngine(
        llm=SimpleNamespace(),
        knowledge_graph=real_kg,
        embedder=_make_embedder_mock(),
        reranker=reranker,
    )
    engine.retriever.query_entities = AsyncMock(
        return_value=[(
            [entity_a, entity_b],
            [EmbeddingHit(id=entity_a.id, distance=0.1), EmbeddingHit(id=entity_b.id, distance=0.9)],
        )]
    )

    result = await engine.search("query", LocalParams(rerank_top_k=1))

    # Only the top reranked entity survives, and derivation sees that reduced set.
    assert [e.id for e in result.result.entities] == [entity_b.id]
    assert [e.id for e in seen["edges"]] == [entity_b.id]


@pytest.mark.asyncio
async def test_find_text_units_fetches_chunks_in_a_single_batched_call():
    """Chunks are resolved with one ``get_by_ids`` call, not one ``get_by_id`` per chunk."""
    from ragu.search_engine import search_functional

    entity_a = Entity(
        entity_name="Alpha", entity_type="Person", description="d",
        source_chunk_id=["chunk-a", "chunk-b"], documents_id=["doc-a"],
    )
    entity_b = Entity(
        entity_name="Beta", entity_type="Place", description="d",
        source_chunk_id=["chunk-c"], documents_id=["doc-b"],
    )

    chunk_store = {
        "chunk-a": {"content": "A", "chunk_order_idx": 0, "doc_id": "doc-a"},
        "chunk-b": {"content": "B", "chunk_order_idx": 1, "doc_id": "doc-a"},
        "chunk-c": {"content": "C", "chunk_order_idx": 2, "doc_id": "doc-b"},
    }
    get_by_ids = AsyncMock(side_effect=lambda ids, fields=None: [chunk_store.get(i) for i in ids])
    get_by_id = AsyncMock(side_effect=lambda i: chunk_store.get(i))

    kg = SimpleNamespace(
        index=SimpleNamespace(
            graph_backend=SimpleNamespace(get_all_edges_for_nodes=AsyncMock(return_value=[[], []])),
            get_nodes=AsyncMock(return_value=[]),
            chunks_kv_storage=SimpleNamespace(get_by_ids=get_by_ids, get_by_id=get_by_id),
        )
    )

    # Neither entity has edges, so the prefetched maps are empty.
    chunks = await search_functional._find_most_related_text_unit_from_entities(
        [entity_a, entity_b], kg, {}, {}
    )

    # One batched fetch over all unique chunk ids; the per-chunk path is gone.
    get_by_ids.assert_awaited_once()
    assert get_by_ids.await_args.args[0] == ["chunk-a", "chunk-b", "chunk-c"]
    get_by_id.assert_not_awaited()

    # Output and seed-entity ordering preserved.
    assert [c.content for c in chunks] == ["A", "B", "C"]
