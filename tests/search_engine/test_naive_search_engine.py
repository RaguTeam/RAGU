from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ragu.chunker.types import Chunk
from ragu.search_engine.base_engine import SearchEngineResponse
from ragu.search_engine.naive_search import (
    NaiveSearchEngine,
    NaiveSearchParams,
    NaiveSearchResult,
    NaiveSearchRetrieve,
)
from ragu.storage.types import EmbeddingHit


def _make_embedder_mock():
    return SimpleNamespace(embed_text=AsyncMock(return_value=[0.0] * 3))


@pytest.mark.asyncio
async def test_naive_search_rerank_and_rerank_top_k(real_kg, kg_fixture_ids):
    chunk_ids = kg_fixture_ids["chunk_ids"]
    reranker = SimpleNamespace(score=AsyncMock(return_value=[(1, 0.95), (0, 0.11)]))
    llm = SimpleNamespace(chat_completion=AsyncMock())

    engine = NaiveSearchEngine(
        llm=llm,
        knowledge_graph=real_kg,
        embedder=_make_embedder_mock(),
        reranker=reranker
    )
    engine.retriever.query_chunks = AsyncMock(
        return_value=[(
            [
                Chunk(content="chunk one", chunk_order_idx=0, doc_id="doc-1"),
                Chunk(content="chunk two", chunk_order_idx=1, doc_id="doc-2"),
            ],
            [
                EmbeddingHit(id=chunk_ids[0], distance=0.2),
                EmbeddingHit(id=chunk_ids[1], distance=0.8),
            ],
        )]
    )
    setattr(engine.retriever.query_chunks.return_value[0][0][0], "id", chunk_ids[0])
    setattr(engine.retriever.query_chunks.return_value[0][0][1], "id", chunk_ids[1])
    result = await engine.search("query", NaiveSearchParams(top_k=3, rerank_top_k=1))

    assert isinstance(result, NaiveSearchRetrieve)
    assert len(result.result.chunks) == 1
    assert result.result.chunks[0].id == chunk_ids[1]
    assert result.result.scores == [0.95]
    assert len(result.result.documents_id) == 1
    assert result.metrics["chunks"] == [
        {"id": chunk_ids[1], "rank": 0, "score": 0.95},
    ]


@pytest.mark.asyncio
async def test_naive_search_empty_returns_empty_result(real_kg):
    engine = NaiveSearchEngine(
        llm=SimpleNamespace(chat_completion=AsyncMock()),
        knowledge_graph=real_kg,
        embedder=_make_embedder_mock(),
    )
    engine.retriever.query_chunks = AsyncMock(return_value=[([], [])])

    result = await engine.search("query")
    assert result.result.chunks == []
    assert result.result.scores == []
    assert result.result.documents_id == []
    assert result.metrics == {}


@pytest.mark.asyncio
async def test_naive_query_uses_llm_response(monkeypatch):
    llm = SimpleNamespace(batch_chat_completion=AsyncMock(return_value=["naive-answer"]))
    kg = SimpleNamespace(
        index=SimpleNamespace(chunks_kv_storage=SimpleNamespace(get_by_ids=AsyncMock(return_value=[]))),
        sparse_embedder=None,
    )
    engine = NaiveSearchEngine(llm=llm, knowledge_graph=kg, embedder=_make_embedder_mock())
    engine.truncation = lambda s: s
    engine.batch_search = AsyncMock(return_value=[NaiveSearchRetrieve(query="question", result=NaiveSearchResult())])

    from ragu.search_engine import naive_search as naive_module
    monkeypatch.setattr(
        naive_module,
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
    assert result.response == "naive-answer"
