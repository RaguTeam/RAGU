from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from ragu.common.prompts.default_models import GlobalSearchContextModel

from ragu.search_engine.base_engine import SearchEngineResponse
from ragu.search_engine.global_search import (
    GlobalSearchEngine,
    GlobalSearchParams,
    GlobalSearchResult,
    GlobalSearchRetrieve,
)


def _stub_community_storages(engine, communities):
    """
    Replace community storages with in-memory stubs.

    :param engine: Engine whose knowledge graph storages are stubbed.
    :param communities: Mapping ``community_id -> (summary, entity_count)``;
        an entity count of ``None`` means the metadata row is missing.
    """
    index = engine.knowledge_graph.index
    ids = list(communities)

    index.community_summary_kv_storage = SimpleNamespace(
        all_keys=AsyncMock(return_value=ids),
        get_by_ids=AsyncMock(return_value=[communities[i][0] for i in ids]),
    )

    async def _get_community_rows(requested_ids):
        rows = []
        for community_id in requested_ids:
            entity_count = communities[community_id][1]
            rows.append(
                None
                if entity_count is None
                else {"entity_ids": [f"ent-{community_id}-{i}" for i in range(entity_count)]}
            )
        return rows

    index.community_kv_storage = SimpleNamespace(get_by_ids=AsyncMock(side_effect=_get_community_rows))
    return index


@pytest.mark.asyncio
async def test_global_search_filters_and_sorts_by_rating(monkeypatch, real_kg):
    engine = GlobalSearchEngine(llm=SimpleNamespace(chat_completion=AsyncMock()), knowledge_graph=real_kg)

    monkeypatch.setattr(
        engine,
        "get_meta_responses",
        AsyncMock(
            return_value=[[
                GlobalSearchContextModel(**{"reasoning": "", "response": "low", "rating": "1"}),
                GlobalSearchContextModel(**{"reasoning": "", "response": "drop", "rating": "0"}),
                GlobalSearchContextModel(**{"reasoning": "", "response": "high", "rating": "5"}),
            ]]
        ),
    )

    result = await engine.search("query")
    assert isinstance(result, GlobalSearchRetrieve)
    assert [r.response for r in result.result.insights] == ["high", "low"]
    assert result.metrics == {
        "insight_0_rating": 5.0,
        "insight_1_rating": 1.0,
    }


@pytest.mark.asyncio
async def test_global_query_returns_llm_response(monkeypatch, real_kg):
    llm = SimpleNamespace(batch_chat_completion=AsyncMock(return_value=["global-answer"]))
    engine = GlobalSearchEngine(llm=llm, knowledge_graph=real_kg)
    engine.truncation = lambda s: s
    engine.batch_search = AsyncMock(
        return_value=[GlobalSearchRetrieve(
            query="question",
            result=GlobalSearchResult(insights=[
                GlobalSearchContextModel(reasoning="", response="x", rating=1),
            ]),
        )]
    )

    from ragu.search_engine import global_search as global_module
    monkeypatch.setattr(
        global_module,
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
    assert result.response == "global-answer"


@pytest.mark.asyncio
async def test_global_search_keeps_all_communities_by_default(real_kg):
    engine = GlobalSearchEngine(llm=SimpleNamespace(chat_completion=AsyncMock()), knowledge_graph=real_kg)
    index = _stub_community_storages(engine, {"com-big": ("big", 5), "com-small": ("small", 1)})
    engine.get_meta_responses = AsyncMock(return_value=[[]])

    await engine.search("query")

    assert engine.get_meta_responses.await_args.args[1] == ["big", "small"]
    index.community_kv_storage.get_by_ids.assert_not_awaited()


@pytest.mark.asyncio
async def test_global_search_skips_communities_below_min_size(real_kg):
    engine = GlobalSearchEngine(llm=SimpleNamespace(chat_completion=AsyncMock()), knowledge_graph=real_kg)
    _stub_community_storages(engine, {"com-big": ("big", 5), "com-small": ("small", 2)})
    engine.get_meta_responses = AsyncMock(return_value=[[]])

    await engine.search("query", GlobalSearchParams(min_cluster_size=3))

    assert engine.get_meta_responses.await_args.args[1] == ["big"]


@pytest.mark.asyncio
async def test_global_search_ignores_params_without_min_cluster_size(real_kg):
    engine = GlobalSearchEngine(llm=SimpleNamespace(chat_completion=AsyncMock()), knowledge_graph=real_kg)
    index = _stub_community_storages(engine, {"com-big": ("big", 5), "com-small": ("small", 1)})
    engine.get_meta_responses = AsyncMock(return_value=[[]])

    await engine.search("query", GlobalSearchParams())

    assert engine.get_meta_responses.await_args.args[1] == ["big", "small"]
    index.community_kv_storage.get_by_ids.assert_not_awaited()


@pytest.mark.asyncio
async def test_global_search_keeps_communities_without_metadata(real_kg):
    engine = GlobalSearchEngine(llm=SimpleNamespace(chat_completion=AsyncMock()), knowledge_graph=real_kg)
    _stub_community_storages(engine, {"com-unknown": ("unknown", None), "com-small": ("small", 1)})
    engine.get_meta_responses = AsyncMock(return_value=[[]])

    await engine.search("query", GlobalSearchParams(min_cluster_size=3))

    assert engine.get_meta_responses.await_args.args[1] == ["unknown"]


@pytest.mark.asyncio
async def test_global_search_min_cluster_size_can_filter_everything(real_kg):
    engine = GlobalSearchEngine(llm=SimpleNamespace(chat_completion=AsyncMock()), knowledge_graph=real_kg)
    _stub_community_storages(engine, {"com-big": ("big", 5)})

    result = await engine.search("query", GlobalSearchParams(min_cluster_size=100))

    assert result.result.insights == []
