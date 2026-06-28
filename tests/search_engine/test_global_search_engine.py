from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from ragu.common.prompts.default_models import GlobalSearchContextModel

from ragu.search_engine.base_engine import SearchEngineResponse
from ragu.search_engine.global_search import GlobalSearchEngine, GlobalSearchResult, GlobalSearchRetrieve


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
    assert [r["response"] for r in result.result.insights] == ["high", "low"]
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
            result=GlobalSearchResult(insights=[{"response": "x", "rating": "1"}]),
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
