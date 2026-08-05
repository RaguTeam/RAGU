from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ragu.common.prompts.default_models import GlobalSearchContextModel
from ragu.common.prompts.messages import ChatMessages, UserMessage
from ragu.search_engine.base_engine import BaseEngine, SearchEngineResponse
from ragu.search_engine.global_search import GlobalSearchResult, GlobalSearchRetrieve
from ragu.search_engine.mix_search import (
    MixQueryParams,
    MixSearchRetrieve,
    MixSearchResult,
    MixSearchEngine,
)
from ragu.search_engine.naive_search import NaiveSearchParams, NaiveSearchResult, NaiveSearchRetrieve


class DummyEngine(BaseEngine):
    def __init__(self, result=None, error: Exception | None = None):
        super().__init__(llm=SimpleNamespace(chat_completion=AsyncMock()), prompts={})
        self._result = result
        self._error = error

    async def batch_search(self, queries, params=None):
        if self._error is not None:
            raise self._error
        return [self._result for _ in queries]

    async def batch_query(self, queries, params=None):
        return [
            SearchEngineResponse(query=query, response="unused", retrieval=self._result)
            for query in queries
        ]


@pytest.mark.asyncio
async def test_mix_search_collects_contexts_in_engine_order():
    naive_result = NaiveSearchRetrieve(
        query="query",
        result=NaiveSearchResult(chunks=[], scores=[], documents_id=["doc-1"]),
    )
    global_result = GlobalSearchRetrieve(
        query="query",
        result=GlobalSearchResult(insights=[
            GlobalSearchContextModel(reasoning="", response="x", rating=3),
        ]),
    )
    engine = MixSearchEngine(
        llm=SimpleNamespace(chat_completion=AsyncMock()),
        engines=[
            DummyEngine(result=naive_result),
            DummyEngine(result=global_result),
        ],
    )

    result = await engine.search("query")

    assert isinstance(result, MixSearchRetrieve)
    assert result.result.results == [naive_result, global_result]
    assert result.metrics == {}


@pytest.mark.asyncio
async def test_mix_search_records_partial_failures():
    ok_result = NaiveSearchRetrieve(query="query", result=NaiveSearchResult())
    engine = MixSearchEngine(
        llm=SimpleNamespace(chat_completion=AsyncMock()),
        engines=[
            DummyEngine(result=ok_result),
            DummyEngine(error=ValueError("broken child engine")),
        ],
        allow_partial_failures=True,
    )

    result = await engine.search("query")

    assert isinstance(result, MixSearchRetrieve)
    assert result.result.results == [ok_result]


@pytest.mark.asyncio
async def test_mix_search_raises_when_all_engines_fail():
    engine = MixSearchEngine(
        llm=SimpleNamespace(chat_completion=AsyncMock()),
        engines=[
            DummyEngine(error=ValueError("first failure")),
            DummyEngine(error=RuntimeError("second failure")),
        ],
        allow_partial_failures=True,
    )

    with pytest.raises(RuntimeError, match="could not retrieve context"):
        await engine.search("query")


@pytest.mark.asyncio
async def test_mix_query_returns_llm_response(monkeypatch):
    llm = SimpleNamespace(batch_chat_completion=AsyncMock(return_value=["mix-answer"]))
    child = DummyEngine(result=NaiveSearchRetrieve(query="question", result=NaiveSearchResult()))
    child.batch_search = AsyncMock(wraps=child.batch_search)
    child.batch_query = AsyncMock(wraps=child.batch_query)
    engine = MixSearchEngine(llm=llm, engines=[child])
    engine.truncation = lambda s: s

    from ragu.search_engine import mix_search as mix_module
    monkeypatch.setattr(
        mix_module,
        "render",
        lambda messages, **kwargs: [ChatMessages.from_messages([UserMessage(content="prompt")])],
    )
    original_get_prompt = engine.get_prompt
    monkeypatch.setattr(
        engine,
        "get_prompt",
        lambda prompt_name: (
            SimpleNamespace(messages=[{"role": "user", "content": "{{query}}"}], pydantic_model=None)
            if prompt_name == "mix_search"
            else original_get_prompt(prompt_name)
        ),
    )

    result = await engine.query("question")
    assert isinstance(result, SearchEngineResponse)
    assert result.response == "mix-answer"
    child.batch_search.assert_awaited_once_with(["question"], None)
    child.batch_query.assert_not_awaited()


@pytest.mark.asyncio
async def test_mix_query_can_ensemble_engine_responses(monkeypatch):
    llm = SimpleNamespace(batch_chat_completion=AsyncMock(return_value=["ensemble-answer"]))
    child = DummyEngine(
        result=SearchEngineResponse(
            query="question",
            response="engine-answer",
            retrieval=NaiveSearchRetrieve(query="question", result=NaiveSearchResult()),
        )
    )
    child.batch_search = AsyncMock(wraps=child.batch_search)
    child.batch_query = AsyncMock(wraps=child.batch_query)
    engine = MixSearchEngine(llm=llm, engines=[child])
    engine.truncation = lambda s: s

    from ragu.search_engine import mix_search as mix_module
    monkeypatch.setattr(
        mix_module,
        "render",
        lambda messages, **kwargs: [ChatMessages.from_messages([UserMessage(content="prompt")])],
    )
    original_get_prompt = engine.get_prompt
    monkeypatch.setattr(
        engine,
        "get_prompt",
        lambda prompt_name: (
            SimpleNamespace(messages=[{"role": "user", "content": "{{query}}"}], pydantic_model=None)
            if prompt_name == "mix_search"
            else original_get_prompt(prompt_name)
        ),
    )

    result = await engine.query("question", MixQueryParams(ensemble_responses=True))
    assert isinstance(result, SearchEngineResponse)
    assert result.response == "ensemble-answer"
    child.batch_query.assert_awaited_once_with(["question"], None)
    child.batch_search.assert_not_awaited()


@pytest.mark.asyncio
async def test_mix_forwards_per_child_engine_params():
    child_a = DummyEngine(result=NaiveSearchRetrieve(query="query", result=NaiveSearchResult()))
    child_b = DummyEngine(result=NaiveSearchRetrieve(query="query", result=NaiveSearchResult()))
    child_a.batch_search = AsyncMock(wraps=child_a.batch_search)
    child_b.batch_search = AsyncMock(wraps=child_b.batch_search)
    params_a = NaiveSearchParams(top_k=3)
    params_b = NaiveSearchParams(top_k=9)

    engine = MixSearchEngine(
        llm=SimpleNamespace(chat_completion=AsyncMock()),
        engines=[child_a, child_b],
        engine_params=[params_a, params_b],
    )

    await engine.search("query")

    child_a.batch_search.assert_awaited_once_with(["query"], params_a)
    child_b.batch_search.assert_awaited_once_with(["query"], params_b)


@pytest.mark.asyncio
async def test_mix_engine_params_length_mismatch_raises():
    with pytest.raises(ValueError, match="engine_params length"):
        MixSearchEngine(
            llm=SimpleNamespace(chat_completion=AsyncMock()),
            engines=[DummyEngine(result=NaiveSearchRetrieve(query="q", result=NaiveSearchResult()))],
            engine_params=[NaiveSearchParams(), NaiveSearchParams()],
        )
