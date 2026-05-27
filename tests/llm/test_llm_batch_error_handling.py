import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from ragu.models.llm import LLM, LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI


class _TestSchema(BaseModel):
    value: str = ""
    count: int = 0


_MSG = [{'role': 'user', 'content': 'hello'}]


def _make_llm():
    mock_client = MagicMock(spec=CachedAsyncOpenAI)
    mock_client.chat_completion = AsyncMock(return_value="ok")
    return LLMOpenAI(client=mock_client, model_name="test-model")


async def test_batch_partial_failure_str():
    llm = _make_llm()

    llm.client.chat_completion = AsyncMock(side_effect=[
        "result-1",
        RuntimeError("transient error"),
        "result-3",
    ])

    conversations = [_MSG, _MSG, _MSG]

    results = await llm.batch_chat_completion(
        conversations=conversations,
        output_schema=str,
        continue_on_error=True,
    )

    assert len(results) == 3
    assert results[0] == "result-1"
    assert results[1] is None
    assert results[2] == "result-3"


async def test_batch_all_fail_str():
    llm = _make_llm()
    llm.client.chat_completion = AsyncMock(side_effect=RuntimeError("boom"))

    results = await llm.batch_chat_completion(
        conversations=[_MSG, _MSG],
        output_schema=str,
        continue_on_error=True,
    )

    assert len(results) == 2
    assert results == [None, None]


async def test_batch_partial_failure_schema():
    llm = _make_llm()

    llm.client.chat_completion = AsyncMock(side_effect=[
        RuntimeError("schema error"),
        _TestSchema(value="ok", count=42),
    ])

    results = await llm.batch_chat_completion(
        conversations=[_MSG, _MSG],
        output_schema=_TestSchema,
        continue_on_error=True,
    )

    assert len(results) == 2
    assert results[0] is None
    assert isinstance(results[1], _TestSchema)
    assert results[1].value == "ok"
    assert results[1].count == 42


async def test_batch_continue_on_error_false():
    llm = _make_llm()
    llm.client.chat_completion = AsyncMock(side_effect=RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        await llm.batch_chat_completion(
            conversations=[_MSG],
            output_schema=str,
            continue_on_error=False,
        )


async def test_batch_default_raises():
    llm = _make_llm()
    llm.client.chat_completion = AsyncMock(side_effect=RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        await llm.batch_chat_completion(
            conversations=[_MSG],
            output_schema=str,
        )


async def test_batch_all_succeed():
    llm = _make_llm()
    llm.client.chat_completion = AsyncMock(return_value="ok")

    results = await llm.batch_chat_completion(
        conversations=[_MSG, _MSG, _MSG],
        output_schema=str,
        continue_on_error=True,
    )

    assert len(results) == 3
    assert all(r == "ok" for r in results)
