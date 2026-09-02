from unittest.mock import AsyncMock

import pytest

from ragu.common.prompts.default_models import RewriteQuery
from ragu.common.prompts.messages import AIMessage, ChatMessages, UserMessage
from ragu.search_engine.query_rewrite import DialogueQueryRewriter


def make_rewriter(responses: list[RewriteQuery] | None = None) -> tuple[DialogueQueryRewriter, AsyncMock]:
    llm = AsyncMock()
    llm.batch_chat_completion = AsyncMock(return_value=responses or [])
    return DialogueQueryRewriter(llm=llm), llm


@pytest.mark.asyncio
async def test_plain_string_query_is_returned_unchanged_without_llm_call():
    rewriter, llm = make_rewriter()

    result = await rewriter.rewrite("Who founded RAGU?")

    assert result == "Who founded RAGU?"
    llm.batch_chat_completion.assert_not_called()


@pytest.mark.asyncio
async def test_dialogue_without_history_is_returned_unchanged_without_llm_call():
    rewriter, llm = make_rewriter()
    dialogue = ChatMessages.from_messages([UserMessage(content="Who founded RAGU?")])

    result = await rewriter.rewrite(dialogue)

    assert result == "Who founded RAGU?"
    llm.batch_chat_completion.assert_not_called()


@pytest.mark.asyncio
async def test_dialogue_with_history_is_rewritten_via_llm():
    rewriter, llm = make_rewriter([RewriteQuery(query="When did the RaguTeam start RAGU?")])
    dialogue = ChatMessages.from_messages([
        UserMessage(content="Who founded RAGU?"),
        AIMessage(content="RAGU was founded by the RaguTeam."),
        UserMessage(content="When did they start it?"),
    ])

    result = await rewriter.rewrite(dialogue)

    assert result == "When did the RaguTeam start RAGU?"
    llm.batch_chat_completion.assert_awaited_once()


@pytest.mark.asyncio
async def test_batch_rewrite_only_calls_llm_for_queries_with_history():
    rewriter, llm = make_rewriter([RewriteQuery(query="When did the RaguTeam start RAGU?")])
    dialogue = ChatMessages.from_messages([
        UserMessage(content="Who founded RAGU?"),
        AIMessage(content="RAGU was founded by the RaguTeam."),
        UserMessage(content="When did they start it?"),
    ])

    results = await rewriter.batch_rewrite(["What is RAGU?", dialogue])

    assert results == ["What is RAGU?", "When did the RaguTeam start RAGU?"]
    conversations = llm.batch_chat_completion.await_args.args[0]
    assert len(conversations) == 1


@pytest.mark.asyncio
async def test_empty_batch_returns_empty_list():
    rewriter, llm = make_rewriter()

    assert await rewriter.batch_rewrite([]) == []
    llm.batch_chat_completion.assert_not_called()


@pytest.mark.asyncio
async def test_empty_dialogue_raises_value_error():
    rewriter, _ = make_rewriter()

    with pytest.raises(ValueError):
        await rewriter.rewrite(ChatMessages.from_messages([]))


@pytest.mark.asyncio
async def test_dialogue_not_ending_in_user_message_raises_value_error():
    rewriter, _ = make_rewriter()
    dialogue = ChatMessages.from_messages([
        UserMessage(content="Who founded RAGU?"),
        AIMessage(content="RAGU was founded by the RaguTeam."),
    ])

    with pytest.raises(ValueError):
        await rewriter.rewrite(dialogue)
