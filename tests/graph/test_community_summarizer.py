from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import ragu.graph.community_summarizer as community_module
from ragu.common.prompts import ChatMessages, UserMessage
from ragu.common.prompts.default_models import CommunityReportModel
from ragu.graph.community_summarizer import CommunitySummarizer
from ragu.graph.types import Community, Entity, Relation


@pytest.mark.asyncio
async def test_community_summarizer_llm_exception_returns_empty(monkeypatch):
    llm = AsyncMock()
    llm.batch_chat_completion = AsyncMock(return_value=[None])
    summarizer = CommunitySummarizer(llm=llm)

    monkeypatch.setattr(
        summarizer,
        "get_prompt",
        lambda _: SimpleNamespace(
            messages=[UserMessage(content="{{ community }}")],
            pydantic_model=CommunityReportModel,
        ),
    )
    monkeypatch.setattr(
        community_module,
        "render",
        lambda messages, **kwargs: [ChatMessages.from_messages([UserMessage(content="prompt")])],
    )

    entity = Entity(
        entity_name="Alice",
        entity_type="Person",
        description="A software engineer",
        source_chunk_id=["chunk-1"],
        documents_id=["doc-1"],
        clusters=[],
    )
    relation = Relation(
        subject_id=entity.id,
        object_id=entity.id,
        subject_name="Alice",
        object_name="Alice",
        relation_type="SELF",
        description="Self reference",
        source_chunk_id=["chunk-1"],
    )
    community = Community(level=1, cluster_id=1, entities=[entity], relations=[relation])

    result = await summarizer.summarize([community])
    assert result == []
