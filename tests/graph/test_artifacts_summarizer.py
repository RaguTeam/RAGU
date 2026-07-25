from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ragu.common.prompts import ChatMessages, UserMessage
from ragu.common.prompts.default_models import EntityDescriptionModel, RelationDescriptionModel
from ragu.graph.artifacts_summarizer import EntitySummarizer, RelationSummarizer
import ragu.graph.artifacts_summarizer as artifacts_module
from ragu.graph.types import Entity, Relation

def test_entity_summarizer_requires_client_when_llm_enabled():
    with pytest.raises(ValueError, match="no client is provided"):
        EntitySummarizer(llm=None, use_llm_summarization=True)


@pytest.mark.asyncio
async def test_entity_summarizer_run_empty_returns_empty_list():
    summarizer = EntitySummarizer(llm=None, use_llm_summarization=False)
    assert await summarizer.run([]) == []


@pytest.mark.asyncio
async def test_entity_summarizer_run_without_llm_deduplicates_entities():
    client = AsyncMock()
    summarizer = EntitySummarizer(
        llm=client,
        use_llm_summarization=False,
        use_clustering=False,
        embedder=AsyncMock(),
        summarize_only_if_more_than=1,
    )
    entities = [
        Entity(
            entity_name="Alice",
            entity_type="Person",
            description="First description",
            source_chunk_id=["chunk-1"],
            documents_id=["doc-1"],
            clusters=[],
        ),
        Entity(
            entity_name="Alice",
            entity_type="Person",
            description="Second description",
            source_chunk_id=["chunk-2"],
            documents_id=["doc-2"],
            clusters=[],
        ),
        Entity(
            entity_name="Bob",
            entity_type="Person",
            description="Bob description",
            source_chunk_id=["chunk-3"],
            documents_id=["doc-3"],
            clusters=[],
        ),
    ]

    result = await summarizer.run(entities)

    assert len(result) == 2
    by_name = {e.entity_name: e for e in result}
    assert set(by_name.keys()) == {"Alice", "Bob"}
    assert "First description" in by_name["Alice"].description
    assert "Second description" in by_name["Alice"].description
    assert set(by_name["Alice"].source_chunk_id) == {"chunk-1", "chunk-2"}
    assert set(by_name["Alice"].documents_id) == {"doc-1", "doc-2"}


@pytest.mark.asyncio
async def test_entity_summarizer_llm_path_updates_description(monkeypatch):
    client = AsyncMock()
    client.batch_chat_completion = AsyncMock(
        return_value=[EntityDescriptionModel(entity_name="Alice", description="LLM summary")]
    )
    summarizer = EntitySummarizer(
        llm=client,
        use_llm_summarization=True,
        use_clustering=False,
        embedder=AsyncMock(),
        summarize_only_if_more_than=1,
    )

    monkeypatch.setattr(
        summarizer,
        "get_prompt",
        lambda _: SimpleNamespace(messages=[UserMessage(content="{{ entity }}")], pydantic_model=EntityDescriptionModel),
    )

    def _fake_render(messages, **kwargs):
        return [ChatMessages.from_messages([UserMessage(content="prompt")]) for _ in kwargs["entity"]]

    monkeypatch.setattr(artifacts_module, "render", _fake_render)

    grouped = EntitySummarizer.group_entities(
        [
            Entity(
                entity_name="Alice",
                entity_type="Person",
                description="First description",
                source_chunk_id=["chunk-1"],
                documents_id=["doc-1"],
                clusters=[],
            ),
            Entity(
                entity_name="Alice",
                entity_type="Person",
                description="Second description",
                source_chunk_id=["chunk-2"],
                documents_id=["doc-2"],
                clusters=[],
            ),
        ]
    )
    result = await summarizer.summarize_entities(grouped)

    assert len(result) == 1
    assert result[0].entity_name == "Alice"
    assert result[0].description == "LLM summary"
    client.batch_chat_completion.assert_awaited_once()


def test_relation_summarizer_requires_client_when_llm_enabled():
    with pytest.raises(ValueError, match="no client is provided"):
        RelationSummarizer(llm=None, use_llm_summarization=True)


@pytest.mark.asyncio
async def test_relation_summarizer_run_empty_returns_empty_list():
    summarizer = RelationSummarizer(llm=None, use_llm_summarization=False)
    assert await summarizer.run([]) == []


def test_relation_group_relations_merges_duplicates():
    grouped = RelationSummarizer.group_relations(
        [
            Relation(
                id="rel-1",
                subject_id="ent-1",
                object_id="ent-2",
                subject_name="ent-1",
                object_name="ent-2",
                relation_type="KNOWS",
                description="A knows B",
                relation_strength=1.0,
                source_chunk_id=["chunk-1"],
            ),
            Relation(
                id="rel-2",
                subject_id="ent-1",
                object_id="ent-2",
                subject_name="ent-1",
                object_name="ent-2",
                relation_type="KNOWS",
                description="A knows B",
                relation_strength=3.0,
                source_chunk_id=["chunk-2"],
            ),
        ]
    )

    assert len(grouped) == 1
    row = grouped.iloc[0]
    assert row["subject_id"] == "ent-1"
    assert row["object_id"] == "ent-2"
    assert row["relation_type"] == "KNOWS"
    assert row["relation_strength"] == pytest.approx(2.0)
    assert set(row["source_chunk_id"]) == {"chunk-1", "chunk-2"}
    assert row["description"] == "A knows B"


@pytest.mark.asyncio
async def test_relation_summarizer_run_without_llm_deduplicates_relations():
    client = AsyncMock()
    summarizer = RelationSummarizer(
        llm=client,
        use_llm_summarization=False,
        summarize_only_if_more_than=1,
    )

    relations = [
        Relation(
            id="rel-1",
            subject_id="ent-1",
            object_id="ent-2",
            subject_name="ent-1",
            object_name="ent-2",
            relation_type="KNOWS",
            description="A knows B",
            relation_strength=1.0,
            source_chunk_id=["chunk-1"],
        ),
        Relation(
            id="rel-2",
            subject_id="ent-1",
            object_id="ent-2",
            subject_name="ent-1",
            object_name="ent-2",
            relation_type="KNOWS",
            description="A met B",
            relation_strength=3.0,
            source_chunk_id=["chunk-2"],
        ),
        Relation(
            id="rel-3",
            subject_id="ent-2",
            object_id="ent-3",
            subject_name="ent-2",
            object_name="ent-3",
            relation_type="KNOWS",
            description="B knows C",
            relation_strength=2.0,
            source_chunk_id=["chunk-3"],
        ),
    ]

    result = await summarizer.run(relations)

    assert len(result) == 2
    by_pair = {(r.subject_id, r.object_id): r for r in result}
    assert ("ent-1", "ent-2") in by_pair
    assert ("ent-2", "ent-3") in by_pair
    assert "A knows B" in by_pair[("ent-1", "ent-2")].description
    assert "A met B" in by_pair[("ent-1", "ent-2")].description
    assert by_pair[("ent-1", "ent-2")].relation_strength == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_relation_summarizer_llm_path_updates_description(monkeypatch):
    client = AsyncMock()
    client.batch_chat_completion = AsyncMock(
        return_value=[
            RelationDescriptionModel(
                subject_name="ent-1",
                object_name="ent-2",
                description="LLM relation summary",
            )
        ]
    )
    summarizer = RelationSummarizer(
        llm=client,
        use_llm_summarization=True,
        summarize_only_if_more_than=1,
    )

    monkeypatch.setattr(
        summarizer,
        "get_prompt",
        lambda _: SimpleNamespace(messages=[UserMessage(content="{{ relation }}")], pydantic_model=RelationDescriptionModel),
    )

    def _fake_render(messages, **kwargs):
        return [ChatMessages.from_messages([UserMessage(content="prompt")]) for _ in kwargs["relation"]]

    monkeypatch.setattr(artifacts_module, "render", _fake_render)

    grouped = RelationSummarizer.group_relations(
        [
            Relation(
                id="rel-1",
                subject_id="ent-1",
                object_id="ent-2",
                subject_name="ent-1",
                object_name="ent-2",
                relation_type="KNOWS",
                description="A knows B",
                relation_strength=1.0,
                source_chunk_id=["chunk-1"],
            ),
            Relation(
                id="rel-2",
                subject_id="ent-1",
                object_id="ent-2",
                subject_name="ent-1",
                object_name="ent-2",
                relation_type="KNOWS",
                description="A met B",
                relation_strength=2.0,
                source_chunk_id=["chunk-2"],
            ),
        ]
    )
    result = await summarizer.summarize_relations(grouped)

    assert len(result) == 1
    assert result[0].description == "LLM relation summary"
    client.batch_chat_completion.assert_awaited_once()


@pytest.mark.asyncio
async def test_entity_summarizer_without_llm_client_at_all():
    """Regression: llm=None with disabled summarization must not raise."""
    summarizer = EntitySummarizer(llm=None, use_llm_summarization=False)
    entities = [
        Entity(
            entity_name="Alice",
            entity_type="Person",
            description=f"Description {i}",
            source_chunk_id=[f"chunk-{i}"],
            documents_id=[],
            clusters=[],
        )
        for i in range(3)
    ]

    result = await summarizer.run(entities)

    assert len(result) == 1
    assert "Description 0" in result[0].description
    assert "Description 2" in result[0].description


@pytest.mark.asyncio
async def test_relation_summarizer_without_llm_client_at_all():
    """Regression: llm=None with disabled summarization must not raise."""
    summarizer = RelationSummarizer(llm=None, use_llm_summarization=False)
    relations = [
        Relation(
            subject_id="ent-s",
            object_id="ent-o",
            subject_name="S",
            object_name="O",
            relation_type="KNOWS",
            description=f"Description {i}",
            source_chunk_id=[f"chunk-{i}"],
        )
        for i in range(3)
    ]

    result = await summarizer.run(relations)

    assert len(result) == 1
    assert "Description 0" in result[0].description
