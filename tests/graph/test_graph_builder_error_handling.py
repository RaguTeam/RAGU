import os
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Tuple

import pytest

from ragu.chunker.types import Chunk
from ragu.common.global_parameters import Settings
from ragu.graph.graph_builder_pipeline import InMemoryGraphBuilder, BuilderArguments
from ragu.graph.types import Entity, Relation, Community, CommunitySummary
from ragu.models.embedder import Embedder
from ragu.models.llm import LLM


def _make_entity(name="Alice", etype="Person"):
    return Entity(
        entity_name=name,
        entity_type=etype,
        description=f"Description of {name}",
        source_chunk_id=["chunk-1"],
        documents_id=[],
        clusters=[],
    )


def _make_relation(subject="Alice", obj="Bob"):
    s = _make_entity(subject)
    o = _make_entity(obj)
    return Relation(
        subject_id=s.id,
        object_id=o.id,
        subject_name=s.entity_name,
        object_name=o.entity_name,
        relation_type="KNOWS",
        description=f"{subject} knows {obj}",
        source_chunk_id=["chunk-1"],
    )


def _make_chunk(text="Hello world"):
    return Chunk(content=text, chunk_order_idx=0, doc_id="doc-1")


def _make_builder(tmp_path, extract_side_effect=None, summarize_entity_side_effect=None):
    mock_embedder = AsyncMock(spec=Embedder)
    mock_embedder.batch_embed_text = AsyncMock(return_value=[[0.1] * 128])
    mock_embedder.embed_text = AsyncMock(return_value=[0.1] * 128)

    mock_llm = AsyncMock(spec=LLM)
    mock_llm.batch_chat_completion = AsyncMock(return_value=[])
    mock_llm.chat_completion = AsyncMock(return_value="")

    extractor = AsyncMock()
    if extract_side_effect:
        extractor.side_effect = extract_side_effect
    else:
        extractor.return_value = ([_make_entity(), _make_entity("Bob", "Person")], [_make_relation()])

    builder = InMemoryGraphBuilder(
        embedder=mock_embedder,
        llm=mock_llm,
        chunker=MagicMock(),
        artifact_extractor=extractor,
        build_parameters=BuilderArguments(
            use_llm_summarization=True,
            make_community_summary=True,
            remove_isolated_nodes=False,
        ),
    )

    entity_summarizer = AsyncMock()
    if summarize_entity_side_effect:
        entity_summarizer.run.side_effect = summarize_entity_side_effect
    else:
        entity_summarizer.run.side_effect = lambda entities: entities

    relation_summarizer = AsyncMock()
    relation_summarizer.run.side_effect = lambda relations: relations

    community_summarizer = AsyncMock()
    community_summarizer.summarize.return_value = []

    builder.artifact_extractor = extractor
    builder.entity_summarizer = entity_summarizer
    builder.relation_summarizer = relation_summarizer
    builder.community_summarizer = community_summarizer

    return builder


async def test_extract_graph_extraction_failure(tmp_path):
    builder = _make_builder(
        tmp_path,
        extract_side_effect=RuntimeError("LLM timeout"),
    )

    entities, relations, summaries, communities, out_chunks = await builder.extract_graph(
        [_make_chunk()]
    )

    assert entities == []
    assert relations == []
    assert summaries == []
    assert communities == []


async def test_extract_graph_entity_summarization_failure(tmp_path):
    entities = [_make_entity(), _make_entity("Bob", "Person")]
    relations = [_make_relation()]

    builder = _make_builder(tmp_path)
    builder.entity_summarizer.run.side_effect = RuntimeError("summarization failed")
    builder.artifact_extractor.return_value = (entities, relations)

    with patch.object(builder, 'cluster_graph', return_value=[]):
        out_entities, out_relations, summaries, communities, _ = await builder.extract_graph(
            [_make_chunk()]
        )

    assert out_entities == entities
    assert out_relations == relations


async def test_extract_graph_community_summarization_failure(tmp_path):
    entities = [_make_entity(), _make_entity("Bob", "Person")]
    relations = [_make_relation()]

    builder = _make_builder(tmp_path)
    builder.artifact_extractor.return_value = (entities, relations)
    builder.community_summarizer.summarize.side_effect = RuntimeError("community failed")

    mock_community = Community(
        level=1, cluster_id=1, entities=entities, relations=relations,
    )
    with patch.object(builder, 'cluster_graph', return_value=[mock_community]):
        out_entities, out_relations, summaries, communities, _ = await builder.extract_graph(
            [_make_chunk()]
        )

    assert communities == [mock_community]
    assert summaries == []


async def test_extract_graph_all_succeed(tmp_path):
    entities = [_make_entity(), _make_entity("Bob", "Person")]
    relations = [_make_relation()]

    builder = _make_builder(tmp_path)
    builder.artifact_extractor.return_value = (entities, relations)

    mock_summary = CommunitySummary(id="summary-1", summary="A community summary")
    builder.community_summarizer.summarize.return_value = [mock_summary]

    mock_community = Community(
        level=1, cluster_id=1, entities=entities, relations=relations,
    )
    with patch.object(builder, 'cluster_graph', return_value=[mock_community]):
        out_entities, out_relations, summaries, communities, _ = await builder.extract_graph(
            [_make_chunk()]
        )

    assert len(out_entities) == 2
    assert len(out_relations) == 1
    assert len(summaries) == 1
    assert len(communities) == 1
