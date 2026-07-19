from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ragu.chunker.types import Chunk
from ragu.graph.graph_builder_pipeline import (
    BuildResult,
    BuilderArguments,
    GraphBuilderModule,
    InMemoryGraphBuilder,
)
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


def _make_builder(tmp_path, extract_side_effect=None):
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


async def test_extract_graph_extraction_failure_propagates(tmp_path):
    builder = _make_builder(
        tmp_path,
        extract_side_effect=RuntimeError("LLM timeout"),
    )

    with pytest.raises(RuntimeError, match="LLM timeout"):
        await builder.extract_graph([_make_chunk()])


async def test_extract_graph_entity_summarization_failure_propagates(tmp_path):
    builder = _make_builder(tmp_path)
    builder.entity_summarizer.run.side_effect = RuntimeError("summarization failed")

    with pytest.raises(RuntimeError, match="summarization failed"):
        await builder.extract_graph([_make_chunk()])


async def test_extract_graph_community_summarization_failure_propagates(tmp_path):
    entities = [_make_entity(), _make_entity("Bob", "Person")]
    relations = [_make_relation()]

    builder = _make_builder(tmp_path)
    builder.artifact_extractor.return_value = (entities, relations)
    builder.community_summarizer.summarize.side_effect = RuntimeError("community failed")

    mock_community = Community(
        level=1, cluster_id=1, entities=entities, relations=relations,
    )
    with patch.object(builder, 'cluster_graph', return_value=[mock_community]):
        with pytest.raises(RuntimeError, match="community failed"):
            await builder.extract_graph([_make_chunk()])


async def test_extract_graph_missing_extractor_raises_value_error(tmp_path):
    builder = _make_builder(tmp_path)
    builder.artifact_extractor = None

    with pytest.raises(ValueError, match="artifact_extractor"):
        await builder.extract_graph([_make_chunk()])


async def test_extract_graph_vector_only_needs_no_extractor(tmp_path):
    mock_embedder = AsyncMock(spec=Embedder)
    builder = InMemoryGraphBuilder(
        embedder=mock_embedder,
        artifact_extractor=None,
        build_parameters=BuilderArguments(build_only_vector_context=True),
    )

    chunks = [_make_chunk()]
    result = await builder.extract_graph(chunks)

    assert isinstance(result, BuildResult)
    assert result.entities == []
    assert result.relations == []
    assert result.chunks == chunks
    assert result.report.chunks_processed == 1


async def test_extract_graph_module_returning_none_raises_type_error(tmp_path):
    class _BadModule(GraphBuilderModule):
        async def run(self, entities, relations, **kwargs):
            return None

    builder = _make_builder(tmp_path)
    builder.additional_pipeline = [_BadModule()]

    with patch.object(builder, 'cluster_graph', return_value=[]):
        with pytest.raises(TypeError, match="_BadModule"):
            await builder.extract_graph([_make_chunk()])


def test_graph_builder_module_is_abstract():
    with pytest.raises(TypeError):
        GraphBuilderModule()


async def test_extract_graph_all_succeed_returns_report(tmp_path):
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
        result = await builder.extract_graph([_make_chunk()])

    assert len(result.entities) == 2
    assert len(result.relations) == 1
    assert len(result.summaries) == 1
    assert len(result.communities) == 1

    report = result.report
    assert report.chunks_processed == 1
    assert report.entities_extracted == 2
    assert report.relations_extracted == 1
    assert report.entities_final == 2
    assert report.relations_final == 1
    assert report.communities_detected == 1
    assert report.community_summaries_generated == 1
    assert report.community_summaries_failed == 0
    assert "entities: 2 extracted -> 2 final" in report.to_text()
