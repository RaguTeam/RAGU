"""
Tests for KnowledgeGraph high-level merge behavior.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ragu.chunker.types import Chunk
from ragu.common.global_parameters import Settings
from ragu.common.prompts.default_models import EntityDescriptionModel, RelationDescriptionModel
from ragu.graph.graph_builder_pipeline import BuilderArguments
from ragu.graph.knowledge_graph import KnowledgeGraph
from ragu.graph.types import Community, CommunitySummary, Entity, Relation
from ragu.models.embedder import Embedder


@pytest.fixture
def mock_embedder():
    embedder = AsyncMock(spec=Embedder)
    embedder.dim = 128
    embedder.embed_text = AsyncMock(return_value=[0.1] * 128)
    embedder.batch_embed_text = AsyncMock(
        side_effect=lambda texts, **kwargs: [[0.1] * 128 for _ in texts]
    )
    return embedder


@pytest.fixture
def builder_settings():
    return BuilderArguments(
        use_llm_summarization=False,
        make_community_summary=False,
        remove_isolated_nodes=False,
    )


@pytest.fixture
def kg(tmp_path, monkeypatch, mock_embedder, builder_settings):
    monkeypatch.setattr(Settings, "storage_folder", str(tmp_path / "storage"))
    return KnowledgeGraph(
        llm=None,
        embedder=mock_embedder,
        builder_settings=builder_settings,
    )


@pytest.mark.asyncio
async def test_insert_entities_merges_with_existing_entity(kg):
    original = Entity(
        entity_name="Alice",
        entity_type="Person",
        description="First description",
        source_chunk_id=["chunk-1"],
        documents_id=["doc-1"],
        clusters=[],
    )
    incoming = Entity(
        entity_name="Alice",
        entity_type="Person",
        description="Second description",
        source_chunk_id=["chunk-2"],
        documents_id=["doc-2"],
        clusters=[],
    )

    await kg.upsert_entities([original])
    await kg.upsert_entities([incoming])

    stored = await kg.index.get_nodes([original.id])

    assert stored[0] is not None
    assert "First description" in stored[0].description
    assert "Second description" in stored[0].description
    assert set(stored[0].source_chunk_id) == {"chunk-1", "chunk-2"}
    assert set(stored[0].documents_id) == {"doc-1", "doc-2"}


@pytest.mark.asyncio
async def test_reindex_descriptions_summarizes_only_long_descriptions(tmp_path, monkeypatch, mock_embedder):
    monkeypatch.setattr(Settings, "storage_folder", str(tmp_path / "storage"))
    llm = AsyncMock()

    async def batch_chat_completion(conversations, output_schema, **kwargs):
        if output_schema is EntityDescriptionModel:
            return [
                EntityDescriptionModel(entity_name="Alice", description="Summarized Alice"),
            ]
        if output_schema is RelationDescriptionModel:
            return [
                RelationDescriptionModel(
                    subject_name="Alice",
                    object_name="Bob",
                    description="Summarized relation",
                ),
            ]
        raise AssertionError(f"Unexpected output schema: {output_schema}")

    llm.batch_chat_completion = AsyncMock(side_effect=batch_chat_completion)
    kg = KnowledgeGraph(
        llm=llm,
        embedder=mock_embedder,
        builder_settings=BuilderArguments(
            use_llm_summarization=True,
            make_community_summary=False,
            remove_isolated_nodes=False,
        ),
    )

    long_entity = Entity(
        id="ent-alice",
        entity_name="Alice",
        entity_type="Person",
        description="Alice is an engineer. She works on graphs.",
        source_chunk_id=["chunk-1"],
        documents_id=["doc-1"],
        clusters=[{"level": 1, "cluster_id": 7}],
    )
    short_entity = Entity(
        id="ent-bob",
        entity_name="Bob",
        entity_type="Person",
        description="Bob is a scientist.",
        source_chunk_id=["chunk-2"],
        documents_id=["doc-2"],
    )
    long_relation = Relation(
        id="rel-alice-bob",
        subject_id=long_entity.id,
        object_id=short_entity.id,
        subject_name="Alice",
        object_name="Bob",
        relation_type="KNOWS",
        description="Alice knows Bob. They work together.",
        relation_strength=0.8,
        source_chunk_id=["chunk-1", "chunk-2"],
    )
    short_relation = Relation(
        id="rel-bob-alice",
        subject_id=short_entity.id,
        object_id=long_entity.id,
        subject_name="Bob",
        object_name="Alice",
        relation_type="KNOWS",
        description="Bob knows Alice.",
        relation_strength=0.4,
        source_chunk_id=["chunk-2"],
    )

    await kg.upsert_entities([long_entity, short_entity])
    await kg.upsert_relations([long_relation, short_relation])

    result = await kg.reindex_descriptions(summarize_only_more_than=1)

    assert result is kg
    stored_entities = await kg.index.get_nodes([long_entity.id, short_entity.id])
    assert stored_entities[0] is not None
    assert stored_entities[1] is not None
    assert stored_entities[0].description == "Summarized Alice"
    assert stored_entities[0].clusters == [{"level": 1, "cluster_id": 7}]
    assert stored_entities[1].description == "Bob is a scientist."

    stored_relations = await kg.index.get_edges([
        (long_relation.subject_id, long_relation.object_id, long_relation.id),
        (short_relation.subject_id, short_relation.object_id, short_relation.id),
    ])
    assert stored_relations[0] is not None
    assert stored_relations[1] is not None
    assert stored_relations[0].description == "Summarized relation"
    assert stored_relations[0].relation_strength == pytest.approx(0.8)
    assert stored_relations[1].description == "Bob knows Alice."

    entity_payloads = await kg.index.nodes_vector_db.get_payloads_by_ids([
        long_entity.id,
        short_entity.id,
    ])
    relation_payloads = await kg.index.edges_vector_db.get_payloads_by_ids([
        long_relation.id,
        short_relation.id,
    ])

    assert entity_payloads[0]["description"] == "Summarized Alice"
    assert entity_payloads[0]["clusters"] == [{"level": 1, "cluster_id": 7}]
    assert entity_payloads[1]["description"] == "Bob is a scientist."
    assert relation_payloads[0]["description"] == "Summarized relation"
    assert relation_payloads[1]["description"] == "Bob knows Alice."

    assert llm.batch_chat_completion.await_count == 2


@pytest.mark.asyncio
async def test_reindex_community_replaces_stale_records_and_persists_clusters_to_vectors(kg):
    alice = Entity(
        id="ent-alice",
        entity_name="Alice",
        entity_type="Person",
        description="Alice",
        source_chunk_id=["chunk-1"],
        documents_id=["doc-1"],
        clusters=[{"level": 99, "cluster_id": 99}],
    )
    bob = Entity(
        id="ent-bob",
        entity_name="Bob",
        entity_type="Person",
        description="Bob",
        source_chunk_id=["chunk-1"],
        documents_id=["doc-1"],
        clusters=[],
    )
    relation = Relation(
        id="rel-alice-bob",
        subject_id=alice.id,
        object_id=bob.id,
        subject_name="Alice",
        object_name="Bob",
        relation_type="KNOWS",
        description="Alice knows Bob.",
        source_chunk_id=["chunk-1"],
    )

    async def summarize(communities):
        return [
            CommunitySummary(id=community.id, summary=f"Summary for {community.id}")
            for community in communities
        ]

    # KnowledgeGraph(llm=None) wires no community summarizer, so stand one in.
    kg.pipeline.community_summarizer = SimpleNamespace(
        summarize=AsyncMock(side_effect=summarize)
    )

    await kg.upsert_entities([alice, bob])
    await kg.upsert_relations([relation])
    await kg.upsert_communities([
        Community(
            id="com-stale",
            level=0,
            cluster_id=99,
            entities=[alice],
            relations=[],
        )
    ])
    await kg.upsert_summaries([
        CommunitySummary(id="com-stale", summary="Stale summary")
    ])

    await kg.reindex_community()

    stored_entities = await kg.get_entities([alice.id, bob.id])
    stale_communities = await kg.get_communities(["com-stale"])
    stale_summaries = await kg.get_summaries(["com-stale"])

    assert stored_entities[0] is not None
    assert stored_entities[1] is not None
    assert stored_entities[0].clusters
    assert stored_entities[1].clusters
    assert stored_entities[0].clusters != [{"level": 99, "cluster_id": 99}]
    assert stale_communities[0] is None
    assert stale_summaries[0] is None

    community_id = Community(
        level=stored_entities[0].clusters[0]["level"],
        cluster_id=stored_entities[0].clusters[0]["cluster_id"],
        entities=[],
        relations=[],
    ).id
    communities = await kg.get_communities([community_id])
    summaries = await kg.get_summaries([community_id])

    assert communities[0] is not None
    assert summaries[0] is not None
    assert summaries[0].id == communities[0].id
    assert {entity.id for entity in communities[0].entities} == {alice.id, bob.id}
    assert {item.id for item in communities[0].relations} == {relation.id}

    entity_payloads = await kg.index.nodes_vector_db.get_payloads_by_ids([
        alice.id,
        bob.id,
    ])
    assert entity_payloads[0]["clusters"] == stored_entities[0].clusters
    assert entity_payloads[1]["clusters"] == stored_entities[1].clusters


@pytest.mark.asyncio
async def test_reindex_community_keeps_existing_communities_when_summarization_fails(kg):
    alice = Entity(
        id="ent-alice",
        entity_name="Alice",
        entity_type="Person",
        description="Alice",
        source_chunk_id=["chunk-1"],
        documents_id=["doc-1"],
        clusters=[],
    )
    bob = Entity(
        id="ent-bob",
        entity_name="Bob",
        entity_type="Person",
        description="Bob",
        source_chunk_id=["chunk-1"],
        documents_id=["doc-1"],
        clusters=[],
    )
    relation = Relation(
        id="rel-alice-bob",
        subject_id=alice.id,
        object_id=bob.id,
        subject_name="Alice",
        object_name="Bob",
        relation_type="KNOWS",
        description="Alice knows Bob.",
        source_chunk_id=["chunk-1"],
    )

    await kg.upsert_entities([alice, bob])
    await kg.upsert_relations([relation])
    await kg.index.upsert_communities([
        Community(
            id="com-existing",
            level=0,
            cluster_id=7,
            entities=[alice, bob],
            relations=[relation],
        )
    ])
    await kg.upsert_summaries([
        CommunitySummary(id="com-existing", summary="Existing summary")
    ])

    kg.pipeline.community_summarizer = SimpleNamespace(
        summarize=AsyncMock(side_effect=RuntimeError("LLM failed"))
    )

    with pytest.raises(RuntimeError, match="LLM failed"):
        await kg.reindex_community()

    communities = await kg.get_communities(["com-existing"])
    summaries = await kg.get_summaries(["com-existing"])

    assert communities[0] is not None
    assert summaries[0] is not None
    assert summaries[0].summary == "Existing summary"


@pytest.mark.asyncio
async def test_insert_relations_merges_with_existing_relation(kg):
    await kg.upsert_entities([
        Entity(
            id="ent-1",
            entity_name="Alice",
            entity_type="Person",
            description="Alice",
            source_chunk_id=["chunk-1"],
            documents_id=["doc-1"],
            clusters=[],
        ),
        Entity(
            id="ent-2",
            entity_name="Bob",
            entity_type="Person",
            description="Bob",
            source_chunk_id=["chunk-1"],
            documents_id=["doc-1"],
            clusters=[],
        ),
    ])

    original = Relation(
        id="rel-1",
        subject_id="ent-1",
        object_id="ent-2",
        subject_name="Alice",
        object_name="Bob",
        relation_type="KNOWS",
        description="Friends",
        relation_strength=1.0,
        source_chunk_id=["chunk-1"],
    )
    incoming = Relation(
        id="rel-1",
        subject_id="ent-1",
        object_id="ent-2",
        subject_name="Alice",
        object_name="Bob",
        relation_type="KNOWS",
        description="Work together",
        relation_strength=0.4,
        source_chunk_id=["chunk-2"],
    )

    await kg.upsert_relations([original])
    await kg.upsert_relations([incoming])

    stored = await kg.index.get_edges([("ent-1", "ent-2", "rel-1")])

    assert stored[0] is not None
    assert "Friends" in stored[0].description
    assert "Work together" in stored[0].description
    assert stored[0].relation_strength == pytest.approx(0.7)
    assert set(stored[0].source_chunk_id) == {"chunk-1", "chunk-2"}


@pytest.mark.asyncio
async def test_build_from_docs_uses_knowledge_graph_merge_path(kg):
    original = Entity(
        entity_name="Alice",
        entity_type="Person",
        description="Stored description",
        source_chunk_id=["chunk-old"],
        documents_id=["doc-old"],
        clusters=[],
    )
    await kg.upsert_entities([original])

    extracted = Entity(
        entity_name="Alice",
        entity_type="Person",
        description="Extracted description",
        source_chunk_id=["chunk-new"],
        documents_id=["doc-new"],
        clusters=[],
    )

    async def fake_extract_graph(chunks):
        from ragu.graph.graph_builder_pipeline import BuildReport, BuildResult
        return BuildResult(
            entities=[extracted],
            relations=[],
            summaries=[],
            communities=[],
            chunks=chunks,
            report=BuildReport(chunks_processed=len(chunks)),
        )

    kg.pipeline.extract_graph = fake_extract_graph

    await kg.build_from_docs(["Alice is mentioned in the new document."])

    stored = await kg.index.get_nodes([original.id])

    assert stored[0] is not None
    assert "Stored description" in stored[0].description
    assert "Extracted description" in stored[0].description
    assert set(stored[0].source_chunk_id) == {"chunk-old", "chunk-new"}
    assert set(stored[0].documents_id) == {"doc-old", "doc-new"}


@pytest.mark.asyncio
async def test_insert_entities_rejects_duplicate_ids_in_one_request(kg):
    entity = Entity(
        id="ent-1",
        entity_name="Alice",
        entity_type="Person",
        description="Alice",
        source_chunk_id=["chunk-1"],
        documents_id=["doc-1"],
        clusters=[],
    )

    with pytest.raises(ValueError, match="duplicated entity IDs"):
        await kg.upsert_entities([entity, Entity(
            id="ent-1",
            entity_name="Alice",
            entity_type="Person",
            description="Alice duplicate",
            source_chunk_id=["chunk-2"],
            documents_id=["doc-2"],
            clusters=[],
        )])


@pytest.mark.asyncio
async def test_update_entities_rejects_duplicate_ids_in_one_request(kg):
    entity = Entity(
        id="ent-1",
        entity_name="Alice",
        entity_type="Person",
        description="Alice",
        source_chunk_id=["chunk-1"],
        documents_id=["doc-1"],
        clusters=[],
    )
    await kg.upsert_entities([entity])

    with pytest.raises(ValueError, match="duplicated entity IDs"):
        await kg.update_entities([entity, Entity(
            id="ent-1",
            entity_name="Alice Updated",
            entity_type="Person",
            description="Alice updated",
            source_chunk_id=["chunk-2"],
            documents_id=["doc-2"],
            clusters=[],
        )])


@pytest.mark.asyncio
async def test_insert_relations_rejects_duplicate_ids_in_one_request(kg):
    await kg.upsert_entities([
        Entity(
            id="ent-1",
            entity_name="Alice",
            entity_type="Person",
            description="Alice",
            source_chunk_id=["chunk-1"],
            documents_id=["doc-1"],
            clusters=[],
        ),
        Entity(
            id="ent-2",
            entity_name="Bob",
            entity_type="Person",
            description="Bob",
            source_chunk_id=["chunk-1"],
            documents_id=["doc-1"],
            clusters=[],
        ),
    ])

    relation = Relation(
        id="rel-1",
        subject_id="ent-1",
        object_id="ent-2",
        subject_name="Alice",
        object_name="Bob",
        relation_type="KNOWS",
        description="Friends",
        relation_strength=1.0,
        source_chunk_id=["chunk-1"],
    )

    with pytest.raises(ValueError, match="duplicated relation IDs"):
        await kg.upsert_relations([relation, Relation(
            id="rel-1",
            subject_id="ent-1",
            object_id="ent-2",
            subject_name="Alice",
            object_name="Bob",
            relation_type="KNOWS",
            description="Coworkers",
            relation_strength=0.5,
            source_chunk_id=["chunk-2"],
        )])


@pytest.mark.asyncio
async def test_update_relations_rejects_duplicate_ids_in_one_request(kg):
    await kg.upsert_entities([
        Entity(
            id="ent-1",
            entity_name="Alice",
            entity_type="Person",
            description="Alice",
            source_chunk_id=["chunk-1"],
            documents_id=["doc-1"],
            clusters=[],
        ),
        Entity(
            id="ent-2",
            entity_name="Bob",
            entity_type="Person",
            description="Bob",
            source_chunk_id=["chunk-1"],
            documents_id=["doc-1"],
            clusters=[],
        ),
    ])
    relation = Relation(
        id="rel-1",
        subject_id="ent-1",
        object_id="ent-2",
        subject_name="Alice",
        object_name="Bob",
        relation_type="KNOWS",
        description="Friends",
        relation_strength=1.0,
        source_chunk_id=["chunk-1"],
    )
    await kg.upsert_relations([relation])

    with pytest.raises(ValueError, match="duplicated relation IDs"):
        await kg.update_relations([relation, Relation(
            id="rel-1",
            subject_id="ent-1",
            object_id="ent-2",
            subject_name="Alice",
            object_name="Bob",
            relation_type="KNOWS",
            description="Updated relation",
            relation_strength=0.7,
            source_chunk_id=["chunk-2"],
        )])


@pytest.mark.asyncio
async def test_get_entities_is_batched(kg):
    await kg.upsert_entities([
        Entity(
            id="ent-1",
            entity_name="Alice",
            entity_type="Person",
            description="Alice",
            source_chunk_id=["chunk-1"],
            documents_id=["doc-1"],
            clusters=[],
        ),
        Entity(
            id="ent-2",
            entity_name="Bob",
            entity_type="Person",
            description="Bob",
            source_chunk_id=["chunk-2"],
            documents_id=["doc-1"],
            clusters=[],
        ),
    ])

    entities = await kg.get_entities(["ent-1", "ent-missing", "ent-2"])

    assert [entity.id if entity is not None else None for entity in entities] == ["ent-1", None, "ent-2"]


@pytest.mark.asyncio
async def test_get_entities_supports_single_item_batch(kg):
    entity = Entity(
        id="ent-1",
        entity_name="Alice",
        entity_type="Person",
        description="Alice",
        source_chunk_id=["chunk-1"],
        documents_id=["doc-1"],
        clusters=[],
    )
    await kg.upsert_entities([entity])

    fetched = await kg.get_entities(["ent-1"])

    assert fetched[0] is not None
    assert fetched[0].id == "ent-1"


@pytest.mark.asyncio
async def test_get_relations_chunks_communities_are_batched(kg):
    await kg.upsert_entities([
        Entity(
            id="ent-1",
            entity_name="Alice",
            entity_type="Person",
            description="Alice",
            source_chunk_id=["chunk-1"],
            documents_id=["doc-1"],
            clusters=[],
        ),
        Entity(
            id="ent-2",
            entity_name="Bob",
            entity_type="Person",
            description="Bob",
            source_chunk_id=["chunk-1"],
            documents_id=["doc-1"],
            clusters=[],
        ),
    ])
    relation = Relation(
        id="rel-1",
        subject_id="ent-1",
        object_id="ent-2",
        subject_name="Alice",
        object_name="Bob",
        relation_type="KNOWS",
        description="Friends",
        relation_strength=1.0,
        source_chunk_id=["chunk-1"],
    )
    await kg.upsert_relations([relation])
    await kg.index.upsert_chunks([
        kg_chunk := Chunk(
            content="Chunk one text",
            chunk_order_idx=0,
            doc_id="doc-1",
        )
    ])
    await kg.index.upsert_communities([
        Community(
            id="com-1",
            level=0,
            cluster_id=1,
            entities=[
                Entity(
                    id="ent-1",
                    entity_name="Alice",
                    entity_type="Person",
                    description="Alice",
                    source_chunk_id=["chunk-1"],
                )
            ],
            relations=[relation],
        )
    ])

    relations = await kg.get_relations([("ent-1", "ent-2", "rel-1")])
    chunks = await kg.get_chunks([kg_chunk.id])
    communities = await kg.get_communities(["com-1"])
    relation = await kg.get_relations([("ent-1", "ent-2", "rel-1")])
    chunk = await kg.get_chunks([kg_chunk.id])
    community = await kg.get_communities(["com-1"])

    assert relations[0] is not None
    assert chunks[0] is not None
    assert communities[0] is not None
    assert relation[0] is not None
    assert chunk[0] is not None
    assert community[0] is not None


@pytest.mark.asyncio
async def test_community_crud_operations_are_exposed_on_knowledge_graph(kg):
    alice = Entity(
        id="ent-1",
        entity_name="Alice",
        entity_type="Person",
        description="Alice",
        source_chunk_id=["chunk-1"],
        documents_id=["doc-1"],
        clusters=[],
    )
    bob = Entity(
        id="ent-2",
        entity_name="Bob",
        entity_type="Person",
        description="Bob",
        source_chunk_id=["chunk-1"],
        documents_id=["doc-1"],
        clusters=[],
    )
    relation = Relation(
        id="rel-1",
        subject_id=alice.id,
        object_id=bob.id,
        subject_name="Alice",
        object_name="Bob",
        relation_type="KNOWS",
        description="Alice knows Bob.",
        source_chunk_id=["chunk-1"],
    )
    community = Community(
        id="com-1",
        level=0,
        cluster_id=1,
        entities=[alice, bob],
        relations=[relation],
    )
    updated_community = Community(
        id="com-1",
        level=1,
        cluster_id=2,
        entities=[alice],
        relations=[],
    )

    await kg.upsert_entities([alice, bob])
    await kg.upsert_relations([relation])

    assert await kg.upsert_communities([community]) is kg

    stored = await kg.get_communities(["com-1"])
    assert stored[0] is not None
    assert stored[0].level == 0
    assert {entity.id for entity in stored[0].entities} == {"ent-1", "ent-2"}
    assert {item.id for item in stored[0].relations} == {"rel-1"}

    assert await kg.update_communities([updated_community]) is kg

    stored = await kg.get_communities(["com-1"])
    assert stored[0] is not None
    assert stored[0].level == 1
    assert stored[0].cluster_id == 2
    assert [entity.id for entity in stored[0].entities] == ["ent-1"]
    assert stored[0].relations == []

    with pytest.raises(ValueError, match="non-existent communities"):
        await kg.update_communities([
            Community(id="com-missing", level=0, cluster_id=9, entities=[], relations=[])
        ])

    await kg.upsert_summaries([CommunitySummary(id="com-1", summary="Community summary")])
    assert await kg.delete_communities(["com-1"]) is kg

    stored = await kg.get_communities(["com-1"])
    summaries = await kg.get_summaries(["com-1"])
    assert stored[0] is None
    assert summaries[0] is None


@pytest.mark.asyncio
async def test_get_summaries_is_batched(kg):
    await kg.upsert_summaries([
        CommunitySummary(id="sum-1", summary="Summary one"),
        CommunitySummary(id="sum-2", summary="Summary two"),
    ])

    summaries = await kg.get_summaries(["sum-1", "sum-missing", "sum-2"])

    assert [summary.id if summary is not None else None for summary in summaries] == ["sum-1", None, "sum-2"]


@pytest.mark.asyncio
async def test_get_summaries_supports_single_item_batch(kg):
    await kg.upsert_summaries([CommunitySummary(id="sum-1", summary="Summary one")])

    summary = await kg.get_summaries(["sum-1"])

    assert summary[0] is not None
    assert summary[0].id == "sum-1"


@pytest.mark.asyncio
async def test_summary_crud_operations_are_exposed_on_knowledge_graph(kg):
    summary = CommunitySummary(id="sum-1", summary="Initial summary")
    updated_summary = CommunitySummary(id="sum-1", summary="Updated summary")

    assert await kg.upsert_summaries([summary]) is kg

    stored = await kg.get_summaries(["sum-1"])
    assert stored[0] is not None
    assert stored[0].summary == "Initial summary"

    assert await kg.update_summaries([updated_summary]) is kg

    stored = await kg.get_summaries(["sum-1"])
    assert stored[0] is not None
    assert stored[0].summary == "Updated summary"

    with pytest.raises(ValueError, match="non-existent summaries"):
        await kg.update_summaries([
            CommunitySummary(id="sum-missing", summary="Missing summary")
        ])

    assert await kg.delete_summaries(["sum-1"]) is kg

    stored = await kg.get_summaries(["sum-1"])
    assert stored[0] is None
