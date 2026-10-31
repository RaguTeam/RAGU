"""
Tests for high-level entity and relation merge policies.
"""

from unittest.mock import AsyncMock

import pytest

from ragu.graph.knowledge_graph import default_merge_entities_policy, default_merge_relations_policy
from ragu.graph.types import Entity, Relation
from ragu.models.embedder import Embedder


@pytest.fixture
def mock_embedder():
    """Create a mock embedder."""
    embedder = AsyncMock(spec=Embedder)
    embedder.dim = 128
    embedder.embed_text = AsyncMock(return_value=[0.1] * 128)
    return embedder


def test_merge_entities_no_duplicates():
    """Test merging when there are no duplicates."""
    entity1 = Entity(
        id="ent-1",
        entity_name="Alice",
        entity_type="Person",
        description="Description 1",
        source_chunk_id=["chunk-1"],
        documents_id=["doc-1"],
        clusters=[],
    )

    merged = default_merge_entities_policy([entity1])

    assert merged.id == "ent-1"
    assert merged.description == "Description 1"


def test_merge_entities_with_duplicates():
    """Test merging entities with duplicates."""
    entity1 = Entity(
        id="ent-1",
        entity_name="Alice",
        entity_type="Person",
        description="Software engineer",
        source_chunk_id=["chunk-1", "chunk-2"],
        documents_id=["doc-1"],
        clusters=[{"level": 0, "cluster_id": 1}],
    )
    entity2 = Entity(
        id="ent-1",
        entity_name="Alice",
        entity_type="Person",
        description="Works at Acme Corp",
        source_chunk_id=["chunk-3"],
        documents_id=["doc-2"],
        clusters=[],
    )

    merged_entity = default_merge_entities_policy([entity1, entity2])

    # Should use primary ID (from entity with most chunks)
    assert merged_entity.id == "ent-1"

    # Should merge descriptions
    assert "Software engineer" in merged_entity.description
    assert "Works at Acme Corp" in merged_entity.description

    # Should union source chunks
    assert set(merged_entity.source_chunk_id) == {"chunk-1", "chunk-2", "chunk-3"}

    # Should union documents
    assert set(merged_entity.documents_id) == {"doc-1", "doc-2"}


def test_merge_entities_sorts_by_richness():
    """Test that merge uses entity with most chunks as primary."""
    entity1 = Entity(
        id="ent-1",
        entity_name="Alice",
        entity_type="Person",
        description="Description 1",
        source_chunk_id=["chunk-1"],
        documents_id=["doc-1"],
        clusters=[],
    )
    entity2 = Entity(
        id="ent-1",
        entity_name="Alice",
        entity_type="Person",
        description="Description 2",
        source_chunk_id=["chunk-2", "chunk-3", "chunk-4"],
        documents_id=["doc-2"],
        clusters=[],
    )

    merged = default_merge_entities_policy([entity1, entity2])

    # Should keep the shared ID while using the richer payload as primary.
    assert merged.id == "ent-1"


def test_merge_relations_no_duplicates():
    """Test merging relations with no duplicates."""
    rel1 = Relation(
        id="rel-1",
        subject_id="ent-1",
        object_id="ent-2",
        subject_name="Alice",
        object_name="Bob",
        relation_type="KNOWS",
        description="Alice knows Bob",
        relation_strength=1.0,
        source_chunk_id=["chunk-1"],
    )

    merged = default_merge_relations_policy([rel1])

    assert merged.id == "rel-1"
    assert merged.description == "Alice knows Bob"


def test_merge_relations_with_duplicates():
    """Test merging duplicate relations."""
    rel1 = Relation(
        id="rel-1",
        subject_id="ent-1",
        object_id="ent-2",
        subject_name="Alice",
        object_name="Bob",
        relation_type="WORKS_WITH",
        description="They work together",
        relation_strength=1.0,
        source_chunk_id=["chunk-1", "chunk-2"],
    )
    rel2 = Relation(
        id="rel-1",
        subject_id="ent-1",
        object_id="ent-2",
        subject_name="Alice",
        object_name="Bob",
        relation_type="WORKS_WITH",
        description="Colleagues on the same project",
        relation_strength=0.8,
        source_chunk_id=["chunk-3"],
    )

    merged_rel = default_merge_relations_policy([rel1, rel2])

    # Should use primary ID
    assert merged_rel.id == "rel-1"

    # Should merge descriptions
    assert "They work together" in merged_rel.description
    assert "Colleagues on the same project" in merged_rel.description

    # Should average strength
    assert merged_rel.relation_strength == pytest.approx(0.9)

    # Should union source chunks
    assert set(merged_rel.source_chunk_id) == {"chunk-1", "chunk-2", "chunk-3"}


def test_merge_entities_deduplicates_descriptions():
    """Test that duplicate descriptions are not repeated."""
    entity1 = Entity(
        id="ent-1",
        entity_name="Alice",
        entity_type="Person",
        description="Software engineer",
        source_chunk_id=["chunk-1"],
        documents_id=["doc-1"],
        clusters=[],
    )
    entity2 = Entity(
        id="ent-1",
        entity_name="Alice",
        entity_type="Person",
        description="Software engineer",
        source_chunk_id=["chunk-2"],
        documents_id=["doc-1"],
        clusters=[],
    )

    merged = default_merge_entities_policy([entity1, entity2])

    assert merged.description == "Software engineer"


def test_merge_relations_deduplicates_descriptions():
    """Test that duplicate relation descriptions are not repeated."""
    rel1 = Relation(
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
    rel2 = Relation(
        id="rel-1",
        subject_id="ent-1",
        object_id="ent-2",
        subject_name="Alice",
        object_name="Bob",
        relation_type="KNOWS",
        description="Friends",
        relation_strength=1.0,
        source_chunk_id=["chunk-2"],
    )

    merged = default_merge_relations_policy([rel1, rel2])

    assert merged.description == "Friends"


def test_merge_is_deterministic():
    """Test that merge produces consistent results."""
    entity1 = Entity(
        id="ent-1",
        entity_name="Alice",
        entity_type="Person",
        description="Desc A",
        source_chunk_id=["chunk-1", "chunk-2"],
        documents_id=["doc-1"],
        clusters=[],
    )
    entity2 = Entity(
        id="ent-1",
        entity_name="Alice",
        entity_type="Person",
        description="Desc B",
        source_chunk_id=["chunk-3"],
        documents_id=["doc-2"],
        clusters=[],
    )

    merged1 = default_merge_entities_policy([entity1, entity2])
    merged2 = default_merge_entities_policy([entity1, entity2])

    assert merged1.id == merged2.id
    assert merged1.description == merged2.description
    assert merged1.source_chunk_id == merged2.source_chunk_id


def test_merge_entities_deduplicates_description_fragments():
    entity1 = Entity(
        id="ent-1",
        entity_name="Alice",
        entity_type="Person",
        description="Software engineer. Works at Acme.",
        source_chunk_id=["chunk-1"],
        documents_id=["doc-1"],
        clusters=[],
    )
    entity2 = Entity(
        id="ent-1",
        entity_name="Alice",
        entity_type="Person",
        description="Software engineer.",
        source_chunk_id=["chunk-2"],
        documents_id=["doc-2"],
        clusters=[],
    )

    merged = default_merge_entities_policy([entity1, entity2])

    assert merged.description.count("Software engineer.") == 1


def test_merge_relations_deduplicates_description_fragments():
    rel1 = Relation(
        id="rel-1",
        subject_id="ent-1",
        object_id="ent-2",
        subject_name="Alice",
        object_name="Bob",
        relation_type="KNOWS",
        description="Friends. Work together.",
        relation_strength=1.0,
        source_chunk_id=["chunk-1"],
    )
    rel2 = Relation(
        id="rel-1",
        subject_id="ent-1",
        object_id="ent-2",
        subject_name="Alice",
        object_name="Bob",
        relation_type="KNOWS",
        description="Friends.",
        relation_strength=1.0,
        source_chunk_id=["chunk-2"],
    )

    merged = default_merge_relations_policy([rel1, rel2])

    assert merged.description.count("Friends.") == 1


def test_merge_entities_rejects_different_ids():
    entity1 = Entity(
        id="ent-1",
        entity_name="Alice",
        entity_type="Person",
        description="Description 1",
        source_chunk_id=["chunk-1"],
        documents_id=["doc-1"],
        clusters=[],
    )
    entity2 = Entity(
        id="ent-2",
        entity_name="Alice",
        entity_type="Person",
        description="Description 2",
        source_chunk_id=["chunk-2"],
        documents_id=["doc-2"],
        clusters=[],
    )

    with pytest.raises(ValueError, match="different IDs"):
        default_merge_entities_policy([entity1, entity2])


def test_merge_relations_rejects_different_ids():
    rel1 = Relation(
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
    rel2 = Relation(
        id="rel-2",
        subject_id="ent-1",
        object_id="ent-2",
        subject_name="Alice",
        object_name="Bob",
        relation_type="KNOWS",
        description="Coworkers",
        relation_strength=1.0,
        source_chunk_id=["chunk-2"],
    )

    with pytest.raises(ValueError, match="different IDs"):
        default_merge_relations_policy([rel1, rel2])


def test_merge_entities_merges_metainfo_richest_wins():
    primary = Entity(
        id="ent-1",
        entity_name="Alice",
        entity_type="Person",
        description="Engineer",
        source_chunk_id=["chunk-1", "chunk-2"],
        metainfo={"source": "wiki", "confidence": 0.9},
    )
    secondary = Entity(
        id="ent-1",
        entity_name="Alice",
        entity_type="Person",
        description="Researcher",
        source_chunk_id=["chunk-3"],
        metainfo={"source": "news", "lang": "en"},
    )

    merged = default_merge_entities_policy([secondary, primary])

    assert merged.metainfo == {"source": "wiki", "confidence": 0.9, "lang": "en"}
    # Inputs are not mutated.
    assert primary.metainfo == {"source": "wiki", "confidence": 0.9}
    assert secondary.metainfo == {"source": "news", "lang": "en"}


def test_merge_relations_merges_metainfo_richest_wins():
    primary = Relation(
        id="rel-1",
        subject_id="ent-1",
        object_id="ent-2",
        subject_name="Alice",
        object_name="Bob",
        relation_type="KNOWS",
        description="Alice knows Bob",
        source_chunk_id=["chunk-1", "chunk-2"],
        metainfo={"evidence": "strong"},
    )
    secondary = Relation(
        id="rel-1",
        subject_id="ent-1",
        object_id="ent-2",
        subject_name="Alice",
        object_name="Bob",
        relation_type="KNOWS",
        description="They met",
        source_chunk_id=["chunk-3"],
        metainfo={"evidence": "weak", "year": 2020},
    )

    merged = default_merge_relations_policy([secondary, primary])

    assert merged.metainfo == {"evidence": "strong", "year": 2020}
