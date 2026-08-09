from unittest.mock import AsyncMock

import pytest

import ragu.graph.graph_builder_pipeline as pipeline_module
from ragu.graph.graph_builder_pipeline import BuilderArguments, InMemoryGraphBuilder
from ragu.graph.types import Entity, Relation
from ragu.models.embedder import Embedder


def _make_entity(name: str) -> Entity:
    return Entity(
        id=f"ent-{name}",
        entity_name=name,
        entity_type="Person",
        description=f"Description of {name}",
        source_chunk_id=["chunk-1"],
        documents_id=["doc-1"],
        clusters=[],
    )


def _make_relation(subject: Entity, obj: Entity) -> Relation:
    return Relation(
        id=f"rel-{subject.entity_name}-{obj.entity_name}",
        subject_id=subject.id,
        object_id=obj.id,
        subject_name=subject.entity_name,
        object_name=obj.entity_name,
        relation_type="KNOWS",
        description=f"{subject.entity_name} knows {obj.entity_name}",
        source_chunk_id=["chunk-1"],
    )


def _make_builder(min_cluster_size: int) -> InMemoryGraphBuilder:
    embedder = AsyncMock(spec=Embedder)
    return InMemoryGraphBuilder(
        embedder=embedder,
        llm=None,
        build_parameters=BuilderArguments(
            use_llm_summarization=False,
            min_cluster_size=min_cluster_size,
        ),
    )


@pytest.fixture
def graph_with_two_clusters(monkeypatch):
    """
    Two disconnected clusters: a three-entity one and a two-entity one.

    ``hierarchical_leiden`` is replaced with a fixed mapping so the assertions
    depend only on the size filter, not on the clustering algorithm.
    """
    alice, bob, carol = _make_entity("Alice"), _make_entity("Bob"), _make_entity("Carol")
    dave, erin = _make_entity("Dave"), _make_entity("Erin")
    entities = [alice, bob, carol, dave, erin]
    relations = [
        _make_relation(alice, bob),
        _make_relation(bob, carol),
        _make_relation(dave, erin),
    ]

    mapping = [
        {"node": alice.id, "cluster": 0, "level": 0},
        {"node": bob.id, "cluster": 0, "level": 0},
        {"node": carol.id, "cluster": 0, "level": 0},
        {"node": dave.id, "cluster": 1, "level": 0},
        {"node": erin.id, "cluster": 1, "level": 0},
    ]
    monkeypatch.setattr(pipeline_module, "hierarchical_leiden", lambda *a, **kw: mapping)

    return entities, relations


async def test_cluster_graph_keeps_all_communities_by_default(graph_with_two_clusters):
    entities, relations = graph_with_two_clusters
    builder = _make_builder(min_cluster_size=1)

    communities = await builder.cluster_graph(entities, relations)

    assert sorted(c.cluster_id for c in communities) == [0, 1]
    assert all(len(entity.clusters) == 1 for entity in entities)


async def test_cluster_graph_drops_communities_below_min_size(graph_with_two_clusters):
    entities, relations = graph_with_two_clusters
    builder = _make_builder(min_cluster_size=3)

    communities = await builder.cluster_graph(entities, relations)

    assert [c.cluster_id for c in communities] == [0]
    assert sorted(e.entity_name for e in communities[0].entities) == ["Alice", "Bob", "Carol"]

    kept_members = {"ent-Alice", "ent-Bob", "ent-Carol"}
    for entity in entities:
        expected = [{"level": 0, "cluster_id": 0}] if entity.id in kept_members else []
        assert entity.clusters == expected


async def test_cluster_graph_can_drop_every_community(graph_with_two_clusters):
    entities, relations = graph_with_two_clusters
    builder = _make_builder(min_cluster_size=10)

    communities = await builder.cluster_graph(entities, relations)

    assert communities == []
    assert all(entity.clusters == [] for entity in entities)
