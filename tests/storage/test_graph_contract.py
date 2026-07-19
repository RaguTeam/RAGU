"""
Behaviour every :class:`BaseGraphStorage` backend must share.

The suite runs against each registered backend, so a divergence between the
in-process NetworkX store and a server-backed one shows up here rather than in
production. Backends needing a server are skipped when it is unreachable.
"""

from __future__ import annotations

import pytest

from ragu.graph.types import Entity, Relation


def _entity(name: str, entity_type: str = "PERSON", **kwargs) -> Entity:
    return Entity(
        id=kwargs.pop("id", name),
        entity_name=name,
        entity_type=entity_type,
        description=kwargs.pop("description", "desc"),
        source_chunk_id=kwargs.pop("source_chunk_id", ["chunk-1"]),
        documents_id=kwargs.pop("documents_id", []),
        clusters=kwargs.pop("clusters", []),
    )


def _relation(subject_id: str, object_id: str, edge_id: str = "rel-1") -> Relation:
    return Relation(
        id=edge_id,
        subject_id=subject_id,
        object_id=object_id,
        subject_name=subject_id,
        object_name=object_id,
        relation_type="WORKS_AT",
        description=f"{subject_id} -> {object_id}",
    )


async def _alice_works_at_acme(store):
    """Seed a single directed edge: Alice -[WORKS_AT]-> Acme."""
    await store.upsert_nodes([_entity("Alice"), _entity("Acme", "ORGANIZATION")])
    await store.upsert_edges([_relation("Alice", "Acme")])


@pytest.mark.asyncio
async def test_graph_contract_nodes_round_trip(graph_storage):
    await graph_storage.upsert_nodes([_entity("Alice")])

    nodes = await graph_storage.get_nodes(["Alice", "missing"])

    assert nodes[0] is not None
    assert nodes[0].id == "Alice"
    assert nodes[0].entity_name == "Alice"
    assert nodes[1] is None


@pytest.mark.asyncio
async def test_graph_contract_structured_fields_round_trip(graph_storage):
    """``clusters`` holds dicts, which not every backend can store natively."""
    clusters = [{"level": 0, "cluster_id": 3}]
    await graph_storage.upsert_nodes([_entity("Alice", clusters=clusters)])

    node = (await graph_storage.get_nodes(["Alice"]))[0]

    assert node is not None
    assert node.clusters == clusters
    assert node.source_chunk_id == ["chunk-1"]


@pytest.mark.asyncio
async def test_graph_contract_edge_direction_is_preserved(graph_storage):
    """
    Edges are directed. Querying from the *target* side must still report the
    original subject and object, not swap them to match the query direction.
    """
    await _alice_works_at_acme(graph_storage)

    edges = (await graph_storage.get_all_edges_for_nodes(["Acme"]))[0]

    assert len(edges) == 1
    assert edges[0].subject_id == "Alice"
    assert edges[0].object_id == "Acme"


@pytest.mark.asyncio
async def test_graph_contract_get_node_edges_includes_incoming(graph_storage):
    """A node's edges are the incident ones, not only those leaving it."""
    await _alice_works_at_acme(graph_storage)

    edges = await graph_storage.get_node_edges("Acme")

    assert [(e.subject_id, e.object_id) for e in edges] == [("Alice", "Acme")]


@pytest.mark.asyncio
async def test_graph_contract_node_id_stays_unique_when_type_changes(graph_storage):
    """
    Re-upserting an id with a different entity type updates the node in place.
    Backends that key storage by type would otherwise end up with two nodes
    sharing one id, and later reads would return an arbitrary one of them.
    """
    await graph_storage.upsert_nodes([_entity("Alice", "PERSON")])
    await graph_storage.upsert_nodes([_entity("Alice", "ORGANIZATION")])

    all_nodes = await graph_storage.get_all_nodes()

    assert [node.id for node in all_nodes] == ["Alice"]
    assert all_nodes[0].entity_type == "ORGANIZATION"


@pytest.mark.asyncio
async def test_graph_contract_edges_round_trip(graph_storage):
    await _alice_works_at_acme(graph_storage)

    edges = await graph_storage.get_all_edges()

    assert len(edges) == 1
    assert edges[0].subject_id == "Alice"
    assert edges[0].object_id == "Acme"
    assert edges[0].relation_type == "WORKS_AT"


@pytest.mark.asyncio
async def test_graph_contract_delete_nodes_removes_incident_edges(graph_storage):
    await _alice_works_at_acme(graph_storage)

    await graph_storage.delete_nodes(["Alice"])

    assert [node.id for node in await graph_storage.get_all_nodes()] == ["Acme"]
    assert await graph_storage.get_all_edges() == []


@pytest.mark.asyncio
async def test_graph_contract_upsert_nodes_is_idempotent(graph_storage):
    await graph_storage.upsert_nodes([_entity("Alice", description="v1")])
    await graph_storage.upsert_nodes([_entity("Alice", description="v2")])

    all_nodes = await graph_storage.get_all_nodes()

    assert len(all_nodes) == 1
    assert all_nodes[0].description == "v2"


@pytest.mark.asyncio
async def test_graph_contract_upsert_edges_is_idempotent(graph_storage):
    await _alice_works_at_acme(graph_storage)
    await graph_storage.upsert_edges([_relation("Alice", "Acme")])

    assert len(await graph_storage.get_all_edges()) == 1
