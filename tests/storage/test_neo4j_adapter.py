import os

import pytest

from ragu.graph.types import Entity, Relation
from ragu.storage.graph_storage_adapters.neo4j_adapter import Neo4jStorage

pytestmark = pytest.mark.integration

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "testpassword")


@pytest.fixture
async def neo4j_store():
    store = Neo4jStorage(
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASSWORD,
        node_cls=Entity,
        edge_cls=Relation,
    )
    await store._verify_connectivity()
    leftovers = await store.get_all_nodes()
    if leftovers:
        await store.delete_nodes([n.id for n in leftovers])

    yield store

    remaining = await store.get_all_nodes()
    if remaining:
        await store.delete_nodes([n.id for n in remaining])
    await store.close()


def _entity(
    entity_name: str,
    entity_type: str = "PERSON",
    description: str = "",
    source_chunk_id: list[str] | None = None,
    id: str | None = None,
) -> Entity:
    return Entity(
        id=id or entity_name,
        entity_name=entity_name,
        entity_type=entity_type,
        description=description,
        source_chunk_id=source_chunk_id or ["chunk-1"],
        documents_id=[],
        clusters=[],
    )


def _relation(
    subject_id: str,
    object_id: str,
    relation_type: str = "KNOWS",
    description: str = "",
    id: str = "rel-1",
) -> Relation:
    return Relation(
        id=id,
        subject_id=subject_id,
        object_id=object_id,
        subject_name=subject_id,
        object_name=object_id,
        relation_type=relation_type,
        description=description,
        relation_strength=1.0,
        source_chunk_id=["chunk-1"],
    )


@pytest.mark.asyncio
async def test_verify_connectivity(neo4j_store):
    assert True


@pytest.mark.asyncio
async def test_upsert_and_get_nodes(neo4j_store):
    entity = _entity("Alice", description="Human Alice")
    await neo4j_store.upsert_nodes([entity])

    got = await neo4j_store.get_nodes(["Alice", "nonexistent"])
    assert got[0] is not None
    assert got[0].entity_name == "Alice"
    assert got[0].description == "Human Alice"
    assert got[1] is None


@pytest.mark.asyncio
async def test_get_all_nodes(neo4j_store):
    entities = [
        _entity("Alice", id="e1"),
        _entity("Bob", id="e2"),
    ]
    await neo4j_store.upsert_nodes(entities)

    all_nodes = await neo4j_store.get_all_nodes()
    assert len(all_nodes) == 2
    assert {n.id for n in all_nodes} == {"e1", "e2"}


@pytest.mark.asyncio
async def test_upsert_and_get_edges(neo4j_store):
    await neo4j_store.upsert_nodes([
        _entity("Alice", id="e1"),
        _entity("Bob", id="e2"),
    ])
    rel = _relation(subject_id="e1", object_id="e2", id="rel-1")
    await neo4j_store.upsert_edges([rel])

    got = await neo4j_store.get_edges([("e1", "e2", "rel-1"), ("e1", "e9", None)])
    assert got[0] is not None
    assert got[0].subject_id == "e1"
    assert got[0].object_id == "e2"
    assert got[0].relation_type == "KNOWS"
    assert got[1] is None


@pytest.mark.asyncio
async def test_get_all_edges(neo4j_store):
    await neo4j_store.upsert_nodes([
        _entity("Alice", id="e1"),
        _entity("Bob", id="e2"),
    ])
    rel = _relation(subject_id="e1", object_id="e2", id="rel-1")
    await neo4j_store.upsert_edges([rel])

    all_edges = await neo4j_store.get_all_edges()
    assert len(all_edges) == 1
    assert all_edges[0].id == "rel-1"


@pytest.mark.asyncio
async def test_get_node_edges_returns_outgoing_only(neo4j_store):
    await neo4j_store.upsert_nodes([
        _entity("Alice", id="e1"),
        _entity("Bob", id="e2"),
        _entity("Charlie", id="e3"),
    ])
    await neo4j_store.upsert_edges([
        _relation(subject_id="e1", object_id="e2", id="rel-1"),
        _relation(subject_id="e3", object_id="e1", id="rel-2"),
    ])

    incident = await neo4j_store.get_node_edges("e1")
    assert len(incident) == 1
    assert incident[0].id == "rel-1"


@pytest.mark.asyncio
async def test_edges_degrees(neo4j_store):
    await neo4j_store.upsert_nodes([
        _entity("Alice", id="e1"),
        _entity("Bob", id="e2"),
        _entity("Charlie", id="e3"),
    ])
    await neo4j_store.upsert_edges([
        _relation(subject_id="e1", object_id="e2", id="rel-1"),
        _relation(subject_id="e1", object_id="e3", id="rel-2"),
    ])

    degrees = await neo4j_store.edges_degrees([
        ("e1", "e2", "rel-1"),
        ("e1", "e3", "rel-2"),
        ("e404", "e405", None),
    ])
    assert degrees == [3, 3, 0]


@pytest.mark.asyncio
async def test_get_all_edges_for_nodes_returns_all_incident(neo4j_store):
    await neo4j_store.upsert_nodes([
        _entity("Alice", id="e1"),
        _entity("Bob", id="e2"),
    ])
    await neo4j_store.upsert_edges([
        _relation(subject_id="e1", object_id="e2", id="rel-1"),
    ])

    grouped = await neo4j_store.get_all_edges_for_nodes(["e1", "e2"])
    assert len(grouped) == 2
    assert len(grouped[0]) == 1
    assert grouped[0][0].id == "rel-1"
    assert len(grouped[1]) == 1
    assert grouped[1][0].id == "rel-1"


@pytest.mark.asyncio
async def test_delete_edges(neo4j_store):
    await neo4j_store.upsert_nodes([
        _entity("Alice", id="e1"),
        _entity("Bob", id="e2"),
    ])
    await neo4j_store.upsert_edges([
        _relation(subject_id="e1", object_id="e2", id="rel-1"),
    ])

    await neo4j_store.delete_edges([("e1", "e2", "rel-1")])
    remaining = await neo4j_store.get_all_edges()
    assert len(remaining) == 0


@pytest.mark.asyncio
async def test_delete_nodes_cascade(neo4j_store):
    await neo4j_store.upsert_nodes([
        _entity("Alice", id="e1"),
        _entity("Bob", id="e2"),
    ])
    await neo4j_store.upsert_edges([
        _relation(subject_id="e1", object_id="e2", id="rel-1"),
    ])

    await neo4j_store.delete_nodes(["e1"])
    remaining_nodes = await neo4j_store.get_all_nodes()
    remaining_edges = await neo4j_store.get_all_edges()
    assert len(remaining_nodes) == 1
    assert remaining_nodes[0].id == "e2"
    assert len(remaining_edges) == 0