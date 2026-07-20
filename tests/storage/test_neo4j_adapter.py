import pytest

from ragu.graph.types import Entity, Relation
from ragu.storage.graph_storage_adapters.neo4j_adapter import Neo4jStorage

from tests.storage.conftest import (
    NEO4J_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    prepare_neo4j_store,
    wipe_neo4j_store,
)

pytestmark = pytest.mark.integration

@pytest.fixture
async def neo4j_store():
    try:
        store = Neo4jStorage(
            uri=NEO4J_URI,
            user=NEO4J_USER,
            password=NEO4J_PASSWORD,
            node_cls=Entity,
            edge_cls=Relation,
        )
    except ImportError as exc:  # optional driver not installed
        pytest.skip(str(exc))

    try:
        await store._verify_connectivity()
    except Exception as exc:  # server not running in this environment
        await store.close()
        pytest.skip(f"Neo4j is not reachable at {NEO4J_URI}: {type(exc).__name__}")

    leftovers = await store.get_all_nodes()
    if leftovers:
        await store.delete_nodes([n.id for n in leftovers])

    await prepare_neo4j_store(store)

    await wipe_neo4j_store(store)
    yield store
    await wipe_neo4j_store(store)
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

    # One list per spec: the first names an existing edge, the second a missing one.
    got = await neo4j_store.get_edges([("e1", "e2", "rel-1"), ("e1", "e9", "rel-404")])
    assert [e.id for e in got[0]] == ["rel-1"]
    assert got[0][0].subject_id == "e1"
    assert got[0][0].object_id == "e2"
    assert got[0][0].relation_type == "KNOWS"
    assert got[1] == []


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


@pytest.mark.asyncio
async def test_upsert_node_with_clusters_roundtrip(neo4j_store):
    entity = Entity(
        id="cluster-test-1",
        entity_name="TestEntity",
        entity_type="TEST",
        description="Entity with clusters",
        source_chunk_id=["chunk-1"],
        documents_id=[],
        clusters=[
            {"level": 0, "cluster_id": 1},
            {"level": 1, "cluster_id": 5},
        ],
    )
    await neo4j_store.upsert_nodes([entity])

    got = await neo4j_store.get_nodes(["cluster-test-1"])
    assert got[0] is not None
    assert got[0].entity_name == "TestEntity"
    assert got[0].clusters == [
        {"level": 0, "cluster_id": 1},
        {"level": 1, "cluster_id": 5},
    ]


@pytest.mark.asyncio
async def test_upsert_node_with_clusters_in_get_all_nodes(neo4j_store):
    entity = Entity(
        id="cluster-test-2",
        entity_name="AnotherEntity",
        entity_type="TEST",
        description="Another entity with clusters",
        source_chunk_id=["chunk-1"],
        documents_id=[],
        clusters=[{"level": 0, "cluster_id": 99}],
    )
    await neo4j_store.upsert_nodes([entity])

    all_nodes = await neo4j_store.get_all_nodes()
    match = [n for n in all_nodes if n.id == "cluster-test-2"]
    assert len(match) == 1
    assert match[0].clusters == [{"level": 0, "cluster_id": 99}]


@pytest.mark.asyncio
async def test_get_label_returns_entity_type(neo4j_store):
    entity = Entity(
        id="label-test-1",
        entity_name="LabelTest",
        entity_type="PERSON",
        description="Label test",
        source_chunk_id=["chunk-1"],
        documents_id=[],
        clusters=[],
    )
    await neo4j_store.upsert_nodes([entity])

    from neo4j import AsyncGraphDatabase
    driver = AsyncGraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    )
    async with driver.session(database=neo4j_store._database) as session:
        result = await session.run(
            "MATCH (n {id: $id}) RETURN labels(n) AS labels",
            id="label-test-1",
        )
        record = await result.single()
        assert record is not None, "Node should exist"
        labels = record["labels"]
        assert "NODE" in labels, "Base label :NODE should be present"
        assert "PERSON" in labels, f"Entity type label :PERSON should be present, got {labels}"
    await driver.close()

@pytest.mark.asyncio
async def test_run_cypher_query_filters_by_type(neo4j_store):
    """The escape hatch used the way type filtering is meant to work."""
    await neo4j_store.upsert_nodes([
        _entity("Alice", entity_type="PERSON", id="e1"),
        _entity("Bob", entity_type="PERSON", id="e2"),
        _entity("Acme", entity_type="ORGANIZATION", id="e3"),
    ])

    rows = await neo4j_store.run_cypher_query(
        "MATCH (n:NODE) WHERE n.entity_type IN $types RETURN n.id AS id ORDER BY id",
        {"types": ["PERSON"]},
    )

    assert [row["id"] for row in rows] == ["e1", "e2"]


@pytest.mark.asyncio
async def test_run_cypher_query_returns_empty_list_for_no_matches(neo4j_store):
    rows = await neo4j_store.run_cypher_query(
        "MATCH (n:NODE {entity_type: $type}) RETURN n.id AS id",
        {"type": "NOTHING_LIKE_THIS"},
    )

    assert rows == []


@pytest.mark.asyncio
async def test_index_start_callback_creates_constraint_and_type_index(neo4j_store):
    """The type index is derived from Entity.label_field, not hardcoded."""
    await neo4j_store.index_start_callback()

    rows = await neo4j_store.run_cypher_query(
        "SHOW INDEXES YIELD name, properties WHERE name STARTS WITH 'ragu' "
        "RETURN name, properties ORDER BY name"
    )

    indexed = {row["name"]: row["properties"] for row in rows}
    assert indexed["ragu_node_id"] == ["id"]
    assert indexed[f"ragu_node_{Entity.label_field}"] == [Entity.label_field]


@pytest.mark.asyncio
async def test_edges_are_written_under_their_relation_type(neo4j_store):
    """Relationship type comes from Relation.label_field, not a fixed constant."""
    await neo4j_store.upsert_nodes([_entity("Alice", id="e1"), _entity("Acme", id="e2")])
    await neo4j_store.upsert_edges([
        _relation("e1", "e2", relation_type="WORKS_AT", id="r1"),
        _relation("e2", "e1", relation_type="EMPLOYS", id="r2"),
    ])

    rows = await neo4j_store.run_cypher_query(
        "MATCH ()-[r]->() RETURN type(r) AS t ORDER BY t"
    )

    assert [row["t"] for row in rows] == ["EMPLOYS", "WORKS_AT"]


@pytest.mark.asyncio
async def test_reads_ignore_relationships_from_other_tools(neo4j_store):
    """
    Typed edges cost the ``:RELATION`` filter, so reads match any relationship
    between our nodes. Only edges carrying an id are ours.
    """
    await neo4j_store.upsert_nodes([_entity("Alice", id="e1"), _entity("Acme", id="e2")])
    await neo4j_store.upsert_edges([_relation("e1", "e2", id="r1")])
    await neo4j_store.run_cypher_query(
        "MATCH (a:NODE {id: 'e1'}), (b:NODE {id: 'e2'}) "
        "CREATE (a)-[:SOME_OTHER_TOOL {note: 'not ours'}]->(b)"
    )

    edges = await neo4j_store.get_all_edges()

    assert [edge.id for edge in edges] == ["r1"]


@pytest.mark.asyncio
async def test_unknown_edge_properties_are_dropped_not_raised(neo4j_store):
    """A property added by hand must not take down the whole read."""
    await neo4j_store.upsert_nodes([_entity("Alice", id="e1"), _entity("Acme", id="e2")])
    await neo4j_store.upsert_edges([_relation("e1", "e2", id="r1")])
    await neo4j_store.run_cypher_query(
        "MATCH ()-[r {id: 'r1'}]->() SET r.added_by_hand = 'x'"
    )

    edge = (await neo4j_store.get_edges([("e1", "e2", "r1")]))[0][0]

    assert edge.id == "r1"
    assert not hasattr(edge, "added_by_hand")
