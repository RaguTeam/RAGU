import pytest

from ragu.graph.types import Entity, Relation
from ragu.storage.graph_storage_adapters.arcadedb_adapter import (
    ArcadeDBStorage,
    EDGE_TYPE,
    VERTEX_TYPE,
    _esc,
    _parse_json_list,
)


@pytest.fixture
def storage():
    return ArcadeDBStorage(url="http://localhost:2480", database="tododb", auth=None)


@pytest.fixture
def sample_entity():
    return Entity(
        id="ent-1",
        entity_name="Alice",
        entity_type="Person",
        description="Alice's profile",
        source_chunk_id=["chunk-1"],
        documents_id=["doc-1"],
        clusters=[],
    )


@pytest.fixture
def sample_relation():
    return Relation(
        id="rel-1",
        subject_id="ent-1",
        object_id="ent-2",
        subject_name="Alice",
        object_name="Bob",
        relation_type="KNOWS",
        description="Alice knows Bob",
        relation_strength=0.9,
        source_chunk_id=["chunk-1"],
    )


def test_parse_json_list_handles_invalid_values():
    assert _parse_json_list(None) == []
    assert _parse_json_list(["a", "b"]) == ["a", "b"]
    assert _parse_json_list('["x", "y"]') == ["x", "y"]
    assert _parse_json_list('{"not":"a-list"}') == []
    assert _parse_json_list("invalid-json") == []


def test_escape_handles_quotes_and_backslashes():
    assert _esc("a'b\\c") == "a\\'b\\\\c"
    assert _esc(None) == ""


@pytest.mark.asyncio
async def test_get_nodes_and_edges_mapping(storage):
    async def fake_command(sql: str):
        if f"SELECT FROM {VERTEX_TYPE} WHERE id = 'ent-1'" in sql:
            return [
                {
                    "id": "ent-1",
                    "entity_name": "Alice",
                    "entity_type": "Person",
                    "description": "Engineer",
                    "source_chunk_id": '["chunk-1"]',
                    "documents_id": '["doc-1"]',
                    "clusters": "[]",
                }
            ]
        if f"SELECT FROM {VERTEX_TYPE} WHERE id = 'missing'" in sql:
            return []
        if (
            f"SELECT FROM {EDGE_TYPE} WHERE out.id = 'ent-1' AND in.id = 'ent-2' "
            "AND id = 'rel-1'"
        ) in sql:
            return [
                {
                    "id": "rel-1",
                    "subject_name": "Alice",
                    "object_name": "Bob",
                    "relation_type": "KNOWS",
                    "description": "Alice knows Bob",
                    "relation_strength": 1.0,
                    "source_chunk_id": '["chunk-1"]',
                }
            ]
        return []

    storage._command = fake_command

    nodes = await storage.get_nodes(["ent-1", "missing"])
    assert nodes[0] is not None
    assert nodes[0].entity_name == "Alice"
    assert nodes[1] is None

    edges = await storage.get_edges([("ent-1", "ent-2", "rel-1"), ("x", "y", None)])
    assert edges[0] is not None
    assert edges[0].id == "rel-1"
    assert edges[0].subject_id == "ent-1"
    assert edges[1] is None


@pytest.mark.asyncio
async def test_upsert_builds_expected_sql(storage, sample_entity, sample_relation):
    commands = []

    async def fake_command(sql: str):
        commands.append(sql)
        return []

    storage._command = fake_command

    await storage.upsert_nodes([sample_entity])
    await storage.upsert_edges([sample_relation])

    assert any(
        sql.startswith(f"UPDATE {VERTEX_TYPE} SET ") and " UPSERT WHERE id = 'ent-1'" in sql
        for sql in commands
    )
    assert any(
        sql.startswith(f"DELETE FROM {EDGE_TYPE} WHERE id = 'rel-1'") for sql in commands
    )
    assert any(
        sql.startswith(f"CREATE EDGE {EDGE_TYPE} ")
        and "FROM (SELECT FROM RaguEntity WHERE id = 'ent-1')" in sql
        and "TO (SELECT FROM RaguEntity WHERE id = 'ent-2')" in sql
        and "relation_type = 'KNOWS'" in sql
        for sql in commands
    )


@pytest.mark.asyncio
async def test_edges_degrees_reads_count(storage):
    async def fake_command(sql: str):
        if "ent-1" in sql:
            return [{"count": 2}]
        if "ent-2" in sql:
            return [{"count": 1}]
        return [{"count": 0}]

    storage._command = fake_command

    degrees = await storage.edges_degrees(
        [("ent-1", "ent-2", "rel-1"), ("ent-404", "ent-405", None)]
    )
    assert degrees == [3, 0]


@pytest.mark.asyncio
async def test_custom_classes_are_reconstructed():
    from dataclasses import dataclass

    @dataclass(slots=True)
    class CustomEntity(Entity):
        custom_field: str = "custom"

    @dataclass(slots=True)
    class CustomRelation(Relation):
        custom_field: str = "custom"

    storage = ArcadeDBStorage(
        node_cls=CustomEntity,
        edge_cls=CustomRelation,
        url="http://localhost:2480",
        database="tododb",
        auth=None,
    )

    async def fake_command(sql: str):
        if f"SELECT FROM {VERTEX_TYPE} WHERE id = 'ent-1'" in sql:
            return [
                {
                    "id": "ent-1",
                    "entity_name": "Alice",
                    "entity_type": "Person",
                    "description": "Engineer",
                    "source_chunk_id": '["chunk-1"]',
                    "documents_id": '["doc-1"]',
                    "clusters": "[]",
                }
            ]
        if f"SELECT FROM {EDGE_TYPE} WHERE out.id = 'ent-1' AND in.id = 'ent-2'" in sql:
            return [
                {
                    "id": "rel-1",
                    "subject_name": "Alice",
                    "object_name": "Bob",
                    "relation_type": "KNOWS",
                    "description": "Alice knows Bob",
                    "relation_strength": 1.0,
                    "source_chunk_id": '["chunk-1"]',
                }
            ]
        return []

    storage._command = fake_command

    nodes = await storage.get_nodes(["ent-1"])
    assert isinstance(nodes[0], CustomEntity)
    assert nodes[0].entity_name == "Alice"

    edges = await storage.get_edges([("ent-1", "ent-2", "rel-1")])
    assert isinstance(edges[0], CustomRelation)
    assert edges[0].id == "rel-1"
