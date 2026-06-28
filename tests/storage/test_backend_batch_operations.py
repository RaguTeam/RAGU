"""
Tests for batch operations in storage backends.
"""
import numpy as np
import pytest
from ragu.graph.types import Entity, Relation
from ragu.storage.graph_storage_adapters.networkx_adapter import NetworkXStorage
from ragu.storage.vdb_storage_adapters.nano_vdb import NanoVectorDBStorage
from ragu.storage.kv_storage_adapters.json_storage import JsonKVStorage


@pytest.fixture
def temp_graph_storage(tmp_path):
    """Create a temporary graph storage."""
    storage_file = tmp_path / "test_graph.gml"
    return NetworkXStorage(
        filename=str(storage_file),
        node_cls=Entity,
        edge_cls=Relation,
    )


@pytest.fixture
def sample_entities():
    """Create sample entities for testing."""
    return [
        Entity(
            id="ent-1",
            entity_name="Alice",
            entity_type="Person",
            description="A software engineer",
            source_chunk_id=["chunk-1"],
            documents_id=["doc-1"],
            clusters=[],
        ),
        Entity(
            id="ent-2",
            entity_name="Bob",
            entity_type="Person",
            description="A data scientist",
            source_chunk_id=["chunk-2"],
            documents_id=["doc-1"],
            clusters=[],
        ),
        Entity(
            id="ent-3",
            entity_name="Acme Corp",
            entity_type="Organization",
            description="A technology company",
            source_chunk_id=["chunk-3"],
            documents_id=["doc-2"],
            clusters=[],
        ),
    ]


@pytest.fixture
def sample_relations(sample_entities):
    """Create sample relations for testing."""
    return [
        Relation(
            id="rel-1",
            subject_id="ent-1",
            object_id="ent-3",
            subject_name="Alice",
            object_name="Acme Corp",
            relation_type="WORKS_FOR",
            description="Alice works for Acme Corp",
            relation_strength=1.0,
            source_chunk_id=["chunk-1", "chunk-3"],
        ),
        Relation(
            id="rel-2",
            subject_id="ent-2",
            object_id="ent-3",
            subject_name="Bob",
            object_name="Acme Corp",
            relation_type="WORKS_FOR",
            description="Bob works for Acme Corp",
            relation_strength=1.0,
            source_chunk_id=["chunk-2", "chunk-3"],
        ),
    ]


@pytest.mark.asyncio
async def test_get_nodes_batch(temp_graph_storage, sample_entities):
    """Test batch node retrieval."""
    # Insert entities
    await temp_graph_storage.upsert_nodes(sample_entities)

    # Retrieve all nodes
    node_ids = [e.id for e in sample_entities]
    retrieved = await temp_graph_storage.get_nodes(node_ids)

    assert len(retrieved) == 3
    assert all(e is not None for e in retrieved)
    assert {e.id for e in retrieved} == {e.id for e in sample_entities}


@pytest.mark.asyncio
async def test_get_nodes_with_missing(temp_graph_storage, sample_entities):
    """Test batch node retrieval with some missing nodes."""
    # Insert only first entity
    await temp_graph_storage.upsert_nodes([sample_entities[0]])

    # Try to retrieve all three
    node_ids = [e.id for e in sample_entities]
    retrieved = await temp_graph_storage.get_nodes(node_ids)

    assert len(retrieved) == 3
    assert retrieved[0] is not None
    assert retrieved[1] is None
    assert retrieved[2] is None


@pytest.mark.asyncio
async def test_upsert_nodes_batch(temp_graph_storage, sample_entities):
    """Test batch node upsert."""
    # Upsert all nodes
    await temp_graph_storage.upsert_nodes(sample_entities)

    # Verify all nodes exist
    node_ids = [e.id for e in sample_entities]
    nodes = await temp_graph_storage.get_nodes(node_ids)
    assert all(n is not None for n in nodes)
    for node, entity in zip(nodes, sample_entities):
        assert node.entity_name == entity.entity_name


@pytest.mark.asyncio
async def test_delete_nodes_batch(temp_graph_storage, sample_entities):
    """Test batch node deletion."""
    # Insert nodes
    await temp_graph_storage.upsert_nodes(sample_entities)

    # Delete first two
    node_ids = [sample_entities[0].id, sample_entities[1].id]
    await temp_graph_storage.delete_nodes(node_ids)

    # Verify deletion
    nodes = await temp_graph_storage.get_nodes(
        [sample_entities[0].id, sample_entities[1].id, sample_entities[2].id]
    )
    assert nodes[0] is None
    assert nodes[1] is None
    assert nodes[2] is not None


@pytest.mark.asyncio
async def test_upsert_edges_batch(temp_graph_storage, sample_entities, sample_relations):
    """Test batch edge upsert."""
    # Insert nodes first
    await temp_graph_storage.upsert_nodes(sample_entities)

    # Insert edges
    await temp_graph_storage.upsert_edges(sample_relations)

    # Verify edges exist
    edge_specs = [(r.subject_id, r.object_id, r.id) for r in sample_relations]
    edges = await temp_graph_storage.get_edges(edge_specs)
    assert all(e is not None for e in edges)
    for edge, relation in zip(edges, sample_relations):
        assert edge.relation_type == relation.relation_type


@pytest.mark.asyncio
async def test_get_edges_batch(temp_graph_storage, sample_entities, sample_relations):
    """Test batch edge retrieval."""
    # Insert nodes and edges
    await temp_graph_storage.upsert_nodes(sample_entities)
    await temp_graph_storage.upsert_edges(sample_relations)

    # Retrieve edges
    edge_specs = [(r.subject_id, r.object_id, r.id) for r in sample_relations]
    retrieved = await temp_graph_storage.get_edges(edge_specs)

    assert len(retrieved) == 2
    assert all(e is not None for e in retrieved)


@pytest.mark.asyncio
async def test_delete_edges_batch(temp_graph_storage, sample_entities, sample_relations):
    """Test batch edge deletion."""
    # Insert nodes and edges
    await temp_graph_storage.upsert_nodes(sample_entities)
    await temp_graph_storage.upsert_edges(sample_relations)

    # Delete edges
    edge_specs = [(r.subject_id, r.object_id, r.id) for r in sample_relations]
    await temp_graph_storage.delete_edges(edge_specs)

    # Verify deletion
    edge_specs = [(r.subject_id, r.object_id, r.id) for r in sample_relations]
    deleted = await temp_graph_storage.get_edges(edge_specs)

    assert all(edge is None for edge in deleted)


@pytest.mark.asyncio
async def test_get_all_edges_for_nodes(temp_graph_storage, sample_entities, sample_relations):
    """Test retrieving all edges for given nodes."""
    # Insert nodes and edges
    await temp_graph_storage.upsert_nodes(sample_entities)
    await temp_graph_storage.upsert_edges(sample_relations)

    # Get all edges for Alice (ent-1)
    edges = await temp_graph_storage.get_all_edges_for_nodes(["ent-1"])
    assert len(edges) == 1
    assert len(edges[0]) == 1
    assert edges[0][0].subject_id == "ent-1"

    # Get all edges for Acme Corp (ent-3) - should have 2 connections
    edges = await temp_graph_storage.get_all_edges_for_nodes(["ent-3"])
    assert len(edges) == 1
    assert len(edges[0]) == 2


@pytest.mark.asyncio
async def test_get_nodes_by_deterministic_entity_id(temp_graph_storage):
    """Test retrieving existing node by deterministic ID from name/type."""
    alice = Entity(
        entity_name="Alice",
        entity_type="Person",
        description="First description",
        source_chunk_id=["chunk-1"],
        documents_id=["doc-1"],
        clusters=[],
    )
    await temp_graph_storage.upsert_nodes([alice])

    duplicate_alice = Entity(
        entity_name="Alice",
        entity_type="Person",
        description="Second description",
        source_chunk_id=["chunk-2"],
        documents_id=["doc-2"],
        clusters=[],
    )

    retrieved = await temp_graph_storage.get_nodes([duplicate_alice.id])

    assert len(retrieved) == 1
    assert retrieved[0] is not None
    assert retrieved[0].id == alice.id


@pytest.mark.asyncio
async def test_kv_delete(tmp_path):
    """Test KV storage delete operation."""
    storage_file = tmp_path / "test_kv.json"
    kv = JsonKVStorage(filename=str(storage_file))

    # Insert data
    await kv.upsert({"key1": {"data": "value1"}, "key2": {"data": "value2"}})

    # Delete key1
    await kv.delete(["key1"])

    # Verify
    assert await kv.get_by_id("key1") is None
    assert await kv.get_by_id("key2") is not None


@pytest.mark.asyncio
async def test_vdb_delete(tmp_path):
    """Test vector DB delete operation."""
    from ragu.storage.types import Point

    dim = 128
    storage_file = tmp_path / "test_vdb.json"
    vdb = NanoVectorDBStorage(embedding_dim=dim, filename=str(storage_file), cosine_threshold=0.0)

    # Insert data
    points = [
        Point(id="id1", dense_embedding=np.array([0.1] * dim), metadata={"content": "test1"}),
        Point(id="id2", dense_embedding=np.array([0.2] * dim), metadata={"content": "test2"}),
    ]
    await vdb.upsert(points)

    # Get initial count
    query_emb = Point(dense_embedding=np.array([0.15] * dim))
    initial_results = (await vdb.query([query_emb], top_k=10))[0]
    assert len(initial_results) == 2

    # Delete id1
    await vdb.delete(["id1"])

    # Verify by querying
    remaining_results = (await vdb.query([query_emb], top_k=10))[0]
    remaining_ids = [r.id for r in remaining_results]
    assert "id1" not in remaining_ids
    assert "id2" in remaining_ids
