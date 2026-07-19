"""
Tests for Index batch CRUD operations and invariant validation.
"""
import numpy as np
import pytest
from ragu.graph.types import Community, Entity, Relation
from ragu.chunker.types import Chunk
from ragu.graph.graph_retrieve_backend import GraphRetriever
from ragu.graph.index import ConsistencyIssue, ConsistencyReport, Index, StorageArguments
from ragu.models.embedder import Embedder
from ragu.storage.types import EmbeddingHit, Point, SparseEmbedding
from unittest.mock import AsyncMock, Mock


@pytest.fixture
def mock_embedder():
    """Create a mock embedder."""
    embedder = AsyncMock(spec=Embedder)
    embedder.dim = 128
    embedder.embed_text = AsyncMock(return_value=[0.1] * 128)
    embedder.batch_embed_text = AsyncMock(
        side_effect=lambda texts, **kwargs: [[0.1] * 128 for _ in texts]
    )
    return embedder


@pytest.fixture
def index(tmp_path, monkeypatch, mock_embedder):
    """Create an Index instance with temporary storage."""
    from ragu.common.global_parameters import Settings
    monkeypatch.setattr(Settings, "storage_folder", str(tmp_path / "storage"))
    storage_args = StorageArguments()
    return Index(embedder=mock_embedder, arguments=storage_args)


@pytest.fixture
def mock_sparse_embedder():
    sparse_embedder = Mock()
    sparse_embedder.embed_document.return_value = [
        SparseEmbedding(indices=[1, 2], values=[0.7, 0.3]),
        SparseEmbedding(indices=[3, 4], values=[0.6, 0.4]),
    ]
    sparse_embedder.embed_query.return_value = [
        SparseEmbedding(indices=[9], values=[1.0]),
    ]
    return sparse_embedder


@pytest.fixture
def sparse_index(tmp_path, monkeypatch, mock_embedder, mock_sparse_embedder):
    from ragu.common.global_parameters import Settings
    monkeypatch.setattr(Settings, "storage_folder", str(tmp_path / "storage"))
    storage_args = StorageArguments()
    return Index(embedder=mock_embedder, arguments=storage_args, sparse_embedder=mock_sparse_embedder)


@pytest.fixture
def sparse_retriever(sparse_index, mock_embedder, mock_sparse_embedder):
    return GraphRetriever(
        Mock(index=sparse_index),
        mock_embedder,
        mock_sparse_embedder,
    )


@pytest.fixture
def sample_entities():
    """Sample entities for testing."""
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
    ]


@pytest.fixture
def sample_relations(sample_entities):
    """Sample relations for testing."""
    return [
        Relation(
            id="rel-1",
            subject_id="ent-1",
            object_id="ent-2",
            subject_name="Alice",
            object_name="Bob",
            relation_type="KNOWS",
            description="Alice knows Bob",
            relation_strength=1.0,
            source_chunk_id=["chunk-1"],
        ),
    ]


@pytest.mark.asyncio
async def test_insert_entities(index, sample_entities):
    """
    Test inserting entities.
    """
    await index.upsert_nodes(sample_entities)

    # Verify entities exist
    retrieved = await index.get_nodes(["ent-1", "ent-2"])
    assert len(retrieved) == 2
    assert all(e is not None for e in retrieved)
    assert [e.id for e in retrieved if e is not None] == ["ent-1", "ent-2"]
    assert [e.entity_name for e in retrieved if e is not None] == ["Alice", "Bob"]
    assert [e.description for e in retrieved if e is not None] == [
        "A software engineer",
        "A data scientist",
    ]

    vector_points = await index.nodes_vector_db.get_points_by_ids(["ent-1", "ent-2"])
    assert [point.id for point in vector_points if point is not None] == ["ent-1", "ent-2"]
    assert [point.metadata["entity_name"] for point in vector_points if point is not None] == ["Alice", "Bob"]


@pytest.mark.asyncio
async def test_insert_entities_does_not_touch_graph_when_vectorization_fails(index, mock_embedder, sample_entities):
    mock_embedder.batch_embed_text = AsyncMock(
        side_effect=RuntimeError("embed failed")
    )

    with pytest.raises(RuntimeError, match="embed failed"):
        await index.upsert_nodes(sample_entities)

    retrieved = await index.get_nodes(["ent-1", "ent-2"])
    assert retrieved == [None, None]


@pytest.mark.asyncio
async def test_build_query_vectors_returns_dense_and_sparse_payload(
    sparse_retriever,
    mock_embedder,
    mock_sparse_embedder,
):
    point = (await sparse_retriever.build_query_vectors(["alpha beta"]))[0]

    assert np.array_equal(np.array(point.dense_embedding), np.array([0.1] * 128))
    assert point.sparse_embedding is not None
    assert point.sparse_embedding.indices == [9]
    mock_embedder.batch_embed_text.assert_awaited_once_with(["alpha beta"])
    mock_sparse_embedder.embed_query.assert_called_once_with(["alpha beta"])


@pytest.mark.asyncio
async def test_insert_entities_passes_sparse_embeddings_to_vector_db(sparse_index, sample_entities, mock_sparse_embedder):
    sparse_index.nodes_vector_db.upsert = AsyncMock()

    await sparse_index.upsert_nodes(sample_entities)

    sparse_index.nodes_vector_db.upsert.assert_awaited_once()
    args, _ = sparse_index.nodes_vector_db.upsert.await_args
    points = args[0]
    assert [point.sparse_embedding.indices for point in points if point.sparse_embedding is not None] == [[1, 2], [3, 4]]
    mock_sparse_embedder.embed_document.assert_called_once()


@pytest.mark.asyncio
async def test_query_entities_passes_sparse_query_to_vector_db(sparse_index, sparse_retriever, sample_entities, mock_sparse_embedder):
    await sparse_index.upsert_nodes(sample_entities)
    sparse_index.nodes_vector_db.query = AsyncMock(
        return_value=[[EmbeddingHit(id="ent-1", distance=0.9)]]
    )

    entities, hits = (await sparse_retriever.query_entities(["alice engineer"], top_k=3))[0]

    assert [entity.id for entity in entities] == ["ent-1"]
    assert [hit.id for hit in hits] == ["ent-1"]
    args, kwargs = sparse_index.nodes_vector_db.query.await_args
    point = args[0][0]
    assert point.sparse_embedding is not None
    assert point.sparse_embedding.indices == [9]
    assert kwargs["top_k"] == 3
    mock_sparse_embedder.embed_query.assert_called_once_with(["alice engineer"])


@pytest.mark.asyncio
async def test_query_chunk_hits_passes_sparse_query_to_vector_db(sparse_index, sparse_retriever, mock_sparse_embedder):
    sparse_index.chunks_vector_db.query = AsyncMock(
        return_value=[[EmbeddingHit(id="chunk-1", distance=0.8, metadata={"doc_id": "doc-1"})]]
    )
    sparse_index.chunks_kv_storage.get_by_ids = AsyncMock(
        return_value=[
            {
                "content": "chunk one",
                "chunk_order_idx": 0,
                "doc_id": "doc-1",
                "num_tokens": 3,
            }
        ]
    )

    chunks, hits = (await sparse_retriever.query_chunks(["hybrid chunk query"], top_k=2))[0]

    assert [chunk.id for chunk in chunks] == ["chunk-1"]
    assert [hit.id for hit in hits] == ["chunk-1"]
    args, kwargs = sparse_index.chunks_vector_db.query.await_args
    point = args[0][0]
    assert point.sparse_embedding is not None
    assert point.sparse_embedding.indices == [9]
    assert kwargs["top_k"] == 2
    sparse_index.chunks_kv_storage.get_by_ids.assert_awaited_once_with(["chunk-1"])
    mock_sparse_embedder.embed_query.assert_called_once_with(["hybrid chunk query"])


@pytest.mark.asyncio
async def test_insert_entities_replaces_existing_payload(index):
    """
    Test storage-level insert replaces existing entity payload by ID.
    """
    entity1 = Entity(
        entity_name="Alice",
        entity_type="Person",
        description="First description",
        source_chunk_id=["chunk-1"],
        documents_id=["doc-1"],
        clusters=[],
    )
    entity2 = Entity(
        entity_name="Alice",
        entity_type="Person",
        description="Second description",
        source_chunk_id=["chunk-2"],
        documents_id=["doc-2"],
        clusters=[],
    )

    # First insert
    await index.upsert_nodes([entity1])

    # Second insert with duplicate
    await index.upsert_nodes([entity2])

    # Should still have one Alice entity, using the latest payload.
    retrieved = await index.get_nodes([entity1.id])
    non_null = [e for e in retrieved if e is not None]
    assert len(non_null) == 1

    stored = non_null[0]
    assert stored.description == "Second description"
    assert stored.source_chunk_id == ["chunk-2"]
    assert stored.documents_id == ["doc-2"]


@pytest.mark.asyncio
async def test_insert_relations(index, sample_entities, sample_relations):
    """
    Test inserting relations.
    """
    await index.upsert_nodes(sample_entities)
    await index.upsert_edges(sample_relations)

    relation = sample_relations[0]
    retrieved = await index.get_edges([(relation.subject_id, relation.object_id, relation.id)])
    assert len(retrieved) == 1
    assert retrieved[0] is not None
    assert retrieved[0].id == relation.id
    assert retrieved[0].subject_id == "ent-1"
    assert retrieved[0].object_id == "ent-2"
    assert retrieved[0].relation_type == "KNOWS"
    assert retrieved[0].description == "Alice knows Bob"

    vector_points = await index.edges_vector_db.get_points_by_ids([relation.id])
    assert vector_points[0] is not None
    assert vector_points[0].id == relation.id
    assert vector_points[0].metadata["subject_id"] == "ent-1"
    assert vector_points[0].metadata["object_id"] == "ent-2"


@pytest.mark.asyncio
async def test_upsert_relations_validates_entities(index, sample_relations):
    """
    Test that upserting relations validates entity existence.
    """
    with pytest.raises(ValueError, match="non-existent nodes"):
        await index.upsert_edges(sample_relations)


@pytest.mark.asyncio
async def test_insert_relations_does_not_touch_graph_when_vectorization_fails(
    index,
    mock_embedder,
    sample_entities,
    sample_relations,
):
    await index.upsert_nodes(sample_entities)
    mock_embedder.batch_embed_text = AsyncMock(
        side_effect=RuntimeError("embed failed")
    )

    with pytest.raises(RuntimeError, match="embed failed"):
        await index.upsert_edges(sample_relations)

    relation = sample_relations[0]
    retrieved = await index.get_edges([(relation.subject_id, relation.object_id, relation.id)])
    assert retrieved == [None]


@pytest.mark.asyncio
async def test_delete_entities(index, sample_entities):
    """
    Test deleting entities.
    """
    await index.upsert_nodes(sample_entities)
    await index.delete_nodes(["ent-1"])

    retrieved = await index.get_nodes(["ent-1", "ent-2"])
    assert retrieved[0] is None
    assert retrieved[1] is not None


@pytest.mark.asyncio
async def test_delete_entities_cascade(index, sample_entities, sample_relations):
    """
    Test that deleting entities cascades to relations.
    """
    await index.upsert_nodes(sample_entities)
    await index.upsert_edges(sample_relations)

    await index.delete_nodes(["ent-1"])

    relation = sample_relations[0]
    retrieved_relations = await index.get_edges([(relation.subject_id, relation.object_id, relation.id)])
    assert retrieved_relations[0] is None


@pytest.mark.asyncio
async def test_delete_relations(index, sample_entities, sample_relations):
    """
    Test deleting relations.
    """
    await index.upsert_nodes(sample_entities)
    await index.upsert_edges(sample_relations)

    relation = sample_relations[0]
    await index.delete_edges([(relation.subject_id, relation.object_id, relation.id)])

    retrieved = await index.get_edges([(relation.subject_id, relation.object_id, relation.id)])
    assert retrieved[0] is None

    entities = await index.get_nodes(["ent-1", "ent-2"])
    assert all(e is not None for e in entities)


@pytest.mark.asyncio
async def test_get_relations_preserves_stored_relation_id(index, sample_entities):
    await index.upsert_nodes(sample_entities)

    relation = Relation(
        id="rel-custom",
        subject_id="ent-1",
        object_id="ent-2",
        subject_name="Alice",
        object_name="Bob",
        relation_type="KNOWS",
        description="Alice knows Bob",
        relation_strength=1.0,
        source_chunk_id=["chunk-1"],
    )

    await index.upsert_edges([relation])

    retrieved = await index.get_edges([("ent-1", "ent-2", "rel-custom")])
    assert retrieved[0] is not None
    assert retrieved[0].id == "rel-custom"


@pytest.mark.asyncio
async def test_delete_relations_removes_relation_vector(index, sample_entities):
    await index.upsert_nodes(sample_entities)

    relation = Relation(
        id="rel-custom",
        subject_id="ent-1",
        object_id="ent-2",
        subject_name="Alice",
        object_name="Bob",
        relation_type="KNOWS",
        description="Alice knows Bob",
        relation_strength=1.0,
        source_chunk_id=["chunk-1"],
    )

    await index.upsert_edges([relation])
    before_delete = (await index.edges_vector_db.query([Point(dense_embedding=[0.1] * 128)], top_k=10))[0]
    assert any(hit.id == "rel-custom" for hit in before_delete)

    await index.delete_edges([("ent-1", "ent-2", "rel-custom")])

    after_delete = (await index.edges_vector_db.query([Point(dense_embedding=[0.1] * 128)], top_k=10))[0]
    assert all(hit.id != "rel-custom" for hit in after_delete)


@pytest.mark.asyncio
async def test_query_relations_returns_vector_hits(sparse_retriever, sparse_index, sample_entities, sample_relations):
    await sparse_index.upsert_nodes(sample_entities)
    await sparse_index.upsert_edges(sample_relations)

    relations, hits = (await sparse_retriever.query_relations(["Alice knows Bob"], top_k=5))[0]

    assert [relation.id for relation in relations] == ["rel-1"]
    assert [hit.id for hit in hits] == ["rel-1"]


@pytest.mark.asyncio
async def test_query_relations_skips_hits_without_endpoint_metadata(index, sample_entities, sample_relations, mock_embedder):
    await index.upsert_nodes(sample_entities)
    await index.upsert_edges(sample_relations)

    retriever = GraphRetriever(Mock(index=index), mock_embedder)
    index.edges_vector_db.query = AsyncMock(
        return_value=[[EmbeddingHit(id="rel-1", distance=0.9, metadata={"content": "Alice knows Bob"})]]
    )

    relations, hits = (await retriever.query_relations(["Alice knows Bob"], top_k=5))[0]

    assert relations == []
    assert hits == []


@pytest.mark.asyncio
async def test_upsert_relations_keeps_non_duplicate_when_duplicates_exist(index, sample_entities):
    await index.upsert_nodes(sample_entities)

    rel_existing = Relation(
        id="rel-dup",
        subject_id="ent-1",
        object_id="ent-2",
        subject_name="Alice",
        object_name="Bob",
        relation_type="KNOWS",
        description="old",
        relation_strength=1.0,
        source_chunk_id=["chunk-1"],
    )
    await index.upsert_edges([rel_existing])

    rel_duplicate_update = Relation(
        id="rel-dup",
        subject_id="ent-1",
        object_id="ent-2",
        subject_name="Alice",
        object_name="Bob",
        relation_type="KNOWS",
        description="new",
        relation_strength=1.0,
        source_chunk_id=["chunk-2"],
    )
    rel_unique_new = Relation(
        id="rel-new",
        subject_id="ent-2",
        object_id="ent-1",
        subject_name="Bob",
        object_name="Alice",
        relation_type="KNOWS",
        description="fresh",
        relation_strength=1.0,
        source_chunk_id=["chunk-2"],
    )

    await index.upsert_edges([rel_duplicate_update, rel_unique_new])

    got = await index.get_edges(
        [
            ("ent-1", "ent-2", "rel-dup"),
            ("ent-2", "ent-1", "rel-new"),
        ]
    )
    assert got[0] is not None
    assert got[1] is not None
    assert got[0].description == "new"
    assert got[0].source_chunk_id == ["chunk-2"]
    assert got[1].description == "fresh"
    assert got[1].subject_id == "ent-2"
    assert got[1].object_id == "ent-1"


@pytest.mark.asyncio
async def test_update_entities_replaces_existing_payload(index):
    original = Entity(
        id="ent-update",
        entity_name="Alice",
        entity_type="Person",
        description="old description",
        source_chunk_id=["chunk-old"],
        documents_id=["doc-old"],
        clusters=[{"level": 1, "cluster_id": "1"}],
    )
    updated = Entity(
        id="ent-update",
        entity_name="Alice Updated",
        entity_type="Person",
        description="new description",
        source_chunk_id=["chunk-new"],
        documents_id=["doc-new"],
        clusters=[{"level": 2, "cluster_id": "2"}],
    )

    await index.upsert_nodes([original])
    await index.update_nodes([updated])

    got = await index.get_nodes([original.id])
    assert got[0] is not None
    assert got[0].description == "new description"
    assert got[0].entity_name == "Alice Updated"
    assert got[0].source_chunk_id == ["chunk-new"]
    assert got[0].documents_id == ["doc-new"]
    assert got[0].clusters == [{"level": 2, "cluster_id": "2"}]


@pytest.mark.asyncio
async def test_update_entities_fails_for_missing_id(index):
    missing = Entity(
        id="ent-missing",
        entity_name="Ghost",
        entity_type="Person",
        description="does not exist",
        source_chunk_id=["chunk-x"],
        documents_id=[],
        clusters=[],
    )

    with pytest.raises(ValueError, match="non-existent nodes"):
        await index.update_nodes([missing])


@pytest.mark.asyncio
async def test_update_relations_replaces_existing_payload(index):
    entities = [
        Entity(id="ent-a", entity_name="A", entity_type="Node", description="A", source_chunk_id=["chunk-a"]),
        Entity(id="ent-b", entity_name="B", entity_type="Node", description="B", source_chunk_id=["chunk-b"]),
        Entity(id="ent-c", entity_name="C", entity_type="Node", description="C", source_chunk_id=["chunk-c"]),
    ]
    await index.upsert_nodes(entities)

    original = Relation(
        id="rel-update",
        subject_id="ent-a",
        object_id="ent-b",
        subject_name="A",
        object_name="B",
        relation_type="LINKS",
        description="old relation",
        relation_strength=1.0,
        source_chunk_id=["chunk-old"],
    )
    updated = Relation(
        id="rel-update",
        subject_id="ent-a",
        object_id="ent-b",
        subject_name="A",
        object_name="B",
        relation_type="LINKS",
        description="new relation",
        relation_strength=7.0,
        source_chunk_id=["chunk-new"],
    )

    await index.upsert_edges([original])
    await index.update_edges([updated])

    updated_edge = await index.get_edges([("ent-a", "ent-b", "rel-update")])

    assert updated_edge[0] is not None
    assert updated_edge[0].description == "new relation"
    assert updated_edge[0].relation_strength == 7.0
    assert updated_edge[0].source_chunk_id == ["chunk-new"]


@pytest.mark.asyncio
async def test_update_relations_fails_for_missing_id(index, sample_entities):
    await index.upsert_nodes(sample_entities)

    missing_relation = Relation(
        id="rel-missing",
        subject_id="ent-1",
        object_id="ent-2",
        subject_name="Alice",
        object_name="Bob",
        relation_type="KNOWS",
        description="missing relation",
        relation_strength=1.0,
        source_chunk_id=["chunk-x"],
    )

    with pytest.raises(ValueError, match="non-existent edges"):
        await index.update_edges([missing_relation])


@pytest.mark.asyncio
async def test_upsert_chunks(index):
    """
    Test upserting chunks.
    """
    chunk1 = Chunk(content="Some text", chunk_order_idx=0, doc_id="doc-1")
    chunk2 = Chunk(content="More text", chunk_order_idx=1, doc_id="doc-1")
    chunks = [chunk1, chunk2]

    await index.upsert_chunks(chunks)

    # Verify chunks exist
    retrieved = await index.get_chunks([chunk1.id, chunk2.id])
    assert len(retrieved) == 2
    assert all(c is not None for c in retrieved)
    assert [chunk.id for chunk in retrieved if chunk is not None] == [chunk1.id, chunk2.id]
    assert [chunk.content for chunk in retrieved if chunk is not None] == ["Some text", "More text"]
    assert [chunk.chunk_order_idx for chunk in retrieved if chunk is not None] == [0, 1]
    assert [chunk.doc_id for chunk in retrieved if chunk is not None] == ["doc-1", "doc-1"]

    vector_points = await index.chunks_vector_db.get_points_by_ids([chunk1.id, chunk2.id])
    assert [point.id for point in vector_points if point is not None] == [chunk1.id, chunk2.id]
    assert [point.metadata["content"] for point in vector_points if point is not None] == ["Some text", "More text"]
    assert [point.metadata["doc_id"] for point in vector_points if point is not None] == ["doc-1", "doc-1"]


@pytest.mark.asyncio
async def test_delete_chunks_cascade(index):
    """
    Test deleting chunks cascades to related entities and relations.
    """
    chunk1 = Chunk(content="Chunk one text", chunk_order_idx=0, doc_id="doc-1")
    chunk2 = Chunk(content="Chunk two text", chunk_order_idx=1, doc_id="doc-1")
    chunk3 = Chunk(content="Chunk three text", chunk_order_idx=2, doc_id="doc-1")

    entities = [
        Entity(
            id="ent-1",
            entity_name="Alice",
            entity_type="Person",
            description="Alice",
            source_chunk_id=[chunk1.id],
            documents_id=["doc-1"],
            clusters=[],
        ),
        Entity(
            id="ent-2",
            entity_name="Bob",
            entity_type="Person",
            description="Bob",
            source_chunk_id=[chunk2.id],
            documents_id=["doc-1"],
            clusters=[],
        ),
        Entity(
            id="ent-3",
            entity_name="Charlie",
            entity_type="Person",
            description="Charlie",
            source_chunk_id=[chunk3.id],
            documents_id=["doc-1"],
            clusters=[],
        ),
    ]

    relations = [
        Relation(
            id="rel-1",
            subject_id="ent-1",
            object_id="ent-2",
            subject_name="Alice",
            object_name="Bob",
            relation_type="KNOWS",
            description="Alice knows Bob",
            relation_strength=1.0,
            source_chunk_id=[chunk1.id],
        ),
        Relation(
            id="rel-2",
            subject_id="ent-2",
            object_id="ent-3",
            subject_name="Bob",
            object_name="Charlie",
            relation_type="KNOWS",
            description="Bob knows Charlie",
            relation_strength=1.0,
            source_chunk_id=[chunk2.id],
        ),
    ]

    await index.upsert_nodes(entities)
    await index.upsert_edges(relations)
    await index.upsert_chunks([chunk1, chunk2, chunk3])

    await index.delete_chunks([chunk1.id])

    chunks_after_delete = await index.get_chunks([chunk1.id, chunk2.id, chunk3.id])
    assert chunks_after_delete[0] is None
    assert chunks_after_delete[1] is not None
    assert chunks_after_delete[2] is not None

    entities_after_delete = await index.get_nodes(["ent-1", "ent-2", "ent-3"])
    assert entities_after_delete[0] is None
    assert entities_after_delete[1] is not None
    assert entities_after_delete[2] is not None

    relations_after_delete = await index.get_edges(
        [
            ("ent-1", "ent-2", "rel-1"),
            ("ent-2", "ent-3", "rel-2"),
        ]
    )
    assert relations_after_delete[0] is None
    assert relations_after_delete[1] is not None


@pytest.mark.asyncio
async def test_delete_chunks_removes_relations_tied_only_to_deleted_chunk(index):
    chunk_entity_a = Chunk(content="Chunk entity A", chunk_order_idx=0, doc_id="doc-1")
    chunk_entity_b = Chunk(content="Chunk entity B", chunk_order_idx=1, doc_id="doc-1")
    chunk_relation = Chunk(content="Chunk relation only", chunk_order_idx=2, doc_id="doc-1")

    entities = [
        Entity(
            id="ent-1",
            entity_name="Alice",
            entity_type="Person",
            description="Alice",
            source_chunk_id=[chunk_entity_a.id],
            documents_id=["doc-1"],
            clusters=[],
        ),
        Entity(
            id="ent-2",
            entity_name="Bob",
            entity_type="Person",
            description="Bob",
            source_chunk_id=[chunk_entity_b.id],
            documents_id=["doc-1"],
            clusters=[],
        ),
    ]
    relation = Relation(
        id="rel-1",
        subject_id="ent-1",
        object_id="ent-2",
        subject_name="Alice",
        object_name="Bob",
        relation_type="KNOWS",
        description="Alice knows Bob",
        relation_strength=1.0,
        source_chunk_id=[chunk_relation.id],
    )

    await index.upsert_nodes(entities)
    await index.upsert_edges([relation])
    await index.upsert_chunks([chunk_entity_a, chunk_entity_b, chunk_relation])

    await index.delete_chunks([chunk_relation.id])

    entities_after_delete = await index.get_nodes(["ent-1", "ent-2"])
    assert entities_after_delete[0] is not None
    assert entities_after_delete[1] is not None

    relations_after_delete = await index.get_edges([("ent-1", "ent-2", "rel-1")])
    assert relations_after_delete[0] is None


@pytest.mark.asyncio
async def test_check_consistency_reports_clean_graph(index):
    chunk = Chunk(content="Chunk one text", chunk_order_idx=0, doc_id="doc-1")
    await index.upsert_chunks([chunk])
    await index.upsert_nodes([
        Entity(
            id="ent-1",
            entity_name="Alice",
            entity_type="Person",
            description="Alice",
            source_chunk_id=[chunk.id],
            documents_id=["doc-1"],
            clusters=[],
        ),
        Entity(
            id="ent-2",
            entity_name="Bob",
            entity_type="Person",
            description="Bob",
            source_chunk_id=[chunk.id],
            documents_id=["doc-1"],
            clusters=[],
        ),
    ])
    await index.upsert_edges([
        Relation(
            id="rel-1",
            subject_id="ent-1",
            object_id="ent-2",
            subject_name="Alice",
            object_name="Bob",
            relation_type="KNOWS",
            description="Alice knows Bob",
            relation_strength=1.0,
            source_chunk_id=[chunk.id],
        )
    ])
    await index.upsert_communities([
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
                    source_chunk_id=[chunk.id],
                )
            ],
            relations=[
                Relation(
                    id="rel-1",
                    subject_id="ent-1",
                    object_id="ent-2",
                    subject_name="Alice",
                    object_name="Bob",
                    relation_type="KNOWS",
                    description="Alice knows Bob",
                    source_chunk_id=[chunk.id],
                )
            ],
        )
    ])

    report = await index.check_consistency()

    assert report.is_consistent
    assert report.errors == []


@pytest.mark.asyncio
async def test_check_consistency_reports_missing_relation_endpoint(index):
    await index.upsert_chunks([Chunk(content="Chunk one text", chunk_order_idx=0, doc_id="doc-1")])
    await index.upsert_nodes([
        Entity(
            id="ent-1",
            entity_name="Alice",
            entity_type="Person",
            description="Alice",
            source_chunk_id=[],
            documents_id=["doc-1"],
            clusters=[],
        )
    ])

    bad_relation = Relation(
        id="rel-1",
        subject_id="ent-1",
        object_id="ent-missing",
        subject_name="Alice",
        object_name="Ghost",
        relation_type="KNOWS",
        description="Alice knows Ghost",
        relation_strength=1.0,
        source_chunk_id=[],
    )
    index.graph_backend.get_all_edges = AsyncMock(return_value=[bad_relation])

    report = await index.check_consistency()
    relation_issue = next(issue for issue in report.errors if issue.check == "relation_endpoints")

    assert not report.is_consistent
    assert any(issue.check == "relation_endpoints" for issue in report.errors)
    assert relation_issue.details["missing_entity_ids"] == ["ent-missing"]
    assert relation_issue.details["relation_ids_with_empty_endpoints"] == ["rel-1"]


@pytest.mark.asyncio
async def test_check_consistency_reports_missing_chunk_and_community_references(index):
    await index.graph_backend.upsert_nodes([
        Entity(
            id="ent-1",
            entity_name="Alice",
            entity_type="Person",
            description="Alice",
            source_chunk_id=["chunk-missing"],
            documents_id=["doc-1"],
            clusters=[],
        )
    ])
    await index.graph_backend.index_done_callback()
    await index.community_kv_storage.upsert({
        "com-1": {
            "level": 0,
            "cluster_id": 1,
            "entity_ids": ["ent-1", "ent-missing"],
            "relation_ids": ["rel-missing"],
        }
    })
    await index.community_kv_storage.index_done_callback()

    report = await index.check_consistency()

    assert not report.is_consistent
    assert any(issue.check == "source_chunk_references" for issue in report.errors)
    assert any(issue.check == "community_references" for issue in report.errors)


@pytest.mark.asyncio
async def test_check_consistency_reports_relation_vector_endpoint_issues(index):
    await index.upsert_chunks([Chunk(content="Chunk one text", chunk_order_idx=0, doc_id="doc-1")])
    await index.upsert_nodes([
        Entity(
            id="ent-1",
            entity_name="Alice",
            entity_type="Person",
            description="Alice",
            source_chunk_id=[],
            documents_id=["doc-1"],
            clusters=[],
        ),
        Entity(
            id="ent-2",
            entity_name="Bob",
            entity_type="Person",
            description="Bob",
            source_chunk_id=[],
            documents_id=["doc-1"],
            clusters=[],
        ),
    ])
    await index.upsert_edges([
        Relation(
            id="rel-1",
            subject_id="ent-1",
            object_id="ent-2",
            subject_name="Alice",
            object_name="Bob",
            relation_type="KNOWS",
            description="Alice knows Bob",
            relation_strength=1.0,
            source_chunk_id=[],
        )
    ])
    await index.edges_vector_db.upsert([
        Point(
            id="rel-bad",
            dense_embedding=[0.1] * 128,
            metadata={"content": "broken relation"},
        )
    ])
    await index.edges_vector_db.index_done_callback()

    report = await index.check_consistency()

    assert not report.is_consistent
    assert any(issue.check == "relation_vector_endpoints" for issue in report.errors)


@pytest.mark.asyncio
async def test_check_consistency_reports_entity_and_chunk_vector_endpoint_issues(index):
    chunk = Chunk(content="Chunk one text", chunk_order_idx=0, doc_id="doc-1")
    await index.upsert_chunks([chunk])
    await index.upsert_nodes([
        Entity(
            id="ent-1",
            entity_name="Alice",
            entity_type="Person",
            description="Alice",
            source_chunk_id=[chunk.id],
            documents_id=["doc-1"],
            clusters=[],
        )
    ])
    existing_relations = await index.graph_backend.get_all_edges()
    index.graph_backend.get_all_edges = AsyncMock(return_value=[
        *existing_relations,
        Relation(
            id="rel-bad-endpoint",
            subject_id="ent-bad",
            object_id="ent-1",
            subject_name="Ghost",
            object_name="Alice",
            relation_type="KNOWS",
            description="Ghost knows Alice",
            relation_strength=1.0,
            source_chunk_id=[],
        )
    ])

    await index.nodes_vector_db.upsert([
        Point(
            id="ent-bad",
            dense_embedding=[0.1] * 128,
            metadata={"content": "broken entity"},
        )
    ])
    await index.nodes_vector_db.index_done_callback()
    await index.chunks_vector_db.upsert([
        Point(
            id="chunk-bad",
            dense_embedding=[0.1] * 128,
            metadata={"content": "broken chunk", "doc_id": "doc-1"},
        )
    ])
    await index.chunks_vector_db.index_done_callback()

    report = await index.check_consistency()
    entity_vector_issue = next(issue for issue in report.errors if issue.check == "entity_vector_endpoints")
    relation_issue = next(issue for issue in report.errors if issue.check == "relation_endpoints")

    assert not report.is_consistent
    assert any(issue.check == "relation_endpoints" for issue in report.errors)
    assert any(issue.check == "entity_vector_endpoints" for issue in report.errors)
    assert any(issue.check == "chunk_vector_endpoints" for issue in report.errors)
    assert relation_issue.details["missing_entity_ids"] == ["ent-bad"]
    assert relation_issue.details["relation_ids_with_empty_endpoints"] == ["rel-bad-endpoint"]
    assert entity_vector_issue.details["orphan_vector_ids"] == ["ent-bad"]


@pytest.mark.asyncio
async def test_check_consistency_reports_missing_vector_representations(index):
    chunk = Chunk(content="Chunk one text", chunk_order_idx=0, doc_id="doc-1")
    await index.upsert_chunks([chunk])
    await index.upsert_nodes([
        Entity(
            id="ent-1",
            entity_name="Alice",
            entity_type="Person",
            description="Alice",
            source_chunk_id=[chunk.id],
            documents_id=["doc-1"],
            clusters=[],
        ),
        Entity(
            id="ent-2",
            entity_name="Bob",
            entity_type="Person",
            description="Bob",
            source_chunk_id=[chunk.id],
            documents_id=["doc-1"],
            clusters=[],
        ),
    ])
    await index.upsert_edges([
        Relation(
            id="rel-1",
            subject_id="ent-1",
            object_id="ent-2",
            subject_name="Alice",
            object_name="Bob",
            relation_type="KNOWS",
            description="Alice knows Bob",
            relation_strength=1.0,
            source_chunk_id=[chunk.id],
        )
    ])

    await index.nodes_vector_db.delete(["ent-1"])
    await index.nodes_vector_db.index_done_callback()
    await index.edges_vector_db.delete(["rel-1"])
    await index.edges_vector_db.index_done_callback()
    await index.chunks_vector_db.delete([chunk.id])
    await index.chunks_vector_db.index_done_callback()

    report = await index.check_consistency()

    assert not report.is_consistent
    assert any(issue.check == "entity_vector_representations" for issue in report.errors)
    assert any(issue.check == "relation_vector_representations" for issue in report.errors)
    assert any(issue.check == "chunk_vector_representations" for issue in report.errors)


def test_consistency_report_str():
    report = ConsistencyReport(errors=[
        ConsistencyIssue(
            check="relation_endpoints",
            message="Relations reference entity endpoints that do not exist in the graph.",
            details={"missing_entity_ids": ["ent-missing"]},
        )
    ])

    rendered = str(report)

    assert "Graph consistency: FAILED" in rendered
    assert "Issues found: 1" in rendered
    assert "- relation_endpoints: Relations reference entity endpoints that do not exist in the graph." in rendered
    assert "missing_entity_ids: ent-missing" in rendered


@pytest.mark.asyncio
async def test_close_releases_every_backend(index):
    """
    Index.close() must reach all seven storages, and stay safe when repeated.

    Backends holding a connection (Neo4j, remote Qdrant) leak a pool otherwise;
    file-backed ones inherit a no-op from BaseStorage.
    """
    closed = []

    for name in (
        "graph_backend",
        "nodes_vector_db",
        "edges_vector_db",
        "chunks_vector_db",
        "chunks_kv_storage",
        "community_summary_kv_storage",
        "community_kv_storage",
    ):
        storage = getattr(index, name)

        async def record(_name=name):
            closed.append(_name)

        storage.close = record

    await index.close()
    await index.close()

    assert sorted(closed) == sorted([
        "graph_backend",
        "nodes_vector_db",
        "edges_vector_db",
        "chunks_vector_db",
        "chunks_kv_storage",
        "community_summary_kv_storage",
        "community_kv_storage",
    ] * 2)
