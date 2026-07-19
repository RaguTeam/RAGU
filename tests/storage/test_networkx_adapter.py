import pytest

from ragu.graph.types import Entity, Relation
from ragu.storage.graph_storage_adapters.networkx_adapter import NetworkXStorage


@pytest.fixture
def storage(tmp_path):
    return NetworkXStorage(
        filename=str(tmp_path / "graph.gml"),
        node_cls=Entity,
        edge_cls=Relation,
    )


@pytest.fixture
def entities():
    return {
        "alice": Entity(
            id="ent-1",
            entity_name="Alice",
            entity_type="Person",
            description="Engineer",
            source_chunk_id=["chunk-1"],
            documents_id=["doc-1"],
            clusters=[],
        ),
        "bob": Entity(
            id="ent-2",
            entity_name="Bob",
            entity_type="Person",
            description="Scientist",
            source_chunk_id=["chunk-2"],
            documents_id=["doc-1"],
            clusters=[],
        ),
        "acme": Entity(
            id="ent-3",
            entity_name="Acme",
            entity_type="Organization",
            description="Company",
            source_chunk_id=["chunk-3"],
            documents_id=["doc-2"],
            clusters=[],
        ),
    }


@pytest.fixture
def relation():
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


@pytest.mark.asyncio
async def test_single_node_and_edge_crud(storage, entities, relation):
    await storage.upsert_nodes([entities["alice"], entities["bob"]])

    nodes = await storage.get_nodes(["ent-1", "ent-999"])
    assert nodes[0] is not None
    assert nodes[0].entity_name == "Alice"
    assert nodes[1] is None

    await storage.upsert_edges([relation])

    edges = await storage.get_edges([("ent-1", "ent-2", "rel-1"), ("ent-2", "ent-3", None)])
    assert edges[0] is not None
    assert edges[0].relation_type == "KNOWS"
    assert edges[1] is None

    await storage.delete_edges([("ent-1", "ent-2", "rel-1")])
    deleted = await storage.get_edges([("ent-1", "ent-2", "rel-1")])
    assert deleted[0] is None

    await storage.delete_nodes(["ent-2"])
    after_delete = await storage.get_nodes(["ent-2"])
    assert after_delete[0] is None


@pytest.mark.asyncio
async def test_edges_and_degree_stub(storage, entities):
    await storage.upsert_nodes([entities["alice"], entities["bob"], entities["acme"]])
    await storage.upsert_edges(
        [
            Relation(
                id="rel-1",
                subject_id="ent-1",
                object_id="ent-2",
                subject_name="Alice",
                object_name="Bob",
                relation_type="KNOWS",
                description="A-B",
                relation_strength=1.0,
                source_chunk_id=[],
            ),
            Relation(
                id="rel-2",
                subject_id="ent-1",
                object_id="ent-3",
                subject_name="Alice",
                object_name="Acme",
                relation_type="WORKS_FOR",
                description="A-C",
                relation_strength=1.0,
                source_chunk_id=[],
            ),
        ]
    )

    node_edges = (await storage.get_all_edges_for_nodes(["ent-1"]))[0]
    assert len(node_edges) == 2
    assert (await storage.get_all_edges_for_nodes(["ent-999"]))[0] == []

    edge_degrees = await storage.edges_degrees(
        [("ent-1", "ent-2", "rel-1"), ("ent-1", "ent-3", "rel-2"), ("ent-404", "ent-405", None)]
    )
    assert edge_degrees == [3, 3, 0]


@pytest.mark.asyncio
async def test_iterators_and_batch_delete(storage, entities):
    await storage.upsert_nodes(list(entities.values()))
    await storage.upsert_edges([
        Relation(
            id="rel-1",
            subject_id="ent-1",
            object_id="ent-2",
            subject_name="Alice",
            object_name="Bob",
            relation_type="KNOWS",
            description="A-B",
            relation_strength=1.0,
            source_chunk_id=[],
        )
    ])

    nodes = await storage.get_all_nodes()
    assert {node.id for node in nodes} == {"ent-1", "ent-2", "ent-3"}

    edges = await storage.get_all_edges()
    assert len(edges) == 1
    assert edges[0].id == "rel-1"

    await storage.delete_nodes(["ent-3", "ent-404"])
    after_delete = await storage.get_nodes(["ent-3"])
    assert after_delete[0] is None


@pytest.mark.asyncio
async def test_index_done_callback_persists_and_reloads(tmp_path, entities, relation):
    filename = tmp_path / "graph.gml"
    storage = NetworkXStorage(
        filename=str(filename),
        node_cls=Entity,
        edge_cls=Relation,
    )

    await storage.upsert_nodes([entities["alice"], entities["bob"]])
    await storage.upsert_edges([relation])
    await storage.index_done_callback()

    reloaded = NetworkXStorage(
        filename=str(filename),
        node_cls=Entity,
        edge_cls=Relation,
    )
    nodes = await reloaded.get_nodes(["ent-1"])
    edges = await reloaded.get_edges([("ent-1", "ent-2", "rel-1")])
    assert nodes[0] is not None
    assert nodes[0].entity_type == "Person"
    assert edges[0] is not None
