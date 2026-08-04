import asyncio
import json

import pytest

from ragu.graph.types import Entity, Relation

try:
    from ragu.storage.graph_storage_adapters.ladybug_adapter import LadybugGraphStorage
except ImportError:
    LadybugGraphStorage = None

pytestmark = pytest.mark.skipif(LadybugGraphStorage is None, reason="ladybug package is not installed")


def _entity(
    entity_id: str,
    name: str,
    entity_type: str = "PERSON",
    clusters: list[dict] | None = None,
) -> Entity:
    return Entity(
        id=entity_id,
        entity_name=name,
        entity_type=entity_type,
        description=f"{name} description",
        source_chunk_id=["chunk-1"],
        documents_id=["doc-1"],
        clusters=clusters or [],
    )


def _relation(
    subject_id: str,
    object_id: str,
    edge_id: str,
    relation_type: str = "KNOWS",
) -> Relation:
    return Relation(
        id=edge_id,
        subject_id=subject_id,
        object_id=object_id,
        subject_name=subject_id,
        object_name=object_id,
        relation_type=relation_type,
        description=f"{subject_id} -> {object_id}",
        relation_strength=1.0,
        source_chunk_id=["chunk-1"],
    )


async def _store(tmp_path, **kwargs):
    store = LadybugGraphStorage(
        filename=str(tmp_path / "graph.lbdb"),
        node_cls=Entity,
        edge_cls=Relation,
        **kwargs,
    )
    await store.index_start_callback()
    return store


def test_ladybug_public_reexports():
    from ragu.storage.graph_storage_adapters import LadybugGraphStorage as AdapterLadybugGraphStorage
    assert AdapterLadybugGraphStorage is LadybugGraphStorage


@pytest.mark.asyncio
async def test_ladybug_persists_across_adapter_recreation(tmp_path):
    first = await _store(tmp_path)
    await first.upsert_nodes([_entity("e1", "Alice"), _entity("e2", "Bob")])
    await first.upsert_edges([_relation("e1", "e2", "r1")])
    await first.index_done_callback()
    await first.close()

    second = await _store(tmp_path)
    nodes = await second.get_nodes(["e1"])
    edges = await second.get_edges([("e1", "e2", "r1")])

    assert nodes[0] is not None
    assert nodes[0].entity_name == "Alice"
    assert [edge.id for edge in edges[0]] == ["r1"]
    await second.close()


@pytest.mark.asyncio
async def test_ladybug_reopens_after_a_session_that_deleted_records(tmp_path):
    """Closing must checkpoint the WAL, or a session with deletions loses the graph.

    ``close()`` used to release only the async connection, leaving the
    write-ahead log unfinished. Every read and write in the session succeeded,
    but reopening the database then failed with ``Checksum verification failed,
    the WAL file is corrupted`` - the stored graph was unrecoverable.

    The size below is deliberate. Deletions are the trigger, but only once the
    WAL outgrows its first checkpoint: a handful of records reopens cleanly even
    with the bug present, which is why the rest of this suite never caught it.
    The corruption appears reproducibly between 1500 and 2000 nodes, so 3000
    keeps a margin while still running in well under a second. Do not shrink
    this workload - a smaller one silently stops testing anything.
    """
    entities = [_entity(f"e{i}", f"Name{i}") for i in range(3000)]

    first = await _store(tmp_path)
    await first.upsert_nodes(entities)
    # Edges live above the deleted range so the cascade does not remove them too.
    await first.upsert_edges([_relation(f"e{1000 + i}", f"e{1001 + i}", f"r{i}") for i in range(5)])
    await first.delete_edges([("e1000", "e1001", "r0")])
    await first.delete_nodes([f"e{i}" for i in range(300)])
    await first.close()

    second = await _store(tmp_path)

    assert len(await second.get_all_nodes()) == 2700
    assert sorted(edge.id for edge in await second.get_all_edges()) == ["r1", "r2", "r3", "r4"]
    await second.close()


@pytest.mark.asyncio
async def test_ladybug_refuses_a_second_handle_on_the_same_file(tmp_path):
    """Opening one database twice in a process must fail loudly.

    Ladybug locks a file against other processes but not against a second
    handle inside one. Both adapters then read from divergent snapshots and
    their writes interleave into corruption - a node row ends up holding
    another record's payload while its own columns are NULL - with nothing
    raising at the time.
    """
    first = await _store(tmp_path)

    with pytest.raises(RuntimeError, match="already open in this process"):
        LadybugGraphStorage(
            filename=str(tmp_path / "graph.lbdb"),
            node_cls=Entity,
            edge_cls=Relation,
        )

    await first.close()


@pytest.mark.asyncio
async def test_ladybug_path_claim_ignores_how_the_path_is_spelled(tmp_path):
    """The guard keys on the resolved path, not the string it was given."""
    first = await _store(tmp_path)
    aliased = str(tmp_path / "sub" / ".." / "graph.lbdb")
    (tmp_path / "sub").mkdir()

    with pytest.raises(RuntimeError, match="already open in this process"):
        LadybugGraphStorage(filename=aliased, node_cls=Entity, edge_cls=Relation)

    await first.close()


@pytest.mark.asyncio
async def test_ladybug_reopens_the_same_file_after_close(tmp_path):
    """Closing releases the claim, so the persistence flow keeps working."""
    first = await _store(tmp_path)
    await first.upsert_nodes([_entity("e1", "Alice")])
    await first.close()

    second = await _store(tmp_path)

    assert (await second.get_nodes(["e1"]))[0] is not None
    await second.close()


@pytest.mark.asyncio
async def test_ladybug_memory_databases_do_not_share_a_claim(tmp_path):
    """``:memory:`` handles are private, so several may coexist."""
    first = LadybugGraphStorage(filename=":memory:", node_cls=Entity, edge_cls=Relation)
    second = LadybugGraphStorage(filename=":memory:", node_cls=Entity, edge_cls=Relation)
    await first.index_start_callback()
    await second.index_start_callback()

    await first.upsert_nodes([_entity("e1", "Alice")])

    assert len(await first.get_all_nodes()) == 1
    assert len(await second.get_all_nodes()) == 0  # separate databases
    await first.close()
    await second.close()


def test_ladybug_failed_construction_leaves_no_claim(tmp_path):
    """A constructor that raises must not leave the path reserved forever."""
    before = len(LadybugGraphStorage._open_databases)
    with pytest.raises(ValueError):
        LadybugGraphStorage(
            filename=str(tmp_path / "graph.lbdb"),
            node_cls=Entity,
            edge_cls=Relation,
            batch_size=0,
        )

    assert len(LadybugGraphStorage._open_databases) == before


@pytest.mark.asyncio
async def test_ladybug_subclass_shares_the_path_registry(tmp_path):
    """A subclass must not get a registry of its own.

    Two handles on one file conflict regardless of which class opened them, so
    the guard reaches the registry through ``LadybugGraphStorage`` by name. A
    subclass that rebinds the attribute - or merely inherits it and is then
    reassigned by someone tidying up - would otherwise open the same file a
    second time without complaint.
    """
    class Subclass(LadybugGraphStorage):
        _open_databases = {}          # deliberately shadowed
        _open_databases_lock = None   # would break a self-routed lookup

    first = await _store(tmp_path)

    with pytest.raises(RuntimeError, match="already open in this process"):
        Subclass(filename=str(tmp_path / "graph.lbdb"), node_cls=Entity, edge_cls=Relation)

    await first.close()


#: Workload for the concurrency tests. Small, but every size tried reproduced
#: the corruption deterministically once the batches interleave, so the numbers
#: only need to give the two calls several chunks each.
_CONCURRENCY_NODES = 200
_CONCURRENCY_BATCH = 20


async def _seeded_store(tmp_path):
    store = await _store(tmp_path, batch_size=_CONCURRENCY_BATCH)
    await store.upsert_nodes([_entity(f"e{i}", f"Name{i}") for i in range(_CONCURRENCY_NODES)])
    await store.upsert_edges([
        _relation(f"e{i}", f"e{(i + 1) % _CONCURRENCY_NODES}", f"r{i}")
        for i in range(_CONCURRENCY_NODES)
    ])
    return store


async def _stub_node_count(store):
    """Count nodes with no payload - the shape a resurrected node takes."""
    rows = await store._query(
        f"MATCH (n:{store._node_table}) WHERE n.payload IS NULL RETURN count(n) AS c",
        None,
        ("c",),
    )
    return rows[0]["c"]


@pytest.mark.asyncio
async def test_ladybug_concurrent_writes_do_not_resurrect_deleted_nodes(tmp_path):
    """Overlapping writes must not recreate deleted nodes as payload-less stubs.

    ``upsert_edges`` confirms its endpoints exist and then merges against them.
    A ``delete_nodes`` landing between those two steps makes the merge recreate
    the deleted nodes with no payload, and every later ``get_all_nodes`` then
    fails on them. Nothing raises while the damage is done.
    """
    store = await _seeded_store(tmp_path)
    doomed = [f"e{i}" for i in range(_CONCURRENCY_NODES // 4)]

    await asyncio.gather(
        store.upsert_edges([
            _relation(f"e{i}", f"e{(i + 1) % _CONCURRENCY_NODES}", f"r{i}")
            for i in range(_CONCURRENCY_NODES)
        ]),
        store.delete_nodes(doomed),
    )

    assert await _stub_node_count(store) == 0
    surviving = await store.get_all_nodes()  # must not raise
    assert len(surviving) == _CONCURRENCY_NODES - len(doomed)
    await store.close()


@pytest.mark.asyncio
async def test_ladybug_concurrency_workload_reproduces_without_the_write_lock(tmp_path):
    """The guard is what keeps the test above green, not a too-small workload.

    Calls the undecorated implementations so the writes interleave as they did
    before ``_serialized_write`` existed. Without this, shrinking the workload
    or dropping the lock would leave the test above passing for the wrong
    reason.
    """
    store = await _seeded_store(tmp_path)
    unguarded_upsert = LadybugGraphStorage.upsert_edges.__wrapped__
    unguarded_delete = LadybugGraphStorage.delete_nodes.__wrapped__

    await asyncio.gather(
        unguarded_upsert(store, [
            _relation(f"e{i}", f"e{(i + 1) % _CONCURRENCY_NODES}", f"r{i}")
            for i in range(_CONCURRENCY_NODES)
        ]),
        unguarded_delete(store, [f"e{i}" for i in range(_CONCURRENCY_NODES // 4)]),
    )

    assert await _stub_node_count(store) > 0
    await store.close()


@pytest.mark.asyncio
async def test_ladybug_close_is_idempotent(tmp_path):
    store = await _store(tmp_path)
    await store.upsert_nodes([_entity("e1", "Alice")])
    await store.close()
    await store.close()


@pytest.mark.asyncio
async def test_ladybug_works_without_an_explicit_index_start_callback(tmp_path):
    """The storage must be usable straight after construction.

    Ladybug is strict-schema, so a query against tables that were never created
    fails with ``Binder exception: Table ... does not exist``. Nothing in RAGU
    calls ``index_start_callback`` - not ``Index``, not ``KnowledgeGraph`` - so
    relying on it made every read and write on a fresh storage fail, including
    the whole ``KnowledgeGraph`` CRUD surface. The rest of this suite calls it
    through ``_store`` and so cannot catch that.
    """
    store = LadybugGraphStorage(
        filename=str(tmp_path / "graph.lbdb"),
        node_cls=Entity,
        edge_cls=Relation,
    )

    assert await store.get_nodes(["missing"]) == [None]
    await store.upsert_nodes([_entity("e1", "Alice"), _entity("e2", "Bob")])
    await store.upsert_edges([_relation("e1", "e2", "r1")])

    assert len(await store.get_all_nodes()) == 2
    assert [edge.id for edge in await store.get_all_edges()] == ["r1"]
    await store.close()


@pytest.mark.asyncio
async def test_ladybug_schema_is_created_once_under_concurrent_first_calls(tmp_path):
    """Concurrent first statements must not race each other into the DDL."""
    store = LadybugGraphStorage(
        filename=str(tmp_path / "graph.lbdb"),
        node_cls=Entity,
        edge_cls=Relation,
    )

    results = await asyncio.gather(*(store.get_nodes([f"e{i}"]) for i in range(8)))

    assert results == [[None]] * 8
    assert store._schema_ready
    await store.upsert_nodes([_entity("e1", "Alice")])
    assert len(await store.get_all_nodes()) == 1
    await store.close()


@pytest.mark.asyncio
async def test_ladybug_supports_configurable_table_names(tmp_path):
    store = await _store(tmp_path, node_table="CustomRaguNode", edge_table="CustomRaguEdge")
    await store.upsert_nodes([_entity("e1", "Alice"), _entity("e2", "Bob")])
    await store.upsert_edges([_relation("e1", "e2", "r1")])

    assert (await store.get_nodes(["e1"]))[0] is not None
    assert [edge.id for edge in (await store.get_all_edges())] == ["r1"]
    await store.close()


@pytest.mark.parametrize("batch_size", [0, -1])
def test_ladybug_rejects_non_positive_batch_size(tmp_path, batch_size):
    """A non-positive batch size must fail loudly, not silently drop writes.

    ``BatchGenerator`` steps a ``range`` by the batch size, so anything below 1
    yields no batches at all and every upsert would report success while
    writing nothing.
    """
    with pytest.raises(ValueError, match="batch_size must be positive"):
        LadybugGraphStorage(
            filename=str(tmp_path / "graph.lbdb"),
            node_cls=Entity,
            edge_cls=Relation,
            batch_size=batch_size,
        )


@pytest.mark.asyncio
async def test_ladybug_honours_configured_batch_size(tmp_path):
    """Writes spanning several batches must persist every record."""
    store = await _store(tmp_path, batch_size=3)
    entities = [_entity(f"e{i}", f"Name{i}") for i in range(10)]
    await store.upsert_nodes(entities)
    await store.upsert_edges([_relation(f"e{i}", f"e{i + 1}", f"r{i}") for i in range(9)])

    assert store._batch_size == 3
    assert len(await store.get_all_nodes()) == 10
    assert len(await store.get_all_edges()) == 9
    await store.close()


@pytest.mark.asyncio
async def test_ladybug_rejects_reserved_graph_name_until_idempotent_setup_exists(tmp_path):
    with pytest.raises(ValueError, match="graph_name is reserved"):
        LadybugGraphStorage(
            filename=str(tmp_path / "graph.lbdb"),
            node_cls=Entity,
            edge_cls=Relation,
            graph_name="tenant_graph",
        )


@pytest.mark.asyncio
async def test_ladybug_structured_payload_and_unknown_fields_round_trip(tmp_path):
    store = await _store(tmp_path)
    clusters = [{"level": 0, "cluster_id": 7}]
    await store.upsert_nodes([_entity("e1", "Alice", clusters=clusters)])

    payload = {
        "entity_name": "Alice",
        "entity_type": "PERSON",
        "description": "edited",
        "source_chunk_id": ["chunk-1"],
        "documents_id": ["doc-1"],
        "clusters": clusters,
        "unknown_field": "ignored",
    }
    await store._run(
        "MATCH (n:RaguNode) WHERE n.id = $id SET n.payload = $payload",
        {"id": "e1", "payload": json.dumps(payload)},
    )

    node = (await store.get_nodes(["e1"]))[0]

    assert node is not None
    assert node.description == "edited"
    assert node.clusters == clusters
    assert not hasattr(node, "unknown_field")
    await store.close()


@pytest.mark.asyncio
async def test_ladybug_mixed_new_and_existing_nodes_all_land(tmp_path):
    """New nodes take the bulk path, already-stored ones must still update.

    ``upsert_nodes`` splits its batch: rows whose id is absent are bulk-loaded
    with ``COPY``, which rejects an existing primary key, and the rest go
    through ``MERGE``. A split that drops either side is the failure this
    guards against.
    """
    store = await _store(tmp_path)
    await store.upsert_nodes([_entity("e1", "Alice"), _entity("e2", "Bob")])

    await store.upsert_nodes([
        _entity("e1", "Alice Updated"),   # existing -> MERGE
        _entity("e3", "Carol"),           # new      -> COPY
    ])

    by_id = {node.id: node for node in await store.get_all_nodes()}
    assert sorted(by_id) == ["e1", "e2", "e3"]
    assert by_id["e1"].entity_name == "Alice Updated"
    assert by_id["e2"].entity_name == "Bob"
    assert by_id["e3"].entity_name == "Carol"
    await store.close()


@pytest.mark.asyncio
async def test_ladybug_writes_are_complete_when_the_bulk_path_is_unavailable(
    tmp_path, monkeypatch
):
    """Falling back from ``COPY`` must still write every record exactly once.

    ``COPY`` is atomic - a rejected load leaves no rows - so the adapter treats
    an unusable bulk path as a speed problem and takes the ``MERGE`` path
    instead. Forcing that branch checks the fallback actually writes, and does
    not double-write what a partial bulk load might have left.
    """
    async def refuse(self, table, columns, rows):
        return False

    monkeypatch.setattr(LadybugGraphStorage, "_copy_rows", refuse)

    store = await _store(tmp_path)
    await store.upsert_nodes([_entity(f"e{i}", f"Name{i}") for i in range(4)])
    await store.upsert_edges([_relation(f"e{i}", f"e{i + 1}", f"r{i}") for i in range(3)])

    assert sorted(node.id for node in await store.get_all_nodes()) == ["e0", "e1", "e2", "e3"]
    assert sorted(edge.id for edge in await store.get_all_edges()) == ["r0", "r1", "r2"]
    await store.close()


@pytest.mark.asyncio
async def test_ladybug_edge_upsert_is_idempotent(tmp_path):
    store = await _store(tmp_path)
    await store.upsert_nodes([_entity("e1", "Alice"), _entity("e2", "Bob")])
    await store.upsert_edges([_relation("e1", "e2", "r1", relation_type="KNOWS")])
    await store.upsert_edges([_relation("e1", "e2", "r1", relation_type="WORKS_WITH")])

    edges = await store.get_edges([("e1", "e2", None)])

    assert len(edges[0]) == 1
    assert edges[0][0].relation_type == "WORKS_WITH"
    await store.close()


@pytest.mark.asyncio
async def test_ladybug_duplicate_node_ids_in_one_batch_keep_last(tmp_path):
    """A repeated id inside one call must resolve to the last one written.

    Ladybug resolves a repeated ``MERGE`` key within a batch to the *first* row,
    so the adapter deduplicates before sending. Without that, batching would
    silently invert the last-write-wins order of the per-node loop.
    """
    store = await _store(tmp_path)
    await store.upsert_nodes([
        _entity("e1", "First", entity_type="PERSON"),
        _entity("e1", "Last", entity_type="ORGANIZATION"),
    ])

    node = (await store.get_nodes(["e1"]))[0]

    assert node is not None
    assert node.entity_name == "Last"
    assert node.entity_type == "ORGANIZATION"
    assert len(await store.get_all_nodes()) == 1
    await store.close()


@pytest.mark.asyncio
async def test_ladybug_duplicate_edge_keys_in_one_batch_keep_last(tmp_path):
    store = await _store(tmp_path)
    await store.upsert_nodes([_entity("e1", "Alice"), _entity("e2", "Bob")])
    await store.upsert_edges([
        _relation("e1", "e2", "r1", relation_type="FIRST"),
        _relation("e1", "e2", "r1", relation_type="LAST"),
    ])

    edges = await store.get_edges([("e1", "e2", "r1")])

    assert [edge.relation_type for edge in edges[0]] == ["LAST"]
    await store.close()


@pytest.mark.asyncio
async def test_ladybug_edge_with_missing_endpoint_creates_no_stub_node(tmp_path):
    """Skipping an edge must not leave a payload-less node behind.

    ``upsert_edges`` merges on its endpoints because that is the form Ladybug
    indexes, which would create empty nodes for unknown ids. Endpoints are
    resolved first to prevent that: a stub node has a NULL payload and would
    make every later read raise while reconstructing the node class.
    """
    store = await _store(tmp_path)
    await store.upsert_nodes([_entity("e1", "Alice"), _entity("e2", "Bob")])
    await store.upsert_edges([
        _relation("e1", "e2", "kept"),
        _relation("e1", "missing", "dropped"),
        _relation("absent", "e2", "dropped-too"),
    ])

    assert sorted(node.id for node in await store.get_all_nodes()) == ["e1", "e2"]
    assert [edge.id for edge in await store.get_all_edges()] == ["kept"]
    await store.close()


@pytest.mark.asyncio
async def test_ladybug_reads_stay_aligned_across_batch_boundaries(tmp_path):
    """Results must realign by position when a request spans several batches.

    Batched statements return rows in storage order, so every read tags its
    rows with the requesting position. A tiny batch size exercises the
    regrouping that a default-sized test would never reach.
    """
    store = await _store(tmp_path, batch_size=2)
    await store.upsert_nodes([_entity(f"e{i}", f"Name{i}") for i in range(5)])
    await store.upsert_edges([_relation(f"e{i}", f"e{i + 1}", f"r{i}") for i in range(4)])

    requested = ["e3", "missing", "e0", "e3", "e1"]
    nodes = await store.get_nodes(requested)
    assert [node.id if node else None for node in nodes] == ["e3", None, "e0", "e3", "e1"]

    groups = await store.get_edges([
        ("e0", "e1", None),
        ("nope", "nope", None),
        ("e2", "e3", "r2"),
        ("e0", "e1", None),
    ])
    assert [sorted(e.id for e in g) for g in groups] == [["r0"], [], ["r2"], ["r0"]]

    incident = await store.get_all_edges_for_nodes(["e0", "missing", "e2", "e0"])
    assert [sorted(e.id for e in g) for g in incident] == [["r0"], [], ["r1", "r2"], ["r0"]]

    assert await store.edges_degrees([("e0", "e1", "r0"), ("gone", "away", None)]) == [3, 0]
    await store.close()


@pytest.mark.asyncio
async def test_ladybug_self_loop_dedup_survives_concurrent_batches(tmp_path):
    """Self-loop deduplication must hold when chunks run concurrently.

    Incident edges are read as two directed lookups, so a loop matches twice and
    is deduplicated per position. The chunks of each direction are now issued
    together via ``asyncio.gather``, so the surviving row must not depend on
    which query finished first. A tiny batch size forces many concurrent chunks.
    """
    store = await _store(tmp_path, batch_size=2)
    await store.upsert_nodes([_entity(f"e{i}", f"Name{i}") for i in range(9)])
    await store.upsert_edges([
        _relation("e0", "e0", "loop0"),
        _relation("e0", "e1", "out0"),
        _relation("e2", "e0", "in0"),
        _relation("e4", "e4", "loop4"),
        _relation("e7", "e7", "loop7"),
    ])

    requested = ["e0", "e4", "e7", "missing", "e0"]
    expected = [
        ["in0", "loop0", "out0"], ["loop4"], ["loop7"], [], ["in0", "loop0", "out0"],
    ]

    # Repeat: gather completion order varies between runs, the result must not.
    for _ in range(5):
        groups = await store.get_all_edges_for_nodes(requested)
        assert [sorted(edge.id for edge in group) for group in groups] == expected

    await store.close()


@pytest.mark.asyncio
async def test_ladybug_preserves_multiple_edges_between_ordered_pair(tmp_path):
    store = await _store(tmp_path)
    await store.upsert_nodes([_entity("e1", "Alice"), _entity("e2", "Bob")])
    await store.upsert_edges([
        _relation("e1", "e2", "r1", relation_type="KNOWS"),
        _relation("e1", "e2", "r2", relation_type="WORKS_WITH"),
    ])

    groups = await store.get_edges([
        ("e1", "e2", None),
        ("e1", "e2", "r2"),
    ])

    assert sorted(edge.id for edge in groups[0]) == ["r1", "r2"]
    assert [edge.id for edge in groups[1]] == ["r2"]
    await store.close()
