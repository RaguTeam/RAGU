from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pytest

from ragu.storage.vdb_storage_adapters.nano_vdb import NanoVectorDBStorage

from tests.storage.qdrant_testkit import load_qdrant_storage


@dataclass(frozen=True)
class VDBBackendCase:
    name: str
    factory: Callable[[Path, pytest.MonkeyPatch], object]
    supports_sparse: bool = False
    supports_persistence: bool = True


def _make_nano_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    return NanoVectorDBStorage(
        embedding_dim=3,
        filename=str(tmp_path / "vdb.json"),
        cosine_threshold=0.0,
    )


def _make_qdrant_dense_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    QdrantVectorDBStorage = load_qdrant_storage(monkeypatch)
    return QdrantVectorDBStorage(
        embedding_dim=3,
        filename=str(tmp_path / "vdb.json"),
    )


def _make_qdrant_sparse_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    QdrantVectorDBStorage = load_qdrant_storage(monkeypatch)
    return QdrantVectorDBStorage(
        embedding_dim=3,
        filename=str(tmp_path / "vdb.json"),
        sparse_type="bm42",
    )


ALL_VDB_CASES = [
    VDBBackendCase(
        name="nano",
        factory=_make_nano_storage,
        supports_sparse=False,
    ),
    VDBBackendCase(
        name="qdrant-dense",
        factory=_make_qdrant_dense_storage,
        supports_sparse=False,
    ),
    VDBBackendCase(
        name="qdrant-bm42",
        factory=_make_qdrant_sparse_storage,
        supports_sparse=True,
    ),
]


def pytest_addoption(parser):
    parser.addoption(
        "--vdb-backend",
        action="append",
        default=[],
        help="Run shared VDB contract tests only for the selected backend ids.",
    )
    parser.addoption(
        "--graph-backend",
        action="append",
        default=[],
        help="Run shared graph contract tests only for the selected backend ids.",
    )


@pytest.fixture(params=ALL_VDB_CASES, ids=lambda case: case.name)
def vdb_backend_case(request, pytestconfig):
    case = request.param
    selected = pytestconfig.getoption("--vdb-backend")
    if selected and case.name not in selected:
        pytest.skip(f"Backend {case.name} not selected")
    return case


@pytest.fixture
def vdb_storage(vdb_backend_case, tmp_path, monkeypatch):
    return vdb_backend_case.factory(tmp_path, monkeypatch)


# --------------------------------------------------------------------------- #
# Shared graph storage backends
# --------------------------------------------------------------------------- #

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "testpassword")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

#: Set to run the graph tests against a database that already holds data.
#: They wipe it, so the default is to skip rather than destroy someone's graph.
ALLOW_WIPE_VAR = "RAGU_TEST_ALLOW_NEO4J_WIPE"


async def prepare_neo4j_store(store):
    """
    Make a Neo4j-backed store usable for a test, or skip.

    Skips when the driver is missing or the server is unreachable, so the shared
    suites still run without one. Also skips when the target database already
    holds data: these tests wipe it, and Neo4j Community offers no second
    database to isolate into, so the working graph of whoever runs pytest is at
    stake. Set ``RAGU_TEST_ALLOW_NEO4J_WIPE=1`` to proceed anyway.

    :param store: Neo4jStorage instance to prepare.
    :raises pytest.skip.Exception: When the store cannot be used safely.
    """
    try:
        await store._verify_connectivity()
    except Exception as exc:
        await store.close()
        pytest.skip(f"Neo4j is not reachable at {NEO4J_URI}: {type(exc).__name__}")

    # Count rather than list: get_all_nodes() materializes typed objects and
    # would fail on any node that does not fit the node class, which is exactly
    # the foreign data this guard exists to notice.
    rows = await store.run_cypher_query("MATCH (n) RETURN count(n) AS total")
    existing = rows[0]["total"] if rows else 0
    if existing and not os.getenv(ALLOW_WIPE_VAR):
        await store.close()
        pytest.skip(
            f"database '{NEO4J_DATABASE}' at {NEO4J_URI} already holds "
            f"{existing} nodes; these tests would delete them. Point "
            f"NEO4J_URI at a scratch server, or set {ALLOW_WIPE_VAR}=1."
        )


async def wipe_neo4j_store(store):
    """
    Remove everything from the store's database.

    Deletes through Cypher rather than ``get_all_nodes`` + ``delete_nodes``:
    materializing typed nodes fails on anything that does not fit the node
    class, and a test fixture must be able to clean up regardless.

    :param store: Neo4jStorage instance to clear.
    """
    await store.run_cypher_query("MATCH (n) DETACH DELETE n")


@dataclass(frozen=True)
class GraphBackendCase:
    name: str
    requires_server: bool = False


ALL_GRAPH_CASES = [
    GraphBackendCase(name="networkx"),
    GraphBackendCase(name="neo4j", requires_server=True),
]


@pytest.fixture(params=ALL_GRAPH_CASES, ids=lambda case: case.name)
def graph_backend_case(request, pytestconfig):
    case = request.param
    selected = pytestconfig.getoption("--graph-backend")
    if selected and case.name not in selected:
        pytest.skip(f"Backend {case.name} not selected")
    return case


@pytest.fixture
async def graph_storage(graph_backend_case, tmp_path):
    """
    A graph storage backend, cleaned before and after each test.

    Backends that need a running server are skipped when it is unreachable, so
    the shared contract suite still runs locally on the in-process backend.
    """
    from ragu.graph.types import Entity, Relation

    if graph_backend_case.name == "networkx":
        from ragu.storage.graph_storage_adapters.networkx_adapter import NetworkXStorage

        yield NetworkXStorage(
            filename=str(tmp_path / "graph.gml"),
            node_cls=Entity,
            edge_cls=Relation,
        )
        return

    from ragu.storage.graph_storage_adapters.neo4j_adapter import Neo4jStorage

    store = Neo4jStorage(
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE,
        node_cls=Entity,
        edge_cls=Relation,
    )
    await prepare_neo4j_store(store)

    await wipe_neo4j_store(store)
    yield store
    await wipe_neo4j_store(store)
    await store.close()
