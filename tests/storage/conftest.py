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
        node_cls=Entity,
        edge_cls=Relation,
    )
    try:
        await store._verify_connectivity()
    except Exception as exc:  # server not running in this environment
        await store.close()
        pytest.skip(f"Neo4j is not reachable at {NEO4J_URI}: {type(exc).__name__}")

    async def wipe():
        nodes = await store.get_all_nodes()
        if nodes:
            await store.delete_nodes([node.id for node in nodes])

    await wipe()
    yield store
    await wipe()
    await store.close()
