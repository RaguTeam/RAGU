"""Adapter-independent lock planning for Index operations."""

from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from contextlib import asynccontextmanager
from copy import deepcopy
from functools import wraps
from inspect import signature
from typing import Any

from ragu.graph._locks import _ElementLocks
from ragu.graph.types import Community
from ragu.storage.base_storage import EdgeSpec


def _index_locked(method: Any) -> Any:
    """Hold the complete operation footprint, including nested facade calls."""
    method_signature = signature(method)

    @wraps(method)
    async def call(self: Any, *args: Any, **kwargs: Any) -> Any:
        args, kwargs = deepcopy((args, kwargs))
        bound = method_signature.bind(self, *args, **kwargs)
        data = next((value for name, value in bound.arguments.items() if name != "self"), ())
        async with self._operation_lock(method.__name__, data):
            for storage in self._write_storages(method.__name__):
                await storage.index_start_callback()
            return await method(self, *args, **kwargs)
    return call


def _graph_locked(method: Any) -> Any:
    """Keep read/merge/write and whole-graph maintenance in one Index scope."""
    operations = {
        "upsert_entities": "upsert_nodes", "upsert_relations": "upsert_edges",
        "reindex_descriptions": "maintenance", "reindex_community": "maintenance",
        "reindex_graph": "maintenance",
    }
    method_signature = signature(method)

    @wraps(method)
    async def call(self: Any, *args: Any, **kwargs: Any) -> Any:
        args, kwargs = deepcopy((args, kwargs))
        bound = method_signature.bind(self, *args, **kwargs)
        data = next((value for name, value in bound.arguments.items() if name != "self"), ())
        async with self.index._operation_lock(operations[method.__name__], data):
            return await method(self, *args, **kwargs)
    return call


class _IndexConcurrency:
    """Shared implementation of fixed read scopes and cascade write scopes."""

    def _init_concurrency(self) -> None:
        self._element_locks = _ElementLocks()

    def _write_storages(self, name: str) -> tuple:
        if name in {"upsert_nodes", "update_nodes"}:
            return self.graph_backend, self.nodes_vector_db
        if name in {"upsert_edges", "update_edges", "delete_edges"}:
            return self.graph_backend, self.edges_vector_db
        if name == "delete_nodes":
            return self.graph_backend, self.nodes_vector_db, self.edges_vector_db
        if name in {"upsert_chunks"}:
            return self.chunks_kv_storage, self.chunks_vector_db
        if name == "delete_chunks":
            return (self.graph_backend, self.nodes_vector_db, self.edges_vector_db,
                    self.chunks_kv_storage, self.chunks_vector_db)
        if name in {"upsert_communities", "update_communities"}:
            return (self.community_kv_storage,)
        if name in {"upsert_summaries", "update_summaries", "delete_summaries"}:
            return (self.community_summary_kv_storage,)
        if name == "delete_communities":
            return self.community_kv_storage, self.community_summary_kv_storage
        return ()

    @asynccontextmanager
    async def _planned_lock(
        self, planner: Callable[[], Awaitable[dict[tuple[str, str], bool]]],
    ) -> AsyncIterator[None]:
        locks = self._element_locks
        if locks.covers({("*", "*"): True}):
            yield
            return
        requirements = await planner()
        if locks.held_requirements() is not None:
            # A nested operation may reuse a scope, never expand or upgrade it.
            async with locks.hold(requirements):
                yield
            return
        while True:
            async with locks.hold(requirements):
                checked = await planner()
                if locks.covers(checked):
                    yield
                    return
            # Discovery never waits for additional elements while holding locks.
            for key, write in checked.items():
                requirements[key] = requirements.get(key, False) or write

    @asynccontextmanager
    async def a_read_lock(
        self,
        node_ids: Iterable[str] = (),
        edge_specs: Iterable[EdgeSpec] = (),
        chunk_ids: Iterable[str] = (),
        community_ids: Iterable[str] = (),
        summary_ids: Iterable[str] = (),
        include_neighbors: bool = False,
        all_summaries: bool = False,
    ) -> AsyncIterator[None]:
        """
        Protect a fixed selection throughout a multi-step read operation.

        :param node_ids: Nodes to protect, including missing IDs.
        :param edge_specs: Relations/pairs to protect with their endpoints.
        :param chunk_ids: Source chunks to protect.
        :param community_ids: Communities and their member nodes/relations.
        :param summary_ids: Community summaries to protect.
        :param include_neighbors: Include incident relations, one-hop neighbors,
            their source chunks, and cluster summaries for the selected nodes.
        :param all_summaries: Protect the complete community/summary collections.
        :raises ValueError: If a nested scope expands or upgrades its parent.
        """
        selection = dict(
            node_ids=tuple(node_ids), edge_specs=tuple(edge_specs),
            chunk_ids=tuple(chunk_ids), community_ids=tuple(community_ids),
            summary_ids=tuple(summary_ids), include_neighbors=include_neighbors,
            all_summaries=all_summaries,
        )
        async with self._planned_lock(lambda: self._read_requirements(**selection)):
            yield

    async def _read_requirements(
        self, node_ids: Iterable[str] = (), edge_specs: Iterable[EdgeSpec] = (),
        chunk_ids: Iterable[str] = (), community_ids: Iterable[str] = (),
        summary_ids: Iterable[str] = (), include_neighbors: bool = False,
        all_summaries: bool = False,
    ) -> dict[tuple[str, str], bool]:
        requirements: dict[tuple[str, str], bool] = {}

        def add(kind: str, ids: Iterable[str]) -> None:
            requirements.update(((kind, item_id), False) for item_id in ids)

        nodes = set(node_ids)
        specs = list(edge_specs)
        add("chunk", chunk_ids)
        add("community", community_ids)
        add("summary", summary_ids)
        if all_summaries:
            add("community", ["*"])
            add("summary", ["*"])
        for subject, object_id, edge_id in specs:
            nodes.update((subject, object_id))
            if edge_id is not None:
                add("edge", [edge_id])
        if community_ids:
            rows = await self.community_kv_storage.get_by_ids(list(community_ids))
            edge_ids: set[str] = set()
            for row in rows:
                if row:
                    nodes.update(row.get("entity_ids", []))
                    edge_ids.update(row.get("relation_ids", []))
            add("edge", edge_ids)
            if edge_ids:
                edges = await self.graph_backend.get_all_edges()
                for edge in edges:
                    if edge.id in edge_ids:
                        nodes.update((edge.subject_id, edge.object_id))
        if include_neighbors and nodes:
            grouped = await self.graph_backend.get_all_edges_for_nodes(list(nodes))
            for group in grouped:
                for edge in group:
                    add("edge", [edge.id])
                    nodes.update((edge.subject_id, edge.object_id))
                    add("chunk", edge.source_chunk_id)
            entities = await self.graph_backend.get_nodes(list(nodes))
            for entity in entities:
                if entity is None:
                    continue
                add("chunk", entity.source_chunk_id)
                for cluster in entity.clusters:
                    try:
                        cluster_id = str(cluster["cluster_id"])
                        community_id = cluster_id if cluster_id.startswith("com-") else Community(
                            level=int(cluster["level"]), cluster_id=int(cluster_id),
                            entities=[], relations=[],
                        ).id
                    except (KeyError, TypeError, ValueError):
                        continue
                    add("community", [community_id])
                    add("summary", [community_id])
        add("node", nodes)
        return requirements

    @asynccontextmanager
    async def _operation_lock(self, name: str, data: Any = ()) -> AsyncIterator[None]:
        async with self._planned_lock(lambda: self._operation_requirements(name, data)):
            yield

    async def _operation_requirements(self, name: str, data: Any) -> dict[tuple[str, str], bool]:
        if name == "build":
            requirements: dict[tuple[str, str], bool] = {}
            for operation, payload in data:
                for key, write in (await self._operation_requirements(operation, payload)).items():
                    requirements[key] = requirements.get(key, False) or write
                if operation == "upsert_communities" and payload:
                    requirements[("community", "*")] = True
                    requirements[("summary", "*")] = True
            return requirements
        if name in {"close", "maintenance"}:
            return {("*", "*"): True}
        if name == "check_consistency":
            return {("*", "*"): False}
        if name in {"get_nodes", "get_chunks", "get_summaries"}:
            kind = {"get_nodes": "node", "get_chunks": "chunk", "get_summaries": "summary"}[name]
            return {(kind, item_id): False for item_id in data}
        if name in {"get_edges", "edges_degrees"}:
            return await self._read_requirements(edge_specs=data)
        if name == "get_edge_specs_by_ids":
            return {("edge", item_id): False for item_id in data}
        if name == "get_communities":
            return await self._read_requirements(community_ids=data)

        requirements: dict[tuple[str, str], bool] = {}

        def add(kind: str, ids: Iterable[str], write: bool = True) -> None:
            for item_id in ids:
                key = (kind, item_id)
                requirements[key] = requirements.get(key, False) or write

        def add_nodes(nodes: Iterable[Any]) -> None:
            for node in nodes:
                if node is not None:
                    add("node", [node.id])
                    add("chunk", node.source_chunk_id, False)

        def add_edges(edges: Iterable[Any]) -> None:
            for edge in edges:
                add("edge", [edge.id])
                add("node", [edge.subject_id, edge.object_id])
                add("chunk", edge.source_chunk_id, False)

        if name in {"upsert_nodes", "update_nodes"}:
            add_nodes(data)
            add_nodes(await self.graph_backend.get_nodes([node.id for node in data]))
        elif name in {"upsert_edges", "update_edges"}:
            add_edges(data)
            specs = [(edge.subject_id, edge.object_id, edge.id) for edge in data]
            groups = await self.graph_backend.get_edges(specs)
            add_edges(edge for group in groups for edge in group)
        elif name == "delete_nodes":
            add("node", data)
            groups = await self.graph_backend.get_all_edges_for_nodes(list(data))
            add_edges(edge for group in groups for edge in group)
        elif name == "delete_edges":
            for subject, object_id, edge_id in data:
                add("node", [subject, object_id])
                if edge_id is not None:
                    add("edge", [edge_id])
            groups = await self.graph_backend.get_edges(list(data))
            add_edges(edge for group in groups for edge in group)
        elif name == "delete_chunks":
            chunk_ids = set(data)
            add("chunk", chunk_ids)
            nodes = await self.graph_backend.get_all_nodes()
            affected = [node for node in nodes if chunk_ids.intersection(node.source_chunk_id)]
            add_nodes(affected)
            node_ids = {node.id for node in affected}
            edges = await self.graph_backend.get_all_edges()
            add_edges(edge for edge in edges if (
                chunk_ids.intersection(edge.source_chunk_id)
                or edge.subject_id in node_ids or edge.object_id in node_ids
            ))
        elif name == "upsert_chunks":
            add("chunk", [chunk.id for chunk in data])
        elif name in {"upsert_communities", "update_communities"}:
            add("community", [community.id for community in data])
        elif name == "delete_communities":
            add("community", data)
            add("summary", data)
        elif name in {"upsert_summaries", "update_summaries"}:
            add("summary", [summary.id for summary in data])
        elif name == "delete_summaries":
            add("summary", data)
        else:
            raise ValueError(f"Unknown index lock operation: {name}")
        return requirements
