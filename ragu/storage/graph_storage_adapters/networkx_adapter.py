from __future__ import annotations

import os
from dataclasses import asdict
from collections.abc import Iterator
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
)
from typing_extensions import override

import networkx as nx

from ragu.storage.base_storage import BaseGraphStorage, EdgeSpec
from ragu.storage.types import Node, Edge

NodeT = TypeVar("NodeT", bound=Node)
EdgeT = TypeVar("EdgeT", bound=Edge)

class NetworkXStorage(BaseGraphStorage[NodeT, EdgeT]):
    """
    NetworkX-based implementation of BaseGraphStorage.
    """

    def __init__(
        self,
        filename: str,
        node_cls: Type[NodeT],
        edge_cls: Type[EdgeT],
        **kwargs: Any,
    ):
        """
        Initialize a new NetworkXStorage.

        :param filename: Path to a `.gml` file used for persistence.
        """
        loaded = nx.read_gml(filename) if os.path.exists(filename) else nx.MultiDiGraph() # type: ignore
        self._graph: nx.MultiDiGraph = (
            loaded if isinstance(loaded, nx.MultiDiGraph) else self._to_multi_di_graph(loaded) # type: ignore
        )
        self._where_to_save = filename
        self._node_cls = node_cls
        self._edge_cls = edge_cls

    @staticmethod
    def _to_multi_di_graph(graph: Any) -> "nx.MultiDiGraph":
        """
        Convert a legacy (non-multi) persisted graph, keying edges by their stored ``id``.

        A plain ``nx.MultiDiGraph(graph)`` conversion assigns integer auto-keys
        (losing stored relation identifiers) and duplicates undirected edges in
        both directions. This helper preserves one edge per stored record and
        restores the ``id`` attribute as the edge key.

        :param graph: Graph loaded from persistence.
        :return: Directed multigraph keyed by stored edge ids.
        """
        converted: "nx.MultiDiGraph" = nx.MultiDiGraph()
        converted.add_nodes_from(graph.nodes(data=True))
        for index, (u, v, data) in enumerate(graph.edges(data=True)):
            attrs = dict(data)
            key = attrs.pop("id", None) or f"edge-{index}"
            converted.add_edge(u, v, key=key, **attrs)
        return converted

    def _iter_incident_edges(self, node_id: str) -> Iterator[tuple[Any, Any, Any, Dict[str, Any]]]:
        """
        Iterate all incoming and outgoing edges for a node.

        In directed graphs, ``edges(node_id)`` returns only outgoing edges.
        This helper merges out/in edges and deduplicates by edge triple.
        """
        seen: set[tuple[str, str, str]] = set()

        for u, v, key, data in self._graph.out_edges(node_id, keys=True, data=True):
            edge_tuple = (str(u), str(v), str(key))
            if edge_tuple in seen:
                continue
            seen.add(edge_tuple)
            yield u, v, key, data

        for u, v, key, data in self._graph.in_edges(node_id, keys=True, data=True):
            edge_tuple = (str(u), str(v), str(key))
            if edge_tuple in seen:
                continue
            seen.add(edge_tuple)
            yield u, v, key, data

    async def index_done_callback(self) -> None:
        """
        Persist the current graph state to disk in GML format.
        """
        nx.write_gml(self._graph, self._where_to_save) # type: ignore

    async def query_done_callback(self) -> None:
        """
        Callback executed after a query is completed.
        Reserved for potential post-processing hooks.
        """
        pass

    async def index_start_callback(self) -> None:
        """
        Callback executed before indexing starts.
        Reserved for potential setup hooks.
        """
        pass

    @override
    async def edges_degrees(self, edge_specs: List[EdgeSpec]) -> List[int]:
        """
        Retrieve degree values for multiple edges.

        For each edge spec, returns ``degree(subject_id) + degree(object_id)``.
        Returns ``0`` when the edge or either endpoint is missing.

        :param edge_specs: edge specifications to evaluate.
        :return: Degree sums aligned with ``edge_specs``.
        """
        degrees: List[int] = []
        for subject_id, object_id, _Edge_id in edge_specs:
            degree = (
                (self._graph.degree(subject_id) if self._graph.has_node(subject_id) else 0)
                + (self._graph.degree(object_id) if self._graph.has_node(object_id) else 0)
            )
            degrees.append(degree)
        return degrees

    @override
    async def upsert_nodes(self, nodes: List[NodeT]) -> None:
        """
        Insert or update multiple nodes in the graph.

        :param nodes: Entities to process.
        """
        for node in nodes:
            attrs = asdict(node)
            node_id = attrs.pop("id")
            self._graph.add_node(node_id, **attrs)

    @override
    async def get_nodes(self, node_ids: List[str]) -> List[Optional[NodeT]]:
        """
        Retrieve multiple nodes by their IDs.

        :param node_ids: List of node identifiers to fetch.
        :return: List of entities (``None`` for missing nodes).
        """
        results: List[Optional[NodeT]] = []
        for node_id in node_ids:
            if not self._graph.has_node(node_id):
                results.append(None)
                continue
            data = self._graph.nodes[node_id]
            results.append(self._node_cls(id=node_id, **data))
        return results

    @override
    async def delete_nodes(self, node_ids: List[str]) -> None:
        """
        Delete multiple nodes from the graph.

        Cascade removes all connected edges.

        :param node_ids: List of node identifiers to remove.
        """
        for node_id in node_ids:
            if self._graph.has_node(node_id):
                self._graph.remove_node(node_id)

    @override
    async def get_edges(self, edge_specs: List[EdgeSpec]) -> List[List[EdgeT]]:
        """
        Retrieve edges by specs, one result list per spec.

        A spec naming an edge yields that edge or an empty list; a spec with
        ``key=None`` yields every edge between the pair, since the graph is a
        multigraph. Results stay aligned with ``edge_specs``.

        :param edge_specs: List of edge specs ``(subject_id, object_id, Edge_id)``.
        :return: One list of edges per spec, in spec order.
        """
        results: List[List[EdgeT]] = []
        for u, v, key in edge_specs:
            matches = self._graph.get_edge_data(u, v, default={}) if self._graph.has_edge(u, v) else {}

            group: List[EdgeT] = []
            if key is not None:
                edge_data = matches.get(key)
                if edge_data is not None:
                    payload = dict(edge_data)
                    payload.pop("id", None)
                    group.append(self._edge_cls(subject_id=u, object_id=v, id=key, **payload))
            else:
                for match_key, edge_data in matches.items():
                    if not edge_data:
                        continue
                    payload = dict(edge_data)
                    payload.pop("id", None)
                    group.append(self._edge_cls(subject_id=u, object_id=v, id=match_key, **payload))
            results.append(group)
        return results

    @override
    async def upsert_edges(self, edges: List[EdgeT]) -> None:
        """
        Insert or update multiple edges in the graph.

        :param edges: Edges to upsert.
        """
        for edge in edges:
            edge_data = asdict(edge)
            edge_data.pop("subject_id", None)
            edge_data.pop("object_id", None)
            edge_key = edge_data.pop("id", None)
            self._graph.add_edge(edge.subject_id, edge.object_id, key=edge_key, **edge_data)

    @override
    async def delete_edges(self, edge_specs: List[EdgeSpec]) -> None:
        """
        Delete multiple edges from the graph.

        Missing edges are tolerated, as in :meth:`delete_nodes`: deletion is
        idempotent, and raising here made this the only removal in the storage
        layer that objected to being asked twice.

        :param edge_specs: List of edge specs (subject_id, object_id, Edge_id).
        """
        for spec in edge_specs:
            u, v, key = spec
            if not self._graph.has_edge(u, v):
                continue

            if key is not None:
                if self._graph.has_edge(u, v, key=key):
                    self._graph.remove_edge(u, v, key=key)
                continue

            edge_dict = self._graph.get_edge_data(u, v, default={})
            keys_to_remove = list(edge_dict.keys())

            for k in keys_to_remove:
                self._graph.remove_edge(u, v, key=k)

    @override
    async def get_all_edges_for_nodes(self, node_ids: List[str]) -> List[List[EdgeT]]:
        """
        Retrieve edges for each given node.

        Returns one EdgeT list per input node ID. No cross-node deduplication
        is performed.

        :param node_ids: List of node identifiers.
        :return: Grouped Edges for each node.
        """
        grouped_relations: List[List[EdgeT]] = []

        for node_id in node_ids:
            node_relations: List[EdgeT] = []
            if not self._graph.has_node(node_id):
                grouped_relations.append(node_relations)
                continue

            for u, v, key, metadata in self._iter_incident_edges(node_id):
                _ = metadata.pop("id", None)
                relation = self._edge_cls(subject_id=str(u), object_id=str(v), id=key, **metadata)
                node_relations.append(relation)

            grouped_relations.append(node_relations)

        return grouped_relations

    @override
    async def get_all_nodes(self) -> List[NodeT]:
        """
        Retrieve all nodes in the graph.

        :return: List of all entities.
        """
        entities: List[NodeT] = []
        for node_id in self._graph.nodes():
            entity = self._node_cls(
                id=node_id,
                **dict(self._graph.nodes[node_id])
            )
            entities.append(entity)
        return entities

    @override
    async def get_all_edges(self) -> List[EdgeT]:
        """
        Retrieve all edges in the graph.

        :return: List of all edges.
        """
        relations: List[EdgeT] = []

        for u, v, key, metadata in self._graph.edges(keys=True, data=True):
            _ = metadata.pop("id", None)
            relation = self._edge_cls(subject_id=str(u), object_id=str(v), id=key, **metadata)
            relations.append(relation)

        return relations
