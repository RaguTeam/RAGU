from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Generic, List, Optional, Set, Type, TypeVar, cast

import numpy as np
from ragu.chunker.types import Chunk
from ragu.common.global_parameters import DEFAULT_FILENAMES
from ragu.common.global_parameters import Settings
from ragu.graph.types import Community, CommunitySummary, Entity, Relation
from ragu.models.embedder import Embedder
from ragu.models.sparse_embedder import SparseEmbedder
from ragu.storage.base_storage import (
    BaseKVStorage,
    BaseVectorStorage,
    BaseGraphStorage,
    EdgeSpec,
)
from ragu.storage.graph_storage_adapters.networkx_adapter import NetworkXStorage
from ragu.storage.kv_storage_adapters.json_storage import JsonKVStorage
from ragu.storage.types import Point
from ragu.storage.vdb_storage_adapters.nano_vdb import NanoVectorDBStorage


@dataclass
class StorageArguments:
    """
    Configuration for Index storage backends.

    :param graph_backend_storage: Storage backend class for graph structure (nodes/edges).
    :param kv_storage_type: Storage backend class for key-value data (chunks, communities, summaries).
    :param vdb_storage_type: Storage backend class for vector embeddings (entities, relations, chunks).
    :param chunks_kv_storage_kwargs: Additional kwargs passed to KV storage for text chunks.
    :param summary_kv_storage_kwargs: Additional kwargs passed to KV storage for community summaries.
    :param communities_kv_storage_kwargs: Additional kwargs passed to KV storage for community metadata.
    :param vdb_storage_kwargs: Additional kwargs passed to vector database instances.
    :param graph_storage_kwargs: Additional kwargs passed to graph backend storage.
    """
    graph_backend_storage: Type[BaseGraphStorage] = NetworkXStorage
    kv_storage_type: Type[BaseKVStorage[Any]] = JsonKVStorage
    vdb_storage_type: Type[BaseVectorStorage] = NanoVectorDBStorage

    chunks_kv_storage_kwargs: Dict[str, Any] = field(default_factory=dict[str, Any])
    summary_kv_storage_kwargs: Dict[str, Any] = field(default_factory=dict[str, Any])
    communities_kv_storage_kwargs: Dict[str, Any] = field(default_factory=dict[str, Any])
    vdb_storage_kwargs: Dict[str, Any] = field(default_factory=dict[str, Any])
    graph_storage_kwargs: Dict[str, Any] = field(default_factory=dict[str, Any])


@dataclass(frozen=True)
class ConsistencyIssue:
    """
    One consistency violation detected during graph storage audit.

    :param check: Stable machine-readable check identifier.
    :param message: Short human-readable explanation of the violation.
    :param details: Additional structured context for the violation.
    """
    check: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict[str, Any])


@dataclass(frozen=True)
class ConsistencyReport:
    """
    Result of :meth:`Index.check_consistency`.

    :param errors: Collected consistency violations.
    """
    errors: List[ConsistencyIssue] = field(default_factory=list[ConsistencyIssue])

    @property
    def is_consistent(self) -> bool:
        return len(self.errors) == 0

    def to_text(self) -> str:
        if self.is_consistent:
            return "Graph consistency: OK\nNo consistency issues found."

        lines = [
            "Graph consistency: FAILED",
            f"Issues found: {len(self.errors)}",
            "",
        ]
        for issue in self.errors:
            lines.append(f"- {issue.check}: {issue.message}")
            for key, value in sorted(issue.details.items()):
                if isinstance(value, list):
                    rendered_value = ", ".join(str(item) for item in value) if value else "-"
                else:
                    rendered_value = str(value)
                lines.append(f"  {key}: {rendered_value}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_text()


NodeT = TypeVar("NodeT", bound=Entity)
EdgeT = TypeVar("EdgeT", bound=Relation)

class Index(Generic[NodeT, EdgeT]):
    """
    Coordinates graph, vector, and KV storage for generic nodes and edges.
    """

    def __init__(
            self,
            arguments: StorageArguments,
            embedder: Embedder | None = None,
            sparse_embedder: SparseEmbedder | None = None,
            node_t: Type[NodeT] = Entity,
            edge_t: Type[EdgeT] = Relation,
    ):
        """
        Initialize storage backends and in-memory reverse indexes.

        :param embedder: Embedder used for text-to-vector conversion.
        :param arguments: Configuration for storage backend implementations.
        """
        Settings.init_storage_folder()
        storage_folder: str = Settings.storage_folder

        self.embedder = embedder
        self.sparse_embedder = sparse_embedder

        # Reverse indexes for cascade operations
        self._chunk_to_nodes: Dict[str, Set[str]] = defaultdict(set)
        self._chunk_to_edges: Dict[str, Set[str]] = defaultdict(set)

        summary_kv_kwargs = self._build_storage_kwargs(
            storage_folder,
            DEFAULT_FILENAMES["community_summary_kv_storage_name"],
            arguments.summary_kv_storage_kwargs,
        )
        community_kv_kwargs = self._build_storage_kwargs(
            storage_folder,
            DEFAULT_FILENAMES["community_kv_storage_name"],
            arguments.communities_kv_storage_kwargs,
        )
        chunks_kv_kwargs = self._build_storage_kwargs(
            storage_folder,
            DEFAULT_FILENAMES["chunks_kv_storage_name"],
            arguments.chunks_kv_storage_kwargs,
        )
        nodes_vdb_kwargs = self._build_storage_kwargs(
            storage_folder,
            DEFAULT_FILENAMES["entity_vdb_name"],
            arguments.vdb_storage_kwargs,
        )
        edges_vdb_kwargs = self._build_storage_kwargs(
            storage_folder,
            DEFAULT_FILENAMES["relation_vdb_name"],
            arguments.vdb_storage_kwargs,
        )
        chunks_vdb_kwargs = self._build_storage_kwargs(
            storage_folder,
            DEFAULT_FILENAMES["chunk_vdb_name"],
            arguments.vdb_storage_kwargs,
        )
        graph_kwargs = self._build_storage_kwargs(
            storage_folder,
            DEFAULT_FILENAMES["knowledge_graph_storage_name"],
            arguments.graph_storage_kwargs,
        )

        # Key-value storages
        self.chunks_kv_storage = arguments.kv_storage_type(**chunks_kv_kwargs)
        self.community_summary_kv_storage = arguments.kv_storage_type(**summary_kv_kwargs)
        self.community_kv_storage = arguments.kv_storage_type(**community_kv_kwargs)

        if self.embedder:
            embedding_dim_from_embedder = embedder.dim

            dimensions_from_kwargs = [
                storage_kwargs.get("embedding_dim") for storage_kwargs in [
                    nodes_vdb_kwargs,
                    edges_vdb_kwargs,
                    chunks_vdb_kwargs
                ] if storage_kwargs.get("embedding_dim")]

            number_of_dimensions = len(dimensions_from_kwargs)
            if number_of_dimensions > 1:
                raise ValueError(f"Dimension mismatch in vdb kwargs: {dimensions_from_kwargs}")
            if number_of_dimensions == 1:
                if dimensions_from_kwargs[0] != embedding_dim_from_embedder:
                    raise ValueError(f"Dimension mismatch in vdb kwargs and embedder setup: "
                                     f"{dimensions_from_kwargs[0]} and {embedding_dim_from_embedder}")

            resolved_dim = embedding_dim_from_embedder

            nodes_vdb_kwargs["embedding_dim"] = resolved_dim
            edges_vdb_kwargs["embedding_dim"] = resolved_dim
            chunks_vdb_kwargs["embedding_dim"] = resolved_dim

        # Vector storages
        self.nodes_vector_db = arguments.vdb_storage_type(**nodes_vdb_kwargs)
        self.edges_vector_db = arguments.vdb_storage_type(**edges_vdb_kwargs)
        self.chunks_vector_db = arguments.vdb_storage_type(**chunks_vdb_kwargs)

        # Graph storage
        self.graph_backend: BaseGraphStorage[NodeT, EdgeT] = arguments.graph_backend_storage(
            node_cls=node_t,
            edge_cls=edge_t,
            **graph_kwargs
        )

    async def upsert_nodes(self, nodes: List[NodeT]) -> "Index[NodeT, EdgeT]":
        """
        Insert or replace nodes in graph and vector DB by ID.

        :param nodes: Nodes to insert.
        :return: Self for method chaining.
        :raises ValueError: If duplicate node IDs are provided in one request.
        """
        assert self.embedder

        if not nodes:
            return self

        incoming_by_id: Dict[str, List[NodeT]] = defaultdict(list)
        for node in nodes:
            assert node.id
            incoming_by_id[node.id].append(node)

        duplicate_ids = [node_id for node_id, group in incoming_by_id.items() if len(group) > 1]
        if duplicate_ids:
            raise ValueError(f"Cannot insert duplicated node IDs in one request: {duplicate_ids}")

        node_ids = list(incoming_by_id.keys())
        existing_nodes = await self.graph_backend.get_nodes(node_ids)
        existing_ids = [
            node_id
            for node_id, existing in zip(node_ids, existing_nodes)
            if existing is not None
        ]

        nodes_to_upsert = [group[0] for group in incoming_by_id.values()]

        dense_embeddings = await self.embedder.batch_embed_text(
            [node.to_text() for node in nodes_to_upsert],
            desc="Nodes vectorization",
        )
        sparse_embeddings = self.sparse_embedder.embed_document(
            [node.to_text() for node in nodes_to_upsert]
        ) if self.sparse_embedder else [None for _ in nodes_to_upsert]

        vdb_data = [
            Point(
                id=node.id,
                dense_embedding=np.array(dense),
                sparse_embedding=sparse,
                metadata=node.to_dict()
            ) for node, dense, sparse in zip(nodes_to_upsert, dense_embeddings, sparse_embeddings)
        ]

        await self.graph_backend.upsert_nodes(nodes_to_upsert)
        await self.nodes_vector_db.upsert(vdb_data)

        await self.graph_backend.index_done_callback()
        await self.nodes_vector_db.index_done_callback()
        await self._update_reverse_indexes(
            deleted_node_ids=existing_ids,
            nodes=nodes_to_upsert,
        )
        return self

    async def update_nodes(self, nodes: List[NodeT]) -> "Index[NodeT, EdgeT]":
        """
        Update nodes by ID using replace semantics.

        Existing nodes are replaced by incoming payloads. No merge with
        previous values is performed.

        :param nodes: Nodes to update.
        :return: Self for method chaining.
        :raises ValueError: If node IDs are missing/duplicated in request or absent in storage.
        """
        assert self.embedder

        if not nodes:
            return self

        incoming_by_id: Dict[str, List[NodeT]] = defaultdict(list)
        for node in nodes:
            assert node.id
            incoming_by_id[node.id].append(node)

        duplicate_ids = [node_id for node_id, group in incoming_by_id.items() if len(group) > 1]
        if duplicate_ids:
            raise ValueError(f"Cannot update duplicated node IDs in one request: {duplicate_ids}")

        node_ids = list(incoming_by_id.keys())
        existing_nodes = await self.graph_backend.get_nodes(node_ids)
        missing_ids = [node_id for node_id, existing in zip(node_ids, existing_nodes) if existing is None]
        if missing_ids:
            raise ValueError(f"Cannot update non-existent nodes: {missing_ids}")

        nodes_to_update = [group[0] for group in incoming_by_id.values()]

        dense_embeddings = await self.embedder.batch_embed_text(
            [node.to_text() for node in nodes_to_update],
            desc="Nodes vectorization",
        )
        sparse_embeddings = self.sparse_embedder.embed_document(
            [node.to_text() for node in nodes_to_update]
        ) if self.sparse_embedder else [None for _ in nodes_to_update]

        vdb_data = [
            Point(
                id=node.id,
                dense_embedding=np.array(dense),
                sparse_embedding=sparse,
                metadata=node.to_dict()
            ) for node, dense, sparse in zip(nodes_to_update, dense_embeddings, sparse_embeddings)
        ]
        await self.graph_backend.upsert_nodes(nodes_to_update)
        await self.nodes_vector_db.upsert(vdb_data)

        await self.graph_backend.index_done_callback()
        await self.nodes_vector_db.index_done_callback()

        await self._update_reverse_indexes(
            deleted_node_ids=node_ids,
            nodes=nodes_to_update,
        )
        return self

    async def upsert_edges(self, edges: List[EdgeT]) -> "Index[NodeT, EdgeT]":
        """
        Insert edges in graph and vector DB.

        :param edges: Edges to insert.
        :return: Self for method chaining.
        :raises ValueError: If edge IDs are duplicated in request or
            referenced nodes don't exist.
        """
        assert self.embedder

        if not edges:
            return self

        await self._validate_edge_endpoints_exist(edges)

        incoming_by_id: Dict[str, List[EdgeT]] = defaultdict(list)
        for edge in edges:
            incoming_by_id[edge.id].append(edge)

        duplicate_ids = [edge_id for edge_id, group in incoming_by_id.items() if len(group) > 1]
        if duplicate_ids:
            raise ValueError(f"Cannot insert duplicated edge IDs in one request: {duplicate_ids}")

        edges_to_upsert = [group[0] for group in incoming_by_id.values()]
        edge_specs = [
            (edge.subject_id, edge.object_id, edge.id)
            for edge in edges_to_upsert
        ]
        existing_edges = await self.graph_backend.get_edges(edge_specs)
        existing_edge_ids = [
            edge.id
            for edge in existing_edges
            if edge is not None
        ]

        dense_embeddings = await self.embedder.batch_embed_text(
            [edge.to_text() for edge in edges_to_upsert],
            desc="Edges vectorization",
        )
        sparse_embeddings = self.sparse_embedder.embed_document(
            [edge.to_text() for edge in edges_to_upsert]
        ) if self.sparse_embedder else [None for _ in edges_to_upsert]

        vdb_data = [
            Point(
                id=edge.id,
                dense_embedding=np.array(dense),
                sparse_embedding=sparse,
                metadata=edge.to_dict()
            ) for edge, dense, sparse in zip(edges_to_upsert, dense_embeddings, sparse_embeddings)
        ]

        await self.graph_backend.upsert_edges(edges_to_upsert)
        await self.edges_vector_db.upsert(vdb_data)

        await self.graph_backend.index_done_callback()
        await self.edges_vector_db.index_done_callback()
        await self._update_reverse_indexes(
            deleted_edge_ids=existing_edge_ids,
            edges=edges_to_upsert,
        )
        return self

    async def update_edges(self, edges: List[EdgeT]) -> "Index[NodeT, EdgeT]":
        """
        Update edges by exact edge spec using replace semantics.

        Existing edges at the same ``(subject_id, object_id, id)`` are
        replaced by incoming payloads. No merge with previous values is
        performed.

        :param edges: Edges to update.
        :return: Self for method chaining.
        :raises ValueError: If edge IDs are missing/duplicated in request,
            matching edge specs are absent in storage, or referenced nodes
            don't exist.
        """
        assert self.embedder

        if not edges:
            return self

        incoming_by_id: Dict[str, List[EdgeT]] = defaultdict(list)
        for edge in edges:
            if not edge.id:
                raise ValueError("Cannot update edge without id")
            incoming_by_id[edge.id].append(edge)

        duplicate_ids = [edge_id for edge_id, group in incoming_by_id.items() if len(group) > 1]
        if duplicate_ids:
            raise ValueError(f"Cannot update duplicated edge IDs in one request: {duplicate_ids}")

        edges_to_update = [group[0] for group in incoming_by_id.values()]
        edge_specs = [
            (edge.subject_id, edge.object_id, edge.id)
            for edge in edges_to_update
        ]
        existing_edges = await self.graph_backend.get_edges(edge_specs)
        missing_specs = [
            edge_spec
            for edge_spec, existing_edge in zip(edge_specs, existing_edges)
            if existing_edge is None
        ]
        if missing_specs:
            raise ValueError(f"Cannot update non-existent edges: {missing_specs}")

        await self._validate_edge_endpoints_exist(edges_to_update)

        dense_embeddings = await self.embedder.batch_embed_text(
            [edge.to_text() for edge in edges_to_update],
            desc="Edges vectorization",
        )
        sparse_embeddings = self.sparse_embedder.embed_document(
            [edge.to_text() for edge in edges_to_update]
        ) if self.sparse_embedder else [None for _ in edges_to_update]

        vdb_data = [
            Point(
                id=edge.id,
                dense_embedding=np.array(dense),
                sparse_embedding=sparse,
                metadata=edge.to_dict()
            ) for edge, dense, sparse in zip(edges_to_update, dense_embeddings, sparse_embeddings)
        ]

        await self.graph_backend.upsert_edges(edges_to_update)
        await self.edges_vector_db.upsert(vdb_data)

        await self.graph_backend.index_done_callback()
        await self.edges_vector_db.index_done_callback()
        await self._update_reverse_indexes(
            deleted_edge_ids=list(incoming_by_id.keys()),
            edges=edges_to_update,
        )
        return self

    async def upsert_chunks(self, chunks: List[Chunk]) -> "Index[NodeT, EdgeT]":
        """
        Insert or update chunks into KV storage (and optionally vector DB).

        :param chunks: Chunks to upsert.
        :return: Self for method chaining.
        """
        assert self.embedder

        if not chunks:
            return self

        # Store in KV
        kv_data: dict[str, dict[str, Any]] = {}
        for chunk in chunks:
            chunk_dict = asdict(chunk)
            chunk_id = cast(str, chunk_dict.pop("id"))
            kv_data[chunk_id] = chunk_dict

        await self.chunks_kv_storage.upsert(kv_data)

        dense_embeddings = await self.embedder.batch_embed_text(
            [c.content for c in chunks],
            desc="Chunks vectorization"
        )
        sparse_embeddings = self.sparse_embedder.embed_document([c.content for c in chunks]) \
            if self.sparse_embedder else [None for _ in chunks]

        vdb_data = [Point(
            id=c.id,
            dense_embedding=np.array(dense),
            sparse_embedding=sparse,
            metadata={"content": c.content, "doc_id": c.doc_id}
        ) for c, dense, sparse in zip(chunks, dense_embeddings, sparse_embeddings)]

        await self.chunks_vector_db.upsert(vdb_data)
        await self.chunks_vector_db.index_done_callback()

        await self.chunks_kv_storage.index_done_callback()
        return self

    async def upsert_communities(self, communities: List[Community]) -> "Index[NodeT, EdgeT]":
        """
        Insert or update communities into KV storage.

        :param communities: Communities to upsert.
        :return: Self for method chaining.
        """
        if not communities:
            return self

        kv_data: dict[str, Any] = {
            c.id: {
                "level": c.level,
                "cluster_id": c.cluster_id,
                "entity_ids": sorted({e.id for e in c.entities}),
                "relation_ids": sorted({r.id for r in c.relations}),
            }
            for c in communities
        }
        await self.community_kv_storage.upsert(kv_data)
        await self.community_kv_storage.index_done_callback()
        return self

    async def upsert_summaries(self, summaries: List[CommunitySummary]) -> "Index[NodeT, EdgeT]":
        """
        Insert or update community summaries into KV storage.

        :param summaries: Summaries to upsert.
        :return: Self for method chaining.
        """
        if not summaries:
            return self

        kv_data = {s.id: s.summary for s in summaries}
        await self.community_summary_kv_storage.upsert(kv_data) # type: ignore
        await self.community_summary_kv_storage.index_done_callback()
        return self

    async def delete_nodes(self, node_ids: List[str]) -> "Index[NodeT, EdgeT]":
        """
        Delete nodes from graph and vector DB.

        All edges connected to the deleted nodes are also removed from the edge vector DB.

        :param node_ids: IDs of nodes to delete.
        :return: Self for method chaining.
        """
        if not node_ids:
            return self

        edges_by_node = await self.graph_backend.get_all_edges_for_nodes(node_ids)
        edge_ids = self._unique_edge_ids_from_grouped(edges_by_node)

        await self.graph_backend.delete_nodes(node_ids)
        await self.nodes_vector_db.delete(node_ids)

        await self.edges_vector_db.delete(edge_ids)
        await self.edges_vector_db.index_done_callback()

        await self.graph_backend.index_done_callback()
        await self.nodes_vector_db.index_done_callback()
        await self._update_reverse_indexes(
            deleted_node_ids=node_ids,
            deleted_edge_ids=edge_ids,
        )
        return self

    async def delete_edges(self, edge_specs: List[EdgeSpec]) -> "Index[NodeT, EdgeT]":
        """
        Delete edges from graph and vector DB.

        :param edge_specs: List of edge specs ``(subject_id, object_id, relation_id)``.
        :return: Self for method chaining.
        """
        if not edge_specs:
            return self

        existing_edges = await self.graph_backend.get_edges(edge_specs)
        found_edge_ids = [edge.id for edge in existing_edges if edge is not None]

        await self.graph_backend.delete_edges(edge_specs)

        if found_edge_ids:
            await self.edges_vector_db.delete(found_edge_ids)

        await self.graph_backend.index_done_callback()
        await self.edges_vector_db.index_done_callback()
        await self._update_reverse_indexes(deleted_edge_ids=found_edge_ids)
        return self

    async def delete_chunks(self, chunk_ids: List[str]) -> "Index[NodeT, EdgeT]":
        """
        Delete chunks from KV and vector storage.

        :param chunk_ids: IDs of chunks to delete.
        :return: Self for method chaining.
        """
        if not chunk_ids:
            return self

        affected_nodes = await self._find_nodes_by_chunk_ids(chunk_ids)
        node_ids = [node.id for node in affected_nodes]
        edge_ids = await self._find_edge_ids_by_chunk_ids(chunk_ids)

        if node_ids:
            edges_by_node = await self.graph_backend.get_all_edges_for_nodes(node_ids)
            edge_ids.extend(self._unique_edge_ids_from_grouped(edges_by_node))

        edge_ids = list(dict.fromkeys(edge_ids))
        delete_specs: List[EdgeSpec] = []
        if edge_ids:
            # TODO: remove full scan here
            delete_specs = await self.get_edge_specs_by_ids(set(edge_ids))

        if delete_specs:
            await self.graph_backend.delete_edges(delete_specs)

        if node_ids:
            await self.graph_backend.delete_nodes(node_ids)
            await self.nodes_vector_db.delete(node_ids)

        if edge_ids:
            await self.edges_vector_db.delete(edge_ids)

        await self.chunks_kv_storage.delete(chunk_ids)
        await self.chunks_vector_db.delete(chunk_ids)

        await self.chunks_kv_storage.index_done_callback()
        await self.chunks_vector_db.index_done_callback()
        if node_ids:
            await self.graph_backend.index_done_callback()
            await self.nodes_vector_db.index_done_callback()
        if edge_ids:
            await self.edges_vector_db.index_done_callback()
        await self._update_reverse_indexes(
            deleted_chunk_ids=chunk_ids,
            deleted_node_ids=node_ids,
            deleted_edge_ids=edge_ids,
        )
        return self

    async def delete_communities(self, community_ids: List[str]) -> "Index[NodeT, EdgeT]":
        """
        Delete communities and their summaries from KV storage.

        :param community_ids: IDs of communities to delete.
        :return: Self for method chaining.
        """
        if not community_ids:
            return self

        await self.community_kv_storage.delete(community_ids)
        await self.community_summary_kv_storage.delete(community_ids)

        await self.community_kv_storage.index_done_callback()
        await self.community_summary_kv_storage.index_done_callback()
        return self

    async def get_nodes(self, node_ids: List[str]) -> List[Optional[NodeT]]:
        """
        Retrieve nodes by their IDs.

        :param node_ids: Node IDs to fetch.
        :return: List of nodes (``None`` for missing).
        """
        return await self.graph_backend.get_nodes(node_ids)

    async def get_edges(self, edge_specs: List[EdgeSpec]) -> List[Optional[EdgeT]]:
        """
        Retrieve edges by edge specs.

        :param edge_specs: List of edge specs ``(subject_id, object_id, relation_id)``.
        :return: List of edges (``None`` for missing).
        """
        return await self.graph_backend.get_edges(edge_specs)

    async def get_chunks(self, chunk_ids: List[str]) -> List[Optional[Chunk]]:
        """
        Retrieve chunks by their IDs.

        :param chunk_ids: Chunk IDs to fetch.
        :return: List of chunks (``None`` for missing).
        """
        chunk_dicts = await self.chunks_kv_storage.get_by_ids(chunk_ids)
        result: list[Chunk | None] = []
        for chunk_dict in chunk_dicts:
            if chunk_dict is None:
                result.append(None)
            else:
                result.append(Chunk(**chunk_dict))
        return result

    async def get_communities(self, community_ids: List[str]) -> List[Optional[Community]]:
        """
        Retrieve communities by their IDs, reconstructing from stored metadata.

        :param community_ids: Community IDs to fetch.
        :return: List of communities (``None`` for missing).
        """
        community_dicts = await self.community_kv_storage.get_by_ids(community_ids)
        communities: list[Community | None] = []

        for community_id, community_dict in zip(community_ids, community_dicts):
            if community_dict is None:
                communities.append(None)
                continue

            entity_ids = community_dict.get("entity_ids", [])
            relation_id_set = set(community_dict.get("relation_ids", []))

            nodes = await self.get_nodes(entity_ids)

            # TODO: remove full scan here
            all_edges = await self.graph_backend.get_all_edges()
            edges = [edge for edge in all_edges if edge and edge.id in relation_id_set]

            entities: List[NodeT] = [entity for entity in nodes if entity is not None]
            relations: List[EdgeT] = [relation for relation in edges if relation is not None]

            communities.append(Community(
                id=community_id,
                level=community_dict["level"],
                cluster_id=community_dict["cluster_id"],
                entities=entities,
                relations=relations,
            ))

        return communities

    async def _validate_edge_endpoints_exist(self, edges: List[EdgeT]) -> None:
        """
        Validate that all edge endpoints exist as nodes.

        :param edges: Edges whose subject/object IDs must exist as nodes.
        :raises ValueError: If at least one referenced node is missing.
        """
        all_node_ids: set[str] = set()
        for edge in edges:
            all_node_ids.add(edge.subject_id)
            all_node_ids.add(edge.object_id)

        existing_nodes = await self.graph_backend.get_nodes(list(all_node_ids))
        existing_ids = {node.id for node in existing_nodes if node is not None}
        missing_ids = all_node_ids - existing_ids

        if missing_ids:
            raise ValueError(
                f"Cannot insert/update edges referencing non-existent nodes: {missing_ids}"
            )

    async def get_edge_specs_by_ids(
        self,
        edge_ids: Set[str],
    ) -> List[EdgeSpec]:
        """
        Retrieve edge specs for existing edges by edge ID.

        :param edge_ids: Edge IDs to search in graph storage.
        :return: Edge specs for matching edges.
        """
        if not edge_ids:
            return []

        all_edges = await self.graph_backend.get_all_edges()
        edge_specs: List[EdgeSpec] = []
        for edge in all_edges:
            assert edge
            assert edge.id
            if edge.id in edge_ids:
                edge_specs.append((edge.subject_id, edge.object_id, edge.id))
        return edge_specs

    async def _update_reverse_indexes(
        self,
        nodes: Optional[List[NodeT]] = None,
        edges: Optional[List[EdgeT]] = None,
        deleted_node_ids: Optional[List[str]] = None,
        deleted_edge_ids: Optional[List[str]] = None,
        deleted_chunk_ids: Optional[List[str]] = None,
    ) -> None:
        """
        Incrementally update reverse indexes from changed entities/relations.

        :param nodes: Upserted nodes to add to chunk-to-node index.
        :param edges: Upserted edges to add to chunk-to-edge index.
        :param deleted_node_ids: Node IDs removed from the graph.
        :param deleted_edge_ids: Edge IDs removed from the graph.
        :param deleted_chunk_ids: Chunk IDs removed from KV/vector storage.
        """
        if deleted_chunk_ids:
            for chunk_id in deleted_chunk_ids:
                self._chunk_to_nodes.pop(chunk_id, None)
                self._chunk_to_edges.pop(chunk_id, None)

        if deleted_node_ids:
            removed_nodes = set(deleted_node_ids)
            for chunk_id, node_ids in list(self._chunk_to_nodes.items()):
                node_ids.difference_update(removed_nodes)
                if not node_ids:
                    self._chunk_to_nodes.pop(chunk_id, None)

        if deleted_edge_ids:
            removed_edges = set(deleted_edge_ids)
            for chunk_id, edge_ids in list(self._chunk_to_edges.items()):
                edge_ids.difference_update(removed_edges)
                if not edge_ids:
                    self._chunk_to_edges.pop(chunk_id, None)

        if nodes:
            nodes_map = self._get_items_map(nodes)
            for chunk_id, node_ids in nodes_map.items():
                self._chunk_to_nodes[chunk_id].update(node_ids)

        if edges:
            edges_map = self._get_items_map(edges)
            for chunk_id, edge_ids in edges_map.items():
                self._chunk_to_edges[chunk_id].update(edge_ids)

    async def _rebuild_reverse_indexes(self) -> None:
        """
        Rebuild reverse indexes by scanning graph data once.

        Used as fallback for cold-start consistency with preloaded graphs.
        """
        self._chunk_to_nodes.clear()
        self._chunk_to_edges.clear()

        all_nodes: List[NodeT] = await self.graph_backend.get_all_nodes()
        all_edges: List[EdgeT] = await self.graph_backend.get_all_edges()

        await self._update_reverse_indexes(
            nodes=all_nodes,
            edges=all_edges,
        )

    async def _find_nodes_by_chunk_ids(self, chunk_ids: List[str]) -> List[NodeT]:
        """
        Find all nodes referencing any of the given chunk IDs.

        :param chunk_ids: Chunk identifiers.
        :return: Nodes referencing these chunks.
        """
        if not self._chunk_to_nodes:
            await self._rebuild_reverse_indexes()

        node_ids = set[str]()
        for chunk_id in chunk_ids:
            node_ids.update(self._chunk_to_nodes.get(chunk_id, set()))

        nodes = await self.graph_backend.get_nodes(list(node_ids))
        return [node for node in nodes if node is not None]

    async def _find_edge_ids_by_chunk_ids(self, chunk_ids: List[str]) -> List[str]:
        """
        Find all edge IDs referencing any of the given chunk IDs.

        :param chunk_ids: Chunk identifiers.
        :return: Edge IDs referencing these chunks.
        """
        if not self._chunk_to_edges:
            await self._rebuild_reverse_indexes()

        edge_ids: List[str] = []
        seen: Set[str] = set()
        for chunk_id in chunk_ids:
            for edge_id in self._chunk_to_edges.get(chunk_id, set()):
                if edge_id in seen:
                    continue
                seen.add(edge_id)
                edge_ids.append(edge_id)

        return edge_ids

    @staticmethod
    def _get_items_map(items: List[NodeT] | List[EdgeT]) -> Dict[str, List[str]]:
        """
        Build reverse mapping chunk_id -> list of item IDs using source_chunk_id.

        :param items: Nodes or edges with ``id`` and ``source_chunk_id`` fields.
        :return: Mapping from chunk ID to list of node/edge IDs.
        """
        chunks_map: Dict[str, List[str]] = defaultdict(list)
        for item in items:
            if not item or not item.id:
                continue
            for chunk_id in item.source_chunk_id:
                chunks_map[chunk_id].append(item.id)
        return dict(chunks_map)

    @staticmethod
    def _unique_edge_ids_from_grouped(edges_by_node: List[List[EdgeT]]) -> List[str]:
        """
        Flatten grouped edges and return unique edge IDs (first-seen order).

        :param edges_by_node: Edges grouped by source node.
        :return: Unique edge IDs preserving first-seen order.
        """
        edge_ids: List[str] = []
        seen: Set[str] = set()
        for edges in edges_by_node:
            for edge in edges:
                edge_id = getattr(edge, "id", None)
                if not edge_id or edge_id in seen:
                    continue
                seen.add(edge_id)
                edge_ids.append(edge_id)
        return edge_ids

    @staticmethod
    def _build_storage_kwargs(
            storage_folder: str,
            filename: str,
            provided_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build effective storage kwargs and ensure a default absolute ``filename``.

        :param storage_folder: Base folder for storage files.
        :param filename: Default storage filename.
        :param provided_kwargs: Optional custom kwargs from user configuration.
        :return: Final kwargs dictionary for storage backend initialization.
        """
        kwargs = dict(provided_kwargs or {})
        kwargs.setdefault(
            "filename",
            os.path.abspath(os.path.join(storage_folder, filename)),
        )
        return kwargs

    async def check_consistency(self) -> ConsistencyReport:
        """
        Audit cross-storage graph consistency and collect invariant violations.

        Checked invariants: relation endpoints exist as entities in a graph;
        ``source_chunk_id`` values exist in chunk storage; community
        entity/relation references exists in the graph; graph and chunk items
        have vector representations; and every vector has a matching entity,
        relation, or chunk endpoint.

        :returns: Structured consistency report.
        """
        all_entities = [entity for entity in await self.graph_backend.get_all_nodes() if entity is not None]
        all_relations = [relation for relation in await self.graph_backend.get_all_edges() if relation is not None]

        all_entity_ids_from_graph = {entity.id for entity in all_entities if entity.id}
        all_relation_ids_from_graph = {relation.id for relation in all_relations if relation.id}
        all_chunk_ids_from_storage = set(await self.chunks_kv_storage.all_keys())

        errors: List[ConsistencyIssue] = []

        referenced_chunk_ids = {
            chunk_id
            for entity in all_entities
            for chunk_id in entity.source_chunk_id
        } | {
            chunk_id
            for relation in all_relations
            for chunk_id in relation.source_chunk_id
        }
        missing_chunk_ids = sorted(referenced_chunk_ids - all_chunk_ids_from_storage)
        if missing_chunk_ids:
            errors.append(
                ConsistencyIssue(
                    check="source_chunk_references",
                    message="Entities or relations reference chunks missing from chunk storage.",
                    details={"missing_chunk_ids": missing_chunk_ids},
                )
            )

        relation_endpoint_ids = {
            endpoint_id
            for relation in all_relations
            for endpoint_id in (relation.subject_id, relation.object_id)
        }
        missing_relation_endpoint_ids = sorted(relation_endpoint_ids - all_entity_ids_from_graph)
        if missing_relation_endpoint_ids:
            missing_relation_endpoint_id_set = set(missing_relation_endpoint_ids)
            affected_relation_ids = sorted([
                relation.id
                for relation in all_relations
                if relation.id and (
                    relation.subject_id in missing_relation_endpoint_id_set
                    or relation.object_id in missing_relation_endpoint_id_set
                )
            ])
            errors.append(
                ConsistencyIssue(
                    check="relation_endpoints",
                    message="Relations reference entity endpoints that do not exist in the graph.",
                    details={
                        "missing_entity_ids": missing_relation_endpoint_ids,
                        "relation_ids_with_empty_endpoints": affected_relation_ids,
                    },
                )
            )

        community_ids = await self.community_kv_storage.all_keys()
        community_rows = await self.community_kv_storage.get_by_ids(community_ids) if community_ids else []
        broken_community_ids: Set[str] = set()
        missing_community_entity_ids: Set[str] = set()
        missing_community_relation_ids: Set[str] = set()
        for community_id, community in zip(community_ids, community_rows):
            if community is None:
                broken_community_ids.add(community_id)
                continue

            missing_entities = set(community.get("entity_ids", [])) - all_entity_ids_from_graph
            missing_relations = set(community.get("relation_ids", [])) - all_relation_ids_from_graph
            if missing_entities or missing_relations:
                broken_community_ids.add(community_id)
                missing_community_entity_ids.update(missing_entities)
                missing_community_relation_ids.update(missing_relations)

        if broken_community_ids:
            errors.append(
                ConsistencyIssue(
                    check="community_references",
                    message="Communities reference entities or relations missing from the graph.",
                    details={
                        "community_ids": sorted(broken_community_ids),
                        "missing_entity_ids": sorted(missing_community_entity_ids),
                        "missing_relation_ids": sorted(missing_community_relation_ids),
                    },
                )
            )

        entity_vector_ids = set(await self.nodes_vector_db.get_all_ids())
        missing_entity_vector_ids = sorted(all_entity_ids_from_graph - entity_vector_ids)
        if missing_entity_vector_ids:
            errors.append(
                ConsistencyIssue(
                    check="entity_vector_representations",
                    message="Graph entities exist without matching entity vectors.",
                    details={"missing_vector_ids": missing_entity_vector_ids},
                )
            )

        orphan_entity_vector_ids = sorted(entity_vector_ids - all_entity_ids_from_graph)
        if orphan_entity_vector_ids:
            errors.append(
                ConsistencyIssue(
                    check="entity_vector_endpoints",
                    message="Entity vectors exist without matching graph entities.",
                    details={"orphan_vector_ids": orphan_entity_vector_ids},
                )
            )

        relation_vector_ids = set(await self.edges_vector_db.get_all_ids())
        missing_relation_vector_ids = sorted(all_relation_ids_from_graph - relation_vector_ids)
        if missing_relation_vector_ids:
            errors.append(
                ConsistencyIssue(
                    check="relation_vector_representations",
                    message="Graph relations exist without matching relation vectors.",
                    details={"missing_vector_ids": missing_relation_vector_ids},
                )
            )

        orphan_relation_vector_ids = sorted(relation_vector_ids - all_relation_ids_from_graph)
        if orphan_relation_vector_ids:
            errors.append(
                ConsistencyIssue(
                    check="relation_vector_endpoints",
                    message="Relation vectors exist without matching graph relations.",
                    details={"orphan_vector_ids": orphan_relation_vector_ids},
                )
            )

        chunks_vdb_ids = await self.chunks_vector_db.get_all_ids()

        # Empty chunk vectors are only valid when chunk storage is also empty.
        if not chunks_vdb_ids and not all_chunk_ids_from_storage:
            return ConsistencyReport(errors=errors)

        chunk_vector_ids = set(chunks_vdb_ids)
        missing_chunk_vector_ids = sorted(all_chunk_ids_from_storage - chunk_vector_ids)
        if missing_chunk_vector_ids:
            errors.append(
                ConsistencyIssue(
                    check="chunk_vector_representations",
                    message="Chunks in storage exist without matching chunk vectors.",
                    details={"missing_vector_ids": missing_chunk_vector_ids},
                )
            )

        orphan_chunk_vector_ids = sorted(chunk_vector_ids - all_chunk_ids_from_storage)
        if orphan_chunk_vector_ids:
            errors.append(
                ConsistencyIssue(
                    check="chunk_vector_endpoints",
                    message="Chunk vectors exist without matching chunk records.",
                    details={"orphan_vector_ids": orphan_chunk_vector_ids},
                )
            )

        return ConsistencyReport(errors=errors)
