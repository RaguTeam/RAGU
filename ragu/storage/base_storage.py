from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import (
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
    Any
)

from ragu.storage.types import Edge, EmbeddingHit, Node, Point

EdgeSpec = Tuple[str, str, Optional[str]]


@dataclass
class BaseStorage(ABC):
    """
    Base contract for all storage backends used by RAGU.
    """

    @abstractmethod
    async def index_start_callback(self):
        """
        Execute pre-indexing initialization hook.
        """
        pass

    @abstractmethod
    async def index_done_callback(self):
        """
        Execute post-indexing finalization hook.
        """
        pass

    @abstractmethod
    async def query_done_callback(self):
        """
        Execute post-query cleanup hook.
        """
        pass

    async def close(self) -> None:
        """
        Release resources held by this backend.

        Backends that keep a connection or a pool (Neo4j, remote Qdrant) must
        override this; file-backed ones need nothing and inherit the no-op.
        """
        return None


@dataclass
class BaseVectorStorage(BaseStorage, ABC):
    """
    Abstract interface for vector storage backends.
    """

    @abstractmethod
    async def query(self, points: List[Point], **kwargs) -> List[List[EmbeddingHit]]:
        """
        Retrieve top-k nearest items for a batch of query embeddings.

        :param points: Query embeddings, one per search.
        :return: Per-query lists of hits with distance score and metadata,
            index-aligned with ``points``.
        """
        ...

    @abstractmethod
    async def upsert(self, data: List[Point], **kwargs: Any) -> None:
        """
        Insert or update embedding records.

        :param data: Embedding records to upsert.
        """
        ...

    @abstractmethod
    async def delete(self, ids: List[str], **kwargs) -> None:
        """
        Delete records by IDs.

        :param ids: Record identifiers to remove.
        """
        ...

    @abstractmethod
    async def get_all_ids(self) -> List[str]:
        """
        Return all stored record IDs.

        :return: List of identifiers currently present in the vector store.
        """
        ...

    @abstractmethod
    async def get_points_by_ids(self, ids: List[str]) -> List[Point | None]:
        """
        Fetch stored points by ID.

        Returns each vector **as the backend stores it**, which is not
        necessarily the vector that was written: backends using cosine
        similarity L2-normalize on write, discarding the original magnitude.
        Callers must therefore treat returned vectors as directions and compare
        them up to a positive scale factor.

        :param ids: Record identifiers to retrieve in order.
        :return: Points aligned with ``ids``; missing IDs mapped to ``None``.
        """
        ...

    @abstractmethod
    async def get_payloads_by_ids(self, ids: List[str]) -> List[Dict | None]:
        """
        Fetch stored payloads by ID.

        :param ids: Record identifiers to retrieve in order.
        :return: Payloads aligned with ``ids``; missing IDs mapped to ``None``.
        """
        ...


T = TypeVar("T")

@dataclass
class BaseKVStorage(Generic[T], BaseStorage, ABC):
    """
    Abstract interface for key-value storage backends.
    """

    @abstractmethod
    async def all_keys(self) -> List[str]:
        """
        Return all currently stored keys.

        :return: List of key strings.
        """
        ...

    @abstractmethod
    async def get_by_id(self, id: str) -> Union[T, None]:
        """
        Fetch one value by key.

        :param id: Key to retrieve.
        :return: Stored value or ``None`` if absent.
        """
        ...

    @abstractmethod
    async def get_by_ids(self, ids: list[str], fields: Union[set[str], None] = None) -> List[Union[T, None]]:
        """
        Fetch multiple values by key.

        :param ids: Keys to retrieve in order.
        :param fields: Optional field projection for dict-like values.
        :return: Values aligned with ``ids``; missing keys mapped to ``None``.
        """
        ...

    @abstractmethod
    async def filter_keys(self, data: List[str]) -> Set[str]:
        """
        Return keys from input that do not exist in storage.

        :param data: Candidate keys.
        :return: Subset of keys that are currently missing.
        """
        ...

    @abstractmethod
    async def upsert(self, data: Dict[str, T]):
        """
        Insert or update key-value entries.

        :param data: Mapping of keys to values.
        """
        ...

    @abstractmethod
    async def delete(self, ids: List[str]) -> None:
        """
        Delete entries by keys.

        :param ids: Keys to delete.
        """
        ...

    @abstractmethod
    async def drop(self):
        """
        Remove all entries from the storage backend.
        """
        ...

NodeT = TypeVar("NodeT", bound=Node)
EdgeT = TypeVar("EdgeT", bound=Edge)

@dataclass
class BaseGraphStorage(Generic[NodeT, EdgeT], BaseStorage, ABC):
    """
    Abstract interface for multigraph storage backends.

    RAGU assumes that every graph implementation is a directed multigraph.
    """

    @abstractmethod
    async def edges_degrees(self, edge_specs: List[EdgeSpec]) -> List[int]:
        """
        Compute degree sums of node's degree for provided edge specifications.

        :param edge_specs: Tuples ``(subject_id, object_id, relation_id)``.
        :return: Degree values aligned with the input order.
        """
        ...


    @abstractmethod
    async def get_nodes(self, node_ids: List[str]) -> List[Optional[NodeT]]:
        """
        Fetch nodes by IDs.

        :param node_ids: Node IDs to retrieve.
        :return: Nodes aligned with input IDs; missing IDs mapped to ``None``.
        """
        ...

    @abstractmethod
    async def upsert_nodes(self, nodes: Iterable[NodeT]) -> None:
        """
        Insert or update nodes.

        :param nodes: Nodes to upsert.
        """
        ...

    @abstractmethod
    async def delete_nodes(self, node_ids: List[str]) -> None:
        """
        Delete nodes by IDs.

        :param node_ids: Node IDs to remove.
        """
        ...

    @abstractmethod
    async def get_edges(self, edge_specs: List[EdgeSpec]) -> List[List[EdgeT]]:
        """
        Fetch edges by specifications, one result list per spec.

        RAGU graphs are directed multigraphs, so a pair of nodes may hold
        several edges. Returning a list per spec keeps the result aligned with
        ``edge_specs`` regardless of how many edges each one matches, so callers
        can safely ``zip`` the two together.

        For a spec ``(subject, object, relation_id)`` the matching list holds:

        * the one named edge, or an empty list, when ``relation_id`` is given;
        * every edge between the pair when ``relation_id`` is ``None``.

        :param edge_specs: Tuples ``(subject_id, object_id, relation_id)``.
        :return: One list of edges per spec, in spec order; empty where nothing
            matched.
        """
        ...

    @abstractmethod
    async def upsert_edges(self, edges: Iterable[EdgeT]) -> None:
        """
        Insert or update edges.

        :param edges: Edges to upsert.
        """
        ...

    @abstractmethod
    async def delete_edges(self, edge_specs: List[EdgeSpec]) -> None:
        """
        Delete edges by specifications. Missing edges are tolerated.

        A spec whose ``relation_id`` is ``None`` removes every edge between the
        pair, mirroring :meth:`get_edges`.

        :param edge_specs: Tuples ``(subject_id, object_id, relation_id)`` to delete.
        """
        ...

    @abstractmethod
    async def get_all_edges_for_nodes(self, node_ids: List[str]) -> List[List[EdgeT]]:
        """
        Fetch all incident edges for each provided node.

        :param node_ids: Node IDs to inspect.
        :return: Edge lists aligned with input node IDs.
        """
        ...

    @abstractmethod
    async def get_all_nodes(self) -> List[NodeT]:
        """
        Fetch all nodes stored in the backend.

        :return: List of nodes.
        """
        ...

    @abstractmethod
    async def get_all_edges(self) -> List[EdgeT]:
        """
        Fetch all edges stored in the backend.

        :return: List of edges.
        """
        ...
