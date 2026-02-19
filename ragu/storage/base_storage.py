from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Union, Generic, TypeVar, List, Set, Optional, Tuple, Dict

from ragu.graph.types import Entity, Relation
from ragu.storage.types import Embedding, EmbeddingHit

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


@dataclass
class BaseVectorStorage(BaseStorage, ABC):
    """
    Abstract interface for vector storage backends.
    """

    @abstractmethod
    async def query(self, vectors: Embedding, top_k: int) -> List[EmbeddingHit]:
        """
        Retrieve top-k nearest items for a batch of embedding vectors.

        :param vectors: Query embedding vector.
        :param top_k: Maximum number of results to return per query vector.
        :return: A list of query hits with distance score and metadata.
        """
        ...

    @abstractmethod
    async def upsert(self, data: List[Embedding]) -> None:
        """
        Insert or update embedding records.

        :param data: Embedding records to upsert.
        """
        ...

    @abstractmethod
    async def delete(self, ids: List[str]) -> None:
        """
        Delete records by IDs.

        :param ids: Record identifiers to remove.
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


@dataclass
class BaseGraphStorage(BaseStorage, ABC):
    """
    Abstract interface for multigraph storage backends.
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
    async def get_nodes(self, node_ids: List[str]) -> List[Optional[Entity]]:
        """
        Fetch nodes by IDs.

        :param node_ids: Node IDs to retrieve.
        :return: Entities aligned with input IDs; missing IDs mapped to ``None``.
        """
        ...

    @abstractmethod
    async def upsert_nodes(self, nodes: List[Entity]) -> None:
        """
        Insert or update nodes.

        :param nodes: Entity nodes to upsert.
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
    async def get_edges(self, edge_specs: List[EdgeSpec]) -> List[Optional[Relation]]:
        """
        Fetch edges by specifications.

        :param edge_specs: Tuples ``(subject_id, object_id, relation_id)``.
        :return: Relations aligned with input specs; missing specs mapped to ``None``.
        """
        ...

    @abstractmethod
    async def upsert_edges(self, edges: List[Relation]) -> None:
        """
        Insert or update edges.

        :param edges: Relations to upsert.
        """
        ...

    @abstractmethod
    async def delete_edges(self, edge_specs: List[EdgeSpec]) -> None:
        """
        Delete edges by specifications.

        :param edge_specs: Tuples ``(subject_id, object_id, relation_id)`` to delete.
        """
        ...

    @abstractmethod
    async def get_all_edges_for_nodes(self, node_ids: List[str]) -> List[List[Relation]]:
        """
        Fetch all incident edges for each provided node.

        :param node_ids: Node IDs to inspect.
        :return: Edge lists aligned with input node IDs.
        """
        ...

    @abstractmethod
    async def get_all_nodes(self) -> List[Entity]:
        """
        Fetch all nodes stored in the backend.

        :return: List of entities.
        """
        ...

    @abstractmethod
    async def get_all_edges(self) -> List[Relation]:
        """
        Fetch all edges stored in the backend.

        :return: List of relations.
        """
        ...
