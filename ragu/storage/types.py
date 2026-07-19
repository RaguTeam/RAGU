import time
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, TypedDict

from ragu.utils.ragu_utils import FLOATS, compute_mdhash_id, serialize


class ClusterInfo(TypedDict):
    """
    Represents graph cluster info.
    """
    level: int
    cluster_id: int


class Node:
    """
    Base graph node type for storage adapters.

    Subclasses are expected to be dataclasses and define ``id``,
    ``source_chunk_id``, and ``clusters`` fields.
    """

    id: str

    #: Name of the field whose value groups nodes of this type. Storage backends
    #: may use it to build labels, indexes or partitions; ``None`` means this
    #: node type has no grouping. Declared as ``ClassVar`` on purpose: a plain
    #: annotation would become a dataclass field on subclasses and leak into
    #: ``asdict()``, which several backends persist verbatim.
    label_field: ClassVar[Optional[str]] = None

    def get_label(self) -> Optional[str]:
        """
        Return the value this node is grouped by.

        :returns: Value of :attr:`label_field`, or ``None`` if the type declares
            no grouping field or the value is empty.
        :rtype: Optional[str]
        """
        if self.label_field is None:
            return None
        return getattr(self, self.label_field, None) or None

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize node to dict.
        """
        return serialize(self)
    
    def to_text(self):
        """
        Convert node to text representation.
        """
        return str(f"{self.id}")


class Edge:
    """
    Base graph edge type for storage adapters.

    Subclasses are expected to be dataclasses and define ``id``,
    ``subject_id`` and ``object_id`` fields.
    """

    id: str
    subject_id: str
    object_id: str

    #: Name of the field whose value groups edges of this type. Same contract as
    #: :attr:`Node.label_field`; see there for why it is a ``ClassVar``.
    label_field: ClassVar[Optional[str]] = None

    def get_label(self) -> Optional[str]:
        """
        Return the value this edge is grouped by.

        :returns: Value of :attr:`label_field`, or ``None`` if the type declares
            no grouping field or the value is empty.
        :rtype: Optional[str]
        """
        if self.label_field is None:
            return None
        return getattr(self, self.label_field, None) or None

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize edge to dict.
        """
        return serialize(self)

    def to_text(self):
        """
        Convert edge to text representation.
        """
        return str(f"{self.subject_id} - {self.object_id}")


DenseEmbedding = FLOATS


@dataclass(slots=True)
class SparseEmbedding:
    indices: List[int]
    values: List[float]

    def __post_init__(self):
        if len(self.indices) != len(self.values):
            raise ValueError("indices and values must have the same length")


@dataclass(slots=True)
class Point:
    """
    Represents embedding point.

    :param id: Matched record identifier.
    :param dense_embedding: Dense embedding.
    :param sparse_embedding: Sparse embedding (TF-IFD, BM25 and so on).
    :param metadata: Additional payload.
    """
    id: str = "auto"
    dense_embedding: DenseEmbedding | None = None
    sparse_embedding: SparseEmbedding | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.id == "auto":
            self.id = compute_mdhash_id(str(time.time_ns()), prefix="pnt")

        if self.dense_embedding is None and self.sparse_embedding is None:
            raise ValueError("Point must contain at least one dense or sparse embedding")


@dataclass(slots=True)
class EmbeddingHit:
    """
    Vector query hit.

    :param id: Matched record identifier.
    :param distance: Similarity/distance score to query embedding.
    :param metadata: Additional payload.
    """
    id: str
    distance: float
    metadata: Dict[str, Any] = field(default_factory=dict[str, Any])
