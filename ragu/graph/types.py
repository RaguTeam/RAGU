"""
Data structures for representing entities, relations, and communities
within the RAGU knowledge graph system.

These lightweight dataclasses define the core graph components used
throughout the indexing, storage, and retrieval pipelines. Each object
includes an auto-generated hashed ID to ensure consistent deduplication
and reproducibility across sessions.

Modules overview
----------------
• **Entity**, **Relation** — primary units representing semantic nodes
  and edges within the knowledge graph.

• **Community**, **CommunitySummary** — represent clustered subgraphs
  (communities) and their human-readable summaries produced during
  community detection or summarization.
"""

from typing import Any, ClassVar, Iterable, List, Mapping
from dataclasses import dataclass, field

from ragu.storage.types import ClusterInfo, Edge, Node
from ragu.utils.ragu_utils import compute_mdhash_id


def merge_metainfo(metainfos: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """
    Shallow-merge several ``metainfo`` dictionaries into a new one.

    Earlier mappings take precedence: a key present in more than one of them
    keeps the value of the first mapping that defines it. The default merge
    policies pass the richest record first, so its metainfo wins on conflicts
    while keys only the other records know about are still kept.

    :param metainfos: Dictionaries to merge, in order of decreasing precedence.
    :return: A new dictionary; the inputs are left untouched.
    """
    merged: dict[str, Any] = {}
    for metainfo in metainfos:
        for key, value in metainfo.items():
            merged.setdefault(key, value)
    return merged


@dataclass(slots=True)
class Entity(Node):
    """
    Represents a semantic entity (graph node).

    :param entity_name: Canonical name of the entity.
    :param entity_type: Type or category of the entity (e.g., Person, Organization).
    :param description: Textual description extracted from the source text.
    :param source_chunk_id: Identifiers of text chunks where the entity was found.
    :param documents_id: Identifiers of documents containing this entity.
    :param clusters: List of cluster memberships from community detection.
    :param id: Unique identifier; auto-generated if not provided.
    :param metainfo: Free-form additional information about the entity
        (provenance, confidence, external identifiers, ...). Not part of the
        identity hash. Persisted as-is by every storage backend, so values
        should stay JSON-serializable; the GML (NetworkX) backend additionally
        requires keys to be identifiers (``[A-Za-z][A-Za-z0-9_]*``) and
        rejects ``None`` values.
    :raises ValueError: If an explicitly provided ``id`` is empty.
    """
    entity_name: str
    entity_type: str
    description: str
    source_chunk_id: list[str]
    documents_id: list[str] = field(default_factory=list[str])
    clusters: list[ClusterInfo] = field(default_factory=list[ClusterInfo])
    id: str = 'auto'
    metainfo: dict[str, Any] = field(default_factory=dict[str, Any])

    label_field: ClassVar[str] = "entity_type"

    def __post_init__(self) -> None:
        """
        Generate a stable MD5-based identifier if not already set.

        :raises ValueError: If an explicitly provided ``id`` is empty.
        """
        if self.id == 'auto':
            self.id = compute_mdhash_id(
                (self.entity_name + " - " + self.entity_type),
                prefix="ent-"
            )
        elif not self.id:
            raise ValueError("Entity id must be a non-empty string")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Entity):
            return NotImplemented
        return self.id == other.id and self.description == other.description

    def __hash__(self) -> int:
        return hash(self.id)

    def to_text(self) -> str:
        return f"{self.entity_name} - {self.description}"


@dataclass(slots=True)
class Relation(Edge):
    """
    Represents a directed relation between two entities.

    :param subject_id: ID of the source (subject) entity.
    :param object_id: ID of the target (object) entity.
    :param subject_name: Display name of the subject entity.
    :param object_name: Display name of the target entity.
    :param description: Description of the relationship.
    :param relation_strength: Numerical weight of the relation (default: 1.0).
    :param source_chunk_id: Identifiers of chunks where this relation was extracted.
    :param id: Unique identifier; auto-generated if not provided.
    :param metainfo: Free-form additional information about the relation. Same
        contract as :attr:`Entity.metainfo`.
    :raises ValueError: If an explicitly provided ``id`` is empty.
    """
    subject_id: str
    object_id: str
    subject_name: str
    object_name: str
    relation_type: str
    description: str
    relation_strength: int | float = 1.0
    source_chunk_id: list[str] = field(default_factory=list[str])

    label_field: ClassVar[str] = "relation_type"
    id: str = 'auto'
    metainfo: dict[str, Any] = field(default_factory=dict[str, Any])

    def __post_init__(self) -> None:
        """
        Generate a stable MD5-based identifier if not already set.

        :raises ValueError: If an explicitly provided ``id`` is empty.
        """
        if self.id == 'auto':
            self.id = compute_mdhash_id(
                (self.subject_id + " -> " + self.object_id + self.relation_type),
                prefix="rel-"
            )
        elif not self.id:
            raise ValueError("Relation id must be a non-empty string")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Relation):
            return NotImplemented
        return self.id == other.id and self.description == other.description

    def __hash__(self) -> int:
        return hash(self.id)

    def to_text(self) -> str:
        return f"{self.description}"


@dataclass(slots=True)
class Community:
    """
    Represents a detected community or cluster in the knowledge graph.

    :param level: Hierarchical clustering level of the community.
    :param cluster_id: Identifier of the cluster within the given level.
    :param entities: List of :class:`Entity` objects belonging to the community.
    :param relations: List of :class:`Relation` objects connecting entities.
    :param id: Unique identifier; auto-generated if not provided.
    """
    level: int
    cluster_id: int
    entities: List[Entity]
    relations: List[Relation]
    id: str = 'auto'

    def __post_init__(self) -> None:
        """
        Generate a stable MD5-based identifier if not already set.

        :raises ValueError: If an explicitly provided ``id`` is empty.
        """
        if self.id == 'auto':
            self.id = compute_mdhash_id(
                f"{self.level}:{self.cluster_id}",
                prefix="com-"
            )
        elif not self.id:
            raise ValueError("Community id must be a non-empty string")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Community):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)


@dataclass(slots=True)
class CommunitySummary:
    """
    Stores a textual summary of a community.

    :param id: Unique identifier of the community.
    :param summary: Generated textual summary of the community content.
    :raises ValueError: If ``id`` is empty.
    """
    id: str
    summary: str

    def __post_init__(self) -> None:
        """
        Validate the summary identifier.

        :raises ValueError: If ``id`` is empty.
        """
        if not self.id:
            raise ValueError("CommunitySummary id must be a non-empty string")