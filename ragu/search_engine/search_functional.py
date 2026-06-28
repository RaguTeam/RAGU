# Based on https://github.com/gusye1234/nano-graphrag/blob/main/nano_graphrag/

from typing import Callable, List, TypeVar

from ragu.chunker.types import Chunk
from ragu.common.prompts.default_models import SubQuery
from ragu.graph.knowledge_graph import KnowledgeGraph
from ragu.graph.types import Entity, Community, CommunitySummary, Relation
from ragu.models.scorer import Scorer


T = TypeVar("T")


async def _find_most_related_edges_from_entities(
    entities: list[Entity],
    knowledge_graph: KnowledgeGraph,
) -> list[Relation]:
    """
    Return unique graph edges adjacent to the seed entities.

    Edges are deduplicated by stored relation ID and sorted by descending ``relation_strength``.
    """
    entity_ids = [entity.id for entity in entities if entity and entity.id]
    if not entity_ids:
        return []

    grouped_edges = await knowledge_graph.index.graph_backend.get_all_edges_for_nodes(entity_ids)
    all_related_edges = [edge for edges in grouped_edges for edge in edges if edge]

    if not all_related_edges:
        return []

    seen_relations = set()
    unique_edges: list[Relation] = []
    for edge in all_related_edges:
        dedup_key = edge.id or (
            edge.subject_id,
            edge.object_id,
            edge.relation_type,
            edge.description,
        )
        if dedup_key in seen_relations:
            continue
        seen_relations.add(dedup_key)
        unique_edges.append(edge)

    return sorted(
        unique_edges,
        key=lambda edge: edge.relation_strength,
        reverse=True
    )


async def _find_most_related_text_unit_from_entities(
        entities: List[Entity],
        knowledge_graph: KnowledgeGraph
) -> list[Chunk]:
    """
    Return source chunks associated with seed entities.

    Chunks are ordered first by the seed entity order, then by how many one-hop
    neighboring entities share the same chunk.
    """
    seed_entities = [entity for entity in entities if entity and entity.id]
    if not seed_entities:
        return []

    chunks_id = [entity.source_chunk_id for entity in seed_entities]
    seed_ids = [entity.id for entity in seed_entities]

    grouped_relations = await knowledge_graph.index.graph_backend.get_all_edges_for_nodes(seed_ids)
    neighbor_ids: List[str] = []
    for seed_id, relations_group in zip(seed_ids, grouped_relations):
        for relation in relations_group:
            if relation is None:
                continue
            if relation.subject_id == seed_id:
                neighbor_ids.append(relation.object_id)
            elif relation.object_id == seed_id:
                neighbor_ids.append(relation.subject_id)
    neighbor_ids = list(dict.fromkeys(neighbor_ids))
    neighbors = await knowledge_graph.index.get_nodes(neighbor_ids)

    all_one_hop_text_units_lookup = {
        neighbor.id : neighbor.source_chunk_id for neighbor in neighbors if neighbor is not None
    }

    all_text_units_lookup = {}
    for index, (seed_id, this_text_units, this_edges) in enumerate(zip(seed_ids, chunks_id, grouped_relations)):
        for c_id in this_text_units:
            if c_id in all_text_units_lookup:
                continue
            relation_counts = 0
            for e in this_edges:
                if e.subject_id == seed_id:
                    neighbor_id = e.object_id
                elif e.object_id == seed_id:
                    neighbor_id = e.subject_id
                else:
                    continue
                if (
                        neighbor_id in all_one_hop_text_units_lookup
                        and c_id in all_one_hop_text_units_lookup[neighbor_id]
                ):
                    relation_counts += 1
            all_text_units_lookup[c_id] = {
                "order": index,
                "relation_counts": relation_counts,
            }

    chunk_ids = list(all_text_units_lookup.keys())
    chunk_payloads = await knowledge_graph.index.chunks_kv_storage.get_by_ids(chunk_ids)
    for c_id, payload in zip(chunk_ids, chunk_payloads):
        all_text_units_lookup[c_id]["data"] = payload

    all_text_units = [
        {"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None
    ]
    chunks = sorted(
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )
    return [
        Chunk(**chunk_data)
        for chunk in chunks
        if (chunk_data := chunk["data"]) is not None
    ]

async def _find_documents_id(entities: List[Entity]):
    """
    Collect unique document IDs referenced by the supplied entities.
    """
    documents_set = set()
    for entity in entities:
        if hasattr(entity, 'documents_id') and entity.documents_id:
            documents_set.update(entity.documents_id)
    return list(documents_set)


async def _find_most_related_community_from_entities(
        entities: List[Entity],
        knowledge_graph: KnowledgeGraph,
        level: int = 2
) -> list[CommunitySummary]:
    """
    Return community summaries linked to seed entity cluster memberships.

    Only clusters with ``level <= level`` are considered. Cluster IDs may be
    stored either as full community IDs or as numeric local cluster IDs that are
    converted to stable :class:`Community` IDs.
    """
    if not entities:
        return []

    desired_community_ids: set[str] = set()
    for entity in entities:
        if not getattr(entity, "clusters", None):
            continue
        for cluster_data in entity.clusters:
            try:
                c_level = int(cluster_data.get("level", 9999))
            except Exception:
                continue
            if c_level <= level:
                cid = cluster_data.get("cluster_id")
                if cid is None:
                    continue

                cid_str = str(cid)
                if cid_str.startswith("com-"):
                    desired_community_ids.add(cid_str)
                    continue

                try:
                    cluster_id = int(cid_str)
                except Exception:
                    continue

                community_id = Community(
                    level=c_level,
                    cluster_id=cluster_id,
                    entities=[],
                    relations=[]
                ).id
                desired_community_ids.add(community_id)

    if not desired_community_ids:
        return []

    summary_store = knowledge_graph.index.community_summary_kv_storage

    community_ids = list(desired_community_ids)
    summaries = await summary_store.get_by_ids(community_ids)
    return [
        CommunitySummary(id=community_id, summary=summary_text)
        for community_id, summary_text in zip(community_ids, summaries)
        if summary_text is not None
    ]

async def _rerank_items(
    query: str,
    items: list[T],
    text_getter: Callable[[T], str],
    reranker: Scorer | None,
) -> list[T]:
    """
    Rerank items with an optional scorer while preserving original items.

    If ``reranker`` is not provided, items are returned unchanged.
    """
    if reranker is None or not items:
        return items

    rerank_results = await reranker.score(
        query,
        [text_getter(item) for item in items],
    )
    return [items[idx] for idx, _ in rerank_results if 0 <= idx < len(items)]

def _topological_sort(subqueries: List[SubQuery]) -> List[SubQuery]:
    """
    Sort subqueries so dependencies appear before dependent subqueries.

    :param subqueries: Query-plan nodes keyed by their ``id`` fields.
    :return: Topologically ordered subquery list.
    """
    by_id = {q.id: q for q in subqueries}
    visited = set()
    ordered: List[SubQuery] = []

    def visit(q: SubQuery):
        if q.id in visited:
            return
        for dep in q.depends_on:
            visit(by_id[dep])
        visited.add(q.id)
        ordered.append(q)

    for q in subqueries:
        visit(q)

    return ordered
