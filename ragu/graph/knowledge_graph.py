from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import (
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple
)

import pandas as pd

from ragu.chunker.base_chunker import BaseChunker
from ragu.chunker.types import Chunk
from ragu.common.logger import logger
from ragu.common.global_parameters import Settings
from ragu.graph.builder_modules import RemoveIsolatedNodes
from ragu.graph.graph_builder_pipeline import (
    InMemoryGraphBuilder,
    BuilderArguments,
    GraphBuilderModule
)
from ragu.graph.types import Community, Entity, Relation, CommunitySummary
from ragu.models.embedder import Embedder
from ragu.models.sparse_embedder import SparseEmbedder
from ragu.models.llm import LLM
from ragu.graph.index import Index, StorageArguments
from ragu.storage.types import ClusterInfo
from ragu.triplet.base_artifact_extractor import BaseArtifactExtractor
from ragu.storage.base_storage import EdgeSpec


def _duplicate_ids(items: Iterable[Entity | Relation | CommunitySummary | Chunk | Community]) -> List[str]:
    counts = Counter(item.id for item in items)
    return [item_id for item_id, count in counts.items() if count > 1]


def _unique_description_fragments(descriptions: Iterable[str]) -> List[str]:
    """
    Split descriptions into normalized fragments and keep first-seen unique ones.

    This prevents repeated sentence fragments when previously merged descriptions
    are merged again with incremental upserts.

    :param descriptions: Description texts to split and normalize.
    :return: Deduplicated description fragments in first-seen order.
    """
    unique_parts: List[str] = []
    seen: set[str] = set()

    for description in descriptions:
        text = (description or "").strip()
        if not text:
            continue
        raw_parts = re.split(r"\n+|(?<=[.!?])\s+", text)
        for part in raw_parts:
            cleaned = re.sub(r"\s+", " ", part).strip()
            if not cleaned:
                continue
            key = cleaned.casefold()
            if key in seen:
                continue
            seen.add(key)
            unique_parts.append(cleaned)

    return unique_parts


# TODO: add an ability to pass merge policy into KG
def default_merge_entities_policy(entities: List[Entity]) -> Entity:
    """
    Default merge policy for entities (class Entity).
    Default policy = concatenate unique descriptions, merge source_chunk_ids, docs ids and clusters.

    :param entities: Entities to merge.
    :return: Merged entities.
    """
    if len(entities) == 0:
        raise ValueError("Cannot merge empty entity list")

    if len(entities) == 1:
        return entities[0]

    entity_ids = {entity.id for entity in entities}
    if len(entity_ids) != 1:
        raise ValueError(f"Cannot merge entities with different IDs: {sorted(entity_ids)}")

    by_richness = sorted(entities, key=lambda e: len(e.source_chunk_id), reverse=True)
    primary = by_richness[0]

    descriptions = _unique_description_fragments(
        [entity.description for entity in by_richness]
    )

    all_chunks: set[str] = set()
    all_docs: set[str] = set()
    all_clusters: list[ClusterInfo] = []
    for entity in by_richness:
        all_chunks.update(entity.source_chunk_id)
        all_docs.update(entity.documents_id)
        all_clusters.extend(entity.clusters)

    deduplicated_clusters: List[ClusterInfo] = []
    seen_cluster_keys: Set[tuple[int, int]] = set()
    for cluster in all_clusters:
        assert isinstance(cluster, dict)
        try:
            level = cluster["level"]
            cluster_id = cluster["cluster_id"]
        except (TypeError, ValueError):
            continue

        cluster_key = (level, cluster_id)
        if cluster_key in seen_cluster_keys:
            continue

        seen_cluster_keys.add(cluster_key)
        normalized_cluster: ClusterInfo = {"level": level, "cluster_id": cluster_id}
        for key, value in cluster.items():
            if key not in normalized_cluster:
                normalized_cluster[key] = value
        deduplicated_clusters.append(normalized_cluster)

    return Entity(
        id=primary.id,
        entity_name=primary.entity_name,
        entity_type=primary.entity_type,
        description=" ".join(descriptions),
        source_chunk_id=sorted(all_chunks),
        documents_id=sorted(all_docs),
        clusters=deduplicated_clusters,
    )

# TODO: add an ability to pass merge policy into KG
def default_merge_relations_policy(relations: List[Relation]) -> Relation:
    """
    Default merge policy for relations (class Relation).
    Default policy = concatenate descriptions, merge source_chunk_ids, docs ids and clusters.

    :param relations: Relations to merge.
    :return: New single relation.
    """
    if len(relations) == 0:
        raise ValueError("Cannot merge empty relation list")

    if len(relations) == 1:
        return relations[0]

    relation_ids = {relation.id for relation in relations}
    if len(relation_ids) != 1:
        raise ValueError(f"Cannot merge relations with different IDs: {sorted(relation_ids)}")

    by_richness = sorted(relations, key=lambda r: len(r.source_chunk_id), reverse=True)
    primary = by_richness[0]

    descriptions = _unique_description_fragments(
        [relation.description for relation in by_richness]
    )

    avg_strength = sum(relation.relation_strength for relation in by_richness) / len(by_richness)

    all_chunks: set[str] = set()
    for relation in by_richness:
        all_chunks.update(relation.source_chunk_id)

    return Relation(
        id=primary.id,
        subject_id=primary.subject_id,
        object_id=primary.object_id,
        subject_name=primary.subject_name,
        object_name=primary.object_name,
        relation_type=primary.relation_type,
        description=" ".join(descriptions),
        relation_strength=avg_strength,
        source_chunk_id=sorted(all_chunks),
    )



class KnowledgeGraph:
    """
    High-level facade for building, storing, and querying a knowledge graph.

    :param llm: LLM client used by extraction and summarization modules.
    :param embedder: Embedder used for vector storage and clustering/similarity steps.
    :param chunker: Optional chunker used to split input documents.
    :param artifact_extractor: Optional extractor used to generate entities/relations from chunks.
    :param builder_settings: Graph-building behavior configuration. Defaults are used if omitted.
    :param storage_settings: Storage backend configuration. Defaults are used if omitted.
    :param additional_modules: Optional post-processing modules for extracted graph items.
    :param language: Optional language override. Defaults to ``Settings.language``.
    """

    def __init__(
        self,
        llm: Optional[LLM],
        embedder: Embedder,
        sparse_embedder: Optional[SparseEmbedder] = None,
        chunker: Optional[BaseChunker] = None,
        artifact_extractor: Optional[BaseArtifactExtractor] = None,
        builder_settings: Optional[BuilderArguments] = None,
        storage_settings: Optional[StorageArguments] = None,
        additional_modules: Optional[List[GraphBuilderModule]] = None,
        language: Optional[str] = None,
    ):
        """
        Initialize KnowledgeGraph with pipeline and storage components.

        :param llm: LLM client used by extraction and summarization modules.
        :param embedder: Embedder used by vector storage and optional clustering.
        :param sparse_embedder: Optional sparse embedder used for hybrid retrieval.
        :param chunker: Optional chunker used to split input documents.
        :param artifact_extractor: Optional entity/relation extractor.
        :param builder_settings: Optional graph builder settings.
        :param storage_settings: Optional storage backend settings.
        :param additional_modules: Optional post-processing modules for graph items.
        :param language: Optional language override. Defaults to ``Settings.language``.
        """
        self.builder_settings = builder_settings or BuilderArguments()
        self.storage_settings = storage_settings or StorageArguments()
        self.language = language or Settings.language
        self.embedder = embedder
        self.sparse_embedder = sparse_embedder

        what_to_add = additional_modules if additional_modules else []

        if self.builder_settings.remove_isolated_nodes:
            what_to_add.append(RemoveIsolatedNodes())

        # Build graph
        self.pipeline = InMemoryGraphBuilder(
            llm=llm,
            chunker=chunker,
            artifact_extractor=artifact_extractor,
            build_parameters=self.builder_settings,
            embedder=embedder,
            additional_pipeline=what_to_add,
            language=self.language,
        )
        # Store graph
        self.index: Index[Entity, Relation] = Index(
            embedder=embedder,
            sparse_embedder=sparse_embedder,
            arguments=self.storage_settings,
            node_t=Entity,
            edge_t=Relation,
        )

        self.make_community_summary = self.builder_settings.make_community_summary
        self.remove_isolated_nodes = self.builder_settings.remove_isolated_nodes
        self.vectorize_chunks = self.builder_settings.vectorize_chunks

    async def build_from_docs(self, docs: List[str]) -> "KnowledgeGraph":
        """
        Build graph and vector context from a list of input documents.

        :param docs: Input documents to process.
        :return: "KnowledgeGraph" for method chaining.
        """
        chunks = self.pipeline.chunker.split(docs) if self.pipeline.chunker else \
            [Chunk(doc, i, doc_id=f"doc_{i}") for i, doc in enumerate(docs)]
        logger.debug(f'Got {len(chunks)} chunks')

        chunks = await self._deduplicate_chunks_by_id(chunks)
        logger.debug(f'Got {len(chunks)} chunks after deduplicating')

        if not chunks:
            logger.warning("Nothing to build.")
            return self

        entities, relations, summaries, communities, chunks = await self.pipeline.extract_graph(chunks)
        logger.debug(f'Extracted {len(entities)} entities')
        logger.debug(f'Extracted {len(relations)} relations')
        logger.debug(f'Extracted {len(communities)} communities')
        logger.debug(f'Extracted {len(chunks)} chunks')

        is_vector_only = self.builder_settings.build_only_vector_context
        should_store_communities = self.make_community_summary and not is_vector_only

        if should_store_communities and communities:
            entities, communities, summaries = await self._reindex_cluster_ids(
                entities,
                communities,
                summaries,
            )

        if not is_vector_only:
            await self.upsert_entities(entities)
            await self.upsert_relations(relations)

        await self.index.upsert_chunks(chunks)

        if should_store_communities:
            await self.index.upsert_communities(communities)
            await self.upsert_summaries(summaries)

        return self

    async def upsert_entities(self, entities: List[Entity]) -> "KnowledgeGraph":
        """
        Add entities to the knowledge graph.

        Existing stored entities with matching IDs are merged before storage.
        Duplicate IDs within the same request are rejected.

        :param entities: Single entity or list of entities to add.
        :return: Self for method chaining.
        """
        duplicate_ids = _duplicate_ids(entities)
        if duplicate_ids:
            raise ValueError(f"Cannot insert duplicated entity IDs in one request: {duplicate_ids}")

        existing_entities = await self.index.get_nodes([entity.id for entity in entities])
        existing_by_id = {
            entity.id: entity
            for entity in existing_entities
            if entity is not None
        }

        merged_entities: List[Entity] = []
        for entity in entities:
            existing = existing_by_id.get(entity.id)
            if existing is not None:
                merged_entities.append(default_merge_entities_policy([entity, existing]))
            else:
                merged_entities.append(entity)

        await self.index.upsert_nodes(merged_entities)
        return self

    async def update_entities(self, entities: List[Entity]) -> "KnowledgeGraph":
        """
        Replace one or more existing entities by ID.

        :param entities: Single entity or list of entities to replace.
        :return: Self for method chaining.
        """
        duplicate_ids = _duplicate_ids(entities)
        if duplicate_ids:
            raise ValueError(f"Cannot update duplicated entity IDs in one request: {duplicate_ids}")

        await self.index.update_nodes(entities)
        return self

    async def get_entities(self, entity_ids: List[str]) -> List[Entity | None]:
        """
        Retrieve one or more entities by ID in one batched operation.

        :param entity_ids: Entity identifier or identifiers.
        :return: Matching entity or list of entities, preserving input order and
            using ``None`` for missing IDs.
        """
        return await self.index.get_nodes(entity_ids)

    async def delete_entities(self, entity_ids: List[str]) -> "KnowledgeGraph":
        """
        Delete an entity from the knowledge graph.

        :param entity_ids: ID of the entity to delete.
        :return: Self for method chaining.
        """
        await self.index.delete_nodes(entity_ids)
        return self

    async def upsert_relations(self, relations: List[Relation]) -> "KnowledgeGraph":
        """
        Add one or more relations to the knowledge graph.

        Existing stored relations with matching IDs are merged before storage.
        Duplicate IDs within the same request are rejected.

        :param relations: Relations to add.
        :return: Self for method chaining.
        """
        for item in relations:
            if not item.id:
                raise ValueError("Cannot insert relation without id")
        duplicate_ids = _duplicate_ids(relations)
        if duplicate_ids:
            raise ValueError(f"Cannot insert duplicated relation IDs in one request: {duplicate_ids}")

        edge_specs = [
            (relation.subject_id, relation.object_id, relation.id)
            for relation in relations
        ]
        existing_relations = await self.index.get_edges(edge_specs)

        merged_relations: List[Relation] = []
        for item, existing_relation in zip(relations, existing_relations):
            if existing_relation is not None:
                merged_relations.append(default_merge_relations_policy([item, existing_relation]))
            else:
                merged_relations.append(item)

        await self.index.upsert_edges(merged_relations)
        return self

    async def update_relations(self, relations: List[Relation]) -> "KnowledgeGraph":
        """
        Replace one or more existing relations by ID.

        :param relations: Relations to replace.
        :return: Self for method chaining.
        """
        for item in relations:
            if not item.id:
                raise ValueError("Cannot update relation without id")
        duplicate_ids = _duplicate_ids(relations)
        if duplicate_ids:
            raise ValueError(f"Cannot update duplicated relation IDs in one request: {duplicate_ids}")

        await self.index.update_edges(relations)
        return self

    async def delete_relations(self, edge_specs: List[EdgeSpec]) -> "KnowledgeGraph":
        """
        Delete relations from the knowledge graph.

        :param edge_specs: Edge specifications ``(subject_id, object_id, relation_id)``.
        :return: Self for method chaining.
        """
        await self.index.delete_edges(edge_specs)
        return self

    async def edges_degrees(self, edge_specs: List[EdgeSpec]) -> List[int]:
        """
        Get degrees for multiple edges.

        Each returned value is ``degree(source) + degree(target)`` for the
        corresponding edge spec, or ``0`` when relation/endpoints are missing.

        :param edge_specs: Edge specifications ``(subject_id, object_id, relation_id)``.
        :return: Degree sums in the same order as input specs.
        """
        return await self.index.graph_backend.edges_degrees(edge_specs)

    async def get_relations(self, edge_specs: List[EdgeSpec]) -> List[Relation | None]:
        """
        Retrieve one or more relations by edge spec in one batched operation.

        :param edge_specs: One edge spec or a list of edge specs.
        :return: Matching relation or list of relations, preserving input order
            and using ``None`` for missing edges.
        """
        return await self.index.get_edges(edge_specs)

    async def get_chunks(self, chunk_ids: List[str]) -> List[Chunk | None]:
        """
        Retrieve one or more chunks by ID in one batched operation.

        :param chunk_ids: Chunk identifier or identifiers.
        :return: Matching chunk or list of chunks, preserving input order and
            using ``None`` for missing IDs.
        """
        return await self.index.get_chunks(chunk_ids)

    async def get_communities(self, community_ids: List[str]) -> List[Community | None]:
        """
        Retrieve one or more communities by ID in one batched operation.

        :param community_ids: Community identifier or identifiers.
        :return: Matching community or list of communities, preserving input
            order and using ``None`` for missing IDs.
        """
        return await self.index.get_communities(community_ids)

    async def upsert_communities(self, communities: List[Community]) -> "KnowledgeGraph":
        """
        Add or replace one or more communities.

        :param communities: Communities to upsert.
        :return: Self for method chaining.
        """
        duplicate_ids = _duplicate_ids(communities)
        if duplicate_ids:
            raise ValueError(f"Cannot upsert duplicated community IDs in one request: {duplicate_ids}")

        await self.index.upsert_communities(communities)
        return self

    async def update_communities(self, communities: List[Community]) -> "KnowledgeGraph":
        """
        Replace one or more existing communities by ID.

        :param communities: Communities to replace.
        :return: Self for method chaining.
        :raises ValueError: If duplicate IDs are provided or a community is missing.
        """
        duplicate_ids = _duplicate_ids(communities)
        if duplicate_ids:
            raise ValueError(f"Cannot update duplicated community IDs in one request: {duplicate_ids}")

        community_ids = [community.id for community in communities]
        existing_communities = await self.index.community_kv_storage.get_by_ids(community_ids)
        missing_ids = [
            community_id
            for community_id, existing in zip(community_ids, existing_communities)
            if existing is None
        ]
        if missing_ids:
            raise ValueError(f"Cannot update non-existent communities: {missing_ids}")

        await self.index.upsert_communities(communities)
        return self

    async def delete_communities(self, community_ids: List[str]) -> "KnowledgeGraph":
        """
        Delete communities and their summaries.

        :param community_ids: IDs of the communities to delete.
        :return: Self for method chaining.
        """
        await self.index.delete_communities(community_ids)
        return self

    async def upsert_summaries(self, summaries: List[CommunitySummary]) -> "KnowledgeGraph":
        """
        Add or replace community summaries.

        :param summaries: Summaries to upsert.
        :return: Self for method chaining.
        """
        duplicate_ids = _duplicate_ids(summaries)
        if duplicate_ids:
            raise ValueError(f"Cannot upsert duplicated summary IDs in one request: {duplicate_ids}")

        await self.index.upsert_summaries(summaries)
        return self

    async def update_summaries(self, summaries: List[CommunitySummary]) -> "KnowledgeGraph":
        """
        Replace one or more existing community summaries by ID.

        :param summaries: Summaries to replace.
        :return: Self for method chaining.
        :raises ValueError: If duplicate IDs are provided or a summary is missing.
        """
        duplicate_ids = _duplicate_ids(summaries)
        if duplicate_ids:
            raise ValueError(f"Cannot update duplicated summary IDs in one request: {duplicate_ids}")

        summary_ids = [summary.id for summary in summaries]
        existing_summaries = await self.index.community_summary_kv_storage.get_by_ids(summary_ids)
        missing_ids = [
            summary_id
            for summary_id, existing in zip(summary_ids, existing_summaries)
            if existing is None
        ]
        if missing_ids:
            raise ValueError(f"Cannot update non-existent summaries: {missing_ids}")

        await self.index.upsert_summaries(summaries)
        return self

    async def get_summaries(self, summary_ids: List[str]) -> List[CommunitySummary | None]:
        """
        Retrieve one or more community summaries by ID in one batched operation.

        :param summary_ids: Summary identifier or identifiers.
        :return: Matching summary or list of summaries, preserving input order
            and using ``None`` for missing IDs.
        """
        results = await self.index.community_summary_kv_storage.get_by_ids(summary_ids)
        summaries = [
            None if summary is None else CommunitySummary(id=summary_id, summary=summary)
            for summary_id, summary in zip(summary_ids, results)
        ]
        return summaries

    async def delete_summaries(self, summary_ids: List[str]) -> "KnowledgeGraph":
        """
        Delete community summaries.

        :param summary_ids: IDs of the summaries to delete.
        :return: Self for method chaining.
        """
        await self.index.community_summary_kv_storage.delete(summary_ids)
        await self.index.community_summary_kv_storage.index_done_callback()
        return self

    async def _deduplicate_chunks_by_id(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Deduplicate chunks by ``chunk.id`` preserving original order.

        :param chunks: Chunks to deduplicate.
        :return: Deduplicated chunk list preserving original order.
        """
        if not chunks:
            return chunks

        already_in_index = await self.index.chunks_kv_storage.all_keys()

        unique_chunks: List[Chunk] = []
        seen_ids: set[str] = set(already_in_index)
        duplicate_count = 0

        for chunk in chunks:
            chunk_id = chunk.id
            if chunk_id in seen_ids:
                duplicate_count += 1
                continue
            seen_ids.add(chunk_id)
            unique_chunks.append(chunk)

        if duplicate_count > 0:
            logger.warning(
                f"Found {duplicate_count}/{len(chunks)} duplicated chunks by id. "
                f"Using {len(unique_chunks)} unique chunks."
            )

        return unique_chunks

    async def reindex_community(self) -> "KnowledgeGraph":
        """
        Run clusterization and community summarization in knowledge graph.
        """
        if not self.pipeline.community_summarizer:
            raise ValueError()

        entities = await self.index.graph_backend.get_all_nodes()
        relations = await self.index.graph_backend.get_all_edges()

        entities = list(
            map(
                lambda node: Entity(
                    id=node.id,
                    entity_name=node.entity_name,
                    description=node.description,
                    entity_type=node.entity_type,
                    source_chunk_id=node.source_chunk_id,
                    documents_id=node.documents_id,
                    clusters=[]),
                entities
            )
        )

        communities = await self.pipeline.cluster_graph(entities=entities, relations=relations)
        summaries = (
            await self.pipeline.community_summarizer.summarize(communities=communities) # type: ignore
            if communities
            else []
        )

        await self.update_entities(entities)

        await self.index.community_kv_storage.drop()
        await self.index.community_summary_kv_storage.drop()

        await self.upsert_communities(communities)
        await self.upsert_summaries(summaries)

        return self

    # TODO: add an ability to summarize any of Entity/Relation (or Node/Edge) child types.
    async def reindex_descriptions(
        self,
        summarize_only_more_than: Optional[int] = None,
    ) -> "KnowledgeGraph":
        """
        Summarize existing entity and relation descriptions that exceed a sentence threshold.

        The method reuses the configured entity/relation summarizers by building
        summarizer-compatible DataFrames for only the long-description items.
        Items at or below the threshold are left unchanged.

        :param summarize_only_more_than: Summarize descriptions with more than
            this many sentences. Defaults to the graph builder setting.
        :return: Self for method chaining.
        """
        entity_summarizer = self.pipeline.entity_summarizer
        relation_summarizer = self.pipeline.relation_summarizer
        if entity_summarizer is None or relation_summarizer is None:
            raise ValueError("Description reindexing requires entity and relation summarizers.")

        if not entity_summarizer.use_llm_summarization or not relation_summarizer.use_llm_summarization: # type: ignore
            raise ValueError("Description reindexing requires LLM summarization to be enabled.")
        threshold = (
            summarize_only_more_than
            if summarize_only_more_than is not None
            else self.builder_settings.summarize_only_if_more_than
        )

        def _description_sentence_count(description: str) -> int:
            text = (description or "").strip()
            if not text:
                return 0

            # TODO: replace with sentenizer
            parts = re.split(r"\n+|(?<=[.!?])\s+", text)
            return sum(1 for part in parts if re.sub(r"\s+", " ", part).strip())

        all_entities = await self.index.graph_backend.get_all_nodes()
        entities_to_summarize = [
            entity for entity in all_entities
            if _description_sentence_count(entity.description) > threshold
        ]

        all_relations = await self.index.graph_backend.get_all_edges()
        relations_to_summarize = [
            relation for relation in all_relations
            if _description_sentence_count(relation.description) > threshold
        ]

        summarized_entities: List[Entity] = []
        if entities_to_summarize:
            entity_rows = [
                {
                    "id": entity.id,
                    "entity_name": entity.entity_name,
                    "entity_type": entity.entity_type,
                    "description": [entity.description],
                    "source_chunk_id": entity.source_chunk_id,
                    "documents_id": entity.documents_id,
                    "clusters": entity.clusters,
                    "duplicate_count": entity_summarizer.summarize_only_if_more_than + 1, # type: ignore
                }
                for entity in entities_to_summarize
            ]
            summarized_entities = await entity_summarizer.summarize_entities(pd.DataFrame(entity_rows)) # type: ignore

        summarized_relations: List[Relation] = []
        if relations_to_summarize:
            relation_rows = [
                {
                    "id": relation.id,
                    "subject_id": relation.subject_id,
                    "object_id": relation.object_id,
                    "subject_name": relation.subject_name,
                    "object_name": relation.object_name,
                    "relation_type": relation.relation_type,
                    "description": relation.description,
                    "relation_strength": relation.relation_strength,
                    "source_chunk_id": relation.source_chunk_id,
                    "duplicate_count": relation_summarizer.summarize_only_if_more_than + 1, # type: ignore
                }
                for relation in relations_to_summarize
            ]
            summarized_relations = await relation_summarizer.summarize_relations(pd.DataFrame(relation_rows)) # type: ignore

        if not summarized_entities and not summarized_relations:
            return self

        if summarized_entities:
            await self.update_entities(summarized_entities)
        if summarized_relations:
            await self.update_relations(summarized_relations)

        return self

    async def reindex_graph(self) -> "KnowledgeGraph":
        """
        Reindex item descriptions and clusters. Useful after upserting a new graph into an existing one.

        Reindexing = summarizing item descriptions + detecting communities and generating their summaries.
        """
        return await (await self.reindex_descriptions()).reindex_community()

    async def _reindex_cluster_ids(
        self,
        entities: List[Entity],
        communities: List[Community],
        summaries: Optional[List[CommunitySummary]] = None,
    ) -> Tuple[List[Entity], List[Community], List[CommunitySummary]]:
        """
        Remap cluster IDs to be globally unique per level across indexing runs.

        Levels are preserved to keep level-based filtering intact.

        :param entities: Entities whose cluster memberships should be remapped.
        :param communities: Newly generated communities with local cluster IDs.
        :param summaries: Optional summaries linked to community IDs.
        :return: Tuple with remapped entities, remapped communities, and remapped summaries.
        """
        if not communities:
            return entities, communities, summaries or []

        existing_keys = await self.index.community_kv_storage.all_keys()
        existing_data = await self.index.community_kv_storage.get_by_ids(existing_keys) if existing_keys else []

        max_cluster_id_by_level: Dict[int, int] = defaultdict(lambda: -1)
        for row in existing_data:
            if not row:
                continue
            try:
                level = int(row.get("level"))
                cluster_id = int(row.get("cluster_id"))
            except (TypeError, ValueError):
                continue
            if cluster_id > max_cluster_id_by_level[level]:
                max_cluster_id_by_level[level] = cluster_id

        local_ids_by_level: Dict[int, List[int]] = defaultdict(list)
        for community in communities:
            level = int(community.level)
            cluster_id = int(community.cluster_id)
            local_ids_by_level[level].append(cluster_id)

        local_to_global: Dict[tuple[int, int], int] = {}
        for level, local_ids in local_ids_by_level.items():
            for local_cluster_id in sorted(set(local_ids)):
                max_cluster_id_by_level[level] += 1
                local_to_global[(level, local_cluster_id)] = max_cluster_id_by_level[level]

        old_to_new_community_id: Dict[str, str] = {}
        remapped_communities: List[Community] = []
        for community in communities:
            level = int(community.level)
            local_cluster_id = int(community.cluster_id)
            global_cluster_id = local_to_global[(level, local_cluster_id)]

            remapped_community = Community(
                level=level,
                cluster_id=global_cluster_id,
                entities=community.entities,
                relations=community.relations,
            )
            if community.id:
                old_to_new_community_id[str(community.id)] = str(remapped_community.id)
            remapped_communities.append(remapped_community)

        valid_cluster_pairs = {
            (int(community.level), int(community.cluster_id))
            for community in remapped_communities
        }

        for entity in entities:
            remapped_memberships: List[ClusterInfo] = []
            seen_memberships: Set[tuple[int, int]] = set()
            for membership in entity.clusters:
                assert isinstance(membership, dict), f"What is membership? {membership}"
                try:
                    level = membership["level"]
                    local_cluster_id = membership["cluster_id"]
                except (TypeError, ValueError):
                    continue

                global_cluster_id = local_to_global.get((level, local_cluster_id), local_cluster_id)
                if (level, global_cluster_id) not in valid_cluster_pairs:
                    continue

                membership_key = (level, global_cluster_id)
                if membership_key in seen_memberships:
                    continue

                seen_memberships.add(membership_key)
                remapped_memberships.append({
                    "level": int(level),
                    "cluster_id": int(global_cluster_id),
                })
            entity.clusters = remapped_memberships

        remapped_summaries: List[CommunitySummary] = []
        if summaries:
            for summary in summaries:
                new_summary_id = old_to_new_community_id.get(str(summary.id), summary.id)
                remapped_summaries.append(
                    CommunitySummary(
                        summary=summary.summary,
                        id=new_summary_id,
                    )
                )

        return entities, remapped_communities, remapped_summaries

__all__ = ["KnowledgeGraph"]
