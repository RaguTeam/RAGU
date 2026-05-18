from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import networkx as nx
from graspologic_native import hierarchical_leiden

from ragu.chunker.base_chunker import BaseChunker
from ragu.chunker.types import Chunk
from ragu.common.global_parameters import Settings
from ragu.common.logger import logger
from ragu.graph.artifacts_summarizer import EntitySummarizer, RelationSummarizer
from ragu.graph.community_summarizer import CommunitySummarizer
from ragu.graph.types import CommunitySummary, Community, Entity, Relation
from ragu.models.embedder import Embedder
from ragu.models.llm import LLM
from ragu.triplet.base_artifact_extractor import BaseArtifactExtractor


@dataclass
class BuilderArguments:
    """
    Configuration settings for the knowledge graph building pipeline.

    This dataclass controls various aspects of graph construction including
    summarization strategies, clustering behavior, and optimization modes.

    :param use_llm_summarization: Enable LLM-based summarization for merging and
        deduplicating similar entity and relation descriptions.
    :param use_clustering: Apply clustering to group similar entities before
        summarization.
    :param build_only_vector_context: Skip entity/relation extraction and build
        only vector context for naive RAG.
    :param make_community_summary: Generate high-level summaries for detected
        graph communities.
    :param remove_isolated_nodes: Remove entities that have no relations.
    :param vectorize_chunks: Generate and store embeddings for text chunks.
    :param cluster_only_if_more_than: Minimum number of entities required before
        clustering is applied.
    :param max_cluster_size: Maximum number of entities per cluster.
    :param random_seed: Random seed for reproducible clustering/community detection.
    """
    use_llm_summarization: bool = True
    use_clustering: bool = False
    build_only_vector_context: bool = False
    make_community_summary: bool = True
    remove_isolated_nodes: bool = True
    vectorize_chunks: bool = False
    cluster_only_if_more_than: int = 10000
    summarize_only_if_more_than: int = 7
    max_cluster_size: int = 128
    random_seed: int = 42


class GraphBuilderModule:
    """
    Abstract interface for modules that extend the graph-building pipeline.

    Each module receives entities and relations
    and can modify, enrich, or filter them before insertion into the graph.

    Typically used for:
      - normalization of entity names
      - filtering noisy relations
      - post-processing after extraction

    Subclasses should override `run` to apply module-specific logic.
    """

    async def run(
            self,
            entities: List[Entity],
            relations: List[Relation],
            **kwargs: Any,
    ) -> Tuple[List[Entity], List[Relation]]:
        """
        Process or update multiple nodes and edges during graph construction.

        :param entities: list of :class:`Entity` objects to insert or modify.
        :param relations: list of :class:`Relation` objects to insert or modify.
        :param kwargs: optional additional parameters specific to the module.
        :return: updated or enriched entities/relations.
        """
        ...


class InMemoryGraphBuilder:
    """
    High-level orchestrator for extracting and summarizing entities and relations
    directly in memory using an LLM client and supporting components.

    The pipeline consists of:
      1. **Chunking** input documents.
      2. **Entity & relation extraction** using a triplet-based artifact extractor.
      3. **Artifact summarization** for merging and deduplicating similar entities.
      4. (Optional) **Additional modules** for graph enrichment.
      5. **Community summarization** (aggregated graph-level summaries).

    When `build_parameters.build_only_vector_context=True`, steps 2-5 are skipped,
    and only chunking is performed. This is useful for naive vector RAG where only
    chunk embeddings are needed without knowledge graph construction.

    :param llm: LLM used for understanding and summarization tasks.
    :param chunker: Module responsible for splitting documents into chunks.
    :param artifact_extractor: Extractor for entities and relations from chunks.
    :param build_parameters: Graph-building settings controlling summarization,
        clustering, and optimization behavior.
    :param embedder: Embedding model used for vectorization and clustering.
    :param additional_pipeline: Optional post-processing modules executed after
        extraction/summarization.
    :param language: Working language for prompts and generation.
    """

    def __init__(
        self,
        embedder: Embedder,
        llm: LLM | None = None,
        chunker: BaseChunker | None = None,
        artifact_extractor: BaseArtifactExtractor | None = None,
        build_parameters: BuilderArguments = BuilderArguments(),
        additional_pipeline: List[GraphBuilderModule] | None = None,
        language: str | None = None
    ):
        self.llm = llm
        self.chunker = chunker
        self.artifact_extractor = artifact_extractor
        self.additional_pipeline = additional_pipeline
        self.embedder = embedder
        self.language = language if language else Settings.language
        self.build_parameters = build_parameters

        if self.build_parameters.build_only_vector_context:
            # No need to create those instances => we are able not to think about its parameters
            self.entity_summarizer, self.relation_summarizer, self.community_summarizer = None, None, None
        else:
            self.entity_summarizer = EntitySummarizer(
                llm,
                use_llm_summarization=self.build_parameters.use_llm_summarization,
                use_clustering=self.build_parameters.use_clustering,
                cluster_only_if_more_than=self.build_parameters.cluster_only_if_more_than,
                summarize_only_if_more_than=self.build_parameters.summarize_only_if_more_than,
                embedder=embedder,
                language=self.language,
            )
            self.relation_summarizer = RelationSummarizer(
                llm,
                use_llm_summarization=self.build_parameters.use_llm_summarization,
                summarize_only_if_more_than=self.build_parameters.summarize_only_if_more_than,
                language=self.language
            )
            self.community_summarizer = CommunitySummarizer(self.llm, language=self.language)

    async def extract_graph(
            self, chunks: List[Chunk]
    ) -> Tuple[List[Entity], List[Relation], List[CommunitySummary], List[Community], List[Chunk]]:
        """
        Run the full extraction pipeline and produce entities, relations,
        community summaries, and communities.

        Pipeline:
          1. Extract entities/relations via :class:`BaseArtifactExtractor`
             (skipped if ``build_only_vector_context=True``).
          2. Summarize or merge similar artifacts.
          3. Detect communities and generate summaries (optional).

        :param chunks: list of input text documents.
        :return:
            A tuple ``(entities, relations, summaries, communities)`` where
              - **entities** (:class:`list[Entity]`) — extracted and summarized entities (empty if build_only_vector_context=True).
              - **relations** (:class:`list[Relation]`) — extracted and summarized relations (empty if build_only_vector_context=True).
              - **summaries** (:class:`list[CommunitySummary]`) — generated summaries for detected communities.
              - **communities** (:class:`list[Community]`) — graph communities detected via Leiden clustering.
              - **chunks** (:class:`list[Chunk]`) - list of chunks extracted from input documents.
        """

        if self.chunker is None:
            logger.info('There is no chunker. Process raw documents.')

        if self.build_parameters.build_only_vector_context:
            return [], [], [], [], chunks

        # Step 1: extract entities and relations
        try:
            entities, relations = await self.artifact_extractor(chunks)
        except Exception as e:
            logger.warning("Extraction failed: {}: {}", type(e).__name__, e)
            entities, relations = [], []

        # Step 2: summarize similar artifacts' descriptions
        try:
            entities = await self.entity_summarizer.run(entities)
        except Exception as e:
            logger.warning("Entity summarization failed: {}: {}", type(e).__name__, e)

        try:
            relations = await self.relation_summarizer.run(relations)
        except Exception as e:
            logger.warning("Relation summarization failed: {}: {}", type(e).__name__, e)

        # Step 3: use additional modules
        if self.additional_pipeline:
            for additional_module in self.additional_pipeline:
                try:
                   entities, relations = await additional_module.run(entities, relations)
                except Exception as e:
                    logger.warning(
                        "Additional module {} failed: {}: {}",
                        type(additional_module).__name__, type(e).__name__, e,
                    )

        # Step 4. get community summary
        communities: List[Community] = []
        summaries: List[CommunitySummary] = []
        if self.build_parameters.make_community_summary:
            try:
                communities = await self.cluster_graph(entities, relations)
            except Exception as e:
                logger.warning("Community detection failed: {}: {}", type(e).__name__, e)
                communities = []

            if communities:
                try:
                    summaries = await self.community_summarizer.summarize(communities)
                except Exception as e:
                    logger.warning("Community summarization failed: {}: {}", type(e).__name__, e)
                    summaries = []

        return entities, relations, summaries, communities, chunks

    async def cluster_graph(
        self,
        entities: List[Entity],
        relations: List[Relation],
    ) -> List[Community]:
        """
        Detect graph communities with hierarchical Leiden clustering.

        Builds an undirected graph from entities/relations and do clusterization.

        :param entities: Entities.
        :param relations: Relations.
        :return: Detected communities.
        """
        if not entities or not relations:
            return []

        graph = nx.Graph()
        entity_by_id: Dict[str, Entity] = {}
        relation_by_id: Dict[str, Relation] = {}

        for entity in entities:
            if not entity.id:
                continue
            entity.clusters = []
            entity_by_id[entity.id] = entity
            graph.add_node(entity.id)

        for relation in relations:
            if not relation.id:
                continue
            if relation.subject_id not in entity_by_id or relation.object_id not in entity_by_id:
                continue
            relation_by_id[relation.id] = relation
            graph.add_edge(relation.subject_id, relation.object_id, relation_id=relation.id)

        if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
            return []

        edges: List[tuple[str, str, float]] = [
            (str(u), str(v), 1.0)
            for u, v in graph.edges()
        ]

        raw_community_mapping = hierarchical_leiden(
            edges,
            starting_communities=None,
            max_cluster_size=self.build_parameters.max_cluster_size,
            seed=self.build_parameters.random_seed,
            resolution=1.0,
            randomness=0.001,
            use_modularity=True,
            iterations=1
        )

        def _extract_mapping_item(item: Any) -> tuple[str, int, int]:
            if hasattr(item, "node") and hasattr(item, "cluster") and hasattr(item, "level"):
                return str(item.node), int(item.cluster), int(item.level)

            if isinstance(item, dict):
                return str(item["node"]), int(item["cluster"]), int(item["level"])

            if isinstance(item, (tuple, list)) and len(item) >= 3:
                return str(item[0]), int(item[1]), int(item[2])

            raise TypeError(f"Unsupported hierarchical_leiden output item: {item!r}")

        clusters = defaultdict(
            lambda: defaultdict(lambda: {"entity_ids": set(), "relation_ids": set()})
        )
        node_membership = defaultdict(set)

        for part in raw_community_mapping:
            node_id, cluster_id, level = _extract_mapping_item(part)

            node = entity_by_id.get(node_id)
            if node is None:
                continue

            node.clusters.append({"level": level, "cluster_id": cluster_id})
            clusters[level][cluster_id]["entity_ids"].add(node_id)
            node_membership[node_id].add((level, cluster_id))

        for relation in relation_by_id.values():
            common = node_membership[relation.subject_id].intersection(
                node_membership[relation.object_id]
            )
            for level, cluster_id in common:
                clusters[level][cluster_id]["relation_ids"].add(relation.id)

        communities: List[Community] = []
        for level in sorted(clusters.keys()):
            for cluster_id in sorted(clusters[level].keys()):
                payload = clusters[level][cluster_id]
                community_entities = [
                    entity_by_id[node_id]
                    for node_id in sorted(payload["entity_ids"])
                    if node_id in entity_by_id
                ]
                community_relations = [
                    relation_by_id[relation_id]
                    for relation_id in sorted(payload["relation_ids"])
                    if relation_id in relation_by_id
                ]
                communities.append(
                    Community(
                        entities=community_entities,
                        relations=community_relations,
                        level=level,
                        cluster_id=cluster_id,
                    )
                )

        return communities
