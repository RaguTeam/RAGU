from typing import Dict, List, NamedTuple

import numpy as np
from ragu.chunker.types import Chunk
from ragu.graph.types import Entity, Relation
from ragu.models.embedder import Embedder
from ragu.models.scorer import Scorer
from ragu.models.sparse_embedder import SparseEmbedder
from ragu.storage.base_storage import EdgeSpec
from ragu.storage.types import Point, EmbeddingHit, SparseEmbedding, DenseEmbedding

from ragu.graph.knowledge_graph import KnowledgeGraph


class EntityHits(NamedTuple):
    """
    One query's matched entities aligned with their vector hits.

    :param entities: Matching entities ordered by relevance.
    :param hits: Vector hits aligned position-for-position with ``entities``.
    """
    entities: List[Entity]
    hits: List[EmbeddingHit]


class RelationHits(NamedTuple):
    """
    One query's matched relations aligned with their vector hits.

    :param relations: Matching relations ordered by relevance.
    :param hits: Vector hits aligned position-for-position with ``relations``.
    """
    relations: List[Relation]
    hits: List[EmbeddingHit]


class ChunkHits(NamedTuple):
    """
    One query's matched chunks aligned with their vector hits.

    :param chunks: Matching chunks ordered by relevance.
    :param hits: Vector hits aligned position-for-position with ``chunks``.
    """
    chunks: List[Chunk]
    hits: List[EmbeddingHit]


class GraphRetriever:
    """
    Query-time retrieval helper for graph vector search.
    """
    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        embedder: Embedder,
        sparse_embedder: SparseEmbedder | None = None,
        reranker: Scorer | None = None,
    ) -> None:
        """
        Initialize a retriever bound to an existing knowledge graph.

        :param knowledge_graph: Graph container exposing storage backends.
        :param embedder: Dense embedder used for query encoding.
        :param sparse_embedder: Optional sparse embedder for hybrid retrieval.
        :param reranker: Optional reranker reserved for post-retrieval scoring.
        """
        self.knowledge_graph = knowledge_graph
        self.embedder = embedder
        self.sparse_embedder = sparse_embedder
        self.reranker = reranker

    async def query_entities(
        self,
        queries: List[str],
        top_k: int = 20,
    ) -> List[EntityHits]:
        """
        Find entities matching a batch of free-text queries.

        Embeddings, the vector search, and entity resolution are each performed
        once for the whole batch: query vectors are encoded together, the vector
        DB is queried in a single call, and matched ids are deduplicated across
        all queries before a single node lookup.

        :param queries: Search query texts.
        :param top_k: Maximum number of results per query.
        :return: Per-query tuples of matching entities ordered by relevance and
            their aligned vector hits, index-aligned with ``queries``.
        """
        if not queries:
            return []

        points = await self.build_query_vectors(queries)
        per_query_hits = await self.knowledge_graph.index.nodes_vector_db.query(
            points,
            top_k=top_k,
        )

        unique_ids = list({hit.id for hits in per_query_hits for hit in hits})
        entities = await self.knowledge_graph.index.get_nodes(unique_ids)
        entity_by_id: Dict[str, Entity] = {
            entity_id: entity
            for entity_id, entity in zip(unique_ids, entities)
            if entity is not None
        }

        results: List[EntityHits] = []
        for hits in per_query_hits:
            filtered_entities: List[Entity] = []
            filtered_hits: List[EmbeddingHit] = []
            for hit in hits:
                entity = entity_by_id.get(hit.id)
                if entity is None:
                    continue
                filtered_entities.append(entity)
                filtered_hits.append(hit)
            results.append(EntityHits(filtered_entities, filtered_hits))
        return results

    async def query_relations(
        self,
        queries: List[str],
        top_k: int = 20,
    ) -> List[RelationHits]:
        """
        Find relations matching a batch of free-text queries.

        Hits missing endpoint metadata are skipped. Edge specifications are
        deduplicated across all queries so relations are resolved in a single
        lookup.

        :param queries: Search query texts.
        :param top_k: Maximum number of results per query.
        :return: Per-query tuples of matching relations ordered by relevance and
            their aligned vector hits, index-aligned with ``queries``.
        """
        if not queries:
            return []

        points = await self.build_query_vectors(queries)
        per_query_hits = await self.knowledge_graph.index.edges_vector_db.query(
            points,
            top_k=top_k,
        )

        unique_specs: Dict[str, EdgeSpec] = {}
        for hits in per_query_hits:
            for hit in hits:
                subject_id = hit.metadata.get("subject_id")
                object_id = hit.metadata.get("object_id")
                if not subject_id or not object_id:
                    continue
                if hit.id not in unique_specs:
                    unique_specs[hit.id] = (str(subject_id), str(object_id), hit.id)

        relation_by_id: Dict[str, Relation] = {}
        if unique_specs:
            spec_ids = list(unique_specs.keys())
            relations = await self.knowledge_graph.index.get_edges(
                [unique_specs[spec_id] for spec_id in spec_ids]
            )
            relation_by_id = {
                spec_id: relation
                for spec_id, relation in zip(spec_ids, relations)
                if relation is not None
            }

        results: List[RelationHits] = []
        for hits in per_query_hits:
            filtered_relations: List[Relation] = []
            aligned_hits: List[EmbeddingHit] = []
            for hit in hits:
                relation = relation_by_id.get(hit.id)
                if relation is None:
                    continue
                filtered_relations.append(relation)
                aligned_hits.append(hit)
            results.append(RelationHits(filtered_relations, aligned_hits))
        return results

    async def query_chunks(
        self,
        queries: List[str],
        top_k: int = 20,
    ) -> List[ChunkHits]:
        """
        Search chunk vectors for a batch of queries and resolve matched chunks.

        Chunk ids are deduplicated across all queries before a single KV lookup.

        :param queries: Search query texts.
        :param top_k: Maximum number of hits per query.
        :return: Per-query tuples of ranked chunks with aligned vector hits,
            index-aligned with ``queries``.
        """
        if not queries:
            return []

        points = await self.build_query_vectors(queries)
        per_query_hits = await self.knowledge_graph.index.chunks_vector_db.query(
            points,
            top_k=top_k,
        )

        unique_ids = list({hit.id for hits in per_query_hits for hit in hits})
        chunk_data_list = await self.knowledge_graph.index.chunks_kv_storage.get_by_ids(unique_ids)
        data_by_id: Dict[str, dict] = {
            chunk_id: chunk_data
            for chunk_id, chunk_data in zip(unique_ids, chunk_data_list)
            if chunk_data is not None
        }

        results: List[ChunkHits] = []
        for hits in per_query_hits:
            chunks: List[Chunk] = []
            filtered_hits: List[EmbeddingHit] = []
            for hit in hits:
                chunk_data = data_by_id.get(hit.id)
                if chunk_data is None:
                    continue
                chunk = Chunk(
                    content=chunk_data.get("content", ""),
                    chunk_order_idx=chunk_data.get("chunk_order_idx", 0),
                    doc_id=chunk_data.get("doc_id", ""),
                    num_tokens=chunk_data.get("num_tokens"),
                )
                setattr(chunk, "id", hit.id)
                chunks.append(chunk)
                filtered_hits.append(hit)
            results.append(ChunkHits(chunks, filtered_hits))
        return results

    async def find_similar_entities(
        self,
        entities: List[Entity],
        top_k: int = 10,
    ) -> List[EntityHits]:
        """
        Find entities semantically similar to each given entity.

        :param entities: Reference entities to search against.
        :param top_k: Maximum number of results per entity.
        :return: Per-entity tuples of similar entities ordered by relevance and
            their aligned vector hits, index-aligned with ``entities``.
        """
        queries = [f"{entity.entity_name} - {entity.description}" for entity in entities]
        return await self.query_entities(queries, top_k=top_k)

    async def find_similar_relations(
        self,
        relations: List[Relation],
        top_k: int = 10,
    ) -> List[RelationHits]:
        """
        Find relations semantically similar to each given relation.

        :param relations: Reference relations to search against.
        :param top_k: Maximum number of results per relation.
        :return: Per-relation tuples of similar relations ordered by relevance
            and their aligned vector hits, index-aligned with ``relations``.
        """
        queries = [relation.description for relation in relations]
        return await self.query_relations(queries, top_k=top_k)

    async def build_query_vectors(self, queries: List[str]) -> List[Point]:
        """
        Encode a batch of queries into dense and optional sparse vectors.

        :param queries: Search query texts.
        :return: Points carrying query-time vector payloads, index-aligned with
            ``queries``.
        :raises ValueError: If the sparse embedder does not return exactly one
            vector per query.
        """
        dense_queries: List[DenseEmbedding] = await self.embedder.batch_embed_text(queries)  # type: ignore
        sparse_queries: List[SparseEmbedding | None] = [None] * len(queries)
        if self.sparse_embedder is not None:
            sparse_vectors = self.sparse_embedder.embed_query(queries)
            if len(sparse_vectors) != len(queries):
                raise ValueError("Sparse query embedder must return exactly one vector per query")
            sparse_queries = list(sparse_vectors)
        return [
            Point(dense_embedding=np.array(dense_query), sparse_embedding=sparse_query)
            for dense_query, sparse_query in zip(dense_queries, sparse_queries)
        ]
