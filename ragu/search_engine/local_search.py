# Partially based on https://github.com/gusye1234/nano-graphrag/blob/main/nano_graphrag/
from dataclasses import dataclass, field
from typing_extensions import override
from textwrap import dedent
from typing import Any, List, Literal

from jinja2 import Template

from ragu.chunker.types import Chunk
from ragu.common.global_parameters import Settings
from ragu.common.logger import logger
from ragu.common.prompts.messages import ChatMessages, render
from ragu.common.prompts.prompt_storage import RAGUInstruction
from ragu.graph.graph_retrieve_backend import GraphRetriever
from ragu.graph.knowledge_graph import KnowledgeGraph
from ragu.graph.types import Entity, Relation
from ragu.models.embedder import Embedder
from ragu.models.llm import LLM
from ragu.models.scorer import Scorer
from ragu.models.sparse_embedder import SparseEmbedder
from ragu.search_engine.base_engine import (
    BaseEngine,
    SearchEngineRetrieve,
    SearchEngineResponse
)
from ragu.search_engine.search_functional import (
    _find_most_related_edges_from_entities,
    _find_most_related_text_unit_from_entities,
    _find_documents_id,
    _find_most_related_community_from_entities,
    _rerank_items,
)


@dataclass(slots=True)
class LocalSearchResult:
    """
    Retrieved graph-local context for a query.

    Entities are the seed retrieval results. Relations, summaries, and chunks
    are derived from those entities and optionally reranked. ``documents_id``
    contains unique source document IDs from the final entity set.
    """
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    summaries: list[Any] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    documents_id: list[str] = field(default_factory=list)


@dataclass(slots=True)
class LocalSearchRetrieve(SearchEngineRetrieve[LocalSearchResult]):
    """
    Retrieval container returned by :class:`LocalSearchEngine`.

    Metrics use ``metrics["entities"]`` with one entry per final entity
    containing ``id``, ``name``, zero-based ``rank``, and vector
    ``relevance_score``.
    """
    result: LocalSearchResult

    _TO_TEXT_TEMPLATE = Template(dedent("""
        **Entities**
        Entity, entity type, entity description
        {%- for e in result.entities %}
        {{ e.entity_name }}, {{ e.entity_type }}, {{ e.description }}
        {%- endfor %}

        **Relations**
        Subject, relation type, object, relation description, rank
        {%- for r in result.relations %}
        {{ r.subject_name }}, {{ r.relation_type }}, {{ r.object_name }} - {{ r.description }}, {{ r.rank }}
        {%- endfor %}

        {%- if result.summaries %}
        **Summary**
        {%- for s in result.summaries %}
        {{ s.summary }}
        {%- endfor %}
        {% endif %}

        {%- if result.chunks %}
        **Chunks**
        {%- for c in result.chunks %}
        {{ c.content }}
        {%- endfor %}
        {% endif %}
    """))

    def to_text(self) -> str:
        """
        Render entities, relations, optional summaries, and optional chunks.
        """
        return self._TO_TEXT_TEMPLATE.render(result=self.result)


class LocalSearchEngine(BaseEngine):
    """
    Performs local retrieval-augmented search (RAG) over a knowledge graph.

    The engine:
      1. Retrieves relevant entities for the query.
      2. Retrieves related items (relations, summary and chunks).
      3. Generates a final response

    Reference
    ---------
    Based on: https://github.com/gusye1234/nano-graphrag/blob/main/nano_graphrag/_op.py#L919
    """

    def __init__(
        self,
        llm: LLM,
        knowledge_graph: KnowledgeGraph,
        embedder: Embedder,
        sparse_embedder: SparseEmbedder | None = None,
        reranker: Scorer | None = None,
        language: str | None = None,
        max_context_length: int | None = None,
        tokenizer_backend: Literal["tiktoken", "local"] | None = None,
        tokenizer_model: str | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Initialize a `LocalSearchEngine`.

        :param llm: LLM used to generate the final answer.
        :param knowledge_graph: Knowledge graph used for entity and relation retrieval.
        :param embedder: Dense embedder used for retrieval queries.
        :param sparse_embedder: Optional sparse embedder used for hybrid retrieval queries.
        :param reranker: Optional reranker used to reorder retrieved context sections.
        :param language: Default output language (fed into prompt template).
        :param max_context_length: Maximum tokens for the assembled context fed to
            the LLM. When ``None``, falls back to ``Settings.llm_context_token_limit``.
        :param tokenizer_backend: Tokenizer backend for context truncation. When
            ``None``, falls back to ``Settings.tokenizer_llm_backend``.
        :param tokenizer_model: Tokenizer model identifier for context truncation.
            When ``None``, falls back to ``Settings.tokenizer_llm_name``.
        """
        _PROMPTS_NAMES = ["local_search"]
        super().__init__(
            llm=llm,
            prompts=_PROMPTS_NAMES,
            max_context_length=max_context_length,
            tokenizer_backend=tokenizer_backend,
            tokenizer_model=tokenizer_model,
            *args,
            **kwargs,
        )

        self.knowledge_graph = knowledge_graph
        self.retriever = GraphRetriever(
            knowledge_graph=knowledge_graph,
            embedder=embedder,
            sparse_embedder=sparse_embedder,
            reranker=reranker,
        )
        self.reranker = reranker
        self.language = language if language else Settings.language

    @override
    async def a_search(self, query: str, top_k: int = 20, *args, **kwargs) -> LocalSearchRetrieve:
        """
        Retrieve local graph context for the given query.

        :param query: Input query string.
        :param top_k: Number of top entities to retrieve from the entity vector DB.
        :return: ``LocalSearchRetrieve`` containing graph-local context and
                 entity relevance metrics.
        """
        entities, entity_hits = await self.retriever.query_entities(query, top_k=top_k)
        entity_scores_by_id = {
            entity.id: hit.distance
            for entity, hit in zip(entities, entity_hits)
            if entity and entity.id
        }

        relations = await _find_most_related_edges_from_entities(entities, self.knowledge_graph)
        relations = [relation for relation in relations if relation is not None]

        relevant_chunks = await _find_most_related_text_unit_from_entities(entities, self.knowledge_graph)
        relevant_chunks = [chunk for chunk in relevant_chunks if chunk is not None]

        summaries = await _find_most_related_community_from_entities(entities, self.knowledge_graph)
        summaries = [summary for summary in summaries if summary is not None]

        entities = await _rerank_items(
            query,
            entities,
            lambda entity: f"{entity.entity_name}\n{entity.entity_type}\n{entity.description}",
            self.reranker,
        )
        relations = await _rerank_items(
            query,
            relations,
            lambda relation: (
                f"{relation.subject_name}\n{relation.relation_type}\n"
                f"{relation.object_name}\n{relation.description}"
            ),
            self.reranker,
        )
        summaries = await _rerank_items(
            query,
            summaries,
            lambda community_summary: community_summary.summary,
            self.reranker,
        )
        relevant_chunks = await _rerank_items(
            query,
            relevant_chunks,
            lambda chunk: chunk.content,
            self.reranker,
        )

        documents_id = await _find_documents_id(entities)

        return LocalSearchRetrieve(
            query=query,
            result=LocalSearchResult(
                entities=entities,
                relations=relations,
                summaries=summaries,
                chunks=relevant_chunks,
                documents_id=documents_id,
            ),
            metrics={
                "entities": [
                    {
                        "id": entity.id,
                        "name": entity.entity_name,
                        "rank": idx,
                        "relevance_score": entity_scores_by_id.get(entity.id),
                    }
                    for idx, entity in enumerate(entities)
                ],
            },
        )

    @override
    async def a_query(
            self,
            query: str,
            top_k: int = 20,
            use_summary: bool = False,
            use_chunks: bool = False
    ) -> SearchEngineResponse:
        """
        Execute a local RAG query.

        :param query: User query in natural language.
        :param top_k: Number of entities to retrieve into context.
        :param use_summary: Whether community summaries are included in the generated context.
        :param use_chunks: Whether source chunks are included in the generated context.
        :return: ``SearchEngineResponse`` containing the generated answer and
                 the ``LocalSearchRetrieve`` used as context.
        """
        context: LocalSearchRetrieve = await self.a_search(query, top_k)

        if not use_summary:
            context.result.summaries = []
        if not use_chunks:
            context.result.chunks = []

        truncated_context: str = self.truncation(str(context))
        instruction: RAGUInstruction = self.get_prompt("local_search")

        rendered_conversations: List[ChatMessages] = render(
            instruction.messages,
            query=query,
            context=truncated_context,
            language=self.language,
        )
        rendered: ChatMessages = rendered_conversations[0]
        response = await self.llm.chat_completion(
            conversation=rendered.to_openai(),
            output_schema=instruction.pydantic_model or str, # type: ignore
        ) # type: ignore

        return SearchEngineResponse(
            query=query,
            response=response,
            retrieval=context,
            payload={}
        )
