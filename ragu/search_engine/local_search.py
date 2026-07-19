# Partially based on https://github.com/gusye1234/nano-graphrag/blob/main/nano_graphrag/
import asyncio

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from textwrap import dedent
from typing import Any, List, Literal
from typing_extensions import override

from jinja2 import Template

from ragu.chunker.types import Chunk
from ragu.common.global_parameters import Settings
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
    SearchEngineResponse,
    SearchEngineStreamEvent,
    EngineParams,
)
from ragu.search_engine.search_functional import (
    _find_most_related_edges_from_entities,
    _find_most_related_text_unit_from_entities,
    _find_documents_id,
    _find_most_related_community_from_entities,
    _rerank_items,
)


@dataclass
class LocalParams(EngineParams):
    """
    Parameters for :class:`LocalSearchEngine`.

    :param top_k: Maximum number of entities to retrieve. (Retrieval-time.)
    :param rerank_top_k: After reranking the retrieved entities, keep only this
        many most-relevant entities before deriving relations, summaries and
        chunks. ``None`` keeps all entities. (Retrieval-time.)
    :param use_summary: Whether community summaries are included in the generated
        context. (Generation-time; ignored by :meth:`batch_search`.)
    :param use_chunks: Whether source chunks are included in the generated
        context. (Generation-time; ignored by :meth:`batch_search`.)
    """
    top_k: int = 20
    rerank_top_k: int | None = None
    use_summary: bool = False
    use_chunks: bool = True


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
    async def batch_search(
        self,
        queries: List[str],
        params: LocalParams | None = None,
    ) -> List[LocalSearchRetrieve]:
        """
        Retrieve local graph context for a batch of queries.

        Entity retrieval is issued once for the whole batch through
        :meth:`GraphRetriever.query_entities`; the per-query assembly (entity
        reranking, related-item derivation and reranking) then runs concurrently.

        :param queries: Input query strings.
        :param params: Retrieval parameters (``top_k``, ``rerank_top_k``). When
            ``None``, defaults to :class:`LocalParams`.
        :return: ``LocalSearchRetrieve`` per query, aligned with ``queries``.
        """
        if not queries:
            return []

        params = params or LocalParams()
        entity_results = await self.retriever.query_entities(queries, top_k=params.top_k)

        return list(await asyncio.gather(*[
            self._assemble(query, entities, entity_hits, params.rerank_top_k)
            for query, (entities, entity_hits) in zip(queries, entity_results)
        ]))

    # TODO: make batch effective
    async def _assemble(self, query, entities, entity_hits, rerank_top_k=None) -> LocalSearchRetrieve:
        """
        Build a :class:`LocalSearchRetrieve` from retrieved entities for one query.

        Entities are reranked first and optionally truncated to the most
        relevant; relations, summaries and chunks are then derived from that
        reduced entity set and reranked in turn.

        :param query: Input query string.
        :param entities: Seed entities retrieved for ``query``.
        :param entity_hits: Vector hits aligned with ``entities``.
        :param rerank_top_k: Keep only this many top entities after reranking;
            ``None`` keeps all.
        :return: Retrieval container with graph-local context and entity metrics.
        """
        entity_scores_by_id = {
            entity.id: hit.distance
            for entity, hit in zip(entities, entity_hits)
            if entity and entity.id
        }

        # 1-2. Rerank the retrieved entities and keep only the most relevant.
        # Truncation applies only when a reranker actually reordered them.
        entities = await _rerank_items(
            query,
            entities,
            lambda entity: f"{entity.entity_name}\n{entity.entity_type}\n{entity.description}",
            self.reranker,
        )
        if self.reranker is not None and rerank_top_k is not None:
            entities = entities[:rerank_top_k]

        # 3. Derive related items from the reduced entity set (document ids are a
        # set union, independent of entity order).
        relations, relevant_chunks, summaries, documents_id = await asyncio.gather(
            _find_most_related_edges_from_entities(entities, self.knowledge_graph),
            _find_most_related_text_unit_from_entities(entities, self.knowledge_graph),
            _find_most_related_community_from_entities(entities, self.knowledge_graph),
            _find_documents_id(entities),
        )
        relations = [relation for relation in relations if relation is not None]
        relevant_chunks = [chunk for chunk in relevant_chunks if chunk is not None]
        summaries = [summary for summary in summaries if summary is not None]

        # 4. Rerank the derived items.
        relations, summaries, relevant_chunks = await asyncio.gather(
            _rerank_items(
                query,
                relations,
                lambda relation: (
                    f"{relation.subject_name}\n{relation.relation_type}\n"
                    f"{relation.object_name}\n{relation.description}"
                ),
                self.reranker,
            ),
            _rerank_items(
                query,
                summaries,
                lambda community_summary: community_summary.summary,
                self.reranker,
            ),
            _rerank_items(
                query,
                relevant_chunks,
                lambda chunk: chunk.content,
                self.reranker,
            ),
        )

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

    def _render_answer_messages(
        self,
        query: str,
        context: LocalSearchRetrieve,
        use_summary: bool,
        use_chunks: bool,
    ) -> ChatMessages:
        """
        Build the final-answer conversation from a retrieved local context.

        Prunes summaries/chunks according to the flags, truncates the rendered
        context to the engine token limit, and renders the ``local_search`` prompt.

        :param query: User query in natural language.
        :param context: Retrieval container produced by :meth:`search`.
        :param use_summary: Whether community summaries are kept in the context.
        :param use_chunks: Whether source chunks are kept in the context.
        :return: Rendered chat messages ready for ``chat_completion``.
        """
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
        return rendered_conversations[0]

    @override
    async def batch_query(
        self,
        queries: List[str],
        params: LocalParams | None = None,
    ) -> List[SearchEngineResponse]:
        """
        Execute local RAG for multiple queries, batching answer generation.

        Retrieval is shared across the batch via :meth:`batch_search`; answer
        generation for all queries is issued through a single
        :meth:`LLM.batch_chat_completion` call. The first failing query aborts
        the whole batch.

        :param queries: User queries in natural language.
        :param params: Query parameters (``top_k``, ``use_summary``,
            ``use_chunks``). When ``None``, defaults to :class:`LocalParams`.
        :return: ``SearchEngineResponse`` objects aligned with ``queries``.
        """
        if not queries:
            return []

        params = params or LocalParams()
        contexts = await self.batch_search(queries, params)

        conversations = [
            self._render_answer_messages(query, context, params.use_summary, params.use_chunks).to_openai()
            for query, context in zip(queries, contexts)
        ]

        instruction: RAGUInstruction = self.get_prompt("local_search")
        answers = await self.llm.batch_chat_completion(
            conversations,
            output_schema=instruction.pydantic_model or str,  # type: ignore[arg-type]
            desc="LocalSearch batch query",
        )

        return [
            SearchEngineResponse(
                query=query,
                response=answer,
                retrieval=context,
                payload={},
            )
            for query, answer, context in zip(queries, answers, contexts)
        ]

    @override
    async def stream_query(
        self,
        query: str,
        params: LocalParams | None = None,
    ) -> AsyncIterator[SearchEngineStreamEvent]:
        """
        Execute local graph RAG and stream the final plain-text answer.

        :param query: User query in natural language.
        :param params: Query parameters. When ``None``, defaults to
            :class:`LocalParams`.
        :returns: Async iterator of text deltas with the retrieval context.
        """
        params = params or LocalParams()
        context = await self.search(query, params)
        conversation = self._render_answer_messages(
            query,
            context,
            params.use_summary,
            params.use_chunks,
        ).to_openai()

        async for delta in self.llm.stream_chat_completion(conversation):
            yield SearchEngineStreamEvent(
                query=query,
                retrieval=context,
                delta=delta,
                payload={},
            )
