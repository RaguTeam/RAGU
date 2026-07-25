import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from textwrap import dedent
from typing import Any, Optional, List, Literal

from jinja2 import Template
from ragu.chunker.types import Chunk
from ragu.common.global_parameters import Settings
from ragu.common.prompts.messages import ChatMessages, render
from ragu.common.prompts.prompt_storage import RAGUInstruction
from ragu.graph.graph_retrieve_backend import GraphRetriever
from ragu.graph.knowledge_graph import KnowledgeGraph
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
from typing_extensions import override


@dataclass
class NaiveSearchParams(EngineParams):
    """
    Retrieval/query parameters for :class:`NaiveSearchEngine`.

    :param top_k: Number of chunks to retrieve.
    :param rerank_top_k: Number of chunks to keep after reranking. ``None`` keeps
        all reranked chunks. Used only when a reranker is configured.
    """
    top_k: int = 20
    rerank_top_k: Optional[int] = None


@dataclass(slots=True)
class NaiveSearchResult:
    """
    Retrieved chunk payload for naive vector search.

    ``chunks`` and ``scores`` are aligned by index after optional reranking and
    truncation by ``rerank_top_k``. ``documents_id`` contains unique document IDs
    present in the final chunk list.
    """
    chunks: list[Chunk] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    documents_id: list[str] = field(default_factory=list)


@dataclass(slots=True)
class NaiveSearchRetrieve(SearchEngineRetrieve[NaiveSearchResult]):
    """
    Retrieval container returned by :class:`NaiveSearchEngine`.

    Metrics use ``metrics["chunks"]`` with one entry per final chunk containing
    ``id``, zero-based ``rank``, and retrieval or reranker ``score``.
    """
    result: NaiveSearchResult

    def to_text(self) -> str:
        """
        Render retrieved chunks and aligned scores for the answer prompt.
        """
        template = Template(dedent("""
            **Retrieved Chunks**
            {%- for chunk, score in zip(result.chunks, result.scores) %}
            [{{ loop.index }}] (score: {{ "%.3f"|format(score) }})
            {{ chunk.content }}
            {%- endfor %}
        """))
        return template.render(result=self.result, zip=zip)


class NaiveSearchEngine(BaseEngine):
    """
    Performs naive vector RAG search over document chunks.

    This engine retrieves chunks most similar to a query using vector embeddings,
    optionally reranks them, and passes the context to an LLM for response generation.
    """

    def __init__(
        self,
        llm: LLM,
        knowledge_graph: KnowledgeGraph,
        embedder: Embedder,
        sparse_embedder: SparseEmbedder | None = None,
        reranker: Optional[Scorer] = None,
        language: str | None = None,
        max_context_length: int | None = None,
        tokenizer_backend: Literal["tiktoken", "local"] | None = None,
        tokenizer_model: str | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Initialize a `NaiveSearchEngine`.

        :param llm: LLM used to generate the final answer.
        :param knowledge_graph: Knowledge graph containing chunk vector DB and chunk KV storage.
        :param embedder: Dense embedder used for retrieval queries.
        :param sparse_embedder: Optional sparse embedder used for hybrid retrieval queries.
        :param reranker: Optional reranker used to improve ranking of retrieved chunks.
        :param language: Default output language.
        :param max_context_length: Maximum tokens for the assembled context fed to
            the LLM. When ``None``, falls back to ``Settings.llm_context_token_limit``.
        :param tokenizer_backend: Tokenizer backend for context truncation. When
            ``None``, falls back to ``Settings.tokenizer_llm_backend``.
        :param tokenizer_model: Tokenizer model identifier for context truncation.
            When ``None``, falls back to ``Settings.tokenizer_llm_name``.
        """
        _PROMPTS_NAMES = ["naive_search"]
        super().__init__(
            llm=llm,
            prompts=_PROMPTS_NAMES,
            max_context_length=max_context_length,
            tokenizer_backend=tokenizer_backend,
            tokenizer_model=tokenizer_model,
            *args,
            **kwargs,
        )

        self.graph = knowledge_graph
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
        params: NaiveSearchParams | None = None,
    ) -> List[NaiveSearchRetrieve]:
        """
        Perform a naive vector search over chunks for a batch of queries.

        Chunk retrieval is issued once for the whole batch through
        :meth:`GraphRetriever.query_chunks`; per-query reranking then runs
        concurrently.

        :param queries: Input query strings.
        :param params: Retrieval parameters (``top_k``, and ``rerank_top_k`` when
            a :class:`NaiveSearchParams` is passed). When ``None``, defaults to
            :class:`NaiveSearchParams`.
        :return: ``NaiveSearchRetrieve`` per query, aligned with ``queries``.
        """
        if not queries:
            return []

        params = params or NaiveSearchParams()
        chunk_results = await self.retriever.query_chunks(queries, top_k=params.top_k)

        return list(await asyncio.gather(*[
            self._assemble(query, chunks, hits, params.rerank_top_k)
            for query, (chunks, hits) in zip(queries, chunk_results)
        ]))

    async def _assemble(self, query, chunks, hits, rerank_top_k) -> NaiveSearchRetrieve:
        """
        Build a :class:`NaiveSearchRetrieve` from retrieved chunks for one query.

        Applies optional reranking and ``rerank_top_k`` truncation.

        :param query: Input query string.
        :param chunks: Retrieved chunks for ``query``.
        :param hits: Vector hits aligned with ``chunks``.
        :param rerank_top_k: Number of chunks to keep after reranking, or ``None``.
        :return: Retrieval container with ranked chunks, scores, document ids and metrics.
        """
        if not hits:
            return NaiveSearchRetrieve(
                query=query,
                result=NaiveSearchResult(),
                metrics={}
            )

        scores = [r.distance for r in hits]
        if self.reranker is not None and chunks:
            chunk_contents = [c.content for c in chunks]
            rerank_results = await self.reranker.score(query, chunk_contents)
            reranked_chunks: list[Chunk] = []
            reranked_scores: list[float] = []
            for idx, score in rerank_results:
                reranked_chunks.append(chunks[idx])
                reranked_scores.append(score)

            chunks = reranked_chunks
            scores = reranked_scores

            if rerank_top_k is not None and rerank_top_k < len(chunks):
                chunks = chunks[:rerank_top_k]
                scores = scores[:rerank_top_k]

        documents_id = list({c.doc_id for c in chunks if c.doc_id})

        return NaiveSearchRetrieve(
            query=query,
            result=NaiveSearchResult(
                chunks=chunks,
                scores=scores,
                documents_id=documents_id,
            ),
            metrics={
                "chunks": [
                    {
                        "id": chunk.id,
                        "rank": idx,
                        "score": score,
                    }
                    for idx, (chunk, score) in enumerate(zip(chunks, scores))
                ],
            },
        )

    def _render_answer_messages(self, query: str, context: NaiveSearchRetrieve) -> ChatMessages:
        """
        Build the final-answer conversation from a retrieved chunk context.

        Truncates the rendered context to the engine token limit and renders the
        ``naive_search`` prompt.

        :param query: User query in natural language.
        :param context: Retrieval container produced by :meth:`search`.
        :return: Rendered chat messages ready for ``chat_completion``.
        """
        truncated_context: str = self.truncation(str(context))
        instruction: RAGUInstruction = self.get_prompt("naive_search")

        rendered_list: List[ChatMessages] = render(
            instruction.messages,
            query=query,
            context=truncated_context,
            language=self.language,
        )
        return rendered_list[0]

    @override
    async def batch_query(
        self,
        queries: List[str],
        params: NaiveSearchParams | None = None,
    ) -> List[SearchEngineResponse]:
        """
        Execute naive vector RAG for multiple queries, batching answer generation.

        Retrieval is shared across the batch via :meth:`batch_search`; answer
        generation for all queries is issued through a single
        :meth:`LLM.batch_chat_completion` call. The first failing query aborts
        the whole batch.

        :param queries: User queries in natural language.
        :param params: Query parameters (``top_k``, ``rerank_top_k``). When
            ``None``, defaults to :class:`NaiveSearchParams`.
        :return: ``SearchEngineResponse`` objects aligned with ``queries``.
        """
        if not queries:
            return []

        params = params or NaiveSearchParams()
        contexts = await self.batch_search(queries, params)

        conversations = [
            self._render_answer_messages(query, context).to_openai()
            for query, context in zip(queries, contexts)
        ]

        instruction: RAGUInstruction = self.get_prompt("naive_search")
        answers = await self.llm.batch_chat_completion(
            conversations,
            output_schema=instruction.pydantic_model,  # type: ignore[arg-type]
            desc="NaiveSearch batch query",
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
        params: NaiveSearchParams | None = None,
    ) -> AsyncIterator[SearchEngineStreamEvent]:
        """
        Execute naive vector RAG and stream the final plain-text answer.

        :param query: User query in natural language.
        :param params: Query parameters. When ``None``, defaults to
            :class:`NaiveSearchParams`.
        :returns: Async iterator of text deltas with the retrieval context.
        """
        context = await self.search(query, params or NaiveSearchParams())
        conversation = self._render_answer_messages(query, context).to_openai()

        async for delta in self.llm.stream_chat_completion(conversation):
            yield SearchEngineStreamEvent(
                query=query,
                retrieval=context,
                delta=delta,
                payload={},
            )
