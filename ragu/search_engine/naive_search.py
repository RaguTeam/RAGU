from dataclasses import dataclass, field
from textwrap import dedent
from typing import Any, Optional, List, Literal

from jinja2 import Template
from ragu.chunker.types import Chunk
from ragu.common.global_parameters import Settings
from ragu.graph.graph_retrieve_backend import GraphRetriever
from ragu.graph.knowledge_graph import KnowledgeGraph
from ragu.models.embedder import Embedder
from ragu.models.llm import LLM
from ragu.models.scorer import Scorer
from ragu.models.sparse_embedder import SparseEmbedder
from ragu.search_engine.base_engine import (
    BaseEngine,
    SearchEngineRetrieve,
    SearchEngineResponse
)
from ragu.common.prompts.prompt_storage import RAGUInstruction
from ragu.common.prompts.messages import ChatMessages, render


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

    async def a_search(
        self,
        query: str,
        top_k: int = 20,
        rerank_top_k: Optional[int] = None,
        *args: Any,
        **kwargs: Any,
    ) -> NaiveSearchRetrieve:
        """
        Perform a naive vector search over chunks.

        :param query: Input query string.
        :param top_k: Number of top chunks to retrieve initially.
        :param rerank_top_k: Number of chunks to keep after reranking.
                             If None, keeps all reranked chunks. Used only when reranker is set.
        :return: ``NaiveSearchRetrieve`` with retrieved chunks, aligned scores,
                 document IDs, and chunk rank metrics.
        """
        chunks, scores = await self.retriever.query_chunks(query, top_k=top_k)

        if not scores:
            return NaiveSearchRetrieve(
                query=query,
                result=NaiveSearchResult(),
                metrics={}
            )

        scores = [r.distance for r in scores]
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

    async def a_query(self, query: str, top_k: int = 20, rerank_top_k: Optional[int] = None) -> SearchEngineResponse:
        """
        Execute a retrieval-augmented query using naive vector search.

        :param query: User query in natural language.
        :param top_k: Number of chunks to search initially (default: 20).
        :param rerank_top_k: Number of chunks to use after reranking (default: None = use all).
        :return: ``SearchEngineResponse`` containing the generated answer and
                 the ``NaiveSearchRetrieve`` used as context.
        """
        context: NaiveSearchRetrieve = await self.a_search(query, top_k, rerank_top_k)
        truncated_context: str = self.truncation(str(context))

        instruction: RAGUInstruction = self.get_prompt("naive_search")

        rendered_list: List[ChatMessages] = render(
            instruction.messages,
            query=query,
            context=truncated_context,
            language=self.language,
        )
        rendered: ChatMessages = rendered_list[0]

        answer = await self.llm.chat_completion(
            conversation=rendered.to_openai(),
            output_schema=instruction.pydantic_model
        ) # type: ignore

        return SearchEngineResponse(
            query=query,
            response=answer,
            retrieval=context,
            payload={}
        )
