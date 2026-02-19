from typing import Optional, List

from openai import embeddings

from ragu.chunker.types import Chunk
from ragu.common.global_parameters import Settings
from ragu.embedder.base_embedder import BaseEmbedder
from ragu.graph.knowledge_graph import KnowledgeGraph
from ragu.llm.base_llm import BaseLLM
from ragu.rerank.base_reranker import BaseReranker
from ragu.search_engine.base_engine import BaseEngine
from ragu.search_engine.types import NaiveSearchResult
from ragu.storage import Embedding
from ragu.utils.token_truncation import TokenTruncation

from ragu.common.prompts.prompt_storage import RAGUInstruction
from ragu.common.prompts.messages import ChatMessages, render


class NaiveSearchEngine(BaseEngine):
    """
    Performs naive vector RAG search over document chunks.

    This engine retrieves chunks most similar to a query using vector embeddings,
    optionally reranks them, and passes the context to an LLM for response generation.
    """

    def __init__(
        self,
        client: BaseLLM,
        knowledge_graph: KnowledgeGraph,
        embedder: BaseEmbedder,
        reranker: Optional[BaseReranker] = None,
        max_context_length: int = 30_000,
        tokenizer_backend: str = "tiktoken",
        tokenizer_model: str = "gpt-4",
        language: str | None = None,
        *args,
        **kwargs
    ):
        """
        Initialize a `NaiveSearchEngine`.

        :param client: LLM client used to generate the final answer.
        :param knowledge_graph: Knowledge graph containing chunk vector DB and chunk KV storage.
        :param embedder: Embedding model (kept for interface parity; retrieval uses graph index DBs).
        :param reranker: Optional reranker used to improve ranking of retrieved chunks.
        :param max_context_length: Max tokens allowed for context after truncation.
        :param tokenizer_backend: Tokenizer backend used for token truncation.
        :param tokenizer_model: Model name used by the tokenizer backend.
        :param language: Default output language
        """
        _PROMPTS_NAMES = ["naive_search"]
        super().__init__(client=client, prompts=_PROMPTS_NAMES, *args, **kwargs)

        self.truncation = TokenTruncation(
            tokenizer_model,
            tokenizer_backend,
            max_context_length,
        )

        self.graph = knowledge_graph
        self.embedder = embedder
        self.reranker = reranker
        self.client = client
        self.language = language if language else Settings.language

    async def a_search(
        self,
        query: str,
        top_k: int = 20,
        rerank_top_k: Optional[int] = None,
        *args,
        **kwargs
    ) -> NaiveSearchResult:
        """
        Perform a naive vector search over chunks.

        :param query: Input query string.
        :param top_k: Number of top chunks to retrieve initially.
        :param rerank_top_k: Number of chunks to keep after reranking.
                             If None, keeps all reranked chunks. Used only when reranker is set.
        :return: NaiveSearchResult with retrieved chunks, scores, and document ids.
        """
        vectorized_query = await self.embedder.embed_single(query)
        results = await self.graph.index.chunk_vector_db.query(
            Embedding(vector=vectorized_query[0]),
        )

        if not results:
            return NaiveSearchResult(chunks=[], scores=[], documents_id=[])

        chunk_ids = [r.id for r in results]
        distances = [r.distance for r in results]

        chunk_data_list = await self.graph.index.chunks_kv_storage.get_by_ids(chunk_ids)

        chunks: List[Chunk] = []
        valid_distances: List[float] = []
        for chunk_id, chunk_data, distance in zip(chunk_ids, chunk_data_list, distances):
            if chunk_data is not None:
                chunk = Chunk(
                    content=chunk_data.get("content", ""),
                    chunk_order_idx=chunk_data.get("chunk_order_idx", 0),
                    doc_id=chunk_data.get("doc_id", ""),
                    num_tokens=chunk_data.get("num_tokens"),
                )
                # Override the auto-generated id with the stored one
                setattr(chunk, "id", chunk_id)
                chunks.append(chunk)
                valid_distances.append(distance)

        scores = valid_distances
        if self.reranker is not None and chunks:
            chunk_contents = [c.content for c in chunks]
            rerank_results = await self.reranker.rerank(query, chunk_contents)
            reranked_chunks = []
            reranked_scores = []
            for idx, score in rerank_results:
                reranked_chunks.append(chunks[idx])
                reranked_scores.append(score)

            chunks = reranked_chunks
            scores = reranked_scores

            if rerank_top_k is not None and rerank_top_k < len(chunks):
                chunks = chunks[:rerank_top_k]
                scores = scores[:rerank_top_k]

        documents_id = list({c.doc_id for c in chunks if c.doc_id})

        return NaiveSearchResult(
            chunks=chunks,
            scores=scores,
            documents_id=documents_id,
        )

    async def a_query(self, query: str, top_k: int = 20, rerank_top_k: Optional[int] = None) -> str:
        """
        Execute a retrieval-augmented query using naive vector search.

        :param query: User query in natural language.
        :param top_k: Number of chunks to search initially (default: 20).
        :param rerank_top_k: Number of chunks to use after reranking (default: None = use all).
        :return: Generated response text from the language model.
        """
        context: NaiveSearchResult = await self.a_search(query, top_k, rerank_top_k)
        truncated_context: str = self.truncation(str(context))

        instruction: RAGUInstruction = self.get_prompt("naive_search")

        rendered_list: List[ChatMessages] = render(
            instruction.messages,
            query=query,
            context=truncated_context,
            language=self.language,
        )
        rendered: ChatMessages = rendered_list[0]

        result: List = await self.client.generate(
            conversations=[rendered],
            response_model=instruction.pydantic_model,
        )

        return result[0].response if hasattr(result[0], "response") else result[0]
