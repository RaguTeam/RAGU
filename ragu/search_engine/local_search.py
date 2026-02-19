# Partially based on https://github.com/gusye1234/nano-graphrag/blob/main/nano_graphrag/
from typing import List

from ragu.common.global_parameters import Settings
from ragu.embedder.base_embedder import BaseEmbedder
from ragu.graph.knowledge_graph import KnowledgeGraph
from ragu.llm.base_llm import BaseLLM
from ragu.search_engine.base_engine import BaseEngine
from ragu.search_engine.search_functional import (
    _find_most_related_edges_from_entities,
    _find_most_related_text_unit_from_entities,
    _find_documents_id,
    _find_most_related_community_from_entities,
)
from ragu.search_engine.types import LocalSearchResult
from ragu.storage import Embedding
from ragu.utils.token_truncation import TokenTruncation

from ragu.common.prompts.prompt_storage import RAGUInstruction
from ragu.common.prompts.messages import ChatMessages, render


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
        client: BaseLLM,
        knowledge_graph: KnowledgeGraph,
        embedder: BaseEmbedder,
        max_context_length: int = 30_000,
        tokenizer_backend: str = "tiktoken",
        tokenizer_model: str = "gpt-4",
        language: str | None = None,
        *args,
        **kwargs,
    ):
        """
        Initialize a `LocalSearchEngine`.

        :param client: LLM client used to generate the final answer.
        :param knowledge_graph: Knowledge graph used for entity and relation retrieval.
        :param embedder: Embedding model used for similarity search.
        :param max_context_length: Max tokens allowed for the final context (after truncation).
        :param tokenizer_backend: Tokenizer backend used for token counting/truncation.
        :param tokenizer_model: Model name used by the tokenizer backend.
        :param language: Default output language (fed into prompt template).
        """
        _PROMPTS_NAMES = ["local_search"]
        super().__init__(client=client, prompts=_PROMPTS_NAMES, *args, **kwargs)

        self.truncation = TokenTruncation(
            tokenizer_model,
            tokenizer_backend,
            max_context_length,
        )

        self.knowledge_graph = knowledge_graph
        self.embedder = embedder
        self.language = language if language else Settings.language

    async def a_search(self, query: str, top_k: int = 20, *args, **kwargs) -> LocalSearchResult:
        """
        Retrieve local graph context for the given query.

        :param query: Input query string.
        :param top_k: Number of top entities to retrieve from the entity vector DB.
        :return: LocalSearchResult containing entities, relations, summaries, chunks, and document ids.
        """
        embedding = await self.embedder.embed_single(query)
        embedding_hits = await self.knowledge_graph.index.entity_vector_db.query(
            Embedding(embedding),
            top_k=top_k,
        )
        entities = await self.knowledge_graph.index.get_entities([hit.id for hit in embedding_hits])

        relations = await _find_most_related_edges_from_entities(entities, self.knowledge_graph)
        relations = [relation for relation in relations if relation is not None]

        relevant_chunks = await _find_most_related_text_unit_from_entities(entities, self.knowledge_graph)
        relevant_chunks = [chunk for chunk in relevant_chunks if chunk is not None]

        summaries = await _find_most_related_community_from_entities(entities, self.knowledge_graph)

        documents_id = await _find_documents_id(entities)

        return LocalSearchResult(
            entities=entities,
            relations=relations,
            summaries=summaries,
            chunks=relevant_chunks,
            documents_id=documents_id,
        )

    async def a_query(self, query: str, top_k: int = 20) -> str:
        """
        Execute a local RAG query.

        :param query: User query in natural language.
        :param top_k: Number of entities to retrieve into context.
        :return: Final model response (string or extracted field if returned model-like).
        """
        context: LocalSearchResult = await self.a_search(query, top_k)
        truncated_context: str = self.truncation(str(context))
        instruction: RAGUInstruction = self.get_prompt("local_search")

        rendered_conversations: List[ChatMessages] = render(
            instruction.messages,
            query=query,
            context=truncated_context,
            language=self.language,
        )
        rendered: ChatMessages = rendered_conversations[0]
        result = await self.client.generate(
            conversations=[rendered],
            response_model=instruction.pydantic_model,
        )

        return result[0].response if hasattr(result[0], "response") else result[0]
