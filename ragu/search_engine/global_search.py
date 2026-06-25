import asyncio
from dataclasses import field, dataclass
from textwrap import dedent
from typing import Any, List, Literal

from jinja2 import Template
from ragu.common.global_parameters import Settings
from ragu.common.logger import logger
from ragu.common.prompts.default_models import GlobalSearchContextModel
from ragu.common.prompts.messages import ChatMessages, render
from ragu.common.prompts.prompt_storage import RAGUInstruction
from ragu.graph.knowledge_graph import KnowledgeGraph
from ragu.models.llm import LLM
from ragu.search_engine.base_engine import (
    BaseEngine,
    SearchEngineRetrieve,
    SearchEngineResponse
)

# TODO: add the ability to use custom schemas instead of GlobalSearchContextModel
@dataclass(slots=True)
class GlobalSearchResult:
    """
    Ranked community-level insights selected for a global query.

    Each insight is expected to contain at least a ``response`` and ``rating``
    field as produced by the global-search context prompt.
    """
    insights: list[GlobalSearchContextModel] = field(default_factory=list)


@dataclass(slots=True)
class GlobalSearchRetrieve(SearchEngineRetrieve[GlobalSearchResult]):
    """
    Retrieval container returned by :class:`GlobalSearchEngine`.

    Metrics include per-insight ratings after filtering and sorting.
    """
    result: GlobalSearchResult

    _TO_TEXT_TEMPLATE = Template(dedent("""
        {%- for insight in result.insights %}
        {{ loop.index }}. Insight: {{ insight.response }}, rating: {{ insight.rating }}
        {%- endfor %}
    """))

    def to_text(self) -> str:
        """
        Render selected community insights for final answer synthesis.
        """
        return self._TO_TEXT_TEMPLATE.render(result=self.result)


class GlobalSearchEngine(BaseEngine):
    """
    Executes global retrieval-augmented search (RAG) across the entire knowledge graph.

    Unlike :class:`LocalSearchEngine`, this engine operates at the level of
    *community summaries*, aggregating and ranking high-level semantic clusters
    before generating a global synthesis via the language model.
    """

    def __init__(
        self,
        llm: LLM,
        knowledge_graph: KnowledgeGraph,
        language: str | None = None,
        max_context_length: int | None = None,
        tokenizer_backend: Literal["tiktoken", "local"] | None = None,
        tokenizer_model: str | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Initialize a new `GlobalSearchEngine`.

        :param llm: Language model client for meta-evaluation and final answer generation.
        :param knowledge_graph: Knowledge graph providing access to community-level summaries.
        :param language: Default output language (fed into prompt templates).
        :param max_context_length: Maximum tokens for the assembled context fed to
            the LLM. When ``None``, falls back to ``Settings.llm_context_token_limit``.
        :param tokenizer_backend: Tokenizer backend for context truncation. When
            ``None``, falls back to ``Settings.tokenizer_llm_backend``.
        :param tokenizer_model: Tokenizer model identifier for context truncation.
            When ``None``, falls back to ``Settings.tokenizer_llm_name``.
        """
        _PROMPTS = ["global_search_context", "global_search"]
        super().__init__(
            llm=llm,
            prompts=_PROMPTS,
            max_context_length=max_context_length,
            tokenizer_backend=tokenizer_backend,
            tokenizer_model=tokenizer_model,
            *args,
            **kwargs,
        )

        self.knowledge_graph = knowledge_graph
        self.language = language if language else Settings.language

    async def a_search(self, query: str, *args, **kwargs) -> GlobalSearchRetrieve:
        """
        Perform a global semantic search across all communities in the knowledge graph.

        This method retrieves all available community summaries, sends them to the LLM
        for meta-evaluation, filters out low-rated responses, and returns a ranked
        concatenation of the top relevant community insights.

        :param query: The input natural language query.
        :return: ``GlobalSearchRetrieve`` containing positively rated insights
                 sorted by descending rating.
        """

        communities_ids = await self.knowledge_graph.index.community_summary_kv_storage.all_keys()
        communities = await self.knowledge_graph.index.community_summary_kv_storage.get_by_ids(communities_ids)
        communities = [c for c in communities if c is not None]

        responses = [r.model_dump() for r in await self.get_meta_responses(query, communities)]

        responses = [r for r in responses if int(r.get("rating", 0)) > 0]
        responses = sorted(responses, key=lambda x: int(x.get("rating", 0)), reverse=True)

        return GlobalSearchRetrieve(
            query=query,
            result=GlobalSearchResult(
                insights=responses,
            ),
            metrics={
                f"insight_{idx}_rating": r.get("rating", 0)
                for idx, r in enumerate(responses)
            },
        )

    async def get_meta_responses(self, query: str, context: List[str]) -> List[GlobalSearchContextModel]:
        """
        Generate and evaluate meta-responses for each community summary.

        The model receives the full list of community summaries and scores each
        according to relevance to the given query. Only positively rated responses
        are retained.

        :param query: The user query used to assess community relevance.
        :param context: A list of community summary texts to evaluate.
        :return: List of structured response dictionaries with fields such as
                 ``response`` and ``rating``.
        """
        instruction: RAGUInstruction = self.get_prompt("global_search_context")

        rendered_list: List[ChatMessages] = render(
            instruction.messages,
            query=query,
            context=context,
            language=self.language,
        )

        meta_results = await asyncio.gather(*[
            self.llm.chat_completion(
                conversation=rendered.to_openai(),
                output_schema=instruction.pydantic_model or str, # type: ignore
            )
            for rendered in rendered_list
        ], return_exceptions=True) # type: ignore

        meta_responses: List[GlobalSearchContextModel] = []
        for i, result in enumerate(meta_results):
            if isinstance(result, Exception):
                logger.warning(
                    "Global search meta-response failed for community {}: {}: {}",
                    i, type(result).__name__, result,
                )
                continue
            meta_responses.append(result)

        return meta_responses

    async def a_query(self, query: str, *args, **kwargs) -> SearchEngineResponse:
        """
        Execute a full global retrieval-augmented generation query.

        - Retrieves all community-level insights.
        - Generates a final global answer.

        :param query: The natural language query from the user.
        :return: ``SearchEngineResponse`` containing the generated answer and
                 the ``GlobalSearchRetrieve`` used as context.
        """
        context = await self.a_search(query)
        truncated_context: str = self.truncation(str(context))

        instruction: RAGUInstruction = self.get_prompt("global_search")

        rendered_list: List[ChatMessages] = render(
            instruction.messages,
            query=query,
            context=truncated_context,
            language=self.language,
        )
        rendered = rendered_list[0]
        answer = await self.llm.chat_completion(
            conversation=rendered.to_openai(),
            output_schema=instruction.pydantic_model or str,  # type: ignore[arg-type]
        )  # type: ignore[assignment]

        return SearchEngineResponse(
            query=query,
            response=answer,
            retrieval=context,
            payload={}
        )
