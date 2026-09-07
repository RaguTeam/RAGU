from collections.abc import AsyncIterator
from dataclasses import field, dataclass
from textwrap import dedent
from typing import List, Literal

from jinja2 import Template
from typing_extensions import override

from ragu.common.global_parameters import Settings
from ragu.common.logger import logger
from ragu.common.prompts.default_models import GlobalSearchContextModel
from ragu.common.prompts.messages import ChatMessages, render
from ragu.common.prompts.prompt_storage import RAGUInstruction, require_prompt_schema
from ragu.graph.knowledge_graph import KnowledgeGraph
from ragu.models.llm import LLM
from ragu.search_engine.base_engine import (
    BaseEngine,
    SearchEngineRetrieve,
    SearchEngineResponse,
    SearchEngineStreamEvent,
    EngineParams,
)


@dataclass
class GlobalSearchParams(EngineParams):
    """
    Per-query parameters for :class:`GlobalSearchEngine`.

    :param min_cluster_size: Minimum number of entities a community must
        contain for its summary to be evaluated. When ``1`` (the default),
        every stored community takes part in retrieval.
    """
    min_cluster_size: int = 1


# TODO: add the ability to use custom schemas instead of GlobalSearchContextModel
@dataclass(slots=True)
class GlobalSearchResult:
    """
    Ranked community-level insights selected for a global query.
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


class GlobalSearchEngine(BaseEngine[GlobalSearchParams, GlobalSearchRetrieve]):
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
            llm,
            prompts=_PROMPTS,
            max_context_length=max_context_length,
            tokenizer_backend=tokenizer_backend,
            tokenizer_model=tokenizer_model,
        )

        self.knowledge_graph = knowledge_graph
        self.language = language if language else Settings.language

    @override
    async def batch_search(
        self,
        queries: List[str],
        params: GlobalSearchParams | None = None,
    ) -> List[GlobalSearchRetrieve]:
        """
        Perform a global semantic search for a batch of queries.

        :param queries: The input natural language queries.
        :param params: Retrieval parameters. Pass :class:`GlobalSearchParams`
            to skip communities smaller than ``min_cluster_size``.
        :return: ``GlobalSearchRetrieve`` per query, aligned with ``queries``.
        """
        if not queries:
            return []

        min_cluster_size = params.min_cluster_size if params else GlobalSearchParams().min_cluster_size
        communities = await self._get_community_summaries(min_cluster_size)

        retrieves: List[GlobalSearchRetrieve] = []
        for query, meta_responses in zip(queries, await self.get_meta_responses(queries, communities)):
            rated = [insight for insight in meta_responses if insight.rating > 0]
            insights = sorted(rated, key=lambda insight: insight.rating, reverse=True)
            retrieves.append(
                GlobalSearchRetrieve(
                    query=query,
                    result=GlobalSearchResult(insights=insights),
                    metrics={
                        f"insight_{idx}_rating": insight.rating
                        for idx, insight in enumerate(insights)
                    },
                )
            )
        return retrieves

    async def _get_community_summaries(self, min_cluster_size: int) -> List[str]:
        """
        Fetch stored community summaries, skipping communities that are too small.

        :param min_cluster_size: Minimum number of entities a community must contain.
        :return: Summaries of the communities that passed the size filter.
        """
        summary_storage = self.knowledge_graph.index.community_summary_kv_storage
        community_ids = await summary_storage.all_keys()
        summaries = await summary_storage.get_by_ids(community_ids)

        kept: List[tuple[str, str]] = [
            (community_id, summary)
            for community_id, summary in zip(community_ids, summaries)
            if summary is not None
        ]
        if min_cluster_size <= 1 or not kept:
            return [summary for _, summary in kept]

        rows = await self.knowledge_graph.index.community_kv_storage.get_by_ids(
            [community_id for community_id, _ in kept]
        )
        filtered = [
            (community_id, summary)
            for (community_id, summary), row in zip(kept, rows)
            if row is None or len(row.get("entity_ids", [])) >= min_cluster_size
        ]

        if len(filtered) != len(kept):
            logger.debug(
                "GlobalSearch: skipped {} of {} communities smaller than {} entities",
                len(kept) - len(filtered),
                len(kept),
                min_cluster_size,
            )
        if kept and not filtered:
            logger.warning(
                "GlobalSearch: every stored community is smaller than "
                "min_cluster_size={}; the answer will be generated without context",
                min_cluster_size,
            )

        return [summary for _, summary in filtered]

    async def get_meta_responses(
        self,
        queries: List[str],
        context: List[str],
    ) -> List[List[GlobalSearchContextModel]]:
        """
        Evaluate every (query, community) pair in a single batched LLM call.

        :param queries: User queries used to assess community relevance.
        :param context: Community summaries to evaluate against every query.
        :return: Per-query lists of structured evaluations, aligned with ``queries``.
        """
        if not queries or not context:
            return [[] for _ in queries]

        instruction: RAGUInstruction = self.get_prompt("global_search_context")

        expanded_queries: List[str] = []
        expanded_context: List[str] = []
        for query in queries:
            for community in context:
                expanded_queries.append(query)
                expanded_context.append(community)

        rendered_list: List[ChatMessages] = render(
            instruction.messages,
            query=expanded_queries,
            context=expanded_context,
            language=self.language,
        )

        answers = await self.llm.batch_chat_completion(
            [rendered.to_openai() for rendered in rendered_list],
            output_schema=require_prompt_schema(
                instruction, "global_search_context", GlobalSearchContextModel
            ),
            continue_on_error=True,
            desc="GlobalSearch batch meta-eval",
        )

        community_count = len(context)
        return [
            [answer for answer in answers[i * community_count:(i + 1) * community_count] if answer is not None]
            for i in range(len(queries))
        ]

    @override
    async def batch_query(
        self,
        queries: List[str],
        params: GlobalSearchParams | None = None,
    ) -> List[SearchEngineResponse]:
        """
        Execute global RAG for multiple queries, batching final synthesis.

        :param queries: The natural language queries from the user.
        :param params: Query parameters forwarded to :meth:`batch_search`; see
            :class:`GlobalSearchParams`.
        :return: ``SearchEngineResponse`` objects aligned with ``queries``.
        """
        if not queries:
            return []

        contexts = await self.batch_search(queries, params)

        instruction: RAGUInstruction = self.get_prompt("global_search")
        conversations: List[ChatMessages] = render(
            instruction.messages,
            query=queries,
            context=[self.truncation(str(context)) for context in contexts],
            language=self.language,
        )
        answers = await self.llm.batch_chat_completion(
            [conversation.to_openai() for conversation in conversations],
            output_schema=instruction.pydantic_model,
            desc="GlobalSearch batch query",
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
        params: GlobalSearchParams | None = None,
    ) -> AsyncIterator[SearchEngineStreamEvent]:
        """
        Execute global RAG and stream the final plain-text synthesis.

        Community meta-evaluation is completed before streaming starts; only the
        final answer synthesis is streamed.

        :param query: The natural language query from the user.
        :param params: Query parameters forwarded to :meth:`search`; see
            :class:`GlobalSearchParams`.
        :returns: Async iterator of text deltas with the retrieval context.
        """
        context = await self.search(query, params)

        instruction: RAGUInstruction = self.get_prompt("global_search")
        conversations: List[ChatMessages] = render(
            instruction.messages,
            query=query,
            context=self.truncation(str(context)),
            language=self.language,
        )
        conversation = conversations[0].to_openai()

        async for delta in self.llm.stream_chat_completion(conversation):
            yield SearchEngineStreamEvent(
                query=query,
                retrieval=context,
                delta=delta,
                payload={},
            )
