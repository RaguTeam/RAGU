import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from textwrap import dedent
from typing import Any, List, Literal

from jinja2 import Template
from typing_extensions import override

from ragu.common.global_parameters import Settings
from ragu.common.logger import logger
from ragu.models.llm import LLM
from ragu.search_engine.base_engine import (
    BaseEngine,
    SearchEngineRetrieve,
    SearchEngineResponse,
    SearchEngineStreamEvent,
    EngineParams,
)
from ragu.common.prompts.prompt_storage import RAGUInstruction
from ragu.common.prompts.messages import ChatMessages, render


@dataclass
class MixQueryParams(EngineParams):
    """
    Query parameters for :class:`MixSearchEngine`.

    :param ensemble_responses: When ``True``, ensemble child-engine *answers*
        (via their ``batch_query``) instead of child-engine retrieval contexts.
    """
    ensemble_responses: bool = False


@dataclass(slots=True)
class MixSearchResult:
    """
    Aggregated child-engine outputs.

    ``results`` contains either retrieval containers from child ``search``
    calls or full ``SearchEngineResponse`` objects from child ``query`` calls,
    depending on the synthesis mode.
    """
    results: list[SearchEngineRetrieve[Any]] | list[SearchEngineResponse] = field(default_factory=list)


@dataclass(slots=True)
class MixSearchRetrieve(SearchEngineRetrieve[MixSearchResult]):
    """
    Retrieval container returned by :class:`MixSearchEngine`.

    Metrics are currently empty; child-engine metrics remain available inside each entry.
    """
    result: MixSearchResult

    _TO_TEXT_TEMPLATE = Template(dedent("""
        {%- for retrieve in result.results %}
        **Engine {{ loop.index }} Context**
        {{ retrieve }}
        {% endfor %}
    """))

    def to_text(self) -> str:
        """
        Render each child engine result as a separate context section.
        """
        return self._TO_TEXT_TEMPLATE.render(result=self.result)


class MixSearchEngine(BaseEngine[MixQueryParams, MixSearchRetrieve]):
    """
    Performs ensemble retrieval-augmented search over multiple engines.

    The engine supports two synthesis modes:
      1. Retrieve raw contexts from each child engine and combine them into one final answer.
      2. Retrieve a full answer from each child engine and combine those answers into one final answer.

    Child engines are executed in the order provided at construction time.
    """

    def __init__(
        self,
        llm: LLM,
        engines: List[BaseEngine[Any, Any]],
        engine_params: List[EngineParams | None] | None = None,
        allow_partial_failures: bool = True,
        language: str | None = None,
        max_context_length: int | None = None,
        tokenizer_backend: Literal["tiktoken", "local"] | None = None,
        tokenizer_model: str | None = None,
    ):
        """
        Initialize a `MixSearchEngine`.

        :param llm: LLM used to generate the final synthesized answer.
        :param engines: Ordered list of child engines used for retrieval or answer ensembling.
        :param engine_params: Optional per-child-engine parameters, aligned with
            ``engines``. Each entry is forwarded to the matching child's
            ``batch_search`` / ``batch_query`` call (so different children can be
            driven with different ``top_k``, reranking, etc.). When ``None``, all
            children use their own defaults; individual entries may also be
            ``None`` to default a single child.
        :param allow_partial_failures: Whether to tolerate failures from individual child engines.
                                       Failed engines are omitted from the result list.
        :param language: Default output language.
        :param max_context_length: Maximum tokens for the assembled context fed to
            the LLM. When ``None``, falls back to ``Settings.llm_context_token_limit``.
            This truncation is applied only to the MixSearchEngine's own final context
            and is NOT propagated to the child engines (each child keeps its own
            tokenizer configuration).
        :param tokenizer_backend: Tokenizer backend for context truncation. When
            ``None``, falls back to ``Settings.tokenizer_llm_backend``.
        :param tokenizer_model: Tokenizer model identifier for context truncation.
            When ``None``, falls back to ``Settings.tokenizer_llm_name``.
        :raises ValueError: If ``engines`` is empty, or ``engine_params`` is
            provided with a length that does not match ``engines``.
        """
        prompts = ["mix_search_context", "mix_search"]
        super().__init__(
            llm,
            prompts=prompts,
            max_context_length=max_context_length,
            tokenizer_backend=tokenizer_backend,
            tokenizer_model=tokenizer_model,
        )

        self.engines = engines
        if not self.engines:
            raise ValueError("MixSearchEngine requires at least one child engine")

        if engine_params is not None and len(engine_params) != len(engines):
            raise ValueError(
                f"engine_params length ({len(engine_params)}) must match engines length ({len(engines)})"
            )
        self.engine_params: List[EngineParams | None] = (
            list(engine_params) if engine_params is not None else [None] * len(engines)
        )

        self.allow_partial_failures = allow_partial_failures
        self.language = language if language else Settings.language

    async def _child_batch(
        self,
        engine: BaseEngine[Any, Any],
        queries: List[str],
        ensemble: bool,
        params: EngineParams | None,
    ) -> list:
        """
        Run a child engine's batch method over ``queries``, tolerating failure.

        :param engine: Child engine to invoke.
        :param queries: Input query strings.
        :param ensemble: When ``True``, call the child's ``batch_query``;
            otherwise ``batch_search``.
        :param params: Parameters forwarded to the child engine (its own
            defaults are used when ``None``).
        :return: Per-query results aligned with ``queries``; on whole-engine
            failure, a list of ``None`` when ``allow_partial_failures`` is set.
        :raises Exception: The child failure when partial failures are disallowed.
        """
        try:
            if ensemble:
                return await engine.batch_query(queries, params)
            return await engine.batch_search(queries, params)
        except Exception as error:
            if not self.allow_partial_failures:
                raise
            logger.warning(
                "MixSearchEngine child engine {} batch failed: {}: {}",
                type(engine).__name__, type(error).__name__, error,
            )
            return [None] * len(queries)

    async def _gather_child_results(
        self,
        queries: List[str],
        ensemble: bool,
    ) -> list[list]:
        """
        Collect per-query child results, transposed to engine-order per query.

        :param queries: Input query strings.
        :param ensemble: Whether to ensemble child answers instead of contexts.
        :return: One list per query, each holding the successful child results in
                 engine order.
        :raises RuntimeError: If, for any query, every child engine failed.
        """
        per_engine = await asyncio.gather(*[
            self._child_batch(engine, queries, ensemble, params)
            for engine, params in zip(self.engines, self.engine_params)
        ])

        per_query: list[list] = []
        for query_idx in range(len(queries)):
            entries = [
                per_engine[engine_idx][query_idx]
                for engine_idx in range(len(self.engines))
                if per_engine[engine_idx][query_idx] is not None
            ]
            if not entries:
                raise RuntimeError("MixSearchEngine could not retrieve context from any child engine")
            per_query.append(entries)
        return per_query

    @override
    async def batch_search(
        self,
        queries: List[str],
        params: MixQueryParams | None = None,
    ) -> List[MixSearchRetrieve]:
        """
        Retrieve raw contexts from all child engines for a batch of queries.

        Each child engine's :meth:`batch_search` is invoked once with the whole
        query list (so children batch internally), then results are transposed
        per query.

        :param queries: Input query strings.
        :param params: Retrieval parameters (unused at the Mix level; children
            use their own defaults).
        :return: ``MixSearchRetrieve`` per query, aligned with ``queries``.
        """
        if not queries:
            return []

        per_query = await self._gather_child_results(queries, ensemble=False)

        # TODO: maybe it is good idea to pass every child engine metrics in 'metrics' field here.
        return [
            MixSearchRetrieve(query=query, result=MixSearchResult(results=entries), metrics={})
            for query, entries in zip(queries, per_query)
        ]

    def _render_synthesis(
        self,
        query: str,
        entries: list,
        ensemble_responses: bool,
    ) -> ChatMessages:
        """
        Build the final synthesis conversation from child results for one query.

        :param query: Input query string.
        :param entries: Successful child contexts or responses in engine order.
        :param ensemble_responses: Whether ``entries`` are child answers (vs contexts).
        :return: Rendered chat messages ready for ``chat_completion``.
        :raises RuntimeError: If the synthesis input cannot be built.
        """
        section_label = "Response" if ensemble_responses else "Context"
        context_instruction: RAGUInstruction = self.get_prompt("mix_search_context")
        rendered_context_list: list[ChatMessages] = render(
            context_instruction.messages,
            payload={"entries": entries},
            section_label=section_label,
        )
        formatted_context = rendered_context_list[0].messages[0].content
        if not formatted_context:
            raise RuntimeError("MixSearchEngine could not build synthesis input from child engines")

        truncated_context = self.truncation(formatted_context)

        instruction: RAGUInstruction = self.get_prompt("mix_search")
        rendered_list: list[ChatMessages] = render(
            instruction.messages,
            query=query,
            context=truncated_context,
            language=self.language,
            ensemble_responses=ensemble_responses,
            section_label=section_label.lower(),
        )
        return rendered_list[0]

    @override
    async def batch_query(
        self,
        queries: List[str],
        params: MixQueryParams | None = None,
    ) -> List[SearchEngineResponse]:
        """
        Execute an ensemble query across child engines for a batch of queries.

        With ``ensemble_responses=False`` (default), child retrieval contexts are
        gathered via each child's ``batch_search`` and synthesized into one final
        answer per query. With ``ensemble_responses=True``, child *answers* are
        gathered via each child's ``batch_query`` and ensembled instead. Final
        synthesis for all queries is issued through a single
        :meth:`LLM.batch_chat_completion` call.

        :param queries: Input query strings.
        :param params: Query parameters (``ensemble_responses``). When ``None``,
            defaults to :class:`MixQueryParams`.
        :return: ``SearchEngineResponse`` objects aligned with ``queries``.
        """
        if not queries:
            return []

        params = params or MixQueryParams()
        ensemble = params.ensemble_responses
        per_query = await self._gather_child_results(queries, ensemble=ensemble)

        conversations = [
            self._render_synthesis(query, entries, ensemble).to_openai()
            for query, entries in zip(queries, per_query)
        ]

        instruction: RAGUInstruction = self.get_prompt("mix_search")
        answers = await self.llm.batch_chat_completion(
            conversations,
            output_schema=instruction.pydantic_model,
            desc="MixSearch batch query",
        )

        # TODO: maybe it is good idea to pass every child engine metrics in 'metrics' field here.
        return [
            SearchEngineResponse(
                query=query,
                response=answer,
                retrieval=MixSearchRetrieve(
                    query=query,
                    result=MixSearchResult(entries),
                    metrics={},
                ),
                payload={},
            )
            for query, answer, entries in zip(queries, answers, per_query)
        ]

    @override
    async def stream_query(
        self,
        query: str,
        params: MixQueryParams | None = None,
    ) -> AsyncIterator[SearchEngineStreamEvent]:
        """
        Execute an ensemble query and stream the final plain-text synthesis.

        Child engines still run through their existing batch methods; only the
        MixSearchEngine final synthesis is streamed.

        :param query: Input query string.
        :param params: Query parameters. When ``None``, defaults to
            :class:`MixQueryParams`.
        :returns: Async iterator of text deltas with the mixed retrieval context.
        """
        params = params or MixQueryParams()
        ensemble = params.ensemble_responses
        entries = (await self._gather_child_results([query], ensemble=ensemble))[0]
        conversation = self._render_synthesis(query, entries, ensemble).to_openai()
        retrieval = MixSearchRetrieve(
            query=query,
            result=MixSearchResult(entries),
            metrics={},
        )

        async for delta in self.llm.stream_chat_completion(conversation):
            yield SearchEngineStreamEvent(
                query=query,
                retrieval=retrieval,
                delta=delta,
                payload={},
            )
