import asyncio
from dataclasses import dataclass, field
from textwrap import dedent
from typing import Any, List, Literal

from jinja2 import Template
from typing_extensions import override

from ragu.common.global_parameters import Settings
from ragu.models.llm import LLM
from ragu.search_engine.base_engine import BaseEngine, SearchEngineRetrieve, SearchEngineResponse
from ragu.common.prompts.prompt_storage import RAGUInstruction
from ragu.common.prompts.messages import ChatMessages, render


@dataclass(slots=True)
class MixSearchResult:
    """
    Aggregated child-engine outputs.

    ``results`` contains either retrieval containers from child ``a_search``
    calls or full ``SearchEngineResponse`` objects from child ``a_query`` calls,
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


class MixSearchEngine(BaseEngine):
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
        engines: List[BaseEngine],
        allow_partial_failures: bool = True,
        language: str | None = None,
        max_context_length: int | None = None,
        tokenizer_backend: Literal["tiktoken", "local"] | None = None,
        tokenizer_model: str | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Initialize a `MixSearchEngine`.

        :param llm: LLM used to generate the final synthesized answer.
        :param engines: Ordered list of child engines used for retrieval or answer ensembling.
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
        """
        prompts = ["mix_search_context", "mix_search"]
        super().__init__(
            llm=llm,
            prompts=prompts,
            max_context_length=max_context_length,
            tokenizer_backend=tokenizer_backend,
            tokenizer_model=tokenizer_model,
            *args,
            **kwargs,
        )

        self.engines = engines
        if not self.engines:
            raise ValueError("MixSearchEngine requires at least one child engine")

        self.allow_partial_failures = allow_partial_failures
        self.language = language if language else Settings.language

    async def _search_all(
        self,
        query: str,
        *args: Any,
        **kwargs: Any,
    ) -> list[SearchEngineRetrieve]:
        """
        Execute ``a_search`` on each child engine.

        :param query: Input query string.
        :return: Ordered list of successful per-engine search contexts. Failed
                 engines are omitted when ``allow_partial_failures=True``.
        :raises RuntimeError: If every child engine fails.
        """
        tasks = [
            engine.a_search(query, *args, **kwargs)
            for engine in self.engines
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        contexts: list[SearchEngineRetrieve] = []
        for result in results:
            if isinstance(result, Exception):
                if not self.allow_partial_failures:
                    raise result
                continue
            contexts.append(result)

        if not contexts:
            raise RuntimeError("MixSearchEngine could not retrieve context from any child engine")

        return contexts

    async def _query_all(
        self,
        query: str,
        *args: Any,
        **kwargs: Any,
    ) -> list[SearchEngineResponse]:
        """
        Execute ``a_query`` on each child engine.

        :param query: Input query string.
        :return: Ordered list of successful per-engine answers. Failed engines
                 are omitted when ``allow_partial_failures=True``.
        :raises RuntimeError: If every child engine fails.
        """
        tasks = [
            engine.a_query(query, *args, **kwargs)
            for engine in self.engines
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        contexts: list[SearchEngineResponse] = []
        for result in results:
            if isinstance(result, Exception):
                if not self.allow_partial_failures:
                    raise result
                continue
            contexts.append(result)

        if not contexts:
            raise RuntimeError("MixSearchEngine could not retrieve context from any child engine")

        return contexts

    @override
    async def a_search(self, query: str, *args: Any, **kwargs: Any) -> MixSearchRetrieve:
        """
        Retrieve raw contexts from all child engines.

        :param query: Input query string.
        :return: ``MixSearchRetrieve`` containing successful child retrieval
                 contexts in engine order.
        """
        results = await self._search_all(query, *args, **kwargs)

        # TODO: maybe it is good idea to pass every child engine metrics in 'metrics' field here.
        return MixSearchRetrieve(
            query=query,
            result=MixSearchResult(results=results),
            metrics={}
        )

    @override
    async def a_query(
        self,
        query: str,
        *args: Any,
        ensemble_responses: bool = False,
        **kwargs: Any,
    ) -> SearchEngineResponse:
        """
        Execute an ensemble query across child engines.

        When ``ensemble_responses=False``, this method retrieves raw contexts from each child
        engine via ``a_search`` and synthesizes one final answer from the combined contexts.

        When ``ensemble_responses=True``, this method first retrieves a full answer from each child
        engine via ``a_query`` and then synthesizes one final answer from those per-engine answers.

        :param query: Input query string.
        :param ensemble_responses: Whether to ensemble child-engine answers instead of child-engine
                                   search contexts.
        :return: ``SearchEngineResponse`` containing the synthesized answer and
                 the child contexts or responses used for synthesis.
        """
        results = await (
            self._query_all(query, *args, **kwargs)
            if ensemble_responses
            else self._search_all(query, *args, **kwargs)
        )
        section_label = "Response" if ensemble_responses else "Context"
        context_instruction: RAGUInstruction = self.get_prompt("mix_search_context")
        rendered_context_list: list[ChatMessages] = render(
            context_instruction.messages,
            payload={"entries": results},
            section_label=section_label,
        )
        rendered_context = rendered_context_list[0]
        formatted_context = rendered_context.messages[0].content
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
        rendered = rendered_list[0]

        response = await self.llm.chat_completion(
            conversation=rendered.to_openai(),
            output_schema=instruction.pydantic_model or str,  # type: ignore[arg-type]
        )  # type: ignore[return-value]

        # TODO: maybe it is good idea to pass every child engine metrics in 'metrics' field here.
        return SearchEngineResponse(
            query=query,
            response=response,
            retrieval=MixSearchRetrieve(
                query=query,
                result=MixSearchResult(results),
                metrics={}
            ),
            payload={}
        )
