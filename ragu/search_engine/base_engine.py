from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar, Generic

from pydantic import BaseModel

from ragu.common.base import RaguGenerativeModule
from ragu.common.global_parameters import Settings
from ragu.models.llm import LLM
from ragu.utils.ragu_utils import deprecated
from ragu.utils.token_truncation import TokenTruncation

ResultT = TypeVar("ResultT")


@dataclass
class EngineParams:
    """
    Base class for engine parameters.

    Carries no fields of its own; concrete engines subclass it to declare only
    the options they actually use (e.g. ``top_k``, reranking limits, generation
    flags). Shared by ``search`` / ``batch_search`` and ``query`` /
    ``batch_query``.
    """


@dataclass(slots=True)
class SearchEngineRetrieve(ABC, Generic[ResultT]):
    """
    Base container for search-only results.

    ``result`` stores the engine-specific retrieval payload, while ``metrics``
    stores optional diagnostics such as relevance scores, ranks, timings, or
    backend-specific retrieval metadata.
    """
    query: str
    result: ResultT
    metrics: dict[str, Any] = field(default_factory=dict)

    @abstractmethod
    def to_text(self) -> str:
        """
        Render the retrieved context as text suitable for prompt injection.
        """
        ...

    def __str__(self) -> str:
        return self.to_text()


@dataclass(slots=True)
class SearchEngineResponse:
    """
    Response from search engine.

    ``response`` is the generated answer, ``retrieval`` is the context used to
    produce it, and ``payload`` carries optional engine-specific metadata.
    """
    query: str
    response: str | BaseModel
    retrieval: SearchEngineRetrieve[Any]
    payload: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        if isinstance(self.response, BaseModel):
            return self.response.model_dump_json(indent=4)
        return self.response


@dataclass(slots=True)
class SearchEngineStreamEvent:
    """
    Incremental text generation event from a search engine.

    ``delta`` contains the next generated text chunk. ``retrieval`` is repeated
    on each event so consumers can access the context even when they process the
    stream one chunk at a time.
    """
    query: str
    retrieval: SearchEngineRetrieve[Any]
    delta: str
    payload: dict[str, Any] = field(default_factory=dict)


class BaseEngine(RaguGenerativeModule, ABC):
    """
    Base interface for RAGU query/search engines.

    Concrete engines implement retrieval (a_search method) and answer generation
    (a_query method) on top of a knowledge graph.
    """

    def __init__(
        self,
        llm: LLM,
        *args: Any,
        max_context_length: int | None = None,
        tokenizer_backend: Literal["tiktoken", "local"] | None = None,
        tokenizer_model: str | None = None,
        **kwargs: Any,
    ):
        """
        Initialize an engine with an LLM and context truncation settings.

        Context truncation parameters default to the corresponding
        :class:`GlobalSettings` fields when ``None`` (the default), so an
        engine built without overrides behaves exactly as before. Pass
        explicit values to configure a specific instance independently of the
        global singleton (e.g. several engines with different LLMs / context
        windows in the same process).

        :param llm: LLM used by concrete engines for answer generation.
        :param max_context_length: Maximum number of tokens the assembled
            context is truncated to. When ``None``, falls back to
            ``Settings.llm_context_token_limit``.
        :param tokenizer_backend: Tokenizer backend (``"tiktoken"`` or
            ``"local"``). When ``None``, falls back to
            ``Settings.tokenizer_llm_backend``.
        :param tokenizer_model: Tokenizer model identifier (e.g.
            ``"gpt-4o"``). When ``None``, falls back to
            ``Settings.tokenizer_llm_name``.
        """
        super().__init__(*args, **kwargs)
        self.llm = llm
        self.truncation = TokenTruncation(
            tokenizer_model or Settings.tokenizer_llm_name,
            tokenizer_backend or Settings.tokenizer_llm_backend,
            max_context_length if max_context_length is not None else Settings.llm_context_token_limit,
        )

    async def search(self, query: str, params: EngineParams | None = None) -> SearchEngineRetrieve:
        """
        Retrieve context relevant to a query without generating an answer.

        Thin single-query delegate; the implementation lives in
        :meth:`batch_search`.

        :param query: Input query string.
        :param params: Engine-specific retrieval parameters. When ``None``, the
            engine's default parameters are used.
        :return: Engine-specific retrieval container with result payload and metrics.
        """
        return (await self.batch_search([query], params))[0]

    async def query(self, query: str, params: EngineParams | None = None) -> SearchEngineResponse:
        """
        Execute retrieval and answer generation for a query.

        Thin single-query delegate; the implementation lives in
        :meth:`batch_query`.

        :param query: Input query string.
        :param params: Engine-specific query parameters. When ``None``, the
            engine's default parameters are used.
        :return: Structured search result containing the final answer and retrieval details.
        """
        return (await self.batch_query([query], params))[0]

    async def stream_query(
        self,
        query: str,
        params: EngineParams | None = None,
    ) -> AsyncIterator[SearchEngineStreamEvent]:
        """
        Execute retrieval and stream plain-text answer generation.

        Concrete engines that can render a final-answer prompt override this
        method. The default keeps the base class backward-compatible for custom
        engines that only implement ``query`` / ``batch_query``.

        :param query: Input query string.
        :param params: Engine-specific query parameters.
        :returns: Async iterator of generated text deltas with retrieval context.
        """

        # Makes this method an async generator, so subclasses share the async-for contract.
        if False: # pragma: no cover
            yield SearchEngineStreamEvent(query=query, retrieval=None, delta="")  # type: ignore[arg-type]
        raise NotImplementedError(f"{type(self).__name__} does not support streaming query")

    @abstractmethod
    async def batch_search(self, queries: list[str], params: EngineParams | None = None) -> list[SearchEngineRetrieve]:
        """
        Retrieve context for multiple queries.

        This is the primary retrieval entry point; concrete engines implement it
        to share retrieval work across the whole batch.

        :param queries: Input query strings.
        :param params: Engine-specific retrieval parameters. When ``None``, the
            engine's default parameters are used.
        :return: Retrieval containers aligned with ``queries``.
        """

    @abstractmethod
    async def batch_query(self, queries: list[str], params: EngineParams | None = None) -> list[SearchEngineResponse]:
        """
        Execute retrieval and answer generation for multiple queries.

        This is the primary query entry point; concrete engines implement it to
        share retrieval across the batch and route answer generation through
        :meth:`LLM.batch_chat_completion`.

        :param queries: Input query strings.
        :param params: Engine-specific query parameters. When ``None``, the
            engine's default parameters are used.
        :return: Responses aligned with ``queries``.
        """

    @deprecated(replacement="search")
    async def a_search(self, query: str, params: EngineParams | None = None) -> SearchEngineRetrieve:
        """
        Deprecated async alias for :meth:`search`.

        :param query: Input query string.
        :param params: Engine-specific retrieval parameters.
        :return: Engine-specific retrieval container with result payload and metrics.
        """
        return await self.search(query, params)

    @deprecated(replacement="query")
    async def a_query(self, query: str, params: EngineParams | None = None) -> SearchEngineResponse:
        """
        Deprecated async alias for :meth:`query`.

        :param query: Input query string.
        :param params: Engine-specific query parameters.
        :return: Structured search result containing the final answer and retrieval details.
        """
        return await self.query(query, params)
