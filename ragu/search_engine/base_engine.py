from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar, Generic

from pydantic import BaseModel
from ragu.common.base import RaguGenerativeModule
from ragu.common.global_parameters import Settings
from ragu.models.llm import LLM
from ragu.utils.ragu_utils import always_get_an_event_loop
from ragu.utils.token_truncation import TokenTruncation


ResultT = TypeVar("ResultT")


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

    @abstractmethod
    async def a_search(
        self,
        query,
        *args,
        **kwargs,
    ) -> SearchEngineRetrieve:
        """
        Retrieve context relevant to a query without generating an answer.

        :param query: Input query string.
        :return: Engine-specific retrieval container with result payload and metrics.
        """
        pass

    @abstractmethod
    async def a_query(self, query: str, *args, **kwargs) -> SearchEngineResponse:
        """
        Execute retrieval and answer generation for a query.

        :param query: Input query string.
        :return: Structured search result containing the final answer and retrieval details.
        """
        pass

    def query(self, query: str, *args, **kwargs) -> SearchEngineResponse:
        """
        Delegate to ``a_query`` through the shared event-loop helper.

        :param query: Input query string.
        :return: Structured search result containing the final answer and retrieval details.
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.a_query(query, *args, **kwargs)
        )

    def search(
        self,
        query,
        *args,
        **kwargs,
    ) -> SearchEngineRetrieve:
        """
        Delegate to ``a_search`` through the shared event-loop helper.

        :param query: Input query string.
        :return: Engine-specific retrieval container with result payload and metrics.
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.a_search(query, *args, **kwargs)
        )
