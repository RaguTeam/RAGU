"""
Backend interface used by the API layer.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ragu.api.models import SourceItem, SubqueryItem
from ragu.search_engine.global_search import GlobalSearchParams
from ragu.search_engine.local_search import LocalParams
from ragu.search_engine.naive_search import NaiveSearchParams


GLOBAL_UNAVAILABLE_MESSAGE = (
    "Global search found no community summaries to answer from. The graph may have been "
    "built without them. Try local or naive search instead."
)
LOCAL_UNAVAILABLE_MESSAGE = (
    "Local search found no entities, relations or chunks for this query. The graph may have "
    "been built without entity extraction or your query was too unrelated to the text corpus."
)
NAIVE_UNAVAILABLE_MESSAGE = (
    "Naive search found no matching chunks. The corpus has nothing on this query, or the "
    "graph carries no chunk index."
)


@dataclass
class SearchOutcome:
    """
    Normalized result of a single search call.
    """

    answer: str
    sources: list[SourceItem] = field(default_factory=list)
    subqueries: list[SubqueryItem] = field(default_factory=list)


class SearchBackend(ABC):
    """
    Owns the knowledge graph and executes searches over it.

    Implementations raise the error types from ``api.errors``; the API
    layer maps them to HTTP status codes.
    """

    @property
    @abstractmethod
    def graph_loaded(self) -> bool:
        """Whether the graph is loaded and searches can be served."""

    async def startup(self) -> None:
        """Load the graph. Called once on service startup."""

    async def shutdown(self) -> None:
        """Release resources on service shutdown."""

    @abstractmethod
    async def search_global(
        self, query: str, *, params: GlobalSearchParams
    ) -> SearchOutcome: ...

    @abstractmethod
    async def search_local(
        self, query: str, *, use_query_plan: bool, params: "LocalParams"
    ) -> SearchOutcome: ...

    @abstractmethod
    async def search_naive(
        self, query: str, *, use_query_plan: bool, params: "NaiveSearchParams"
    ) -> SearchOutcome: ...
