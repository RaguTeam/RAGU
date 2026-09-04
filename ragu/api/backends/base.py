"""
Backend interface used by the API layer.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any

from ragu.api.config import ServiceSettings
from ragu.api.errors import CapabilityUnavailableError
from ragu.api.models import (
    GraphStatsResponse,
    SearchMode,
    SourceItem,
    SubqueryItem,
)
from ragu.search_engine.global_search import GlobalSearchParams
from ragu.search_engine.local_search import LocalParams
from ragu.search_engine.naive_search import NaiveSearchParams

# The graph has no store this mode can read at all.
GLOBAL_MISSING_MESSAGE = (
    "This graph carries no community summaries, so global search cannot run. Rebuild it "
    "with make_community_summary enabled, or use local or naive search."
)
LOCAL_MISSING_MESSAGE = (
    "This graph carries no entity index, so local search cannot run. Rebuild it with "
    "entity extraction enabled, or use naive search."
)
NAIVE_MISSING_MESSAGE = (
    "This graph carries no chunk index, so naive search cannot run. Rebuild it with chunk "
    "vectorization enabled, or use local or global search."
)

# The store exists, but this query retrieved nothing from it.
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


@dataclass(frozen=True, slots=True)
class ModeRequirement:
    """
    What one search mode needs from the graph, and what to say when it is absent.

    :param capability: Name reported as ``missing_capability`` when the store
        this mode reads is empty.
    :param missing_message: Explanation when the graph cannot serve the mode.
    :param no_evidence_message: Explanation when the graph can serve the mode
        but this query retrieved nothing.
    """

    capability: str
    missing_message: str
    no_evidence_message: str


MODE_REQUIREMENTS: dict[SearchMode, ModeRequirement] = {
    "global": ModeRequirement(
        capability="community_summaries",
        missing_message=GLOBAL_MISSING_MESSAGE,
        no_evidence_message=GLOBAL_UNAVAILABLE_MESSAGE,
    ),
    "local": ModeRequirement(
        capability="entity_graph",
        missing_message=LOCAL_MISSING_MESSAGE,
        no_evidence_message=LOCAL_UNAVAILABLE_MESSAGE,
    ),
    "naive": ModeRequirement(
        capability="vector_index",
        missing_message=NAIVE_MISSING_MESSAGE,
        no_evidence_message=NAIVE_UNAVAILABLE_MESSAGE,
    ),
}


@dataclass(frozen=True, slots=True)
class GraphStats:
    """
    Sizes of the stores the search modes read, counted once at startup.

    Counting them up front is what lets the service refuse a mode the graph
    cannot serve *before* paying for a generation call, and what makes
    ``graph_loaded`` mean "can answer searches" rather than "an object was
    constructed".
    """

    entities: int = 0
    relations: int = 0
    chunks: int = 0
    community_summaries: int = 0

    @property
    def is_empty(self) -> bool:
        """
        Whether the graph has nothing any mode could read.
        """
        return not (self.entities or self.chunks or self.community_summaries)

    def supports(self, mode: SearchMode) -> bool:
        """
        Whether the store this mode reads holds anything.

        :param mode: Search mode to check.
        :return: ``True`` when the mode can be served.
        """
        if mode == "global":
            return self.community_summaries > 0
        if mode == "local":
            return self.entities > 0
        return self.chunks > 0

    def to_response(self) -> GraphStatsResponse:
        """
        Render the counts for the health endpoint.
        """
        return GraphStatsResponse(
            entities=self.entities,
            relations=self.relations,
            chunks=self.chunks,
            community_summaries=self.community_summaries,
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

    Implementations raise the error types from ``ragu.api.errors``; the API
    layer maps them to HTTP status codes.
    """

    def __init__(self, settings: ServiceSettings):
        """
        :param settings: Service settings, source of the request bounds every
            backend shares.
        """
        self.settings = settings
        self._stats: GraphStats | None = None

    @property
    def graph_loaded(self) -> bool:
        """
        Whether the graph is loaded, non-empty, and searches can be served.
        """
        return self._stats is not None and not self._stats.is_empty

    @property
    def stats(self) -> GraphStats | None:
        """
        Graph sizes measured at startup, or ``None`` before it completed.
        """
        return self._stats

    async def startup(self) -> None:
        """Load the graph. Called once on service startup."""

    async def shutdown(self) -> None:
        """Release resources on service shutdown."""

    def bound_params(self, params: Any) -> Any:
        """
        Clamp client-supplied parameters to what this service will serve.

        The engine parameter classes are plain dataclasses with no bounds and
        they arrive straight from the request body, so every backend — the stub
        included — has to apply the same ceilings before acting on them.

        :param params: Engine parameter object from the request.
        :return: ``params`` unchanged, or a clamped copy.
        """
        changes: dict[str, Any] = {}

        top_k = getattr(params, "top_k", None)
        if isinstance(top_k, int) and not isinstance(top_k, bool):
            if top_k > self.settings.max_top_k:
                changes["top_k"] = self.settings.max_top_k

        rerank_top_k = getattr(params, "rerank_top_k", None)
        if isinstance(rerank_top_k, int) and not isinstance(rerank_top_k, bool):
            if rerank_top_k > self.settings.max_top_k:
                changes["rerank_top_k"] = self.settings.max_top_k

        min_cluster_size = getattr(params, "min_cluster_size", None)
        if isinstance(min_cluster_size, int) and not isinstance(min_cluster_size, bool):
            if min_cluster_size < self.settings.min_cluster_size_floor:
                changes["min_cluster_size"] = self.settings.min_cluster_size_floor

        if not changes:
            return params
        return replace(params, **changes)

    def require_capability(self, mode: SearchMode) -> None:
        """
        Refuse a mode the loaded graph cannot serve, before generating anything.

        :param mode: Search mode about to run.
        :raises CapabilityUnavailableError: If the store this mode reads is empty.
        """
        stats = self._stats
        if stats is None or stats.supports(mode):
            return
        requirement = MODE_REQUIREMENTS[mode]
        raise CapabilityUnavailableError(
            requirement.missing_message,
            mode=mode,
            missing_capability=requirement.capability,
        )

    @staticmethod
    def no_evidence(mode: SearchMode) -> CapabilityUnavailableError:
        """
        Build the error for a query that retrieved nothing.

        ``missing_capability`` stays ``None``: the graph does support the mode,
        this query simply had no evidence behind it.

        :param mode: Search mode that came back empty.
        :return: The error to raise.
        """
        return CapabilityUnavailableError(
            MODE_REQUIREMENTS[mode].no_evidence_message,
            mode=mode,
            missing_capability=None,
        )

    @abstractmethod
    async def search_global(
        self, query: str, *, params: GlobalSearchParams
    ) -> SearchOutcome: ...

    @abstractmethod
    async def search_local(
        self, query: str, *, use_query_plan: bool, params: LocalParams
    ) -> SearchOutcome: ...

    @abstractmethod
    async def search_naive(
        self, query: str, *, use_query_plan: bool, params: NaiveSearchParams
    ) -> SearchOutcome: ...
