"""
Backend interface used by the API layer.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from ragu.api.config import ServiceSettings
from ragu.api.errors import CapabilityUnavailableError, InvalidRequestError
from ragu.api.models import (
    Capability,
    EngineReport,
    GraphStatsResponse,
    SearchMode,
    SourceItem,
    SubqueryItem,
)
from ragu.search_engine.global_search import GlobalSearchParams
from ragu.search_engine.local_search import LocalParams
from ragu.search_engine.mix_search import MixQueryParams
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

MIX_MISSING_MESSAGE = (
    "Mixed search ensembles the local and naive engines, so it needs both an entity index "
    "and a chunk index. Use whichever single mode this graph supports."
)
MIX_UNAVAILABLE_MESSAGE = (
    "Mixed search found nothing for this query in either the entity index or the chunk "
    "index."
)


@dataclass(frozen=True, slots=True)
class ModeRequirement:
    """
    What one search mode needs from the graph, and what to say when it is absent.

    :param requires: Capabilities the mode needs, all of them. The first one
        the graph lacks is reported as ``missing_capability``.
    :param missing_message: Explanation when the graph cannot serve the mode.
    :param no_evidence_message: Explanation when the graph can serve the mode
        but this query retrieved nothing.
    """

    requires: tuple[Capability, ...]
    missing_message: str
    no_evidence_message: str


MODE_REQUIREMENTS: dict[SearchMode, ModeRequirement] = {
    "global": ModeRequirement(
        requires=("community_summaries",),
        missing_message=GLOBAL_MISSING_MESSAGE,
        no_evidence_message=GLOBAL_UNAVAILABLE_MESSAGE,
    ),
    "local": ModeRequirement(
        requires=("entity_graph",),
        missing_message=LOCAL_MISSING_MESSAGE,
        no_evidence_message=LOCAL_UNAVAILABLE_MESSAGE,
    ),
    "naive": ModeRequirement(
        requires=("vector_index",),
        missing_message=NAIVE_MISSING_MESSAGE,
        no_evidence_message=NAIVE_UNAVAILABLE_MESSAGE,
    ),
    "mix": ModeRequirement(
        requires=("entity_graph", "vector_index"),
        missing_message=MIX_MISSING_MESSAGE,
        no_evidence_message=MIX_UNAVAILABLE_MESSAGE,
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

    def has(self, capability: Capability) -> bool:
        """
        Whether the store behind one capability holds anything.

        :param capability: Capability to check.
        :return: ``True`` when the store is non-empty.
        """
        if capability == "community_summaries":
            return self.community_summaries > 0
        if capability == "entity_graph":
            return self.entities > 0
        return self.chunks > 0

    def supports(self, mode: SearchMode) -> bool:
        """
        Whether the graph holds everything this mode reads.

        :param mode: Search mode to check.
        :return: ``True`` when the mode can be served.
        """
        return self.missing_for(mode) is None

    def missing_for(self, mode: SearchMode) -> Capability | None:
        """
        The first capability this mode needs and the graph does not have.

        :param mode: Search mode to check.
        :return: The missing capability, or ``None`` when the mode is supported.
        """
        for capability in MODE_REQUIREMENTS[mode].requires:
            if not self.has(capability):
                return capability
        return None

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


@dataclass(frozen=True, slots=True)
class SearchCall:
    """
    One resolved search request, independent of how it arrived.

    Collapsing the per-mode backend methods into a single call object keeps the
    backend contract at two methods as retrieval, batching and streaming are
    added, and gives later work (graph selection, per-request budgets) one place
    to grow.

    :param mode: Search mode to run.
    :param queries: The client's queries. The engines batch internally — and
        ``QueryPlanEngine`` merges independent subqueries across different
        top-level queries into one child batch — so the batch is carried all the
        way down rather than being split here.
    :param params: The engine's own parameter object for this mode.
    :param local_params: Local child parameters; ``mix`` only.
    :param naive_params: Naive child parameters; ``mix`` only.
    :param use_query_plan: Whether to decompose the query first. Ignored by
        retrieval: ``QueryPlanEngine.batch_search`` delegates straight to the
        wrapped engine and does no planning.
    """

    mode: SearchMode
    queries: tuple[str, ...]
    params: Any = None
    local_params: LocalParams | None = None
    naive_params: NaiveSearchParams | None = None
    use_query_plan: bool = False

    @property
    def query(self) -> str:
        """
        The first query. A batch of one is the common case.
        """
        return self.queries[0]


@dataclass(frozen=True, slots=True)
class SearchStreamEvent:
    """
    One event of a streamed answer, before it is framed as SSE.

    ``meta`` carries the retrieval and the engine report and arrives once, before
    any text; ``delta`` carries the next chunk of the answer; ``done`` closes the
    stream with the final engine report; ``error`` reports a failure that struck
    after the response headers were already sent.
    """

    event: Literal["meta", "delta", "done", "error"]
    data: dict[str, Any]


@dataclass
class SearchOutcome:
    """
    Normalized result of a single search call.
    """

    answer: str
    sources: list[SourceItem] = field(default_factory=list)
    subqueries: list[SubqueryItem] = field(default_factory=list)
    engines: EngineReport | None = None


@dataclass
class RetrieveOutcome:
    """
    Normalized result of a retrieval-only call: context, no generation.
    """

    sources: list[SourceItem] = field(default_factory=list)
    engines: EngineReport | None = None


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

    def require_batch_size(self, size: int) -> None:
        """
        Reject a batch larger than this service will serve.

        :param size: Number of queries in the batch.
        :raises InvalidRequestError: If the batch is too large.
        """
        if size > self.settings.max_batch_size:
            raise InvalidRequestError(
                f"queries: batch of {size} exceeds the limit of "
                f"{self.settings.max_batch_size}"
            )

    def require_capability(self, mode: SearchMode) -> None:
        """
        Refuse a mode the loaded graph cannot serve, before generating anything.

        :param mode: Search mode about to run.
        :raises CapabilityUnavailableError: If the store this mode reads is empty.
        """
        stats = self._stats
        if stats is None:
            return
        missing = stats.missing_for(mode)
        if missing is None:
            return
        raise CapabilityUnavailableError(
            MODE_REQUIREMENTS[mode].missing_message,
            mode=mode,
            missing_capability=missing,
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

    def require_evidence(self, mode: SearchMode, outcome: SearchOutcome | RetrieveOutcome) -> None:
        """
        Refuse an outcome that rests on nothing.

        The engines answer confidently from an empty context, so a single-query
        route turns "no evidence at all" into a 409 rather than returning an
        answer built on nothing. Batch routes report it per query instead, since
        one empty query must not fail the whole batch.

        :param mode: Mode that produced the outcome.
        :param outcome: The outcome to check.
        :raises CapabilityUnavailableError: If the outcome has no sources.
        """
        if not outcome.sources:
            raise self.no_evidence(mode)

    @abstractmethod
    def stream(self, call: SearchCall) -> AsyncIterator[SearchStreamEvent]:
        """
        Stream the answer to a single query.

        Implementations are async generators. Readiness and capability are
        checked by the caller before the response starts, so that a graph which
        cannot serve the mode fails with a status code rather than inside an
        already-open stream.

        :param call: The resolved request; only the first query is streamed.
        :return: Async iterator of stream events.
        """

    @abstractmethod
    async def search(self, call: SearchCall) -> list[SearchOutcome]:
        """
        Retrieve context and generate an answer for every query in the call.

        :param call: The resolved request.
        :return: One outcome per query, aligned with ``call.queries``.
        """

    @abstractmethod
    async def retrieve(self, call: SearchCall) -> list[RetrieveOutcome]:
        """
        Retrieve context for every query, generating nothing.

        :param call: The resolved request. ``use_query_plan`` has no effect.
        :return: One outcome per query, aligned with ``call.queries``.
        """
