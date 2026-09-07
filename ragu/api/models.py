"""
Request and response schemas of the search API.
"""

from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field
from ragu.search_engine.global_search import GlobalSearchParams
from ragu.search_engine.local_search import LocalParams
from ragu.search_engine.mix_search import MixQueryParams
from ragu.search_engine.naive_search import NaiveSearchParams

SearchMode = Literal["global", "local", "naive", "mix"]

# What a search mode needs the graph to hold. Named here rather than in the
# backends so that the configuration layer can validate against the same set.
Capability = Literal["entity_graph", "community_summaries", "vector_index"]

CAPABILITIES: frozenset[str] = frozenset(get_args(Capability))


class GlobalSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, description="Search query")
    params: GlobalSearchParams = Field(
        default_factory=GlobalSearchParams,
        description="GlobalSearchEngine retrieval parameters: min_cluster_size",
    )


class LocalSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, description="Search query")
    use_query_plan: bool = Field(
        default=True, description="Decompose the query into subqueries"
    )
    params: LocalParams = Field(
        default_factory=LocalParams,
        description="LocalSearchEngine retrieval parameters: top_k, rerank_top_k, "
        "use_summary, use_chunks",
    )


class NaiveSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, description="Search query")
    use_query_plan: bool = Field(
        default=True, description="Decompose the query into subqueries"
    )
    params: NaiveSearchParams = Field(
        default_factory=NaiveSearchParams,
        description="NaiveSearchEngine retrieval parameters: top_k, rerank_top_k",
    )


class MixSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, description="Search query")
    use_query_plan: bool = Field(
        default=True, description="Decompose the query into subqueries"
    )
    params: MixQueryParams = Field(
        default_factory=MixQueryParams,
        description="MixSearchEngine parameters: ensemble_responses",
    )
    local_params: LocalParams = Field(
        default_factory=LocalParams,
        description="Parameters for the local child engine. MixSearchEngine reads "
        "child parameters from its constructor, not from the request-time params, "
        "so they are named separately here.",
    )
    naive_params: NaiveSearchParams = Field(
        default_factory=NaiveSearchParams,
        description="Parameters for the naive child engine",
    )


class ChildEngineReport(BaseModel):
    """
    What one child engine of an ensemble actually did.
    """

    engine: str = Field(description="Child engine class name")
    mode: str | None = Field(
        default=None, description="Search mode the child corresponds to, when known"
    )
    ok: bool = Field(description="Whether the child produced a result")
    error: str | None = Field(
        default=None, description="Why the child failed, when it did"
    )


class EngineReport(BaseModel):
    """
    What ran, as opposed to what was asked for.

    ``MixSearchEngine`` tolerates child failures by design, so a request that
    asked for graph plus chunks can silently be answered from chunks alone.
    This reports the difference instead of hiding it.
    """

    requested: SearchMode = Field(description="Mode the client asked for")
    used: str = Field(description="Engine class that produced the answer")
    query_plan: bool = Field(
        default=False, description="Whether the query was decomposed first"
    )
    degraded: bool = Field(
        default=False, description="True when some child engine did not contribute"
    )
    children: list[ChildEngineReport] = Field(
        default_factory=list, description="Per-child outcome, for ensemble engines"
    )


class SourceItem(BaseModel):
    id: str = Field(
        description="Stable source identifier, e.g. chunk_42 or community_3"
    )
    type: str = Field(
        description="Source kind: chunk, entity, relation, community_summary"
    )
    content: str = Field(default="", description="Source text")
    score: float | None = Field(
        default=None, description="Retrieval score when the engine provides one"
    )


class SubqueryItem(BaseModel):
    query: str = Field(description="Subquery produced by the query plan")
    answer: str = Field(default="", description="Intermediate answer to the subquery")


class SearchResponse(BaseModel):
    query: str
    mode: SearchMode
    used_query_plan: bool
    answer: str
    sources: list[SourceItem] = Field(default_factory=list)
    subqueries: list[SubqueryItem] = Field(default_factory=list)
    engines: EngineReport = Field(
        description="What actually ran, including child-engine failures"
    )


class GlobalRetrieveRequest(BaseModel):
    """
    Retrieval-only request.

    No ``use_query_plan``: ``QueryPlanEngine.batch_search`` delegates straight to
    the wrapped engine and does no planning, so accepting the flag would promise
    a decomposition that never happens.
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, description="Search query")
    params: GlobalSearchParams = Field(default_factory=GlobalSearchParams)


class LocalRetrieveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, description="Search query")
    params: LocalParams = Field(default_factory=LocalParams)


class NaiveRetrieveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, description="Search query")
    params: NaiveSearchParams = Field(default_factory=NaiveSearchParams)


class MixRetrieveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, description="Search query")
    params: MixQueryParams = Field(default_factory=MixQueryParams)
    local_params: LocalParams = Field(default_factory=LocalParams)
    naive_params: NaiveSearchParams = Field(default_factory=NaiveSearchParams)


class RetrieveResponse(BaseModel):
    """
    Context gathered for a query, with no answer generated for it.
    """

    query: str
    mode: SearchMode
    sources: list[SourceItem] = Field(default_factory=list)
    engines: EngineReport = Field(
        description="What actually ran, including child-engine failures"
    )


class BatchQueries(BaseModel):
    """
    Shared shape of the batch requests: many queries, one mode, one parameter set.

    One mode and one parameter set per batch is what makes the batch worth
    making: the engines share retrieval across the whole list, and
    ``QueryPlanEngine`` merges independent subqueries from different top-level
    queries into the same child batch.
    """

    model_config = ConfigDict(extra="forbid")

    queries: list[str] = Field(
        min_length=1, description="Queries to answer in one pass"
    )


class GlobalBatchRequest(BatchQueries):
    params: GlobalSearchParams = Field(default_factory=GlobalSearchParams)


class LocalBatchRequest(BatchQueries):
    use_query_plan: bool = Field(default=True)
    params: LocalParams = Field(default_factory=LocalParams)


class NaiveBatchRequest(BatchQueries):
    use_query_plan: bool = Field(default=True)
    params: NaiveSearchParams = Field(default_factory=NaiveSearchParams)


class MixBatchRequest(BatchQueries):
    use_query_plan: bool = Field(default=True)
    params: MixQueryParams = Field(default_factory=MixQueryParams)
    local_params: LocalParams = Field(default_factory=LocalParams)
    naive_params: NaiveSearchParams = Field(default_factory=NaiveSearchParams)


class BatchSearchItem(BaseModel):
    """
    One query's result inside a batch.

    A query that retrieved nothing carries ``error`` instead of an answer: one
    empty query must not fail the whole batch.
    """

    query: str
    answer: str = ""
    sources: list[SourceItem] = Field(default_factory=list)
    subqueries: list[SubqueryItem] = Field(default_factory=list)
    error: "ErrorBody | None" = None


class BatchSearchResponse(BaseModel):
    mode: SearchMode
    used_query_plan: bool
    engines: EngineReport
    results: list[BatchSearchItem]


class ErrorBody(BaseModel):
    code: str
    mode: str | None = None
    missing_capability: str | None = None
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody


class GraphStatsResponse(BaseModel):
    """
    Sizes of the stores each search mode reads, counted once at startup.
    """

    entities: int = Field(description="Vectorized entities backing local search")
    relations: int = Field(description="Vectorized relations backing local search")
    chunks: int = Field(description="Vectorized chunks backing naive search")
    community_summaries: int = Field(
        description="Community summaries backing global search"
    )


class HealthResponse(BaseModel):
    status: str = Field(description="'ok' when searches can be served, else 'degraded'")
    graph_loaded: bool
    stats: GraphStatsResponse | None = Field(
        default=None, description="Graph sizes; absent until the graph is loaded"
    )
    error: str | None = Field(
        default=None, description="Why the backend is not ready, when it is not"
    )


BatchSearchItem.model_rebuild()
