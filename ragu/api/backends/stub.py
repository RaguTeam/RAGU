"""
In-memory backend for local development and tests.

Serves deterministic canned answers without a real graph, so the agent side can
be exercised end-to-end without building a knowledge graph.
"""

from ragu.api.backends.base import (
    GraphStats,
    RetrieveOutcome,
    SearchBackend,
    SearchCall,
    SearchOutcome,
)
from ragu.api.config import ServiceSettings
from ragu.api.models import (
    ChildEngineReport,
    EngineReport,
    SearchMode,
    SourceItem,
    SubqueryItem,
)
from ragu.common.logger import logger
from ragu.search_engine.global_search import GlobalSearchParams
from ragu.search_engine.local_search import LocalParams
from ragu.search_engine.mix_search import MixQueryParams
from ragu.search_engine.naive_search import NaiveSearchParams

# How a simulated missing capability shows up in the graph sizes the base class
# reads. Driving the stub through the same GraphStats the real backend measures
# keeps both backends on one capability code path instead of two.
_EMPTY_STORE_FOR_CAPABILITY = {
    "community_summaries": "community_summaries",
    "entity_graph": "entities",
    "vector_index": "chunks",
}

_FULL_STATS = GraphStats(
    entities=1, relations=1, chunks=1, community_summaries=1
)


class StubBackend(SearchBackend):
    """Canned backend. Which modes fail is driven by
    ``RAGU_API_STUB_MISSING_CAPABILITIES``."""

    def __init__(self, settings: ServiceSettings | None = None):
        super().__init__(settings or ServiceSettings(backend="stub"))
        self._missing = self.settings.missing_capabilities()

    async def startup(self) -> None:
        self._stats = self._simulated_stats()
        logger.warning(
            "StubBackend started: canned answers only, no real knowledge graph"
        )

    def _simulated_stats(self) -> GraphStats:
        """
        Build graph sizes that reproduce the configured missing capabilities.

        :return: Sizes where each simulated-missing store counts zero.
        """
        empty = {
            _EMPTY_STORE_FOR_CAPABILITY[capability]
            for capability in self._missing
            if capability in _EMPTY_STORE_FOR_CAPABILITY
        }
        return GraphStats(
            entities=0 if "entities" in empty else _FULL_STATS.entities,
            relations=_FULL_STATS.relations,
            chunks=0 if "chunks" in empty else _FULL_STATS.chunks,
            community_summaries=(
                0 if "community_summaries" in empty else _FULL_STATS.community_summaries
            ),
        )

    @staticmethod
    def _subqueries(query: str, use_query_plan: bool) -> list[SubqueryItem]:
        if not use_query_plan:
            return []
        return [SubqueryItem(query=query, answer=f"stub answer for '{query}'")]

    def _sources_for(self, call: SearchCall) -> list[SourceItem]:
        """
        Canned retrieval for one mode, shaped by the request parameters.
        """
        if call.mode == "global":
            params = self.bound_params(call.params or GlobalSearchParams())
            return [
                SourceItem(
                    id="community_1",
                    type="community_summary",
                    content=f"stub community summary (min_cluster_size={params.min_cluster_size})",
                )
            ]

        if call.mode == "local":
            params = self.bound_params(call.params or LocalParams())
            sources = [SourceItem(id="entity_1", type="entity", content="stub entity")]
            if params.use_chunks:
                sources.append(
                    SourceItem(
                        id="chunk_1", type="chunk", content="stub chunk", score=0.87
                    )
                )
            if params.use_summary:
                sources.append(
                    SourceItem(
                        id="community_1",
                        type="community_summary",
                        content="stub community summary",
                    )
                )
            return sources

        if call.mode == "mix":
            naive_params = self.bound_params(call.naive_params or NaiveSearchParams())
            return [
                SourceItem(id="entity_1", type="entity", content="stub entity")
            ] + self._chunks(naive_params.top_k)

        params = self.bound_params(call.params or NaiveSearchParams())
        return self._chunks(params.top_k)

    @staticmethod
    def _chunks(top_k: int) -> list[SourceItem]:
        return [
            SourceItem(
                id=f"chunk_{i}",
                type="chunk",
                content=f"stub chunk {i}",
                score=round(0.9 - i / 100, 2),
            )
            for i in range(1, top_k + 1)
        ]

    def _children_for(self, call: SearchCall) -> list[ChildEngineReport]:
        if call.mode != "mix":
            return []
        return [
            ChildEngineReport(engine="LocalSearchEngine", mode="local", ok=True),
            ChildEngineReport(engine="NaiveSearchEngine", mode="naive", ok=True),
        ]

    def _report(self, call: SearchCall, *, query_plan: bool) -> EngineReport:
        children = self._children_for(call)
        return EngineReport(
            requested=call.mode,
            used="StubBackend",
            query_plan=query_plan,
            degraded=any(not child.ok for child in children),
            children=children,
        )

    async def search(self, call: SearchCall) -> list[SearchOutcome]:
        self.require_capability(call.mode)
        report = self._report(call, query_plan=call.use_query_plan)
        return [
            SearchOutcome(
                answer=f"[stub {call.mode}] {query}",
                sources=self._sources_for(call),
                subqueries=self._subqueries(query, call.use_query_plan),
                engines=report,
            )
            for query in call.queries
        ]

    async def retrieve(self, call: SearchCall) -> list[RetrieveOutcome]:
        self.require_capability(call.mode)
        report = self._report(call, query_plan=False)
        return [
            RetrieveOutcome(sources=self._sources_for(call), engines=report)
            for _ in call.queries
        ]
