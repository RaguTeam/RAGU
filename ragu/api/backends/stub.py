"""
In-memory backend for local development and tests.

Serves deterministic canned answers without a real graph, so the agent side can
be exercised end-to-end without building a knowledge graph.
"""

from ragu.api.backends.base import GraphStats, SearchBackend, SearchOutcome
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

    def _finish(
        self,
        mode: SearchMode,
        answer: str,
        sources: list[SourceItem],
        subqueries: list[SubqueryItem],
        children: list[ChildEngineReport] | None = None,
    ) -> SearchOutcome:
        """
        Apply the same no-evidence rule the real backend applies.

        :raises CapabilityUnavailableError: If the canned retrieval is empty.
        """
        if not sources:
            raise self.no_evidence(mode)
        reports = children or []
        return SearchOutcome(
            answer=answer,
            sources=sources,
            subqueries=subqueries,
            engines=EngineReport(
                requested=mode,
                used="StubBackend",
                query_plan=bool(subqueries),
                degraded=any(not report.ok for report in reports),
                children=reports,
            ),
        )

    async def search_global(
        self, query: str, *, params: GlobalSearchParams
    ) -> SearchOutcome:
        self.require_capability("global")
        params = self.bound_params(params)
        return self._finish(
            "global",
            f"[stub global] {query}",
            [
                SourceItem(
                    id="community_1",
                    type="community_summary",
                    content=f"stub community summary (min_cluster_size={params.min_cluster_size})",
                )
            ],
            [],
        )

    async def search_local(
        self, query: str, *, use_query_plan: bool, params: LocalParams
    ) -> SearchOutcome:
        self.require_capability("local")
        params = self.bound_params(params)
        sources = [SourceItem(id="entity_1", type="entity", content="stub entity")]
        if params.use_chunks:
            sources.append(
                SourceItem(id="chunk_1", type="chunk", content="stub chunk", score=0.87)
            )
        if params.use_summary:
            sources.append(
                SourceItem(
                    id="community_1",
                    type="community_summary",
                    content="stub community summary",
                )
            )
        return self._finish(
            "local",
            f"[stub local] {query}",
            sources,
            self._subqueries(query, use_query_plan),
        )

    async def search_naive(
        self, query: str, *, use_query_plan: bool, params: NaiveSearchParams
    ) -> SearchOutcome:
        self.require_capability("naive")
        params = self.bound_params(params)
        sources = [
            SourceItem(
                id=f"chunk_{i}",
                type="chunk",
                content=f"stub chunk {i}",
                score=round(0.9 - i / 100, 2),
            )
            for i in range(1, params.top_k + 1)
        ]
        return self._finish(
            "naive",
            f"[stub naive] {query}",
            sources,
            self._subqueries(query, use_query_plan),
        )

    async def search_mix(
        self,
        query: str,
        *,
        use_query_plan: bool,
        params: MixQueryParams,
        local_params: LocalParams,
        naive_params: NaiveSearchParams,
    ) -> SearchOutcome:
        self.require_capability("mix")
        local_params = self.bound_params(local_params)
        naive_params = self.bound_params(naive_params)
        sources = [SourceItem(id="entity_1", type="entity", content="stub entity")]
        sources += [
            SourceItem(
                id=f"chunk_{i}",
                type="chunk",
                content=f"stub chunk {i}",
                score=round(0.9 - i / 100, 2),
            )
            for i in range(1, naive_params.top_k + 1)
        ]
        return self._finish(
            "mix",
            f"[stub mix] {query}",
            sources,
            self._subqueries(query, use_query_plan),
            children=[
                ChildEngineReport(engine="LocalSearchEngine", mode="local", ok=True),
                ChildEngineReport(engine="NaiveSearchEngine", mode="naive", ok=True),
            ],
        )
