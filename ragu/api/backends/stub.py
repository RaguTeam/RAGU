"""
In-memory backend for local development and tests.

Serves deterministic canned answers without a real graph, so the agent side can
be exercised end-to-end without building a knowledge graph.
"""

import logging

from ragu.api.backends.base import (
    GLOBAL_UNAVAILABLE_MESSAGE,
    LOCAL_UNAVAILABLE_MESSAGE,
    NAIVE_UNAVAILABLE_MESSAGE,
    SearchBackend,
    SearchOutcome,
)
from ragu.api.config import ServiceSettings
from ragu.api.errors import CapabilityUnavailableError
from ragu.api.models import SourceItem, SubqueryItem
from ragu.search_engine.global_search import GlobalSearchParams
from ragu.search_engine.local_search import LocalParams
from ragu.search_engine.naive_search import NaiveSearchParams

logger = logging.getLogger(__name__)


class StubBackend(SearchBackend):
    """Canned backend. Which modes fail is driven by
    ``RAGU_STUB_MISSING_CAPABILITIES``."""

    def __init__(self, settings: ServiceSettings | None = None):
        self.settings = settings or ServiceSettings(backend="stub")
        self._missing = self.settings.missing_capabilities()
        self._loaded = False

    @property
    def graph_loaded(self) -> bool:
        return self._loaded

    async def startup(self) -> None:
        self._loaded = True
        logger.warning(
            "StubBackend started: canned answers only, no real knowledge graph"
        )

    @staticmethod
    def _subqueries(query: str, use_query_plan: bool) -> list[SubqueryItem]:
        if not use_query_plan:
            return []
        return [SubqueryItem(query=query, answer=f"stub answer for '{query}'")]

    async def search_global(
        self, 
        query: str, 
        *, 
        params: GlobalSearchParams
    ) -> SearchOutcome:
        if "community_summaries" in self._missing:
            raise CapabilityUnavailableError(
                GLOBAL_UNAVAILABLE_MESSAGE,
                mode="global",
                missing_capability="community_summaries",
            )
        return SearchOutcome(
            answer=f"[stub global] {query}",
            sources=[
                SourceItem(
                    id="community_1",
                    type="community_summary",
                    content=f"stub community summary (min_cluster_size={params.min_cluster_size})",
                )
            ],
        )

    async def search_local(
        self, 
        query: str, 
        *, 
        use_query_plan: bool, 
        params: LocalParams
    ) -> SearchOutcome:
        if "entity_graph" in self._missing:
            raise CapabilityUnavailableError(
                LOCAL_UNAVAILABLE_MESSAGE,
                mode="local",
                missing_capability="entity_graph",
            )
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
        return SearchOutcome(
            answer=f"[stub local] {query}",
            sources=sources,
            subqueries=self._subqueries(query, use_query_plan),
        )

    async def search_naive(
        self, 
        query: str, 
        *, 
        use_query_plan: bool, 
        params: NaiveSearchParams
    ) -> SearchOutcome:
        if "vector_index" in self._missing:
            raise CapabilityUnavailableError(
                NAIVE_UNAVAILABLE_MESSAGE, mode="naive", missing_capability=None
            )
        sources = [
            SourceItem(
                id=f"chunk_{i}",
                type="chunk",
                content=f"stub chunk {i}",
                score=round(0.9 - i / 100, 2),
            )
            for i in range(1, params.top_k + 1)
        ]
        return SearchOutcome(
            answer=f"[stub naive] {query}",
            sources=sources,
            subqueries=self._subqueries(query, use_query_plan),
        )
