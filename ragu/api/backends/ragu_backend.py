"""
Adapter over the RAGU engines.
"""

import logging
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from ragu.api.backends.base import (
    GLOBAL_UNAVAILABLE_MESSAGE,
    LOCAL_UNAVAILABLE_MESSAGE,
    NAIVE_UNAVAILABLE_MESSAGE,
    SearchBackend,
    SearchOutcome,
)
from ragu.api.config import ServiceSettings
from ragu.api.errors import (
    BackendExecutionError,
    CapabilityUnavailableError,
    ServiceNotReadyError,
)
from ragu.api.models import SourceItem, SubqueryItem
from ragu.search_engine.global_search import GlobalSearchParams
from ragu.search_engine.local_search import LocalParams
from ragu.search_engine.naive_search import NaiveSearchParams
from ragu.search_engine.query_plan import QueryPlanEngine

logger = logging.getLogger(__name__)

_NO_EVIDENCE = {
    "global": (GLOBAL_UNAVAILABLE_MESSAGE, "community_summaries"),
    "local": (LOCAL_UNAVAILABLE_MESSAGE, "entity_graph"),
    "naive": (NAIVE_UNAVAILABLE_MESSAGE, None),
}

class RaguBackend(SearchBackend):
    """
    Real backend: a loaded graph plus one engine per search mode.
    """

    def __init__(self, settings: ServiceSettings):
        self.settings = settings
        self.graph: Any = None
        self._engines: dict[str, Any] = {}

    @property
    def graph_loaded(self) -> bool:
        return self.graph is not None

    async def startup(self) -> None:
        try:
            from ragu import (
                GlobalSearchEngine,
                KnowledgeGraph,
                LocalSearchEngine,
                NaiveSearchEngine,
                Settings,
            )
            from ragu.common.env import Env
            from ragu.models.embedder import EmbedderOpenAI
            from ragu.models.llm import LLMOpenAI
            from ragu.models.openai import CachedAsyncOpenAI
        except ImportError as exc:  # pragma: no cover - depends on the deployment env
            raise ServiceNotReadyError(
                "The 'ragu' package is not importable. Install it or run the service with RAGU_API_BACKEND=stub."
            ) from exc

        settings = self.settings
        if settings.settings_file:
            Settings.load(settings.settings_file)
        Settings.storage_folder = settings.storage_folder
        Settings.language = settings.language

        try:
            env = Env.from_env()
        except Exception as exc:
            raise ServiceNotReadyError(
                "LLM credentials are missing: set LLM_MODEL_NAME, LLM_BASE_URL and LLM_API_KEY "
                f"in the environment or in .env ({exc})."
            ) from exc

        llm_client = CachedAsyncOpenAI(
            base_url=env.llm_base_url,
            api_key=env.llm_api_key,
            rate_min_delay=settings.rate_min_delay,
            rate_max_simultaneous=settings.rate_max_simultaneous,
            cache=settings.llm_cache,
        )
        embedder_client = (
            llm_client
            if not env.embedder_base_url
            else CachedAsyncOpenAI(
                base_url=env.embedder_base_url,
                api_key=env.embedder_api_key or env.llm_api_key,
            )
        )

        llm = LLMOpenAI(client=llm_client, model_name=env.llm_model_name)
        embedder = EmbedderOpenAI(
            client=embedder_client,
            model_name=env.embedder_model_name or env.llm_model_name,
            dim=settings.embedder_dim,
        )
        if settings.embedder_dim is None:
            try:
                await embedder.initialize()
            except Exception as exc:
                raise ServiceNotReadyError(
                    "Could not detect the embedding dimension: the embedder endpoint is unreachable. "
                    "Set RAGU_API_EMBEDDER_DIM to the dimension the graph was built with, "
                    f"or make the endpoint reachable ({exc})."
                ) from exc

        try:
            graph = KnowledgeGraph(
                llm=llm, embedder=embedder, language=settings.language
            )
        except Exception as exc:
            raise ServiceNotReadyError(
                f"Failed to load the graph from '{settings.storage_folder}': {exc}"
            ) from exc

        self.graph = graph
        self._engines = {
            "global": GlobalSearchEngine(
                llm=llm, knowledge_graph=graph, language=settings.language
            ),
            "local": LocalSearchEngine(
                llm=llm,
                knowledge_graph=graph,
                embedder=embedder,
                language=settings.language,
            ),
            "naive": NaiveSearchEngine(
                llm=llm,
                knowledge_graph=graph,
                embedder=embedder,
                language=settings.language,
            ),
        }
        logger.info(
            f"Graph loaded from '{settings.storage_folder}' (language={settings.language})"
        )

    async def _query(
        self, mode: str, 
        query: str, 
        params: Any, 
        *, 
        use_query_plan: bool
    ) -> SearchOutcome:
        if not self.graph_loaded:
            raise ServiceNotReadyError("Knowledge graph is not loaded yet.", mode=mode)
        engine = self._engines[mode]
        try:
            if use_query_plan:
                engine = QueryPlanEngine(engine)
            response = await engine.query(query, params)
        except Exception as exc:
            logger.exception(f"RAGU {mode} search failed")
            raise BackendExecutionError(
                f"{mode} search failed: {exc}", mode=mode
            ) from exc

        outcome = to_outcome(response)
        if not outcome.sources:
            # TODO: when the builder records graph metadata (entity/chunk counts,
            # make_community_summary), read it at startup and refuse an
            # unsupported mode before spending a generation call on it.
            raise CapabilityUnavailableError(
                _NO_EVIDENCE[mode][0],
                mode=mode,
                missing_capability=_NO_EVIDENCE[mode][1],
            )
        return outcome

    def _capped(self, params: Any) -> Any:
        """
        Clamp client-supplied ``top_k`` to what this service will serve.

        The engine parameter classes are plain dataclasses with no bounds, and
        they now arrive straight from the request body, so an unbounded
        ``top_k`` would otherwise reach the retriever.
        """
        top_k = getattr(params, "top_k", None)
        if isinstance(top_k, int) and top_k > self.settings.max_top_k:
            logger.info(f"Clamping top_k {top_k} to {self.settings.max_top_k}")
            return replace(params, top_k=self.settings.max_top_k)
        return params

    async def search_global(
        self, 
        query: str,
        *, 
        params: GlobalSearchParams
    ) -> SearchOutcome:
        return await self._query("global", query, params, use_query_plan=False)

    async def search_local(
        self, 
        query: str, 
        *, 
        use_query_plan: bool, 
        params: LocalParams
    ) -> SearchOutcome:
        return await self._query(
            "local", query, self._capped(params), use_query_plan=use_query_plan
        )

    async def search_naive(
        self, 
        query: str, 
        *, 
        use_query_plan: bool, 
        params: NaiveSearchParams
    ) -> SearchOutcome:
        return await self._query(
            "naive", query, self._capped(params), use_query_plan=use_query_plan
        )

def _field(item: Any, name: str, default: Any = None) -> Any:
    """
    Read a field from an engine result item, dict or object alike.
    """
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


_KNOWN_RESULT_FIELDS = ("insights", "entities", "relations", "summaries", "chunks")


def _sources_from_result(result: Any) -> list[SourceItem]:
    """
    Flatten an engine-specific retrieval result into flat sources.

    Handles the three result types the engines return today
    (``LocalSearchResult``, ``NaiveSearchResult``, ``GlobalSearchResult``) and
    leaves anything else to the caller's fallback.
    """
    sources: list[SourceItem] = []

    for index, insight in enumerate(getattr(result, "insights", []) or [], start=1):
        sources.append(
            SourceItem(
                id=f"insight_{index}",
                type="community_summary",
                content=str(_field(insight, "response", "")),
                score=_as_score(_field(insight, "rating")),
            )
        )

    for entity in getattr(result, "entities", []) or []:
        sources.append(
            SourceItem(
                id=str(getattr(entity, "id", ""))
                or str(getattr(entity, "entity_name", "entity")),
                type="entity",
                content=f"{getattr(entity, 'entity_name', '')} — {getattr(entity, 'description', '')}".strip(
                    " —"
                ),
            )
        )

    for relation in getattr(result, "relations", []) or []:
        sources.append(
            SourceItem(
                id=str(getattr(relation, "id", "")) or "relation",
                type="relation",
                content=(
                    f"{getattr(relation, 'subject_name', '')} "
                    f"{getattr(relation, 'relation_type', '')} "
                    f"{getattr(relation, 'object_name', '')}: {getattr(relation, 'description', '')}"
                ).strip(),
            )
        )

    for index, summary in enumerate(getattr(result, "summaries", []) or [], start=1):
        sources.append(
            SourceItem(
                id=str(_field(summary, "id", "")) or f"summary_{index}",
                type="community_summary",
                content=str(_field(summary, "summary", summary)),
            )
        )

    scores = list(getattr(result, "scores", []) or [])
    for index, chunk in enumerate(getattr(result, "chunks", []) or []):
        sources.append(
            SourceItem(
                id=str(getattr(chunk, "id", "")) or f"chunk_{index + 1}",
                type="chunk",
                content=str(getattr(chunk, "content", chunk)),
                score=_as_score(scores[index]) if index < len(scores) else None,
            )
        )

    return sources


def _as_score(value: Any) -> float | None:
    return (
        float(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else None
    )


def extract_sources(retrieval: Any) -> list[SourceItem]:
    """Convert a ``SearchEngineRetrieve`` into flat sources."""
    if retrieval is None:
        return []

    result = getattr(retrieval, "result", None)
    if any(hasattr(result, field) for field in _KNOWN_RESULT_FIELDS):
        return _sources_from_result(result)

    # Unknown result type (a new engine, or GST): keep the rendered context as
    # one source rather than dropping the evidence.
    to_text = getattr(retrieval, "to_text", None)
    text = to_text() if callable(to_text) else None
    if text and text.strip():
        return [SourceItem(id="retrieval_context", type="context", content=str(text))]
    return []


def extract_subqueries(payload: Any) -> list[SubqueryItem]:
    """
    Pull the query plan's intermediate answers out of the payload.

    ``QueryPlanEngine.batch_query`` sets ``payload`` to ``{subquery_id:
    SearchEngineResponse}``; the streaming path nests the same mapping under
    ``answers``.
    """
    if not isinstance(payload, dict):
        return []

    answers = (
        payload.get("answers") if isinstance(payload.get("answers"), dict) else payload
    )

    subqueries = []
    for item in answers.values():
        query = getattr(item, "query", None)
        if not query:
            continue
        subqueries.append(
            SubqueryItem(
                query=str(query), answer=_response_text(getattr(item, "response", ""))
            )
        )
    return subqueries


def _response_text(response: Any) -> str:
    """Engines may answer with a pydantic model instead of a string."""
    if response is None:
        return ""
    dump = getattr(response, "model_dump_json", None)
    return dump() if callable(dump) else str(response)


def to_outcome(response: Any) -> SearchOutcome:
    return SearchOutcome(
        answer=_response_text(getattr(response, "response", "")),
        sources=extract_sources(getattr(response, "retrieval", None)),
        subqueries=extract_subqueries(getattr(response, "payload", None)),
    )
