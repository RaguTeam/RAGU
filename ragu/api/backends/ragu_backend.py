"""
Adapter over the RAGU engines.
"""

import asyncio
import os
from typing import Any

from ragu import (
    CachedAsyncOpenAI,
    EmbedderOpenAI,
    Env,
    GlobalSearchEngine,
    KnowledgeGraph,
    LLMOpenAI,
    LocalSearchEngine,
    NaiveSearchEngine,
    Settings,
)
from ragu.api.backends.base import GraphStats, SearchBackend, SearchOutcome
from ragu.api.config import ServiceSettings
from ragu.api.errors import BackendExecutionError, ServiceNotReadyError
from ragu.api.mapping import to_outcome
from ragu.api.models import SearchMode
from ragu.common.logger import logger
from ragu.search_engine.base_engine import BaseEngine
from ragu.search_engine.global_search import GlobalSearchParams
from ragu.search_engine.local_search import LocalParams
from ragu.search_engine.naive_search import NaiveSearchParams
from ragu.search_engine.query_plan import QueryPlanEngine


class RaguBackend(SearchBackend):
    """
    Real backend: a loaded graph plus one engine per search mode.
    """

    def __init__(self, settings: ServiceSettings):
        super().__init__(settings)
        self.graph: KnowledgeGraph | None = None
        self._engines: dict[SearchMode, BaseEngine[Any, Any]] = {}
        self._clients: list[CachedAsyncOpenAI] = []

    async def startup(self) -> None:
        """
        Load the graph and build one engine per search mode.

        :raises ServiceNotReadyError: If credentials, the embedder endpoint or
            the storage folder make the graph unusable.
        """
        settings = self.settings

        self._require_storage_folder(settings.storage_folder)

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
        self._clients.append(llm_client)

        embedder_client = llm_client
        if env.embedder_base_url:
            embedder_client = CachedAsyncOpenAI(
                base_url=env.embedder_base_url,
                api_key=env.embedder_api_key or env.llm_api_key,
            )
            self._clients.append(embedder_client)

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
            # Constructing the graph opens every storage and reads the whole
            # graph file, which takes minutes on a large corpus. Off the event
            # loop, so /health keeps answering while it happens.
            graph = await asyncio.to_thread(
                KnowledgeGraph,
                llm=llm,
                embedder=embedder,
                language=settings.language,
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
        self._stats = await self._measure(graph)

        if self._stats.is_empty:
            raise ServiceNotReadyError(
                f"The graph at '{settings.storage_folder}' is empty: no entities, chunks or "
                "community summaries. Point RAGU_API_STORAGE_FOLDER at a built graph."
            )

        logger.info(
            "Graph loaded from '{}' (language={}): {}",
            settings.storage_folder,
            settings.language,
            self._stats,
        )

    async def shutdown(self) -> None:
        """
        Close the graph storages and the model clients.

        File-backed storages do nothing here, but server-backed ones (Neo4j,
        remote Qdrant) and the HTTP pool behind every model client stay open
        until closed.
        """
        if self.graph is not None:
            try:
                await self.graph.index.close()
            except Exception:
                logger.exception("Failed to close the graph index")
        for client in self._clients:
            try:
                await client.client.close()
            except Exception:
                logger.exception("Failed to close a model client")
        self._clients.clear()
        self.graph = None
        self._stats = None

    @staticmethod
    def _require_storage_folder(storage_folder: str) -> None:
        """
        Reject a storage folder that holds no graph.

        ``Index.__init__`` calls ``Settings.init_storage_folder()``, which
        *creates* the folder when it is missing. Without this check a typo in
        ``RAGU_API_STORAGE_FOLDER`` produces an empty directory and a backend
        that looks healthy while every search comes back empty.

        :param storage_folder: Folder the service was pointed at.
        :raises ServiceNotReadyError: If it is missing, not a directory, or empty.
        """
        if not os.path.isdir(storage_folder):
            raise ServiceNotReadyError(
                f"Storage folder '{storage_folder}' does not exist. Set "
                "RAGU_API_STORAGE_FOLDER to the folder a build run produced; the service "
                "serves a prebuilt graph and never builds one."
            )
        if not os.listdir(storage_folder):
            raise ServiceNotReadyError(
                f"Storage folder '{storage_folder}' is empty. Set RAGU_API_STORAGE_FOLDER "
                "to the folder a build run produced."
            )

    @staticmethod
    async def _measure(graph: KnowledgeGraph) -> GraphStats:
        """
        Count what each search mode has to read.

        Vector-store ids are used rather than materialized entities: the counts
        only decide which modes can run, and listing ids stays cheap on a large
        graph.

        :param graph: The loaded graph.
        :return: Sizes of the stores behind the three search modes.
        """
        index = graph.index
        entities, relations, chunks, summaries = await asyncio.gather(
            index.nodes_vector_db.get_all_ids(),
            index.edges_vector_db.get_all_ids(),
            index.chunks_kv_storage.all_keys(),
            index.community_summary_kv_storage.all_keys(),
        )
        return GraphStats(
            entities=len(entities),
            relations=len(relations),
            chunks=len(chunks),
            community_summaries=len(summaries),
        )

    async def _query(
        self,
        mode: SearchMode,
        query: str,
        params: Any,
        *,
        use_query_plan: bool,
    ) -> SearchOutcome:
        """
        Run one search and normalize the engine response.

        :param mode: Search mode to run.
        :param query: Client query.
        :param params: Engine parameters, already clamped.
        :param use_query_plan: Whether to decompose the query first.
        :return: Normalized outcome.
        :raises ServiceNotReadyError: If the graph is not loaded.
        :raises CapabilityUnavailableError: If the graph cannot serve this mode,
            or this query retrieved nothing.
        :raises BackendExecutionError: If the engine itself failed.
        """
        if not self.graph_loaded:
            raise ServiceNotReadyError("Knowledge graph is not loaded yet.", mode=mode)

        # Refuse before generating: the engines answer confidently from an empty
        # context, and a generation call on a graph that cannot serve the mode
        # is paid for and then discarded.
        self.require_capability(mode)

        engine: BaseEngine[Any, Any] = self._engines[mode]
        try:
            if use_query_plan:
                engine = QueryPlanEngine(engine)
            response = await engine.query(query, params)
        except Exception as exc:
            logger.exception("RAGU {} search failed", mode)
            raise BackendExecutionError(mode=mode, detail=str(exc)) from exc

        answer, sources, subqueries = to_outcome(
            response, used_query_plan=use_query_plan
        )
        if not sources:
            raise self.no_evidence(mode)
        return SearchOutcome(answer=answer, sources=sources, subqueries=subqueries)

    async def search_global(
        self, query: str, *, params: GlobalSearchParams
    ) -> SearchOutcome:
        return await self._query(
            "global", query, self.bound_params(params), use_query_plan=False
        )

    async def search_local(
        self, query: str, *, use_query_plan: bool, params: LocalParams
    ) -> SearchOutcome:
        return await self._query(
            "local", query, self.bound_params(params), use_query_plan=use_query_plan
        )

    async def search_naive(
        self, query: str, *, use_query_plan: bool, params: NaiveSearchParams
    ) -> SearchOutcome:
        return await self._query(
            "naive", query, self.bound_params(params), use_query_plan=use_query_plan
        )
