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
    MixSearchEngine,
    NaiveSearchEngine,
    Settings,
)
from ragu.api.backends.base import GraphStats, SearchBackend, SearchOutcome
from ragu.api.config import ServiceSettings
from ragu.api.errors import BackendExecutionError, ServiceNotReadyError
from ragu.api.mapping import to_outcome
from ragu.api.models import ChildEngineReport, EngineReport, SearchMode
from ragu.common.logger import logger
from ragu.models.llm import LLM
from ragu.search_engine.base_engine import BaseEngine
from ragu.search_engine.global_search import GlobalSearchParams
from ragu.search_engine.local_search import LocalParams
from ragu.search_engine.mix_search import MixQueryParams
from ragu.search_engine.naive_search import NaiveSearchParams
from ragu.search_engine.query_plan import QueryPlanEngine


class RecordingEngine:
    """
    Child-engine proxy that remembers whether the child actually contributed.

    ``MixSearchEngine`` is constructed with ``allow_partial_failures=True``: a
    child that raises is logged and dropped from the ensemble, so from the
    outside "graph and chunks" is indistinguishable from "chunks only". This
    records the failure on the way past, before the ensemble swallows it.
    """

    def __init__(self, engine: BaseEngine[Any, Any], mode: SearchMode):
        self.engine = engine
        self.mode = mode
        self.error: str | None = None
        self.called = False

    @property
    def llm(self) -> LLM:
        return self.engine.llm

    async def batch_search(self, queries: list[str], params: Any = None) -> Any:
        return await self._record(self.engine.batch_search, queries, params)

    async def batch_query(self, queries: list[str], params: Any = None) -> Any:
        return await self._record(self.engine.batch_query, queries, params)

    async def _record(self, call: Any, queries: list[str], params: Any) -> Any:
        self.called = True
        try:
            return await call(queries, params)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            raise

    def report(self) -> ChildEngineReport:
        """
        Describe what this child did, for the response's engine report.
        """
        return ChildEngineReport(
            engine=type(self.engine).__name__,
            mode=self.mode,
            ok=self.called and self.error is None,
            error=self.error,
        )


class RaguBackend(SearchBackend):
    """
    Real backend: a loaded graph plus one engine per search mode.
    """

    def __init__(self, settings: ServiceSettings):
        super().__init__(settings)
        self.graph: KnowledgeGraph | None = None
        self._engines: dict[SearchMode, BaseEngine[Any, Any]] = {}
        self._clients: list[CachedAsyncOpenAI] = []
        self._llm: LLM | None = None

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
        self._llm = llm
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
        self._llm = None
        self._engines = {}
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
        engine: BaseEngine[Any, Any],
        query: str,
        params: Any,
        *,
        use_query_plan: bool,
        children: tuple[RecordingEngine, ...] = (),
    ) -> SearchOutcome:
        """
        Run one search and normalize the engine response.

        :param mode: Search mode to run.
        :param engine: Engine that answers the query.
        :param query: Client query.
        :param params: Engine parameters, already clamped.
        :param use_query_plan: Whether to decompose the query first.
        :param children: Recording proxies of an ensemble's child engines, so
            the response can report which of them contributed.
        :return: Normalized outcome.
        :raises CapabilityUnavailableError: If this query retrieved nothing.
        :raises BackendExecutionError: If the engine itself failed.
        """
        engine_name = type(engine).__name__
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
        reports = [child.report() for child in children]
        if not sources:
            raise self.no_evidence(mode)
        return SearchOutcome(
            answer=answer,
            sources=sources,
            subqueries=subqueries,
            engines=EngineReport(
                requested=mode,
                used=engine_name,
                query_plan=use_query_plan,
                degraded=any(not report.ok for report in reports),
                children=reports,
            ),
        )

    def _ready(self, mode: SearchMode) -> BaseEngine[Any, Any]:
        """
        Resolve the engine for a mode, refusing what this graph cannot serve.

        The capability check runs before generation: the engines answer
        confidently from an empty context, so a graph that cannot serve the mode
        would otherwise be paid for and then discarded.

        :param mode: Search mode about to run.
        :return: The cached engine for that mode.
        :raises ServiceNotReadyError: If the graph is not loaded.
        :raises CapabilityUnavailableError: If the graph cannot serve the mode.
        """
        if not self.graph_loaded:
            raise ServiceNotReadyError("Knowledge graph is not loaded yet.", mode=mode)
        self.require_capability(mode)
        return self._engines[mode]

    async def search_global(
        self, query: str, *, params: GlobalSearchParams
    ) -> SearchOutcome:
        engine = self._ready("global")
        return await self._query(
            "global", engine, query, self.bound_params(params), use_query_plan=False
        )

    async def search_local(
        self, query: str, *, use_query_plan: bool, params: LocalParams
    ) -> SearchOutcome:
        engine = self._ready("local")
        return await self._query(
            "local",
            engine,
            query,
            self.bound_params(params),
            use_query_plan=use_query_plan,
        )

    async def search_naive(
        self, query: str, *, use_query_plan: bool, params: NaiveSearchParams
    ) -> SearchOutcome:
        engine = self._ready("naive")
        return await self._query(
            "naive",
            engine,
            query,
            self.bound_params(params),
            use_query_plan=use_query_plan,
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
        # MixSearchEngine reads its children's parameters from the constructor —
        # batch_search ignores the params argument entirely and batch_query reads
        # only ensemble_responses from it — so the ensemble is assembled per
        # request around the cached child engines rather than cached itself.
        children = (
            RecordingEngine(self._ready("local"), "local"),
            RecordingEngine(self._ready("naive"), "naive"),
        )
        self.require_capability("mix")
        engine = MixSearchEngine(
            llm=self._require_llm(),
            engines=list(children),
            engine_params=[
                self.bound_params(local_params),
                self.bound_params(naive_params),
            ],
            language=self.settings.language,
        )
        return await self._query(
            "mix",
            engine,
            query,
            params,
            use_query_plan=use_query_plan,
            children=children,
        )

    def _require_llm(self) -> LLM:
        """
        Return the LLM built at startup.

        :raises ServiceNotReadyError: If the backend has not started.
        """
        if self._llm is None:
            raise ServiceNotReadyError("Knowledge graph is not loaded yet.", mode="mix")
        return self._llm
