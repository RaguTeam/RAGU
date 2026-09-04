"""Tests for the HTTP search API."""

from __future__ import annotations

import pytest

pytest.importorskip(
    "fastapi", reason="install the 'api' extra to test the search service"
)

from fastapi.testclient import TestClient  # noqa: E402

from ragu.api.app import UNHANDLED_ERROR_MESSAGE, create_app
from ragu.api.backends.base import GraphStats
from ragu.api.config import ServiceSettings
from ragu.api.mapping import extract_sources, extract_subqueries, to_outcome
from ragu.chunker.types import Chunk
from ragu.common.prompts.default_models import GlobalSearchContextModel
from ragu.graph.types import CommunitySummary, Entity, Relation
from ragu.search_engine.base_engine import SearchEngineResponse
from ragu.search_engine.global_search import (
    GlobalSearchParams,
    GlobalSearchResult,
    GlobalSearchRetrieve,
)
from ragu.search_engine.local_search import (
    LocalParams,
    LocalSearchResult,
    LocalSearchRetrieve,
)
from ragu.search_engine.naive_search import (
    NaiveSearchParams,
    NaiveSearchResult,
    NaiveSearchRetrieve,
)


def build_client(missing: str = "", **overrides) -> TestClient:
    settings = ServiceSettings(
        backend="stub", stub_missing_capabilities=missing, **overrides
    )
    return TestClient(create_app(settings))


def make_chunk(content: str, index: int = 0) -> Chunk:
    return Chunk(content=content, chunk_order_idx=index, doc_id="doc-1")


def make_entity(name: str) -> Entity:
    return Entity(
        entity_name=name,
        entity_type="Person",
        description=f"{name} description",
        source_chunk_id=["c1"],
    )


def make_retrieval(*contents: str) -> NaiveSearchRetrieve:
    return NaiveSearchRetrieve(
        query="sub",
        result=NaiveSearchResult(chunks=[make_chunk(c) for c in contents]),
    )


class TestHealth:
    def test_health_reports_loaded_graph(self):
        with build_client() as client:
            body = client.get("/health").json()
        assert (body["status"], body["graph_loaded"], body["error"]) == (
            "ok",
            True,
            None,
        )

    def test_health_reports_graph_sizes(self):
        with build_client() as client:
            stats = client.get("/health").json()["stats"]
        assert stats == {
            "entities": 1,
            "relations": 1,
            "chunks": 1,
            "community_summaries": 1,
        }

    def test_readiness_is_503_until_the_graph_is_loaded(self):
        # No lifespan run: the app has no backend yet.
        client = TestClient(create_app(ServiceSettings(backend="stub")))
        response = client.get("/health/ready")
        assert response.status_code == 503
        assert response.headers["Retry-After"] == "30"
        assert response.json()["graph_loaded"] is False

    def test_readiness_is_200_once_loaded(self):
        with build_client() as client:
            assert client.get("/health/ready").status_code == 200

    def test_liveness_answers_while_degraded(self):
        # Liveness must not restart a service that is still loading a graph.
        client = TestClient(create_app(ServiceSettings(backend="stub")))
        response = client.get("/health/live")
        assert response.status_code == 200
        assert response.json()["status"] == "degraded"

    def test_search_answers_503_until_the_graph_is_loaded(self):
        client = TestClient(create_app(ServiceSettings(backend="stub")))
        response = client.post("/v1/search/naive", json={"query": "q"})
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "SERVICE_NOT_READY"

    def test_a_startup_failure_is_reported_instead_of_being_swallowed(self):
        class ExplodingBackend:
            graph_loaded = False
            stats = None

            async def startup(self):
                raise RuntimeError("embedder endpoint unreachable")

            async def shutdown(self):
                pass

        with TestClient(
            create_app(ServiceSettings(backend="stub"), backend=ExplodingBackend())
        ) as client:
            health = client.get("/health").json()
            search = client.post("/v1/search/naive", json={"query": "q"})

        assert "embedder endpoint unreachable" in health["error"]
        assert "embedder endpoint unreachable" in search.json()["error"]["message"]


class TestSearchRoutes:
    def test_global_search_passes_its_params(self):
        with build_client() as client:
            body = client.post(
                "/v1/search/global",
                json={"query": "тренды", "params": {"min_cluster_size": 4}},
            ).json()
        assert "min_cluster_size=4" in body["sources"][0]["content"]

    def test_global_search_never_plans(self):
        with build_client() as client:
            body = client.post("/v1/search/global", json={"query": "тренды"}).json()
        assert body["mode"] == "global"
        assert body["used_query_plan"] is False
        assert body["subqueries"] == []
        assert body["sources"][0]["type"] == "community_summary"

    def test_global_search_rejects_the_removed_query_plan_field(self):
        # The mode has no query plan; a client still sending the field learns so
        # instead of getting an answer that quietly ignored it.
        with build_client() as client:
            response = client.post(
                "/v1/search/global", json={"query": "тренды", "use_query_plan": True}
            )
        assert response.status_code == 400
        assert "use_query_plan" in response.json()["error"]["message"]

    def test_local_search_honours_context_flags(self):
        with build_client() as client:
            body = client.post(
                "/v1/search/local",
                json={
                    "query": "кто написал роман",
                    "params": {"use_summary": False, "use_chunks": False},
                },
            ).json()
        assert body["used_query_plan"] is True
        assert [source["type"] for source in body["sources"]] == ["entity"]

    def test_naive_search_respects_top_k(self):
        with build_client() as client:
            body = client.post(
                "/v1/search/naive",
                json={"query": "версия api", "params": {"top_k": 3}},
            ).json()
        assert len(body["sources"]) == 3
        assert body["subqueries"][0]["query"] == "версия api"

    def test_omitted_params_fall_back_to_engine_defaults(self):
        # The request models embed the engine parameter classes, so a client
        # that sends no params gets LocalParams()/NaiveSearchParams() as-is —
        # including use_summary=False, which the API used to default to True.
        with build_client() as client:
            local = client.post("/v1/search/local", json={"query": "q"}).json()
            naive = client.post("/v1/search/naive", json={"query": "q"}).json()
        assert [source["type"] for source in local["sources"]] == ["entity", "chunk"]
        assert len(naive["sources"]) == NaiveSearchParams().top_k

    def test_a_misplaced_field_is_rejected_rather_than_ignored(self):
        # The pre-params flat shape: top_k belongs inside params now. Ignoring it
        # would silently serve the engine default instead of what was asked.
        with build_client() as client:
            response = client.post("/v1/search/naive", json={"query": "q", "top_k": 3})
        assert response.status_code == 400
        assert "top_k" in response.json()["error"]["message"]

    def test_unknown_engine_parameter_is_rejected(self):
        with build_client() as client:
            response = client.post(
                "/v1/search/naive", json={"query": "q", "params": {"top_k": "many"}}
            )
        assert response.status_code == 400
        assert "params.top_k" in response.json()["error"]["message"]

    def test_an_unknown_key_inside_params_is_rejected(self):
        # Documented as "silently ignored" for a long time; it is not, and a
        # client that mistypes a parameter must not be served the default.
        with build_client() as client:
            response = client.post(
                "/v1/search/naive", json={"query": "q", "params": {"topk": 3}}
            )
        assert response.status_code == 400
        assert "params.topk" in response.json()["error"]["message"]

    @pytest.mark.parametrize(
        "path, payload",
        [
            ("/v1/search/global", {"query": ""}),
            ("/v1/search/local", {}),
            ("/v1/search/naive", {"query": "q", "params": {"top_k": []}}),
        ],
    )
    def test_invalid_requests_answer_400(self, path, payload):
        with build_client() as client:
            response = client.post(path, json=payload)
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "INVALID_REQUEST"


class TestRequestBounds:
    """Service-level ceilings apply to every backend, not just the real one."""

    def test_top_k_is_clamped_for_the_stub_too(self):
        # The bound used to live in RaguBackend, so the stub happily allocated
        # one SourceItem per requested top_k.
        with build_client(max_top_k=5) as client:
            body = client.post(
                "/v1/search/naive",
                json={"query": "q", "params": {"top_k": 10_000}},
            ).json()
        assert len(body["sources"]) == 5

    def test_min_cluster_size_is_raised_to_the_floor(self):
        # Global rates every surviving community with its own LLM call, so the
        # floor is the only cap on the cost of one request.
        with build_client(min_cluster_size_floor=7) as client:
            body = client.post(
                "/v1/search/global",
                json={"query": "q", "params": {"min_cluster_size": 1}},
            ).json()
        assert "min_cluster_size=7" in body["sources"][0]["content"]

    def test_a_request_within_the_bounds_is_untouched(self):
        with build_client(max_top_k=100, min_cluster_size_floor=1) as client:
            body = client.post(
                "/v1/search/global",
                json={"query": "q", "params": {"min_cluster_size": 3}},
            ).json()
        assert "min_cluster_size=3" in body["sources"][0]["content"]


class TestCapabilityErrors:
    def test_missing_community_summaries_answers_409(self):
        with build_client(missing="community_summaries") as client:
            response = client.post("/v1/search/global", json={"query": "q"})
        assert response.status_code == 409
        error = response.json()["error"]
        assert error["code"] == "CAPABILITY_UNAVAILABLE"
        assert error["mode"] == "global"
        assert error["missing_capability"] == "community_summaries"

    def test_missing_entity_graph_answers_409(self):
        with build_client(missing="entity_graph") as client:
            error = client.post("/v1/search/local", json={"query": "q"}).json()["error"]
        assert error["missing_capability"] == "entity_graph"

    def test_missing_chunk_index_answers_409(self):
        with build_client(missing="vector_index") as client:
            error = client.post("/v1/search/naive", json={"query": "q"}).json()["error"]
        assert error["missing_capability"] == "vector_index"

    def test_other_modes_stay_available(self):
        with build_client(missing="community_summaries") as client:
            assert (
                client.post("/v1/search/naive", json={"query": "q"}).status_code == 200
            )

    def test_a_query_without_evidence_names_no_capability(self):
        # The graph supports the mode; this particular query retrieved nothing.
        with build_client() as client:
            response = client.post(
                "/v1/search/naive", json={"query": "q", "params": {"top_k": 0}}
            )
        assert response.status_code == 409
        assert response.json()["error"]["missing_capability"] is None

    def test_a_graph_with_nothing_at_all_is_not_ready(self):
        missing = "community_summaries,entity_graph,vector_index"
        with build_client(missing=missing) as client:
            assert client.get("/health/ready").status_code == 503
            assert (
                client.post("/v1/search/naive", json={"query": "q"}).status_code == 503
            )


class TestErrorEnvelope:
    def test_an_unhandled_error_does_not_leak_its_message(self):
        # Engine and LLM-client errors quote the endpoint URL and parts of the
        # request; only the log may see them.
        class ExplodingBackend:
            graph_loaded = True
            stats = None

            async def startup(self):
                pass

            async def shutdown(self):
                pass

            async def search_naive(self, query, *, use_query_plan, params):
                raise RuntimeError("https://llm.internal/v1 rejected sk-secret")

        app = create_app(ServiceSettings(backend="stub"), backend=ExplodingBackend())
        with TestClient(app, raise_server_exceptions=False) as client:
            body = client.post("/v1/search/naive", json={"query": "q"}).json()

        assert body["error"]["message"] == UNHANDLED_ERROR_MESSAGE
        assert "sk-secret" not in str(body)

    def test_every_error_shares_one_envelope_shape(self):
        with build_client(missing="entity_graph") as client:
            errors = [
                client.post("/v1/search/local", json={"query": "q"}).json()["error"],
                client.post("/v1/search/naive", json={"query": ""}).json()["error"],
            ]
        for error in errors:
            assert set(error) == {"code", "mode", "missing_capability", "message"}


class TestResponseConversion:
    """The adapter is pinned to the real engine result types."""

    def test_naive_chunks_keep_their_ids_and_scores(self):
        chunks = [make_chunk("first"), make_chunk("second", 1)]
        retrieval = NaiveSearchRetrieve(
            query="q",
            result=NaiveSearchResult(
                chunks=chunks, scores=[0.9, 0.7], documents_id=["doc-1"]
            ),
        )
        sources = extract_sources(retrieval)
        assert [source.type for source in sources] == ["chunk", "chunk"]
        assert [source.id for source in sources] == [chunk.id for chunk in chunks]
        assert [source.score for source in sources] == [0.9, 0.7]
        assert sources[0].content == "first"

    def test_local_result_yields_entities_relations_summaries_and_chunks(self):
        entity = make_entity("Сенкевич")
        relation = Relation(
            subject_id="a",
            object_id="b",
            subject_name="Сенкевич",
            object_name="Польша",
            relation_type="born_in",
            description="родился в",
        )
        retrieval = LocalSearchRetrieve(
            query="q",
            result=LocalSearchResult(
                entities=[entity],
                relations=[relation],
                summaries=[CommunitySummary(id="com-1", summary="сводка")],
                chunks=[make_chunk("chunk text")],
            ),
        )
        sources = extract_sources(retrieval)
        assert [source.type for source in sources] == [
            "entity",
            "relation",
            "community_summary",
            "chunk",
        ]
        assert sources[0].id == entity.id
        assert "Сенкевич" in sources[1].content
        assert sources[2].content == "сводка"

    def test_global_insights_become_community_summaries_with_ratings(self):
        insight = GlobalSearchContextModel(
            reasoning="r", response="общий вывод", rating=8.0
        )
        retrieval = GlobalSearchRetrieve(
            query="q", result=GlobalSearchResult(insights=[insight])
        )
        sources = extract_sources(retrieval)
        assert (sources[0].type, sources[0].content, sources[0].score) == (
            "community_summary",
            "общий вывод",
            8.0,
        )

    def test_empty_retrieval_yields_no_sources(self):
        retrieval = NaiveSearchRetrieve(query="q", result=NaiveSearchResult())
        assert extract_sources(retrieval) == []

    def test_an_unmodelled_result_keeps_its_rendered_context(self):
        class UnknownResult:
            pass

        class UnknownRetrieve:
            result = UnknownResult()

            def to_text(self):
                return "rendered context"

        sources = extract_sources(UnknownRetrieve())
        assert [(s.id, s.type, s.content) for s in sources] == [
            ("retrieval_context", "context", "rendered context")
        ]

    def test_query_plan_payload_becomes_subqueries(self):
        retrieval = make_retrieval("text")
        payload = {
            "sq-1": SearchEngineResponse(
                query="Кто написал?", response="Сенкевич", retrieval=retrieval
            ),
            "sq-2": SearchEngineResponse(
                query="Из какой страны?", response="Польша", retrieval=retrieval
            ),
        }
        assert [(item.query, item.answer) for item in extract_subqueries(payload)] == [
            ("Кто написал?", "Сенкевич"),
            ("Из какой страны?", "Польша"),
        ]

    def test_a_response_without_a_plan_has_no_subqueries(self):
        answer, sources, subqueries = to_outcome(
            SearchEngineResponse(
                query="q", response="ответ", retrieval=make_retrieval("text")
            ),
            used_query_plan=False,
        )
        assert answer == "ответ"
        assert subqueries == []
        assert len(sources) == 1

    def test_a_plan_keeps_the_evidence_of_every_subquery(self):
        # The plan answers the top-level query from the sink subquery alone, so
        # without this the evidence behind the other subqueries is dropped even
        # though their answers are returned.
        sink_retrieval = make_retrieval("sink evidence")
        payload = {
            "sq-1": SearchEngineResponse(
                query="Кто написал?",
                response="Сенкевич",
                retrieval=make_retrieval("branch evidence"),
            ),
            "sq-2": SearchEngineResponse(
                query="Итог?", response="ответ", retrieval=sink_retrieval
            ),
        }
        _, sources, subqueries = to_outcome(
            SearchEngineResponse(
                query="q", response="ответ", retrieval=sink_retrieval, payload=payload
            ),
            used_query_plan=True,
        )
        assert len(subqueries) == 2
        assert {source.content for source in sources} == {
            "sink evidence",
            "branch evidence",
        }

    def test_sources_shared_between_subqueries_appear_once(self):
        shared = make_retrieval("shared evidence")
        payload = {
            "sq-1": SearchEngineResponse(query="a", response="a", retrieval=shared),
            "sq-2": SearchEngineResponse(query="b", response="b", retrieval=shared),
        }
        _, sources, _ = to_outcome(
            SearchEngineResponse(
                query="q", response="ответ", retrieval=shared, payload=payload
            ),
            used_query_plan=True,
        )
        assert len(sources) == 1

    def test_an_empty_sink_still_reports_the_subquery_evidence(self):
        # An empty sink retrieval used to make the whole request a 409 even when
        # the branches had retrieved plenty.
        payload = {
            "sq-1": SearchEngineResponse(
                query="Кто написал?",
                response="Сенкевич",
                retrieval=make_retrieval("branch evidence"),
            )
        }
        _, sources, _ = to_outcome(
            SearchEngineResponse(
                query="q",
                response="ответ",
                retrieval=NaiveSearchRetrieve(query="q", result=NaiveSearchResult()),
                payload=payload,
            ),
            used_query_plan=True,
        )
        assert [source.content for source in sources] == ["branch evidence"]


class TestEngineInvocation:
    """The adapter passes what today's engines accept.

    Engine parameter classes come and go between RAGU versions (0.0.5 dropped
    ``GlobalSearchParams``), and a mismatch only shows up as a 500 at request
    time, so the call into the engine is exercised here.
    """

    class FakeEngine:
        def __init__(self):
            self.calls = []
            self.llm = object()
            self.result = NaiveSearchResult(chunks=[make_chunk("text")], scores=[0.5])

        async def query(self, query, params=None):
            self.calls.append((query, params))
            retrieval = NaiveSearchRetrieve(query=query, result=self.result)
            return SearchEngineResponse(
                query=query, response="ответ", retrieval=retrieval
            )

    def build_backend(self, **overrides):
        from ragu.api.backends.ragu_backend import RaguBackend

        backend = RaguBackend(ServiceSettings(backend="ragu", **overrides))
        backend.graph = object()
        backend._stats = GraphStats(
            entities=1, relations=1, chunks=1, community_summaries=1
        )
        engines = {mode: self.FakeEngine() for mode in ("global", "local", "naive")}
        backend._engines = engines
        return backend, engines

    async def test_local_search_passes_context_flags_as_params(self):
        backend, engines = self.build_backend()

        outcome = await backend.search_local(
            "кто написал",
            use_query_plan=False,
            params=LocalParams(top_k=5, use_summary=False, use_chunks=True),
        )

        query, params = engines["local"].calls[0]
        assert query == "кто написал"
        assert (params.use_summary, params.use_chunks) == (False, True)
        assert params.top_k == 5
        assert outcome.answer == "ответ"
        assert outcome.sources[0].type == "chunk"

    async def test_naive_search_passes_top_k(self):
        backend, engines = self.build_backend()

        await backend.search_naive(
            "версия", use_query_plan=False, params=NaiveSearchParams(top_k=7)
        )

        _, params = engines["naive"].calls[0]
        assert params.top_k == 7

    async def test_global_search_passes_its_params(self):
        backend, engines = self.build_backend()
        params = GlobalSearchParams(min_cluster_size=3)

        await backend.search_global("тренды", params=params)

        assert engines["global"].calls == [("тренды", params)]

    @pytest.mark.parametrize("mode", ["global", "local", "naive"])
    async def test_a_search_without_evidence_is_reported_not_answered(self, mode):
        # Every engine answers from whatever it retrieved, empty included, so an
        # empty retrieval must not reach the client as an answer.
        from ragu.api.errors import CapabilityUnavailableError

        backend, engines = self.build_backend()
        engines[mode].result = NaiveSearchResult()

        with pytest.raises(CapabilityUnavailableError) as failure:
            if mode == "global":
                await backend.search_global("q", params=GlobalSearchParams())
            elif mode == "local":
                await backend.search_local(
                    "q", use_query_plan=False, params=LocalParams()
                )
            else:
                await backend.search_naive(
                    "q", use_query_plan=False, params=NaiveSearchParams()
                )
        assert failure.value.mode == mode
        assert failure.value.status_code == 409
        assert failure.value.missing_capability is None

    @pytest.mark.parametrize(
        "mode, stats, capability",
        [
            ("global", GraphStats(entities=1, chunks=1), "community_summaries"),
            ("local", GraphStats(chunks=1, community_summaries=1), "entity_graph"),
            ("naive", GraphStats(entities=1, community_summaries=1), "vector_index"),
        ],
    )
    async def test_an_unsupported_mode_is_refused_before_generating(
        self, mode, stats, capability
    ):
        # The whole point of measuring the graph at startup: no generation call
        # is paid for on a graph that cannot serve the mode.
        from ragu.api.errors import CapabilityUnavailableError

        backend, engines = self.build_backend()
        backend._stats = stats

        with pytest.raises(CapabilityUnavailableError) as failure:
            if mode == "global":
                await backend.search_global("q", params=GlobalSearchParams())
            elif mode == "local":
                await backend.search_local(
                    "q", use_query_plan=False, params=LocalParams()
                )
            else:
                await backend.search_naive(
                    "q", use_query_plan=False, params=NaiveSearchParams()
                )

        assert failure.value.missing_capability == capability
        assert engines[mode].calls == []

    async def test_evidence_is_returned_as_an_answer(self):
        backend, _ = self.build_backend()

        outcome = await backend.search_naive(
            "q", use_query_plan=False, params=NaiveSearchParams()
        )

        assert outcome.answer == "ответ"
        assert len(outcome.sources) == 1

    async def test_query_plan_wraps_the_engine_with_no_extra_arguments(
        self, monkeypatch
    ):
        wrapped = {}

        class FakePlanEngine:
            def __init__(self, engine, *args, **kwargs):
                wrapped["engine"] = engine
                wrapped["args"] = (args, kwargs)
                self.engine = engine

            async def query(self, query, params=None):
                return await self.engine.query(query, params)

        # The backend imports the symbol into its own namespace, so patching
        # ``ragu.QueryPlanEngine`` would not reach the code under test.
        monkeypatch.setattr(
            "ragu.api.backends.ragu_backend.QueryPlanEngine", FakePlanEngine
        )
        backend, engines = self.build_backend()

        await backend.search_naive(
            "версия", use_query_plan=True, params=NaiveSearchParams(top_k=3)
        )

        # RAGU 0.0.5 dropped the `language` argument; anything extra here is a
        # TypeError at request time, on the default path of two of three modes.
        assert wrapped["engine"] is engines["naive"]
        assert wrapped["args"] == ((), {})
        assert engines["naive"].calls[0][1].top_k == 3

    async def test_engine_construction_failures_report_their_mode(self, monkeypatch):
        from ragu.api.errors import BackendExecutionError

        class ExplodingPlanEngine:
            def __init__(self, engine, *args, **kwargs):
                raise TypeError("unexpected keyword argument")

        monkeypatch.setattr(
            "ragu.api.backends.ragu_backend.QueryPlanEngine", ExplodingPlanEngine
        )
        backend, _ = self.build_backend()

        with pytest.raises(BackendExecutionError) as failure:
            await backend.search_local("q", use_query_plan=True, params=LocalParams())
        assert failure.value.mode == "local"
        assert failure.value.detail == "unexpected keyword argument"
        # The exception text belongs in the log, not in the client's message.
        assert "unexpected keyword argument" not in failure.value.message

    async def test_top_k_is_clamped_to_the_service_limit(self):
        # The engine parameter classes carry no bounds and now arrive straight
        # from the request body.
        backend, engines = self.build_backend(max_top_k=50)

        await backend.search_naive(
            "версия", use_query_plan=False, params=NaiveSearchParams(top_k=10_000)
        )

        assert engines["naive"].calls[0][1].top_k == 50

    async def test_rerank_top_k_is_clamped_too(self):
        backend, engines = self.build_backend(max_top_k=50)

        await backend.search_naive(
            "версия",
            use_query_plan=False,
            params=NaiveSearchParams(top_k=10, rerank_top_k=9_000),
        )

        assert engines["naive"].calls[0][1].rerank_top_k == 50

    async def test_params_within_the_limit_are_passed_through_untouched(self):
        backend, engines = self.build_backend()
        params = NaiveSearchParams(top_k=5, rerank_top_k=2)

        await backend.search_naive("версия", use_query_plan=False, params=params)

        assert engines["naive"].calls[0][1] is params

    async def test_global_search_never_reaches_the_planner(self, monkeypatch):
        class ForbiddenPlanEngine:
            def __init__(self, engine, *args, **kwargs):
                raise AssertionError(
                    "global search must not be wrapped in a query plan"
                )

        monkeypatch.setattr(
            "ragu.api.backends.ragu_backend.QueryPlanEngine", ForbiddenPlanEngine
        )
        backend, engines = self.build_backend()

        await backend.search_global("тренды", params=GlobalSearchParams())

        assert len(engines["global"].calls) == 1


class TestStorageFolderValidation:
    """A typo in the storage folder must not look like a healthy service."""

    def test_a_missing_folder_is_refused_and_not_created(self, tmp_path):
        from ragu.api.backends.ragu_backend import RaguBackend
        from ragu.api.errors import ServiceNotReadyError

        missing = tmp_path / "typo"
        with pytest.raises(ServiceNotReadyError) as failure:
            RaguBackend._require_storage_folder(str(missing))

        # Index.__init__ would have created it via Settings.init_storage_folder().
        assert not missing.exists()
        assert "does not exist" in failure.value.message

    def test_an_empty_folder_is_refused(self, tmp_path):
        from ragu.api.backends.ragu_backend import RaguBackend
        from ragu.api.errors import ServiceNotReadyError

        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ServiceNotReadyError) as failure:
            RaguBackend._require_storage_folder(str(empty))
        assert "is empty" in failure.value.message

    def test_a_populated_folder_is_accepted(self, tmp_path):
        from ragu.api.backends.ragu_backend import RaguBackend

        populated = tmp_path / "graph"
        populated.mkdir()
        (populated / "knowledge_graph.gml").write_text("graph [ ]", encoding="utf-8")
        RaguBackend._require_storage_folder(str(populated))


class TestShutdown:
    async def test_shutdown_closes_the_index_and_the_model_clients(self):
        from ragu.api.backends.ragu_backend import RaguBackend

        closed = []

        class FakeIndex:
            async def close(self):
                closed.append("index")

        class FakeGraph:
            index = FakeIndex()

        class FakeHTTPClient:
            async def close(self):
                closed.append("client")

        class FakeClient:
            client = FakeHTTPClient()

        backend = RaguBackend(ServiceSettings(backend="ragu"))
        backend.graph = FakeGraph()
        backend._stats = GraphStats(entities=1)
        backend._clients = [FakeClient()]

        await backend.shutdown()

        assert closed == ["index", "client"]
        assert backend.graph_loaded is False


class TestLogging:
    """The service logs through loguru, like the rest of RAGU."""

    def test_the_api_package_does_not_use_stdlib_logging(self):
        import pathlib

        import ragu.api

        package = pathlib.Path(ragu.api.__file__).parent
        offenders = [
            path.name
            for path in package.rglob("*.py")
            # logging_setup.py is the bridge itself; it must import logging.
            if path.name != "logging_setup.py"
            and "logging.getLogger" in path.read_text(encoding="utf-8")
        ]
        assert offenders == []

    def test_stdlib_records_are_re_emitted_through_loguru(self):
        import logging

        from ragu.api.logging_setup import InterceptHandler
        from ragu.common.logger import logger

        captured = []
        sink_id = logger.add(lambda message: captured.append(message), level="DEBUG")
        try:
            record = logging.LogRecord(
                name="uvicorn.error",
                level=logging.WARNING,
                pathname=__file__,
                lineno=1,
                msg="listening on %s",
                args=("127.0.0.1:8020",),
                exc_info=None,
            )
            InterceptHandler().emit(record)
        finally:
            logger.remove(sink_id)

        assert len(captured) == 1
        assert "listening on 127.0.0.1:8020" in captured[0]
        assert captured[0].record["level"].name == "WARNING"

    def test_set_level_replaces_the_sink(self):
        from ragu.common.logger import DEFAULT_LEVEL, set_level

        try:
            set_level("debug")
            with pytest.raises(ValueError):
                set_level("not-a-level")
        finally:
            set_level(DEFAULT_LEVEL)

    def test_set_level_survives_a_bare_logger_remove(self):
        # logger.remove() with no argument is the usual way to reconfigure
        # loguru, and it invalidates the sink id set_level tracks.
        from ragu.common.logger import DEFAULT_LEVEL, logger, set_level

        try:
            logger.remove()
            set_level("warning")
            assert logger._core.handlers
        finally:
            set_level(DEFAULT_LEVEL)
