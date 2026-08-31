"""Tests for the HTTP search API."""

from __future__ import annotations

import pytest

pytest.importorskip(
    "fastapi", reason="install the 'api' extra to test the search service"
)

from fastapi.testclient import TestClient  # noqa: E402

from ragu.api.app import create_app
from ragu.api.backends.ragu_backend import (
    extract_sources,
    extract_subqueries,
    to_outcome,
)
from ragu.api.config import ServiceSettings
from ragu.chunker.types import Chunk
from ragu.common.prompts.default_models import GlobalSearchContextModel
from ragu.graph.types import Entity, Relation
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


def build_client(missing: str = "") -> TestClient:
    settings = ServiceSettings(backend="stub", stub_missing_capabilities=missing)
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


class TestHealth:
    def test_health_reports_loaded_graph(self):
        with build_client() as client:
            assert client.get("/health").json() == {
                "status": "ok",
                "graph_loaded": True,
            }

    def test_search_answers_503_until_the_graph_is_loaded(self):
        # No lifespan run: the app has no backend yet.
        client = TestClient(create_app(ServiceSettings(backend="stub")))
        response = client.post("/v1/search/naive", json={"query": "q"})
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "SERVICE_NOT_READY"


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

    def test_other_modes_stay_available(self):
        with build_client(missing="community_summaries") as client:
            assert (
                client.post("/v1/search/naive", json={"query": "q"}).status_code == 200
            )

    def test_a_search_without_evidence_answers_409(self):
        with build_client(missing="vector_index") as client:
            response = client.post("/v1/search/naive", json={"query": "q"})
        assert response.status_code == 409
        assert response.json()["error"]["missing_capability"] is None


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
                summaries=[],
                chunks=[make_chunk("chunk text")],
            ),
        )
        sources = extract_sources(retrieval)
        assert [source.type for source in sources] == ["entity", "relation", "chunk"]
        assert sources[0].id == entity.id
        assert "Сенкевич" in sources[1].content

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

    def test_global_insights_are_dicts_in_practice(self):
        # GlobalSearchEngine.batch_search stores `model_dump()` results, not the
        # models themselves.
        insight = GlobalSearchContextModel(
            reasoning="r", response="общий вывод", rating=8.0
        ).model_dump()
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

    def test_query_plan_payload_becomes_subqueries(self):
        retrieval = NaiveSearchRetrieve(
            query="sub", result=NaiveSearchResult(chunks=[make_chunk("text")])
        )
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

    def test_streaming_payload_shape_is_also_understood(self):
        retrieval = NaiveSearchRetrieve(query="sub", result=NaiveSearchResult())
        payload = {
            "answers": {
                "sq-1": SearchEngineResponse(
                    query="Кто написал?", response="Сенкевич", retrieval=retrieval
                )
            },
            "plan": ["ignored"],
        }
        assert [item.query for item in extract_subqueries(payload)] == ["Кто написал?"]

    def test_response_without_plan_has_no_subqueries(self):
        retrieval = NaiveSearchRetrieve(
            query="q", result=NaiveSearchResult(chunks=[make_chunk("text")])
        )
        outcome = to_outcome(
            SearchEngineResponse(query="q", response="ответ", retrieval=retrieval)
        )
        assert outcome.answer == "ответ"
        assert outcome.subqueries == []
        assert len(outcome.sources) == 1


class TestEngineInvocation:
    """The adapter passes what today's engines accept.

    Engine parameter classes come and go between RAGU versions (0.0.5 dropped
    ``GlobalSearchParams``), and a mismatch only shows up as a 500 at request
    time, so the call into the engine is exercised here.
    """

    class FakeEngine:
        def __init__(self):
            self.calls = []
            self.result = NaiveSearchResult(chunks=[make_chunk("text")], scores=[0.5])

        async def query(self, query, params=None):
            self.calls.append((query, params))
            retrieval = NaiveSearchRetrieve(query=query, result=self.result)
            return SearchEngineResponse(
                query=query, response="ответ", retrieval=retrieval
            )

    def build_backend(self):
        from ragu.api.backends.ragu_backend import RaguBackend

        backend = RaguBackend(ServiceSettings(backend="ragu"))
        backend.graph = object()
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
        import ragu

        wrapped = {}

        class FakePlanEngine:
            def __init__(self, engine, *args, **kwargs):
                wrapped["engine"] = engine
                wrapped["args"] = (args, kwargs)
                self.engine = engine

            async def query(self, query, params=None):
                return await self.engine.query(query, params)

        monkeypatch.setattr(ragu, "QueryPlanEngine", FakePlanEngine)
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
        import ragu

        from ragu.api.errors import BackendExecutionError

        class ExplodingPlanEngine:
            def __init__(self, engine, *args, **kwargs):
                raise TypeError("unexpected keyword argument")

        monkeypatch.setattr(ragu, "QueryPlanEngine", ExplodingPlanEngine)
        backend, _ = self.build_backend()

        with pytest.raises(BackendExecutionError) as failure:
            await backend.search_local("q", use_query_plan=True, params=LocalParams())
        assert failure.value.mode == "local"

    async def test_top_k_is_clamped_to_the_service_limit(self):
        # The engine parameter classes carry no bounds and now arrive straight
        # from the request body.
        backend, engines = self.build_backend()
        backend.settings.max_top_k = 50

        await backend.search_naive(
            "версия", use_query_plan=False, params=NaiveSearchParams(top_k=10_000)
        )

        assert engines["naive"].calls[0][1].top_k == 50

    async def test_params_within_the_limit_are_passed_through_untouched(self):
        backend, engines = self.build_backend()
        params = NaiveSearchParams(top_k=5, rerank_top_k=2)

        await backend.search_naive("версия", use_query_plan=False, params=params)

        assert engines["naive"].calls[0][1] is params

    async def test_global_search_never_reaches_the_planner(self, monkeypatch):
        import ragu

        class ForbiddenPlanEngine:
            def __init__(self, engine, *args, **kwargs):
                raise AssertionError(
                    "global search must not be wrapped in a query plan"
                )

        monkeypatch.setattr(ragu, "QueryPlanEngine", ForbiddenPlanEngine)
        backend, engines = self.build_backend()

        await backend.search_global("тренды", params=GlobalSearchParams())

        assert len(engines["global"].calls) == 1
