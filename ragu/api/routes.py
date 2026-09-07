"""
HTTP routes: one physical resource per search mode.

Separate paths let a gateway apply per-mode timeouts and rate limits without
any change in the service (see the spec, "Per-route policy").
"""

import json
from typing import Any

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import StreamingResponse

from ragu.api.backends.base import (
    RetrieveOutcome,
    SearchStreamEvent,
    SearchBackend,
    SearchCall,
    SearchOutcome,
)
from ragu.api.errors import RETRY_AFTER_SECONDS, ServiceNotReadyError
from ragu.api.models import (
    BatchSearchItem,
    BatchSearchResponse,
    EngineReport,
    ErrorBody,
    ErrorResponse,
    GlobalBatchRequest,
    GlobalRetrieveRequest,
    GlobalSearchRequest,
    HealthResponse,
    LocalBatchRequest,
    LocalRetrieveRequest,
    LocalSearchRequest,
    MixBatchRequest,
    MixRetrieveRequest,
    MixSearchRequest,
    NaiveBatchRequest,
    NaiveRetrieveRequest,
    NaiveSearchRequest,
    RetrieveResponse,
    SearchMode,
    SearchResponse,
)

router = APIRouter()

SEARCH_RESPONSES = {
    409: {
        "model": ErrorResponse,
        "description": "Capability unavailable for this graph",
    },
    503: {
        "model": ErrorResponse,
        "description": "Graph is not loaded / service not ready",
    },
    500: {"model": ErrorResponse, "description": "Internal error"},
}


def _response(call: SearchCall, outcome: SearchOutcome) -> SearchResponse:
    """
    Render one search outcome as the wire response.

    :param call: The resolved request.
    :param outcome: Normalized backend outcome.
    :return: The response body.
    """
    return SearchResponse(
        query=call.query,
        mode=call.mode,
        used_query_plan=call.use_query_plan,
        answer=outcome.answer,
        sources=outcome.sources,
        subqueries=outcome.subqueries,
        engines=outcome.engines
        or EngineReport(
            requested=call.mode, used="unknown", query_plan=call.use_query_plan
        ),
    )


def _backend_of(request: Request) -> SearchBackend | None:
    return getattr(request.app.state, "backend", None)


def _health_of(request: Request) -> HealthResponse:
    """
    Describe the current readiness of the service.

    :param request: Incoming request, for the application state.
    :return: Health payload, including why the backend is not ready.
    """
    backend = _backend_of(request)
    graph_loaded = bool(backend and backend.graph_loaded)
    stats = backend.stats if backend is not None else None
    return HealthResponse(
        status="ok" if graph_loaded else "degraded",
        graph_loaded=graph_loaded,
        stats=stats.to_response() if stats is not None else None,
        error=getattr(request.app.state, "startup_error", None),
    )


def get_backend(request: Request) -> SearchBackend:
    """
    Resolve a backend that can serve searches.

    :raises ServiceNotReadyError: If no graph is loaded, quoting the startup
        failure when there was one.
    """
    backend = _backend_of(request)
    if backend is None or not backend.graph_loaded:
        reason = getattr(request.app.state, "startup_error", None)
        raise ServiceNotReadyError(
            f"Knowledge graph is not loaded: {reason}"
            if reason
            else "Knowledge graph is not loaded yet."
        )
    return backend


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """
    Report readiness with a 200 regardless, for probes that read the body.
    """
    return _health_of(request)


@router.get("/health/live", response_model=HealthResponse)
async def health_live(request: Request) -> HealthResponse:
    """
    Liveness: the process is up and serving. Always 200 while it can answer.
    """
    return _health_of(request)


@router.get(
    "/health/ready",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse, "description": "Graph not loaded"}},
)
async def health_ready(request: Request, response: Response) -> HealthResponse:
    """
    Readiness: 200 only when searches can actually be served, 503 otherwise.

    Loading a large graph takes minutes, so an orchestrator needs this split
    from liveness to avoid restarting a service that is still starting up.
    """
    payload = _health_of(request)
    if not payload.graph_loaded:
        response.status_code = 503
        response.headers["Retry-After"] = str(RETRY_AFTER_SECONDS)
    return payload


def _call(mode: SearchMode, payload: Any) -> SearchCall:
    """
    Build a backend call from any of the search or retrieve request models.

    :param mode: Mode the route serves.
    :param payload: The validated request body.
    :return: The resolved call.
    """
    return SearchCall(
        mode=mode,
        queries=(payload.query,),
        params=payload.params,
        local_params=getattr(payload, "local_params", None),
        naive_params=getattr(payload, "naive_params", None),
        use_query_plan=getattr(payload, "use_query_plan", False),
        language=getattr(payload, "language", None),
    )


async def _one(backend: SearchBackend, awaitable: Any, mode: SearchMode) -> Any:
    """
    Take the single outcome of a one-query call, refusing an empty one.

    :raises CapabilityUnavailableError: If the query retrieved nothing.
    """
    outcome = (await awaitable)[0]
    backend.require_evidence(mode, outcome)
    return outcome


def _retrieved(call: SearchCall, outcome: RetrieveOutcome) -> RetrieveResponse:
    return RetrieveResponse(
        query=call.query,
        mode=call.mode,
        sources=outcome.sources,
        engines=outcome.engines or EngineReport(requested=call.mode, used="unknown"),
    )


@router.post(
    "/v1/search/global", response_model=SearchResponse, responses=SEARCH_RESPONSES
)
async def search_global(
    payload: GlobalSearchRequest,
    backend: SearchBackend = Depends(get_backend),
) -> SearchResponse:
    call = _call("global", payload)
    return _response(call, await _one(backend, backend.search(call), call.mode))


@router.post(
    "/v1/search/local", response_model=SearchResponse, responses=SEARCH_RESPONSES
)
async def search_local(
    payload: LocalSearchRequest,
    backend: SearchBackend = Depends(get_backend),
) -> SearchResponse:
    call = _call("local", payload)
    return _response(call, await _one(backend, backend.search(call), call.mode))


@router.post(
    "/v1/search/naive", response_model=SearchResponse, responses=SEARCH_RESPONSES
)
async def search_naive(
    payload: NaiveSearchRequest,
    backend: SearchBackend = Depends(get_backend),
) -> SearchResponse:
    call = _call("naive", payload)
    return _response(call, await _one(backend, backend.search(call), call.mode))


@router.post(
    "/v1/search/mix", response_model=SearchResponse, responses=SEARCH_RESPONSES
)
async def search_mix(
    payload: MixSearchRequest,
    backend: SearchBackend = Depends(get_backend),
) -> SearchResponse:
    """
    Ensemble the local and naive engines over one query.

    Child failures are tolerated by the ensemble, so ``engines`` in the response
    reports which children actually contributed.
    """
    call = _call("mix", payload)
    return _response(call, await _one(backend, backend.search(call), call.mode))


@router.post(
    "/v1/search/global/retrieve",
    response_model=RetrieveResponse,
    responses=SEARCH_RESPONSES,
)
async def retrieve_global(
    payload: GlobalRetrieveRequest,
    backend: SearchBackend = Depends(get_backend),
) -> RetrieveResponse:
    call = _call("global", payload)
    return _retrieved(call, await _one(backend, backend.retrieve(call), call.mode))


@router.post(
    "/v1/search/local/retrieve",
    response_model=RetrieveResponse,
    responses=SEARCH_RESPONSES,
)
async def retrieve_local(
    payload: LocalRetrieveRequest,
    backend: SearchBackend = Depends(get_backend),
) -> RetrieveResponse:
    call = _call("local", payload)
    return _retrieved(call, await _one(backend, backend.retrieve(call), call.mode))


@router.post(
    "/v1/search/naive/retrieve",
    response_model=RetrieveResponse,
    responses=SEARCH_RESPONSES,
)
async def retrieve_naive(
    payload: NaiveRetrieveRequest,
    backend: SearchBackend = Depends(get_backend),
) -> RetrieveResponse:
    call = _call("naive", payload)
    return _retrieved(call, await _one(backend, backend.retrieve(call), call.mode))


@router.post(
    "/v1/search/mix/retrieve",
    response_model=RetrieveResponse,
    responses=SEARCH_RESPONSES,
)
async def retrieve_mix(
    payload: MixRetrieveRequest,
    backend: SearchBackend = Depends(get_backend),
) -> RetrieveResponse:
    """
    Gather context from both child engines without generating an answer.
    """
    call = _call("mix", payload)
    return _retrieved(call, await _one(backend, backend.retrieve(call), call.mode))


def _batch_call(mode: SearchMode, payload: Any) -> SearchCall:
    return SearchCall(
        mode=mode,
        queries=tuple(payload.queries),
        params=payload.params,
        local_params=getattr(payload, "local_params", None),
        naive_params=getattr(payload, "naive_params", None),
        use_query_plan=getattr(payload, "use_query_plan", False),
        language=getattr(payload, "language", None),
    )


async def _batched(
    mode: SearchMode, payload: Any, backend: SearchBackend
) -> BatchSearchResponse:
    """
    Answer a batch, reporting per-query emptiness instead of failing the batch.
    """
    call = _batch_call(mode, payload)
    backend.require_batch_size(len(call.queries))
    outcomes = await backend.search(call)

    results = []
    for query, outcome in zip(call.queries, outcomes):
        if outcome.sources:
            results.append(
                BatchSearchItem(
                    query=query,
                    answer=outcome.answer,
                    sources=outcome.sources,
                    subqueries=outcome.subqueries,
                )
            )
        else:
            empty = backend.no_evidence(mode)
            results.append(
                BatchSearchItem(
                    query=query, error=ErrorBody(**empty.to_envelope()["error"])
                )
            )

    engines = outcomes[0].engines if outcomes else None
    return BatchSearchResponse(
        mode=mode,
        used_query_plan=call.use_query_plan,
        engines=engines or EngineReport(requested=mode, used="unknown"),
        results=results,
    )


@router.post(
    "/v1/search/global/batch",
    response_model=BatchSearchResponse,
    responses=SEARCH_RESPONSES,
)
async def batch_global(
    payload: GlobalBatchRequest,
    backend: SearchBackend = Depends(get_backend),
) -> BatchSearchResponse:
    return await _batched("global", payload, backend)


@router.post(
    "/v1/search/local/batch",
    response_model=BatchSearchResponse,
    responses=SEARCH_RESPONSES,
)
async def batch_local(
    payload: LocalBatchRequest,
    backend: SearchBackend = Depends(get_backend),
) -> BatchSearchResponse:
    return await _batched("local", payload, backend)


@router.post(
    "/v1/search/naive/batch",
    response_model=BatchSearchResponse,
    responses=SEARCH_RESPONSES,
)
async def batch_naive(
    payload: NaiveBatchRequest,
    backend: SearchBackend = Depends(get_backend),
) -> BatchSearchResponse:
    """
    Answer many queries in one pass.

    This is where the engines earn their keep: retrieval is shared across the
    whole list and, with a query plan, independent subqueries from different
    top-level queries are answered in the same child batch.
    """
    return await _batched("naive", payload, backend)


@router.post(
    "/v1/search/mix/batch",
    response_model=BatchSearchResponse,
    responses=SEARCH_RESPONSES,
)
async def batch_mix(
    payload: MixBatchRequest,
    backend: SearchBackend = Depends(get_backend),
) -> BatchSearchResponse:
    return await _batched("mix", payload, backend)


def _sse(event: SearchStreamEvent) -> str:
    """
    Frame one event as a Server-Sent Event.
    """
    payload = json.dumps(event.data, ensure_ascii=False)
    return f"event: {event.event}\ndata: {payload}\n\n"


async def _stream(mode: SearchMode, payload: Any, backend: SearchBackend) -> Response:
    """
    Stream one answer, refusing an unservable mode before the response starts.
    """
    call = _call(mode, payload)
    # Inside an open stream a 409 could only be an SSE event, so the capability
    # is checked while a status code can still carry it.
    backend.require_capability(mode)

    async def events():
        async for event in backend.stream(call):
            yield _sse(event)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/v1/search/global/stream", responses=SEARCH_RESPONSES)
async def stream_global(
    payload: GlobalSearchRequest,
    backend: SearchBackend = Depends(get_backend),
) -> Response:
    return await _stream("global", payload, backend)


@router.post("/v1/search/local/stream", responses=SEARCH_RESPONSES)
async def stream_local(
    payload: LocalSearchRequest,
    backend: SearchBackend = Depends(get_backend),
) -> Response:
    return await _stream("local", payload, backend)


@router.post("/v1/search/naive/stream", responses=SEARCH_RESPONSES)
async def stream_naive(
    payload: NaiveSearchRequest,
    backend: SearchBackend = Depends(get_backend),
) -> Response:
    """
    Stream the answer as Server-Sent Events.

    One ``meta`` event carries the retrieval and the engine report, then
    ``delta`` events carry the text, then ``done`` closes with the final engine
    report. A failure after the headers are sent arrives as an ``error`` event.
    """
    return await _stream("naive", payload, backend)


@router.post("/v1/search/mix/stream", responses=SEARCH_RESPONSES)
async def stream_mix(
    payload: MixSearchRequest,
    backend: SearchBackend = Depends(get_backend),
) -> Response:
    return await _stream("mix", payload, backend)
