"""
HTTP routes: one physical resource per search mode.

Separate paths let a gateway apply per-mode timeouts and rate limits without
any change in the service (see the spec, "Per-route policy").
"""

from typing import Any

from fastapi import APIRouter, Depends, Request, Response

from ragu.api.backends.base import (
    RetrieveOutcome,
    SearchBackend,
    SearchCall,
    SearchOutcome,
)
from ragu.api.errors import RETRY_AFTER_SECONDS, ServiceNotReadyError
from ragu.api.models import (
    EngineReport,
    ErrorResponse,
    GlobalRetrieveRequest,
    GlobalSearchRequest,
    HealthResponse,
    LocalRetrieveRequest,
    LocalSearchRequest,
    MixRetrieveRequest,
    MixSearchRequest,
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
        query=payload.query,
        params=payload.params,
        local_params=getattr(payload, "local_params", None),
        naive_params=getattr(payload, "naive_params", None),
        use_query_plan=getattr(payload, "use_query_plan", False),
    )


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
    return _response(call, await backend.search(call))


@router.post(
    "/v1/search/local", response_model=SearchResponse, responses=SEARCH_RESPONSES
)
async def search_local(
    payload: LocalSearchRequest,
    backend: SearchBackend = Depends(get_backend),
) -> SearchResponse:
    call = _call("local", payload)
    return _response(call, await backend.search(call))


@router.post(
    "/v1/search/naive", response_model=SearchResponse, responses=SEARCH_RESPONSES
)
async def search_naive(
    payload: NaiveSearchRequest,
    backend: SearchBackend = Depends(get_backend),
) -> SearchResponse:
    call = _call("naive", payload)
    return _response(call, await backend.search(call))


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
    return _response(call, await backend.search(call))


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
    return _retrieved(call, await backend.retrieve(call))


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
    return _retrieved(call, await backend.retrieve(call))


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
    return _retrieved(call, await backend.retrieve(call))


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
    return _retrieved(call, await backend.retrieve(call))
