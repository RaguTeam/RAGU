"""
HTTP routes: one physical resource per search mode.

Separate paths let a gateway apply per-mode timeouts and rate limits without
any change in the service (see the spec, "Per-route policy").
"""

from fastapi import APIRouter, Depends, Request, Response

from ragu.api.backends.base import SearchBackend
from ragu.api.errors import ServiceNotReadyError
from ragu.api.models import (
    ErrorResponse,
    GlobalSearchRequest,
    HealthResponse,
    LocalSearchRequest,
    NaiveSearchRequest,
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
        response.headers["Retry-After"] = "30"
    return payload


@router.post(
    "/v1/search/global", response_model=SearchResponse, responses=SEARCH_RESPONSES
)
async def search_global(
    payload: GlobalSearchRequest,
    backend: SearchBackend = Depends(get_backend),
) -> SearchResponse:
    outcome = await backend.search_global(payload.query, params=payload.params)
    return SearchResponse(
        query=payload.query,
        mode="global",
        used_query_plan=False,
        answer=outcome.answer,
        sources=outcome.sources,
        subqueries=outcome.subqueries,
    )


@router.post(
    "/v1/search/local", response_model=SearchResponse, responses=SEARCH_RESPONSES
)
async def search_local(
    payload: LocalSearchRequest,
    backend: SearchBackend = Depends(get_backend),
) -> SearchResponse:
    outcome = await backend.search_local(
        payload.query,
        use_query_plan=payload.use_query_plan,
        params=payload.params,
    )
    return SearchResponse(
        query=payload.query,
        mode="local",
        used_query_plan=payload.use_query_plan,
        answer=outcome.answer,
        sources=outcome.sources,
        subqueries=outcome.subqueries,
    )


@router.post(
    "/v1/search/naive", response_model=SearchResponse, responses=SEARCH_RESPONSES
)
async def search_naive(
    payload: NaiveSearchRequest,
    backend: SearchBackend = Depends(get_backend),
) -> SearchResponse:
    outcome = await backend.search_naive(
        payload.query,
        use_query_plan=payload.use_query_plan,
        params=payload.params,
    )
    return SearchResponse(
        query=payload.query,
        mode="naive",
        used_query_plan=payload.use_query_plan,
        answer=outcome.answer,
        sources=outcome.sources,
        subqueries=outcome.subqueries,
    )
