"""
HTTP routes: one physical resource per search mode.

Separate paths let a gateway apply per-mode timeouts and rate limits without
any change in the service (see the spec, "Per-route policy").
"""

import logging

from fastapi import APIRouter, Depends, Request

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

logger = logging.getLogger(__name__)

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


def get_backend(request: Request) -> SearchBackend:
    backend: SearchBackend | None = getattr(request.app.state, "backend", None)
    if backend is None or not backend.graph_loaded:
        raise ServiceNotReadyError("Knowledge graph is not loaded yet.")
    return backend


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    backend: SearchBackend | None = getattr(request.app.state, "backend", None)
    graph_loaded = bool(backend and backend.graph_loaded)
    return HealthResponse(
        status="ok" if graph_loaded else "degraded", graph_loaded=graph_loaded
    )


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
