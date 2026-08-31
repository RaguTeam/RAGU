"""
FastAPI application factory.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ragu.api.backends import build_backend
from ragu.api.config import ServiceSettings
from ragu.api.errors import RaguServiceError
from ragu.api.routes import router

logger = logging.getLogger(__name__)


def create_app(settings: ServiceSettings | None = None, backend=None) -> FastAPI:
    """
    Build the service application.

    params: settings: Service settings; read from the environment when omitted.
    params: backend: Pre-built backend, used by tests to bypass graph loading.
    """
    settings = settings or ServiceSettings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.backend = backend or build_backend(settings)
        try:
            await app.state.backend.startup()
        except Exception as exc:
            logger.error(f"Backend startup failed: {exc}", exc_info=True)
        yield
        await app.state.backend.shutdown()

    app = FastAPI(
        title="RAGU Search Service",
        description="Graph-RAG search over a prebuilt knowledge graph: global, local and naive modes.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.exception_handler(RaguServiceError)
    async def _service_error_handler(_: Request, exc: RaguServiceError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.to_envelope())

    @app.exception_handler(RequestValidationError)
    async def _validation_error_handler(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # The contract specifies 400 for an invalid request, not FastAPI's default 422.
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": "INVALID_REQUEST",
                    "mode": None,
                    "missing_capability": None,
                    "message": "; ".join(
                        f"{'.'.join(str(part) for part in err['loc'][1:])}: {err['msg']}"
                        for err in exc.errors()
                    )
                    or "Invalid request",
                }
            },
        )

    @app.exception_handler(Exception)
    async def _unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled service error")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "mode": None,
                    "missing_capability": None,
                    "message": str(exc),
                }
            },
        )

    app.include_router(router)
    return app
