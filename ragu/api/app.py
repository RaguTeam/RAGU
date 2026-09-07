"""
FastAPI application factory.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ragu.api.backends import build_backend
from ragu.api.backends.base import SearchBackend
from ragu.api.config import ServiceSettings
from ragu.api.errors import InvalidRequestError, RaguServiceError
from ragu.api.routes import router
from ragu.common.logger import logger

# Returned instead of the exception text: engine and LLM-client errors routinely
# quote the endpoint URL and parts of the request body.
UNHANDLED_ERROR_MESSAGE = (
    "The service failed to handle this request. See the service log for details."
)


def create_app(
    settings: ServiceSettings | None = None,
    backend: SearchBackend | None = None,
) -> FastAPI:
    """
    Build the service application.

    :param settings: Service settings; read from the environment when omitted.
    :param backend: Pre-built backend, used by tests to bypass graph loading.
    :return: The configured application.
    """
    settings = settings or ServiceSettings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.startup_error = None
        app.state.backend = backend or build_backend(settings)
        try:
            await app.state.backend.startup()
        except Exception as exc:
            # The service still starts so that /health can report *why* it is
            # not ready; without this the operator only ever sees a container
            # that restarts.
            app.state.startup_error = str(exc)
            logger.opt(exception=True).error("Backend startup failed: {}", exc)
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
        detail = getattr(exc, "detail", None)
        if detail:
            logger.error("{} ({}): {}", exc.code, exc.mode, detail)
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_envelope(),
            headers=exc.headers or None,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error_handler(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # The contract specifies 400 for an invalid request, not FastAPI's default 422.
        message = "; ".join(
            f"{'.'.join(str(part) for part in err['loc'][1:])}: {err['msg']}"
            for err in exc.errors()
        )
        error = InvalidRequestError(message or "Invalid request")
        return JSONResponse(status_code=error.status_code, content=error.to_envelope())

    @app.exception_handler(Exception)
    async def _unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled service error")
        error = RaguServiceError(UNHANDLED_ERROR_MESSAGE)
        return JSONResponse(status_code=error.status_code, content=error.to_envelope())

    app.include_router(router)
    return app
