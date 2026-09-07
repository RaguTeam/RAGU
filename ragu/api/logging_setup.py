"""
Route the service's logs into RAGU's loguru sink.

The engines log through loguru while uvicorn, httpx and the OpenAI client log
through stdlib ``logging``. Without this the two halves of one request land in
different sinks with different formats, and a trace cannot be followed across
them. Installing an intercept handler on the stdlib root gives the whole
process a single sink.

Only ``python -m ragu.api`` calls :func:`configure_logging`. Importing
``create_app`` into another server leaves that server's logging alone.
"""

import logging

from ragu.common.logger import logger, set_level

# Loggers that install their own handlers; those handlers have to go, or their
# records are emitted twice — once by uvicorn's formatter, once by loguru.
_LOGGERS_TO_INTERCEPT = (
    "uvicorn",
    "uvicorn.error",
    "uvicorn.access",
    "uvicorn.asgi",
    "fastapi",
    "httpx",
    "httpcore",
)


class InterceptHandler(logging.Handler):
    """
    stdlib ``logging`` handler that re-emits every record through loguru.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Walk out of the logging machinery so loguru reports the real caller.
        frame, depth = logging.currentframe(), 2
        while frame is not None and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def configure_logging(level: str = "info") -> None:
    """
    Send stdlib log records to loguru and set the loguru sink's level.

    :param level: Level name, e.g. ``"debug"`` (case-insensitive).
    :raises ValueError: If the level name is not known to loguru.
    """
    set_level(level)

    numeric_level = logging.getLevelName(level.upper())
    logging.root.handlers = [InterceptHandler()]
    logging.root.setLevel(numeric_level)

    for name in _LOGGERS_TO_INTERCEPT:
        intercepted = logging.getLogger(name)
        intercepted.handlers = []
        intercepted.propagate = True
