import logging
import sys

import openai

openai._utils._logs.logger.setLevel(logging.WARNING)
openai._utils._logs.httpx_logger.setLevel(logging.WARNING)

from loguru import logger

LOG_FORMAT = (
    "<cyan>{time:HH:mm:ss}</cyan> | <level>{level: <8}</level> | <level>{message}</level>"
)

DEFAULT_LEVEL = "INFO"

logger.remove()
_handler_id = logger.add(
    sys.stdout,
    colorize=True,
    enqueue=True,
    level=DEFAULT_LEVEL,
    format=LOG_FORMAT,
)


def set_level(level: str) -> None:
    """
    Replace the default stdout sink with one at a different level.

    A loguru sink cannot be re-levelled in place, so the existing one is
    removed and re-added with the same destination and format.

    :param level: Level name, e.g. ``"DEBUG"`` or ``"warning"`` (case-insensitive).
    :raises ValueError: If the level name is not known to loguru.
    """
    global _handler_id

    level = level.upper()
    logger.level(level)  # Raises ValueError on an unknown name.

    try:
        logger.remove(_handler_id)
    except ValueError:
        # A bare logger.remove() elsewhere already dropped every sink, so the
        # tracked id is stale. Adding the new sink is still the right outcome.
        pass
    _handler_id = logger.add(
        sys.stdout,
        colorize=True,
        enqueue=True,
        level=level,
        format=LOG_FORMAT,
    )


__all__ = ["logger", "set_level", "LOG_FORMAT", "DEFAULT_LEVEL"]
