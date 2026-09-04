"""
Search backends.
"""

from ragu.api.backends.base import GraphStats, SearchBackend, SearchOutcome
from ragu.api.backends.ragu_backend import RaguBackend
from ragu.api.backends.stub import StubBackend
from ragu.api.config import ServiceSettings


def build_backend(settings: ServiceSettings) -> SearchBackend:
    """
    Instantiate the backend selected by ``RAGU_API_BACKEND``.

    :param settings: Service settings.
    :return: The configured backend, not yet started.
    """
    if settings.backend == "stub":
        return StubBackend(settings)
    return RaguBackend(settings)


__all__ = [
    "GraphStats",
    "RaguBackend",
    "SearchBackend",
    "SearchOutcome",
    "StubBackend",
    "build_backend",
]
