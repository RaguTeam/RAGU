"""
Search backends.
"""

from ragu.api.backends.base import SearchBackend, SearchOutcome
from ragu.api.backends.stub import StubBackend
from ragu.api.config import ServiceSettings


def build_backend(settings: ServiceSettings) -> SearchBackend:
    """Instantiate the backend selected by ``RAGU_BACKEND``."""
    if settings.backend == "stub":
        return StubBackend(settings)
    from ragu.api.backends.ragu_backend import RaguBackend

    return RaguBackend(settings)


__all__ = ["SearchBackend", "SearchOutcome", "StubBackend", "build_backend"]
