"""
Error types mapped to the HTTP error contract of the service.

Every error renders through :class:`~ragu.api.models.ErrorResponse`, so the
envelope has one definition rather than one per call site. Messages carried
here are authored for clients; details that come from an exception stay in
``detail`` and are logged, never returned.
"""

from ragu.api.models import Capability, ErrorBody, ErrorResponse

# How long a client should wait before retrying a service that is not ready.
# Loading a graph takes minutes, so this is a hint to back off, not to spin.
RETRY_AFTER_SECONDS = 30


class RaguServiceError(Exception):
    """
    Base class for errors rendered as the service error envelope.
    """

    code: str = "INTERNAL_ERROR"
    status_code: int = 500

    def __init__(self, message: str, *, mode: str | None = None):
        """
        :param message: Client-facing explanation. Must not embed exception text.
        :param mode: Search mode the error belongs to, when it has one.
        """
        super().__init__(message)
        self.message = message
        self.mode = mode

    @property
    def missing_capability(self) -> str | None:
        """
        Capability the graph lacks, for errors that identify one.
        """
        return None

    @property
    def headers(self) -> dict[str, str]:
        """
        Extra HTTP headers to send with this error.
        """
        return {}

    def to_envelope(self) -> dict:
        """
        Render the error as the service error envelope.

        :return: Envelope body ready to serialize.
        """
        return ErrorResponse(
            error=ErrorBody(
                code=self.code,
                mode=self.mode,
                missing_capability=self.missing_capability,
                message=self.message,
            )
        ).model_dump()


class InvalidRequestError(RaguServiceError):
    """
    The request could not be validated (400).
    """

    code = "INVALID_REQUEST"
    status_code = 400


class CapabilityUnavailableError(RaguServiceError):
    """
    The graph cannot serve this search mode (409).

    Raised in two situations, told apart by ``missing_capability``: the graph
    was built without the storage this mode reads (the capability is named), or
    the graph has that storage but retrieved nothing for this query (``None``).
    The client turns either into a plain SEARCH_UNAVAILABLE tool result so the
    agent can switch to another search mode by itself.
    """

    code = "CAPABILITY_UNAVAILABLE"
    status_code = 409

    def __init__(self, message: str, *, mode: str, missing_capability: Capability | None):
        super().__init__(message, mode=mode)
        self._missing_capability = missing_capability

    @property
    def missing_capability(self) -> str | None:
        return self._missing_capability


class ServiceNotReadyError(RaguServiceError):
    """
    The graph is not loaded yet or the backend failed to start (503).

    Carries ``Retry-After`` so a client backs off for the minutes a graph load
    takes instead of retrying immediately.
    """

    code = "SERVICE_NOT_READY"
    status_code = 503

    @property
    def headers(self) -> dict[str, str]:
        return {"Retry-After": str(RETRY_AFTER_SECONDS)}


class BackendExecutionError(RaguServiceError):
    """
    The search engine failed: LLM unavailable, embedder timeout, etc (500).

    Unlike the base class it always names a mode, and it keeps the underlying
    exception text in ``detail`` for the log instead of putting it in the
    response: engine and LLM-client errors routinely quote the endpoint URL and
    parts of the request.
    """

    code = "INTERNAL_ERROR"
    status_code = 500

    def __init__(self, *, mode: str, detail: str):
        """
        :param mode: Search mode that failed.
        :param detail: Underlying exception text, for logging only.
        """
        super().__init__(f"The {mode} search engine failed.", mode=mode)
        self.detail = detail
