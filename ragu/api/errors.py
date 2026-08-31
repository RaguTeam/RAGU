"""
Error types mapped to the HTTP error contract of the service.
"""


class RaguServiceError(Exception):
    """
    Base class for errors rendered as the service error envelope.
    """

    code = "INTERNAL_ERROR"
    status_code = 500

    def __init__(self, message: str, *, mode: str | None = None):
        super().__init__(message)
        self.message = message
        self.mode = mode

    def to_envelope(self) -> dict:
        return {
            "error": {
                "code": self.code,
                "mode": self.mode,
                "missing_capability": None,
                "message": self.message,
            }
        }


class CapabilityUnavailableError(RaguServiceError):
    """
    The graph cannot serve this search mode (409).

    The client turns this into a plain SEARCH_UNAVAILABLE tool result so the
    agent can switch to another search mode by itself.
    """

    code = "CAPABILITY_UNAVAILABLE"
    status_code = 409

    def __init__(self, message: str, *, mode: str, missing_capability: str):
        super().__init__(message, mode=mode)
        self.missing_capability = missing_capability

    def to_envelope(self) -> dict:
        envelope = super().to_envelope()
        envelope["error"]["missing_capability"] = self.missing_capability
        return envelope


class ServiceNotReadyError(RaguServiceError):
    """
    The graph is not loaded yet or the backend failed to start (503).
    """

    code = "SERVICE_NOT_READY"
    status_code = 503


class BackendExecutionError(RaguServiceError):
    """
    The search engine failed: LLM unavailable, embedder timeout, etc (500).
    """

    code = "INTERNAL_ERROR"
    status_code = 500
