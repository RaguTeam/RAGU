"""Process-local element locks shared by tasks in any event loop or thread."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any


_LockKey = tuple[str, str]
_Requirements = dict[_LockKey, bool]


@dataclass(slots=True, eq=False)
class _Request:
    owner: asyncio.Task[Any]
    requirements: _Requirements
    loop: asyncio.AbstractEventLoop
    ready: asyncio.Future[None]


def _conflicts(left: _Requirements, right: _Requirements) -> bool:
    if not left or not right:
        return False
    # Exact requests need four dictionary lookups. Build aggregate indexes only
    # for wildcard requests, keeping total work linear in both footprints.
    namespaces: dict[str, bool] | None = None
    ids: dict[str, bool] | None = None
    for (namespace, item_id), write in left.items():
        if namespace == "*" and item_id == "*":
            if write or any(right.values()):
                return True
        if namespace == "*":
            if ids is None:
                ids = {}
                for (_, key_id), right_write in right.items():
                    ids[key_id] = ids.get(key_id, False) or right_write
            if any(key in ids and (write or ids[key]) for key in (item_id, "*")):
                return True
        elif item_id == "*":
            if namespaces is None:
                namespaces = {}
                for (key_namespace, _), right_write in right.items():
                    namespaces[key_namespace] = namespaces.get(key_namespace, False) or right_write
            if any(key in namespaces and (write or namespaces[key]) for key in (namespace, "*")):
                return True
        elif any(
            key in right and (write or right[key])
            for key in ((namespace, item_id), (namespace, "*"), ("*", item_id), ("*", "*"))
        ):
            return True
    return False


def _covers(held: _Requirements, requested: _Requirements) -> bool:
    return all(
        any(
            key in held and (held[key] or not requested_write)
            for key in (
                (namespace, item_id), (namespace, "*"), ("*", item_id), ("*", "*"),
            )
        )
        for (namespace, item_id), requested_write in requested.items()
    )


class _ElementLocks:
    """
    Atomically acquire reader/writer grants over namespaced element IDs.

    A requirement maps ``(namespace, id)`` to ``True`` for exclusive access
    or ``False`` for shared access. Either part may be ``"*"``. Requests
    conflicting with an earlier waiter cannot overtake it; unrelated requests
    remain free to proceed. Ownership belongs to the current asyncio task,
    rather than a context variable that child tasks would inherit.

    The mutex protects scheduling metadata only. Waiting never blocks an event
    loop or holds a thread mutex across an await.
    """

    def __init__(self) -> None:
        self._mutex = threading.RLock()
        self._active: dict[asyncio.Task[Any], _Request] = {}
        self._waiting: list[_Request] = []

    @staticmethod
    def _current_task() -> asyncio.Task[Any] | None:
        try:
            return asyncio.current_task()
        except RuntimeError:
            return None

    def held_requirements(self) -> _Requirements | None:
        """Return a copy of the current task's grants, or ``None``."""
        owner = self._current_task()
        with self._mutex:
            request = self._active.get(owner) if owner is not None else None
            return None if request is None else dict(request.requirements)

    def covers(self, requirements: _Requirements) -> bool:
        """Return whether the current task already owns every requested grant."""
        held = self.held_requirements()
        return held is not None and _covers(held, requirements)

    @staticmethod
    def _resolve(request: _Request) -> None:
        if not request.ready.done():
            request.ready.set_result(None)

    def _drain(self) -> None:
        """Grant every eligible waiter while the metadata mutex is held."""
        blocked: list[_Request] = []
        for request in self._waiting:
            if request.ready.cancelled() or request.loop.is_closed():
                continue
            if any(
                _conflicts(request.requirements, active.requirements)
                for active in self._active.values()
            ) or any(
                _conflicts(request.requirements, earlier.requirements)
                for earlier in blocked
            ):
                blocked.append(request)
                continue

            self._active[request.owner] = request
            try:
                request.loop.call_soon_threadsafe(self._resolve, request)
            except RuntimeError:
                # The owning loop may close between is_closed() and wake-up.
                self._active.pop(request.owner, None)
        self._waiting = blocked

    @asynccontextmanager
    async def hold(self, requirements: _Requirements) -> AsyncIterator[None]:
        """
        Hold all grants until the context exits, releasing on cancellation.

        Nested scopes must be covered by this task's existing grants. Rejecting
        expansion and read-to-write upgrades prevents deadlocks between scopes.

        :param requirements: Element keys mapped to their exclusive/shared mode.
        :raises ValueError: If keys/modes are invalid or a nested scope expands.
        """
        requirements = dict(requirements)
        if any(
            not isinstance(key, tuple)
            or len(key) != 2
            or not all(isinstance(part, str) for part in key)
            or not isinstance(write, bool)
            for key, write in requirements.items()
        ):
            raise ValueError("Lock requirements must map (namespace, id) strings to booleans.")

        owner = self._current_task()
        if owner is None:
            raise ValueError("Element locks require an asyncio task.")
        request: _Request | None = None
        with self._mutex:
            held = self._active.get(owner)
            if held is not None:
                if not _covers(held.requirements, requirements):
                    raise ValueError(
                        "Nested lock scopes cannot add elements or upgrade read grants to writes."
                    )
            else:
                loop = asyncio.get_running_loop()
                request = _Request(owner, requirements, loop, loop.create_future())
                self._waiting.append(request)
                self._drain()

        if request is None:
            yield
            return

        try:
            await request.ready
            yield
        finally:
            with self._mutex:
                if self._active.get(owner) is request:
                    self._active.pop(owner)
                if request in self._waiting:
                    self._waiting.remove(request)
                self._drain()
