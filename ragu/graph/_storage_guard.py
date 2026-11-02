"""Short adapter-call guards for backends with shared in-memory state."""

import asyncio
import threading
from concurrent.futures import Future
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
from functools import wraps
from inspect import iscoroutinefunction
from typing import Any

from ragu.graph._locks import _ElementLocks
from ragu.storage.base_storage import BaseStorage


@dataclass(slots=True)
class _StorageLease:
    active: bool = True
    pending: dict[Future[None], asyncio.Task[Any]] = field(default_factory=dict)
    mutex: Any = field(default_factory=threading.Lock)

    def borrow(self) -> Future[None] | None:
        """Register an internal call only while its enclosing guard is live."""
        task = asyncio.current_task()
        assert task is not None
        with self.mutex:
            if not self.active:
                return None
            completed: Future[None] = Future()
            self.pending[completed] = task
            return completed

    def release(self, completed: Future[None]) -> None:
        with self.mutex:
            self.pending.pop(completed)
        completed.set_result(None)

    def _cancel(self) -> None:
        with self.mutex:
            tasks = set(self.pending.values())
        for task in tasks:
            if not task.done():
                task.get_loop().call_soon_threadsafe(task.cancel)

    async def finish(self, cancel: bool = False) -> None:
        """Finish delegated calls before the outer guard can be released."""
        interrupted = False
        while True:
            with self.mutex:
                completed = list(self.pending)
                if not completed:
                    self.active = False
                    break
            if cancel:
                self._cancel()
            waiting = asyncio.gather(*[asyncio.wrap_future(call) for call in completed])
            while True:
                try:
                    await asyncio.shield(waiting)
                    break
                except asyncio.CancelledError:
                    interrupted = cancel = True
                    self._cancel()
        if interrupted:
            raise asyncio.CancelledError


def _guard_storage(storage: BaseStorage) -> None:
    """Keep adapter instances/types intact while synchronizing their async API."""
    locks = _ElementLocks()
    current_lease: ContextVar[_StorageLease | None] = ContextVar("ragu_storage_lease", default=None)
    read_methods = {
        "query", "get_all_ids", "get_points_by_ids", "get_payloads_by_ids",
        "all_keys", "get_by_id", "get_by_ids", "filter_keys", "edges_degrees",
        "get_nodes", "get_edges", "get_all_edges_for_nodes", "get_all_nodes",
        "get_all_edges",
    }
    write_methods = {
        "upsert", "delete", "drop", "upsert_nodes", "upsert_edges",
        "delete_nodes", "delete_edges", "index_start_callback",
        "index_done_callback", "query_done_callback", "close",
    }

    def guarded(method: Any, write: bool) -> Any:
        @wraps(method)
        async def call(*args: Any, **kwargs: Any) -> Any:
            inherited = current_lease.get()
            borrowed = None if inherited is None else inherited.borrow()
            if borrowed is not None:
                # Public adapter methods may delegate to each other, including
                # through child tasks. They share the enclosing adapter call.
                try:
                    return await method(*args, **kwargs)
                finally:
                    inherited.release(borrowed)
            async with locks.hold({("backend", "*"): write}):
                lease = _StorageLease()
                token = current_lease.set(lease)
                try:
                    # Adapters may retain payloads or return live containers.
                    if write:
                        args, kwargs = deepcopy((args, kwargs))
                    try:
                        result = await method(*args, **kwargs)
                    except BaseException:
                        await lease.finish(cancel=True)
                        raise
                    await lease.finish()
                    return result if write else deepcopy(result)
                finally:
                    lease.active = False
                    current_lease.reset(token)
        return call

    for name in read_methods | write_methods:
        method = getattr(storage, name, None)
        if iscoroutinefunction(method):
            setattr(storage, name, guarded(method, name in write_methods))
