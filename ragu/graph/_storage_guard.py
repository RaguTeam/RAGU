"""Short adapter-call guards for backends with shared in-memory state."""

from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from functools import wraps
from inspect import iscoroutinefunction
from typing import Any

from ragu.graph._locks import _ElementLocks
from ragu.storage.base_storage import BaseStorage


@dataclass(slots=True)
class _StorageLease:
    active: bool = True


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
            if inherited is not None and inherited.active:
                # Public adapter methods may delegate to each other, including
                # through child tasks. They share the enclosing adapter call.
                return await method(*args, **kwargs)
            async with locks.hold({("backend", "*"): write}):
                lease = _StorageLease()
                token = current_lease.set(lease)
                try:
                    # Adapters may retain payloads or return live containers.
                    if write:
                        args, kwargs = deepcopy((args, kwargs))
                    result = await method(*args, **kwargs)
                    return result if write else deepcopy(result)
                finally:
                    lease.active = False
                    current_lease.reset(token)
        return call

    for name in read_methods | write_methods:
        method = getattr(storage, name, None)
        if iscoroutinefunction(method):
            setattr(storage, name, guarded(method, name in write_methods))
