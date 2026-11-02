"""Short adapter-call guards for backends with shared in-memory state."""

from copy import deepcopy
from functools import wraps
from inspect import iscoroutinefunction
from typing import Any

from ragu.graph._locks import _ElementLocks
from ragu.storage.base_storage import BaseStorage


def _guard_storage(storage: BaseStorage) -> None:
    """Keep adapter instances/types intact while synchronizing their async API."""
    locks = _ElementLocks()
    read_methods = {
        "query", "get_all_ids", "get_points_by_ids", "get_payloads_by_ids",
        "all_keys", "get_by_id", "get_by_ids", "filter_keys", "edges_degrees",
        "get_nodes", "get_edges", "get_all_edges_for_nodes", "get_all_nodes",
        "get_all_edges",
    }

    def guarded(method: Any, write: bool) -> Any:
        @wraps(method)
        async def call(*args: Any, **kwargs: Any) -> Any:
            async with locks.hold({("backend", "*"): write}):
                # Adapters may retain payloads or return live nested containers.
                if write:
                    args, kwargs = deepcopy((args, kwargs))
                result = await method(*args, **kwargs)
                return result if write else deepcopy(result)
        return call

    for name in dir(storage):
        if not name.startswith("_"):
            method = getattr(storage, name)
            if iscoroutinefunction(method):
                setattr(storage, name, guarded(method, name not in read_methods))
