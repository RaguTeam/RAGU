from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

from diskcache_community import Index, SafeDisk


_CACHES: dict[str, Index] = {}

def get_cache(dir: str | Path) -> MutableMapping[str, Any]:
    """Get or create a key-value cache which uses the specified directory."""
    dir = str(dir)
    if (cache := _CACHES.get(dir, None)) is None:
        cache = _CACHES[dir] = Index(dir, disk=SafeDisk.with_options(on_error='miss'), sqlite_synchronous=2)
    return cache