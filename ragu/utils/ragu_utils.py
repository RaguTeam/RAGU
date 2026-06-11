import asyncio
import functools
import json
import logging
import time
from collections.abc import Awaitable, Collection, MutableMapping
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from dataclasses import is_dataclass, asdict
from hashlib import md5
from pathlib import Path
from typing import Any, Iterable, Iterator
from typing import Callable, TypeVar, cast
from typing import List

import loguru
import numpy as np
import numpy.typing as npt
from diskcache import Index  # pyright: ignore[reportMissingTypeStubs]

from ragu.common.logger import logger

FLOATS = npt.NDArray[np.floating[Any]]
"""A typization for numpy array of floats"""

INTS = npt.NDArray[np.integer[Any]]
"""A typization for numpy array of integers"""

_dish_caches: dict[str, Index] = {}

def get_disk_cache(dir: str | Path) -> MutableMapping[str, Any]:
    """Get or create a DiskCache by a directory name.
    Cache is shared between multiple `get_disk_cache` calls.
    """
    path = str(Path(dir).resolve())
    if (cache := _dish_caches.get(path, None)):
        return cache
    _dish_caches[path] = cache = Index(path, sqlite_synchronous=2)
    return cache


T_fn = TypeVar('T_fn', bound=Callable[..., Awaitable[Any]])

def attach_async_contexts(
    func: T_fn,
    *contexts: AbstractAsyncContextManager[Any],
) -> T_fn:
    """Wraps the `func` into the given async contexts."""
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        # logger.debug('attach_async_contexts: entering context...')
        async with AsyncExitStack() as stack:
            for mgr in contexts:
                await stack.enter_async_context(mgr)
            # logger.debug('attach_async_contexts: entered context!')
            return await func(*args, **kwargs)
            
    return cast(T_fn, wrapper)

def save_args_on_exception(func: T_fn, storage: MutableMapping[str, Any]) -> T_fn:
    """Wraps an async function. If it raises an exception, saves
    the input args and kwargs into a global dict before re-raising.
    """
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            current_time = str(time.time_ns())
            logger.debug(f'Saved args that caused {e.__class__.__name__} as {current_time}')
            storage[current_time] = {
                'function_name': func.__qualname__,
                'args': args,
                'kwargs': kwargs,
                'exception': e,
            }
            raise
            
    return cast(T_fn, wrapper)


class LoguruAdapter(logging.Logger):
    # is neeed where some tool requires a logging.Logger, but we have loguru
    def __init__(self, name: str):
        super().__init__(name)
        
    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False): # type: ignore
        # We override the internal _log method to intercept standard logging calls
        # and redirect them to loguru.
        
        # Map integer levels to loguru level names/values
        # Note: stacklevel=2 usually helps valid source file reporting
        loguru_opts = loguru.logger.opt(depth=2, exception=exc_info) # type: ignore
        
        # Handle standard logging's printf style formatting (%s) 
        # vs Loguru's mechanism
        try:
             # Standard logging expands args eagerly usually, 
             # but here we might just pass the message formatted
             formatted_msg = msg % args if args else msg # type: ignore
        except TypeError:
             formatted_msg = msg # Fallback # type: ignore

        loguru_opts.log(level, formatted_msg) # type: ignore


def compute_mdhash_id(*args: str, prefix: str = '', **kwargs: str) -> str:
    """A unique string hash for the given combination of arguments.
    Invariant to kwargs order.
    """
    string = ''
    for x in args:
        assert isinstance(x, str)
        string += '\0' + x
    for key, x in sorted(kwargs.items(), key=lambda item: item[0]):
        assert isinstance(x, str)
        string += '\0' + key + '\1' + x
    return prefix + md5(string.encode()).hexdigest()


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    try:
        current_loop = asyncio.get_event_loop()
        if current_loop.is_closed():
            raise RuntimeError()
        return current_loop

    except RuntimeError:
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        return new_loop


def read_text_from_files(directory: str | Path, file_extensions: Collection[str] | None = None) -> List[str]:
    texts: list[str] = []
    directory = Path(directory)
    for file_path in directory.rglob('*'):
        if file_path.is_file() and (file_extensions is None or file_path.suffix in file_extensions):
            try:
                with file_path.open('r', encoding='utf-8') as f:
                    texts.append(f.read())
            except (UnicodeDecodeError, PermissionError) as e:
                print(f"⚠️ Cannot read file {file_path}: {e}")

    return texts

def serialize(obj: Any) -> Any:
    if obj is None or isinstance(obj, (int, float, str, bool)):
        return obj

    if is_dataclass(obj):
        return {k: serialize(v) for k, v in asdict(obj).items()}

    if isinstance(obj, dict):
        return {serialize(k): serialize(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [serialize(v) for v in obj]

    if hasattr(obj, "__dict__"):
        return {
            k: serialize(v)
            for k, v in vars(obj).items()
            if not k.startswith("_")  # опционально
        }

    if hasattr(obj, "__slots__"):
        return {
            slot: serialize(getattr(obj, slot))
            for slot in obj.__slots__
            if hasattr(obj, slot)
        }

    return str(obj)

def serialized_size(obj) -> int:
    """
    Estimate size of object after JSON serialization (bytes).
    """
    try:
        return len(json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    except TypeError:
        return len(str(obj).encode("utf-8"))

T = TypeVar("T")
def split_on_batches_by_size(
    objects: Iterable[T],
    max_size_in_bytes: int,
) -> Iterator[List[T]]:
    current_batch: list[T] = []
    current_size = 0

    for obj in objects:
        size = serialized_size(obj)
        if size > max_size_in_bytes:
            if current_batch:
                yield current_batch
                current_batch = []
                current_size = 0
            yield [obj]
            continue

        if current_size + size > max_size_in_bytes:
            if current_batch:
                yield current_batch
            current_batch = [obj]
            current_size = size
        else:
            current_batch.append(obj)
            current_size += size

    if current_batch:
        yield current_batch

