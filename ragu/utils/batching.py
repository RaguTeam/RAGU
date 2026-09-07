"""
Bounded fan-out shared by the media backends.

Every one of them needs the same shape — run a few items at once, then the next
few — and the same two concessions to running a model on a local GPU: show that
something is happening during calls that take minutes, and survive a batch that
fails as a whole.
"""
import asyncio
import gc
import sys
from typing import (
    Any,
    Awaitable,
    Callable,
    List,
    Literal,
    Sequence,
    TypeVar,
    overload,
)

from tqdm import tqdm

from ragu.common.batch_generator import BatchGenerator
from ragu.common.logger import logger

T = TypeVar("T")
R = TypeVar("R")


def release_gpu_memory() -> None:
    """
    Hand cached GPU blocks back before retrying a batch that exhausted them.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return

    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - freeing memory must not mask the real failure
        pass


@overload
async def run_batched(
    items: Sequence[T],
    call: Callable[[T], Awaitable[R]],
    batch_size: int,
    desc: str | None = ...,
    continue_on_error: Literal[False] = ...,
) -> List[R]: ...


@overload
async def run_batched(
    items: Sequence[T],
    call: Callable[[T], Awaitable[R]],
    batch_size: int,
    desc: str | None = ...,
    continue_on_error: bool = ...,
) -> List[R | None]: ...


async def run_batched(
    items: Sequence[T],
    call: Callable[[T], Awaitable[R]],
    batch_size: int,
    desc: str | None = None,
    continue_on_error: bool = False,
) -> List[Any]:
    """
    Fan a coroutine out over a sequence, `batch_size` items at a time.

    The shape every media backend needs and each used to spell out for itself:
    items within a batch run concurrently, batches run one after the other.
    Handing the whole sequence to :func:`asyncio.gather` instead would put every
    item in flight at once, which is how a hosted backend hits its rate limit
    and a local one runs out of memory.

    Progress goes to a single bar over the items, not one per batch, and it is
    updated as each item finishes rather than as each batch does — the point is
    to show movement during the slowest operations in the library, where a
    batch may take minutes.

    `continue_on_error` follows
    :meth:`~ragu.models.llm.LLM.batch_chat_completion`: a failed item is logged
    and comes back as ``None``, so the caller can tell it apart from a
    legitimately empty result and keeps everything that did succeed.

    :param items: Items to process, in order.
    :type items: Sequence[T]
    :param call: Coroutine function applied to each item.
    :type call: Callable[[T], Awaitable[R]]
    :param batch_size: Items processed concurrently per batch.
    :type batch_size: int
    :param desc: Progress bar description.
    :type desc: str | None
    :param continue_on_error: Log and yield ``None`` for a failed item instead
        of raising on the first failure. Left at ``False`` no item can be
        ``None``, and the overloads narrow the result to ``List[R]`` so callers
        that cannot fail need no guard for a case that cannot arise.
    :type continue_on_error: bool
    :returns: Results in input order, ``None`` where an item failed and
        `continue_on_error` is set.
    :rtype: List[R | None]
    :raises ValueError: If `batch_size` is not positive.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    ordered = list(items)
    results: List[Any] = []

    with tqdm(total=len(ordered), desc=desc) as progress:

        async def tracked(item: T) -> R:
            try:
                return await call(item)
            finally:
                progress.update(1)

        for batch in BatchGenerator(ordered, batch_size).get_batches():
            outcomes = await asyncio.gather(
                *(tracked(item) for item in batch),
                return_exceptions=continue_on_error,
            )

            base = len(results)
            for offset, outcome in enumerate(outcomes):
                if isinstance(outcome, BaseException):
                    logger.warning(
                        "Batched call failed for item {}: {}: {}",
                        base + offset,
                        type(outcome).__name__,
                        outcome,
                    )
                    results.append(None)
                else:
                    results.append(outcome)

    return results
